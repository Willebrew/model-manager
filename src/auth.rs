//! Dashboard authentication: Argon2id password + TOTP, cookie sessions,
//! per-IP lockout. No plaintext secrets in the config — the TOTP secret is
//! AES-256-GCM-encrypted under `~/.config/model-manager/secret.key` (0600,
//! generated on first use), and API/admin keys are stored as SHA-256 hashes.

use anyhow::{Context, Result};
use hmac::{Hmac, Mac};
use sha2::Sha256;
use std::collections::HashMap;
use std::net::IpAddr;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use aes_gcm::aead::{Aead, KeyInit, OsRng};
use aes_gcm::{AeadCore, Aes256Gcm, Key, Nonce};
use base64::Engine;

/// Session idle timeout and absolute lifetime.
pub const SESSION_IDLE: Duration = Duration::from_secs(15 * 60);
pub const SESSION_ABS: Duration = Duration::from_secs(8 * 60 * 60);
/// Failed logins per peer IP before a 15-minute lock.
pub const LOCKOUT_FAILS: u32 = 5;
pub const LOCKOUT_TIME: Duration = Duration::from_secs(15 * 60);
/// TOTP step window (±1) and step size.
pub const TOTP_STEP: u64 = 30;
pub const TOTP_WINDOW: u64 = 1;

pub struct Session {
    pub created: Instant,
    pub last: Instant,
}

struct FailRec {
    fails: u32,
    locked_until: Option<Instant>,
    last: Instant,
}

/// All mutable auth state, held in `AppState`.
pub struct AuthState {
    pub sessions: Mutex<HashMap<String, Session>>,
    fails: Mutex<HashMap<IpAddr, FailRec>>,
    /// Highest TOTP step already accepted (anti-replay).
    last_totp_step: Mutex<u64>,
}

impl Default for AuthState {
    fn default() -> Self {
        Self {
            sessions: Mutex::new(HashMap::new()),
            fails: Mutex::new(HashMap::new()),
            last_totp_step: Mutex::new(0),
        }
    }
}

impl AuthState {
    /// Mint a session, returning the cookie token.
    pub fn new_session(&self) -> String {
        let mut buf = [0u8; 32];
        use rand::RngCore;
        rand::thread_rng().fill_bytes(&mut buf);
        let id = buf.iter().map(|b| format!("{b:02x}")).collect::<String>();
        let now = Instant::now();
        if let Ok(mut s) = self.sessions.lock() {
            s.insert(
                id.clone(),
                Session {
                    created: now,
                    last: now,
                },
            );
        }
        id
    }

    /// Validate + refresh a session token.
    pub fn check_session(&self, token: &str) -> bool {
        let now = Instant::now();
        match self.sessions.lock() {
            Ok(mut s) => match s.get_mut(token) {
                Some(sess)
                    if now.duration_since(sess.last) <= SESSION_IDLE
                        && now.duration_since(sess.created) <= SESSION_ABS =>
                {
                    sess.last = now;
                    true
                }
                Some(_) => {
                    s.remove(token);
                    false
                }
                None => false,
            },
            Err(_) => false,
        }
    }

    pub fn drop_session(&self, token: &str) {
        if let Ok(mut s) = self.sessions.lock() {
            s.remove(token);
        }
    }

    /// Is this peer currently locked out?
    pub fn is_locked(&self, ip: IpAddr) -> bool {
        match self.fails.lock() {
            Ok(m) => m
                .get(&ip)
                .and_then(|r| r.locked_until)
                .map(|t| Instant::now() < t)
                .unwrap_or(false),
            Err(_) => true, // fail closed
        }
    }

    /// Record a login outcome; returns seconds to back off (global-ish).
    pub fn record(&self, ip: IpAddr, ok: bool) -> u64 {
        let Ok(mut m) = self.fails.lock() else {
            return 0;
        };
        if ok {
            m.remove(&ip);
            return 0;
        }
        let now = Instant::now();
        let rec = m.entry(ip).or_insert(FailRec {
            fails: 0,
            locked_until: None,
            last: now,
        });
        // Decay stale failures after the lockout window.
        if now.duration_since(rec.last) > LOCKOUT_TIME {
            rec.fails = 0;
            rec.locked_until = None;
        }
        rec.fails += 1;
        rec.last = now;
        if rec.fails >= LOCKOUT_FAILS {
            rec.locked_until = Some(now + LOCKOUT_TIME);
        }
        (rec.fails.min(8) as u64) * 250
    }

    /// Anti-replay check for a TOTP step that verified.
    pub fn claim_totp_step(&self, step: u64) -> bool {
        match self.last_totp_step.lock() {
            Ok(mut last) => {
                if step <= *last {
                    false
                } else {
                    *last = step;
                    true
                }
            }
            Err(_) => false,
        }
    }
}

// ---------- password (Argon2id) ----------

pub fn hash_password(password: &str) -> Result<String> {
    use argon2::password_hash::{PasswordHasher, SaltString};
    let params = argon2::Params::new(65_536, 3, 1, None).map_err(|e| anyhow::anyhow!("argon2 params: {e}"))?;
    let argon2 = argon2::Argon2::new(
        argon2::Algorithm::Argon2id,
        argon2::Version::V0x13,
        params,
    );
    let salt = SaltString::generate(&mut OsRng);
    Ok(argon2
        .hash_password(password.as_bytes(), &salt)
        .map_err(|e| anyhow::anyhow!("argon2 hash: {e}"))?
        .to_string())
}

pub fn verify_password(hash: &str, password: &str) -> bool {
    use argon2::password_hash::{PasswordHash, PasswordVerifier};
    let Ok(parsed) = PasswordHash::new(hash) else {
        return false;
    };
    argon2::Argon2::default()
        .verify_password(password.as_bytes(), &parsed)
        .is_ok()
}

// ---------- encrypted secrets (AES-256-GCM under secret.key) ----------

fn key_path() -> PathBuf {
    crate::config::Config::dir().join("secret.key")
}

/// Load or generate the 256-bit wrapping key (0600 file).
pub fn secret_key() -> Result<[u8; 32]> {
    let p = key_path();
    if p.exists() {
        let raw = std::fs::read(&p).with_context(|| format!("reading {}", p.display()))?;
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(String::from_utf8_lossy(&raw).trim())
            .context("decoding secret.key")?;
        return bytes
            .try_into()
            .map_err(|_| anyhow::anyhow!("secret.key is not 32 bytes"));
    }
    let mut k = [0u8; 32];
    use rand::RngCore;
    rand::thread_rng().fill_bytes(&mut k);
    let enc = base64::engine::general_purpose::STANDARD.encode(k);
    std::fs::write(&p, enc).with_context(|| format!("writing {}", p.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o600))
            .with_context(|| format!("chmod 600 {}", p.display()))?;
    }
    Ok(k)
}

/// Encrypt arbitrary bytes; returns base64(nonce || ciphertext).
pub fn seal(key: &[u8; 32], plaintext: &[u8]) -> Result<String> {
    let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(key));
    let nonce = Aes256Gcm::generate_nonce(&mut OsRng);
    let ct = cipher
        .encrypt(&nonce, plaintext)
        .map_err(|e| anyhow::anyhow!("encrypt: {e}"))?;
    let mut blob = nonce.to_vec();
    blob.extend(ct);
    Ok(base64::engine::general_purpose::STANDARD.encode(blob))
}

/// Decrypt a base64(nonce || ciphertext) blob.
pub fn open(key: &[u8; 32], blob_b64: &str) -> Result<Vec<u8>> {
    let blob = base64::engine::general_purpose::STANDARD
        .decode(blob_b64.trim())
        .context("base64")?;
    if blob.len() < 12 + 16 {
        anyhow::bail!("sealed blob too short");
    }
    let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(key));
    let nonce = Nonce::from_slice(&blob[..12]);
    cipher
        .decrypt(nonce, &blob[12..])
        .map_err(|e| anyhow::anyhow!("decrypt: {e}"))
}

// ---------- TOTP (RFC 6238, HMAC-SHA1, 6 digits, 30 s) ----------

/// Generate a fresh 20-byte TOTP secret.
pub fn gen_totp_secret() -> [u8; 20] {
    let mut s = [0u8; 20];
    use rand::RngCore;
    rand::thread_rng().fill_bytes(&mut s);
    s
}

fn hotp(secret: &[u8], counter: u64) -> u32 {
    let mut mac = <Hmac<sha1::Sha1> as Mac>::new_from_slice(secret).expect("hmac key");
    mac.update(&counter.to_be_bytes());
    let out = mac.finalize().into_bytes();
    let off = (out[19] & 0x0f) as usize;
    ((u32::from(out[off]) & 0x7f) << 24
        | (u32::from(out[off + 1])) << 16
        | (u32::from(out[off + 2])) << 8
        | u32::from(out[off + 3]))
        % 1_000_000
}

/// Verify a 6-digit code within ±TOTP_WINDOW steps. Returns the matching
/// step on success (caller must claim it via `claim_totp_step`).
pub fn verify_totp(secret: &[u8], code: &str, at_unix: u64) -> Option<u64> {
    let code: u32 = code.trim().parse().ok()?;
    let step = at_unix / TOTP_STEP;
    for s in (step.saturating_sub(TOTP_WINDOW))..=(step + TOTP_WINDOW) {
        if hotp(secret, s) == code % 1_000_000 {
            return Some(s);
        }
    }
    None
}

/// RFC 4648 base32 (no padding) — for the otpauth:// URI only.
pub fn base32(data: &[u8]) -> String {
    const A: &[u8; 32] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";
    let mut out = String::new();
    let (mut buf, mut bits) = (0u32, 0u32);
    for &b in data {
        buf = (buf << 8) | b as u32;
        bits += 8;
        while bits >= 5 {
            out.push(A[((buf >> (bits - 5)) & 0x1f) as usize] as char);
            bits -= 5;
        }
    }
    if bits > 0 {
        out.push(A[((buf << (5 - bits)) & 0x1f) as usize] as char);
    }
    out
}

pub fn totp_uri(secret: &[u8]) -> String {
    format!(
        "otpauth://totp/model-manager:grace?secret={}&issuer=model-manager&digits=6&period={}",
        base32(secret),
        TOTP_STEP
    )
}

// ---------- API keys ----------

pub fn sha256_hex(data: &[u8]) -> String {
    use sha2::Digest;
    let d = Sha256::digest(data);
    d.iter().map(|b| format!("{b:02x}")).collect()
}

/// Constant-time equality for hex digests.
pub fn ct_eq(a: &str, b: &str) -> bool {
    use subtle::ConstantTimeEq;
    if a.len() != b.len() {
        return false;
    }
    a.as_bytes().ct_eq(b.as_bytes()).into()
}

/// New gateway key: `mmk_<id8>_<32 random bytes, base64url>`.
/// Returns (id, full secret to show once, sha256 of the secret part).
pub fn gen_gateway_key() -> (String, String, String) {
    use rand::RngCore;
    let mut idb = [0u8; 4];
    rand::thread_rng().fill_bytes(&mut idb);
    let id = idb.iter().map(|b| format!("{b:02x}")).collect::<String>();
    let mut sec = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut sec);
    let secret = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(sec);
    let hash = sha256_hex(secret.as_bytes());
    (id.clone(), format!("mmk_{id}_{secret}"), hash)
}

/// Split `mmk_<id>_<secret>` into parts for lookup.
pub fn parse_gateway_key(token: &str) -> Option<(String, String)> {
    let rest = token.strip_prefix("mmk_")?;
    let (id, secret) = rest.split_once('_')?;
    if id.is_empty() || secret.is_empty() {
        return None;
    }
    Some((id.to_string(), secret.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn password_roundtrip() {
        let h = hash_password("hunter2").unwrap();
        assert!(verify_password(&h, "hunter2"));
        assert!(!verify_password(&h, "hunter3"));
        assert!(!verify_password("not-a-hash", "hunter2"));
    }

    #[test]
    fn totp_window_and_replay() {
        let secret = gen_totp_secret();
        let t = 1_700_000_000u64;
        let step = t / TOTP_STEP;
        let code = format!("{:06}", hotp(&secret, step));
        // Current step verifies; a made-up code does not.
        assert_eq!(verify_totp(&secret, &code, t), Some(step));
        assert_eq!(verify_totp(&secret, "000000", t), None);
        // ±1 step window.
        let prev = format!("{:06}", hotp(&secret, step - 1));
        let next = format!("{:06}", hotp(&secret, step + 1));
        assert!(verify_totp(&secret, &prev, t).is_some());
        assert!(verify_totp(&secret, &next, t).is_some());
        // Replay: claiming the same step twice fails.
        let st = AuthState::default();
        assert!(st.claim_totp_step(step));
        assert!(!st.claim_totp_step(step));
    }

    #[test]
    fn session_expiry() {
        let st = AuthState::default();
        let tok = st.new_session();
        assert!(st.check_session(&tok));
        // Idle expiry: backdate last-seen past SESSION_IDLE.
        {
            let mut m = st.sessions.lock().unwrap();
            let s = m.get_mut(&tok).unwrap();
            s.last = Instant::now() - SESSION_IDLE - Duration::from_secs(1);
        }
        assert!(!st.check_session(&tok));
        // Absolute expiry.
        let tok2 = st.new_session();
        {
            let mut m = st.sessions.lock().unwrap();
            let s = m.get_mut(&tok2).unwrap();
            s.created = Instant::now() - SESSION_ABS - Duration::from_secs(1);
        }
        assert!(!st.check_session(&tok2));
    }

    #[test]
    fn lockout_after_five() {
        let st = AuthState::default();
        let ip: IpAddr = "100.64.1.2".parse().unwrap();
        for _ in 0..LOCKOUT_FAILS {
            st.record(ip, false);
        }
        assert!(st.is_locked(ip));
        // A different IP is unaffected.
        assert!(!st.is_locked("100.64.1.3".parse().unwrap()));
        // Success clears.
        let st2 = AuthState::default();
        st2.record(ip, false);
        st2.record(ip, true);
        assert!(!st2.is_locked(ip));
    }

    #[test]
    fn key_parse_and_ct_eq() {
        let (id, full, hash) = gen_gateway_key();
        assert!(full.starts_with(&format!("mmk_{id}_")));
        let (pid, secret) = parse_gateway_key(&full).unwrap();
        assert_eq!(pid, id);
        assert!(ct_eq(&sha256_hex(secret.as_bytes()), &hash));
        assert!(!ct_eq(&hash, &sha256_hex(b"wrong")));
        // Old fallback forms are not mmk_ keys at all.
        assert!(parse_gateway_key("local").is_none());
        assert!(parse_gateway_key("sk-local").is_none());
        assert!(parse_gateway_key("sk-mm-abcdef").is_none());
        assert!(parse_gateway_key("Willrocks18!").is_none());
    }

    #[test]
    fn seal_open_roundtrip() {
        let key = [7u8; 32];
        let blob = seal(&key, b"totp-secret").unwrap();
        assert_eq!(open(&key, &blob).unwrap(), b"totp-secret");
        assert!(open(&[8u8; 32], &blob).is_err());
    }

    #[test]
    fn base32_rfc4648() {
        // RFC 4648 test vector: "foobar" -> MZXW6YTBOI======
        assert_eq!(base32(b"foobar"), "MZXW6YTBOI");
    }
}
