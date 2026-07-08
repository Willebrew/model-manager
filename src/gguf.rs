//! Minimal GGUF metadata reader — just enough to estimate a model's memory
//! footprint (weights from file sizes, KV cache from hyperparameters). It
//! streams the header and seeks past large arrays (tokenizer vocab/merges)
//! rather than loading the whole multi-GB file.

use anyhow::{anyhow, bail, Context, Result};
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF" little-endian

// GGUF metadata value types.
const T_UINT8: u32 = 0;
const T_INT8: u32 = 1;
const T_UINT16: u32 = 2;
const T_INT16: u32 = 3;
const T_UINT32: u32 = 4;
const T_INT32: u32 = 5;
const T_FLOAT32: u32 = 6;
const T_BOOL: u32 = 7;
const T_STRING: u32 = 8;
const T_ARRAY: u32 = 9;
const T_UINT64: u32 = 10;
const T_INT64: u32 = 11;
const T_FLOAT64: u32 = 12;

#[derive(Debug, Clone, Default)]
pub struct GgufInfo {
    pub arch: String,
    pub n_layer: u64,
    pub n_head: u64,
    pub n_head_kv: u64,
    pub key_length: u64,
    pub value_length: u64,
    pub embedding_length: u64,
    pub context_length: u64,
}

impl GgufInfo {
    /// Effective per-head K/V dimensions, falling back to embedding/head_count
    /// when the model doesn't specify key/value length explicitly.
    fn head_dims(&self) -> (u64, u64) {
        if self.key_length > 0 && self.value_length > 0 {
            (self.key_length, self.value_length)
        } else if self.n_head > 0 && self.embedding_length > 0 {
            let d = self.embedding_length / self.n_head;
            (d, d)
        } else {
            (128, 128) // last-resort default
        }
    }

    /// Estimated KV-cache bytes for a given context length and cache quant type.
    pub fn kv_bytes(&self, context: u64, kv_type: &str) -> u64 {
        let (kd, vd) = self.head_dims();
        let heads = self.n_head_kv.max(1);
        let layers = self.n_layer.max(1);
        let per_elem = kv_bytes_per_elem(kv_type);
        let elems = layers * context * heads * (kd + vd);
        (elems as f64 * per_elem) as u64
    }
}

/// Approximate bytes-per-element for a KV cache type (block-quant overhead
/// included where relevant).
pub fn kv_bytes_per_elem(kv_type: &str) -> f64 {
    match kv_type {
        "f32" => 4.0,
        "f16" | "bf16" => 2.0,
        "q8_0" => 34.0 / 32.0,   // 1.0625
        "q5_1" => 24.0 / 32.0,   // 0.75
        "q5_0" => 22.0 / 32.0,   // 0.6875
        "q4_1" => 20.0 / 32.0,   // 0.625
        "q4_0" | "iq4_nl" => 18.0 / 32.0, // 0.5625
        _ => 2.0,
    }
}

struct Reader<R: Read + Seek> {
    inner: R,
}

impl<R: Read + Seek> Reader<R> {
    fn u32(&mut self) -> Result<u32> {
        let mut b = [0u8; 4];
        self.inner.read_exact(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }
    fn u64(&mut self) -> Result<u64> {
        let mut b = [0u8; 8];
        self.inner.read_exact(&mut b)?;
        Ok(u64::from_le_bytes(b))
    }
    fn skip(&mut self, n: i64) -> Result<()> {
        self.inner.seek(SeekFrom::Current(n))?;
        Ok(())
    }
    fn string(&mut self) -> Result<String> {
        let len = self.u64()? as usize;
        // Guard against absurd lengths from a malformed/misaligned read.
        if len > 64 * 1024 * 1024 {
            bail!("gguf string length {} unreasonable (misaligned parse?)", len);
        }
        let mut buf = vec![0u8; len];
        self.inner.read_exact(&mut buf)?;
        Ok(String::from_utf8_lossy(&buf).into_owned())
    }

    /// Read a scalar of the given type as an f64 (for numeric metadata) or None
    /// for non-numeric. Advances the cursor past the value in all cases.
    fn scalar_as_f64(&mut self, vtype: u32) -> Result<Option<f64>> {
        let v = match vtype {
            T_UINT8 | T_INT8 | T_BOOL => {
                let mut b = [0u8; 1];
                self.inner.read_exact(&mut b)?;
                Some(b[0] as f64)
            }
            T_UINT16 | T_INT16 => {
                let mut b = [0u8; 2];
                self.inner.read_exact(&mut b)?;
                Some(u16::from_le_bytes(b) as f64)
            }
            T_UINT32 | T_INT32 => Some(self.u32()? as f64),
            T_FLOAT32 => {
                let mut b = [0u8; 4];
                self.inner.read_exact(&mut b)?;
                Some(f32::from_le_bytes(b) as f64)
            }
            T_UINT64 | T_INT64 => Some(self.u64()? as f64),
            T_FLOAT64 => {
                let mut b = [0u8; 8];
                self.inner.read_exact(&mut b)?;
                Some(f64::from_le_bytes(b))
            }
            T_STRING => {
                let _ = self.string()?;
                None
            }
            _ => bail!("unexpected scalar type {vtype}"),
        };
        Ok(v)
    }

    fn scalar_size(vtype: u32) -> Option<i64> {
        match vtype {
            T_UINT8 | T_INT8 | T_BOOL => Some(1),
            T_UINT16 | T_INT16 => Some(2),
            T_UINT32 | T_INT32 | T_FLOAT32 => Some(4),
            T_UINT64 | T_INT64 | T_FLOAT64 => Some(8),
            _ => None, // string / array: variable
        }
    }
}

/// Parse the interesting hyperparameters from a GGUF file header.
pub fn parse_metadata(path: &Path) -> Result<GgufInfo> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut r = Reader {
        inner: BufReader::new(file),
    };

    let magic = r.u32()?;
    if magic != GGUF_MAGIC {
        bail!("not a GGUF file (bad magic) : {}", path.display());
    }
    let _version = r.u32()?;
    let _tensor_count = r.u64()?;
    let kv_count = r.u64()?;

    let mut info = GgufInfo::default();
    // First pass: read general.architecture and all numeric scalars we might
    // need, keyed by full name. We resolve arch-specific keys after.
    let mut numbers: std::collections::HashMap<String, f64> = std::collections::HashMap::new();

    for _ in 0..kv_count {
        let key = r.string()?;
        let vtype = r.u32()?;
        if vtype == T_ARRAY {
            let elem_type = r.u32()?;
            let count = r.u64()? as i64;
            // We only need the first element of *.head_count_kv arrays.
            if key.ends_with("attention.head_count_kv") && count > 0 {
                if let Some(v) = r.scalar_as_f64(elem_type)? {
                    numbers.insert(key.clone(), v);
                }
                // skip the remaining elements
                if let Some(sz) = Reader::<BufReader<File>>::scalar_size(elem_type) {
                    r.skip(sz * (count - 1))?;
                } else {
                    // array of strings: read+discard each
                    for _ in 0..(count - 1) {
                        let _ = r.string()?;
                    }
                }
            } else {
                // skip whole array
                if let Some(sz) = Reader::<BufReader<File>>::scalar_size(elem_type) {
                    r.skip(sz * count)?;
                } else if elem_type == T_STRING {
                    for _ in 0..count {
                        let _ = r.string()?;
                    }
                } else {
                    bail!("nested arrays not supported (key {key})");
                }
            }
            continue;
        }
        if vtype == T_STRING {
            let s = r.string()?;
            if key == "general.architecture" {
                info.arch = s;
            }
            continue;
        }
        if let Some(v) = r.scalar_as_f64(vtype)? {
            numbers.insert(key, v);
        }
    }

    let arch = if info.arch.is_empty() {
        // Fall back: guess from any key like "<arch>.block_count"
        numbers
            .keys()
            .find_map(|k| k.strip_suffix(".block_count").map(|a| a.to_string()))
            .unwrap_or_default()
    } else {
        info.arch.clone()
    };

    let get = |suffix: &str| -> u64 {
        numbers
            .get(&format!("{arch}.{suffix}"))
            .copied()
            .map(|v| v as u64)
            .unwrap_or(0)
    };

    info.arch = arch.clone();
    info.n_layer = get("block_count");
    info.n_head = get("attention.head_count");
    info.n_head_kv = {
        let v = get("attention.head_count_kv");
        if v == 0 {
            info.n_head // MHA: kv heads == heads
        } else {
            v
        }
    };
    info.key_length = get("attention.key_length");
    info.value_length = get("attention.value_length");
    info.embedding_length = get("embedding_length");
    info.context_length = get("context_length");

    if info.n_layer == 0 {
        return Err(anyhow!(
            "could not read block_count from GGUF (arch='{}')",
            info.arch
        ));
    }
    Ok(info)
}

/// Sum the byte sizes of all shards belonging to a (possibly split) GGUF,
/// given the path to the first shard.
pub fn weights_bytes(first_shard: &Path) -> Result<u64> {
    let shards = shard_paths(first_shard)?;
    let mut total = 0u64;
    for p in &shards {
        let md = std::fs::metadata(p).with_context(|| format!("stat {}", p.display()))?;
        total += md.len();
    }
    Ok(total)
}

/// Enumerate every shard path for a split GGUF, or just the single file.
pub fn shard_paths(first_shard: &Path) -> Result<Vec<PathBuf>> {
    let name = first_shard
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| anyhow!("bad gguf path"))?;
    let dir = first_shard.parent().unwrap_or_else(|| Path::new("."));

    // Match "<base>-<idx>-of-<count>.gguf"
    if let Some((base, count)) = parse_shard_name(name) {
        let mut out = Vec::new();
        for i in 1..=count {
            let fname = format!("{base}-{i:05}-of-{count:05}.gguf");
            out.push(dir.join(fname));
        }
        Ok(out)
    } else {
        Ok(vec![first_shard.to_path_buf()])
    }
}

fn parse_shard_name(name: &str) -> Option<(String, u32)> {
    // e.g. Model-UD-IQ4_XS-00001-of-00004.gguf
    let stem = name.strip_suffix(".gguf")?;
    let idx = stem.rfind("-of-")?;
    let count: u32 = stem[idx + 4..].parse().ok()?;
    let left = &stem[..idx]; // "<base>-00001"
    let dash = left.rfind('-')?;
    let base = &left[..dash];
    Some((base.to_string(), count))
}

/// Total weight bytes for a model path that may be a GGUF (single or sharded)
/// file, or a HuggingFace-format directory (sum of *.safetensors/*.bin/*.gguf).
pub fn model_weight_bytes(model_path: &Path) -> Result<u64> {
    if model_path.is_dir() {
        let mut total = 0u64;
        for entry in std::fs::read_dir(model_path)
            .with_context(|| format!("reading dir {}", model_path.display()))?
        {
            let entry = entry?;
            let p = entry.path();
            if let Some(ext) = p.extension().and_then(|e| e.to_str()) {
                if matches!(ext, "safetensors" | "bin" | "gguf" | "pt") {
                    // Follow symlinks (HF cache snapshots symlink into ../blobs).
                    total += std::fs::metadata(&p)?.len();
                }
            }
        }
        if total == 0 {
            bail!(
                "no weight files (*.safetensors/*.bin/*.gguf) in {}",
                model_path.display()
            );
        }
        Ok(total)
    } else {
        weights_bytes(model_path)
    }
}

/// Locate a GGUF file for a llama.cpp model whose path may be a directory.
pub fn resolve_gguf(model_path: &Path) -> Result<PathBuf> {
    if model_path.is_file() {
        return Ok(model_path.to_path_buf());
    }
    if model_path.is_dir() {
        let mut ggufs: Vec<PathBuf> = std::fs::read_dir(model_path)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("gguf"))
            .collect();
        ggufs.sort();
        return ggufs
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("no .gguf in {}", model_path.display()));
    }
    Err(anyhow!("model path not found: {}", model_path.display()))
}

/// Parse a HuggingFace `config.json` (next to, or inside, the model path) into
/// the same hyperparameter shape used for KV estimation. For vLLM models.
pub fn parse_hf_config(model_path: &Path) -> Result<GgufInfo> {
    let dir = if model_path.is_dir() {
        model_path.to_path_buf()
    } else {
        model_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf()
    };
    let cfg_path = dir.join("config.json");
    let text = std::fs::read_to_string(&cfg_path)
        .with_context(|| format!("reading {}", cfg_path.display()))?;
    let v: serde_json::Value = serde_json::from_str(&text)?;
    let getu = |k: &str| v.get(k).and_then(|x| x.as_u64());

    let n_head = getu("num_attention_heads").unwrap_or(0);
    let n_layer = getu("num_hidden_layers").unwrap_or(0);
    let n_head_kv = getu("num_key_value_heads").unwrap_or(n_head);
    let hidden = getu("hidden_size").unwrap_or(0);
    let head_dim = getu("head_dim").unwrap_or(if n_head > 0 { hidden / n_head } else { 0 });

    if n_layer == 0 {
        bail!("config.json missing num_hidden_layers");
    }
    Ok(GgufInfo {
        arch: v
            .get("model_type")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string(),
        n_layer,
        n_head,
        n_head_kv,
        key_length: head_dim,
        value_length: head_dim,
        embedding_length: hidden,
        context_length: getu("max_position_embeddings").unwrap_or(0),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shard_name_parsing() {
        let (base, count) =
            parse_shard_name("MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf").unwrap();
        assert_eq!(base, "MiniMax-M2.7-UD-IQ4_XS");
        assert_eq!(count, 4);
        assert!(parse_shard_name("single.gguf").is_none());
    }
}
