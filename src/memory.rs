//! System memory sampling + per-model footprint estimation and OOM guarding.

use crate::config::{Engine, ModelDef, ModelKind};
use crate::gguf;
use serde::Serialize;
use std::path::Path;
use sysinfo::System;

const MIB: u64 = 1024 * 1024;

#[derive(Serialize, Clone, Debug)]
pub struct MemSnapshot {
    pub total_mib: u64,
    pub used_mib: u64,
    pub available_mib: u64,
    pub swap_used_mib: u64,
    pub swap_total_mib: u64,
}

pub fn snapshot(sys: &mut System) -> MemSnapshot {
    sys.refresh_memory();
    MemSnapshot {
        total_mib: sys.total_memory() / MIB,
        used_mib: sys.used_memory() / MIB,
        available_mib: sys.available_memory() / MIB,
        swap_used_mib: sys.used_swap() / MIB,
        swap_total_mib: sys.total_swap() / MIB,
    }
}

#[derive(Serialize, Clone, Debug)]
pub struct MemEstimate {
    pub weights_mib: u64,
    pub kv_mib: u64,
    pub overhead_mib: u64,
    /// The number used for OOM decisions: measured peak if we have one,
    /// otherwise weights + kv + overhead.
    pub total_mib: u64,
    /// True when `total_mib` comes from a real measured load, not an estimate.
    pub measured: bool,
    /// Non-fatal note if we couldn't read GGUF hyperparameters for the KV term.
    pub note: Option<String>,
}

/// Estimate a model's memory footprint. Weights come from summing shard file
/// sizes (exact); KV comes from GGUF hyperparameters (approximate). If the
/// model has a measured peak from a prior load, that wins for `total_mib`.
pub fn estimate(model: &ModelDef, overhead_mib: u64, total_system_mib: u64) -> MemEstimate {
    let path = Path::new(&model.model_path);

    let weights_mib = gguf::model_weight_bytes(path).map(|b| b / MIB).unwrap_or(0);

    // KV hyperparameters come from GGUF metadata (llama.cpp) or config.json (vLLM).
    let info = match model.engine {
        Engine::Llamacpp => gguf::resolve_gguf(path).and_then(|g| gguf::parse_metadata(&g)),
        Engine::Vllm => gguf::parse_hf_config(path),
    };
    let (kv_mib, mut note) = match info {
        Ok(info) => (
            info.kv_bytes(model.context as u64, &model.kv_type) / MIB,
            None,
        ),
        Err(e) => (0, Some(format!("KV estimate unavailable: {e}"))),
    };

    let mut estimated_total = weights_mib + kv_mib + overhead_mib;

    // vLLM LLMs pre-reserve `gpu_memory_utilization` of memory for weights + KV
    // cache regardless of the weight size, so their real footprint is at least
    // that fraction of the machine's memory. Embedding models are small and
    // meant to run alongside an LLM, so we use their actual weight footprint
    // instead (don't block them on the reservation).
    if model.engine == Engine::Vllm && model.kind == ModelKind::Llm {
        let reserved = (model.gpu_mem_util_or_default() as f64 * total_system_mib as f64) as u64;
        if reserved > estimated_total {
            estimated_total = reserved;
            let pct = (model.gpu_mem_util_or_default() * 100.0) as u32;
            note = Some(format!(
                "vLLM reserves ~{pct}% of memory (gpu_memory_utilization)"
            ));
        }
    }

    match model.measured_peak_mib {
        Some(m) => MemEstimate {
            weights_mib,
            kv_mib,
            overhead_mib,
            total_mib: m,
            measured: true,
            note,
        },
        None => MemEstimate {
            weights_mib,
            kv_mib,
            overhead_mib,
            total_mib: estimated_total,
            measured: false,
            note,
        },
    }
}

#[derive(Serialize, Clone, Debug)]
pub struct OomVerdict {
    pub would_oom: bool,
    pub needed_mib: u64,
    pub available_mib: u64,
    /// available - needed - safety_margin. Negative means predicted OOM.
    pub headroom_mib: i64,
    pub safety_margin_mib: u64,
}

pub fn oom_check(needed_mib: u64, snap: &MemSnapshot, safety_mib: u64) -> OomVerdict {
    let headroom = snap.available_mib as i64 - needed_mib as i64 - safety_mib as i64;
    OomVerdict {
        would_oom: headroom < 0,
        needed_mib,
        available_mib: snap.available_mib,
        headroom_mib: headroom,
        safety_margin_mib: safety_mib,
    }
}
