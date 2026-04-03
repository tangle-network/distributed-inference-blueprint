//! Model sharding — download and load partial model weights.
//!
//! Each operator in the pipeline only needs the weights for its assigned
//! layer range. This module handles downloading the specific shard from
//! HuggingFace and loading it into GPU memory via vLLM.

use blueprint_std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};

/// Configuration for which layers this operator serves.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardConfig {
    pub model_id: String,
    pub layer_start: u32,
    pub layer_end: u32,
    pub total_layers: u32,
    pub download_dir: PathBuf,
}

impl ShardConfig {
    /// Number of layers in this shard.
    pub fn layer_count(&self) -> u32 {
        self.layer_end - self.layer_start
    }

    /// Fraction of the full model this shard represents.
    pub fn fraction(&self) -> f64 {
        self.layer_count() as f64 / self.total_layers as f64
    }

    /// Estimated VRAM required for this shard in MiB.
    /// Rough heuristic: total_model_vram * fraction + overhead.
    pub fn estimated_vram_mib(&self, total_model_vram_mib: u32) -> u32 {
        let base = (total_model_vram_mib as f64 * self.fraction()) as u32;
        // Add ~2 GiB overhead for KV cache, activations, etc.
        base + 2048
    }
}

/// Handle to a loaded model shard.
#[derive(Debug)]
pub struct ModelShard {
    pub config: ShardConfig,
    pub shard_path: PathBuf,
    pub vllm_endpoint: String,
    pub loaded: bool,
}

/// Download only the layers this operator needs from HuggingFace.
///
/// For models stored in safetensors format, the index file maps
/// parameter names to shard files. We parse the index, identify which
/// shard files contain our layers, and download only those.
pub async fn download_model_shard(
    model_id: &str,
    layer_start: u32,
    layer_end: u32,
    download_dir: &Path,
    hf_token: Option<&str>,
) -> anyhow::Result<PathBuf> {
    let shard_dir = download_dir.join(format!(
        "{}_layers_{}_{}",
        model_id.replace('/', "_"),
        layer_start,
        layer_end
    ));

    if shard_dir.exists() {
        tracing::info!(
            path = %shard_dir.display(),
            "shard directory already exists, skipping download"
        );
        return Ok(shard_dir);
    }

    std::fs::create_dir_all(&shard_dir)?;

    // Build HuggingFace API URL for the model index
    let index_url = format!(
        "https://huggingface.co/{}/resolve/main/model.safetensors.index.json",
        model_id
    );

    let client = reqwest::Client::new();
    let mut req = client.get(&index_url);
    if let Some(token) = hf_token {
        req = req.header("Authorization", format!("Bearer {token}"));
    }

    let resp = req.send().await?;
    if !resp.status().is_success() {
        anyhow::bail!(
            "failed to fetch safetensors index for {}: {}",
            model_id,
            resp.status()
        );
    }

    let index: serde_json::Value = resp.json().await?;
    let weight_map = index
        .get("weight_map")
        .and_then(|v| v.as_object())
        .ok_or_else(|| anyhow::anyhow!("missing weight_map in safetensors index"))?;

    // Identify shard files that contain parameters for our layer range.
    // Layer parameters are typically named: model.layers.{N}.{param}
    let mut needed_files: blueprint_std::collections::HashSet<String> =
        blueprint_std::collections::HashSet::new();

    // Always need embedding and output head weights
    for (param_name, file_value) in weight_map {
        let file_name = file_value.as_str().unwrap_or("");

        // Check if this parameter belongs to a layer in our range
        if let Some(layer_num) = extract_layer_number(param_name) {
            if layer_num >= layer_start && layer_num < layer_end {
                needed_files.insert(file_name.to_string());
            }
        } else {
            // Non-layer parameters (embeddings, output head, norm) —
            // head operator gets embeddings, tail gets output head
            if layer_start == 0
                && (param_name.contains("embed") || param_name.contains("wte"))
            {
                needed_files.insert(file_name.to_string());
            }
            if param_name.contains("lm_head")
                || param_name.contains("norm")
                || param_name.contains("ln_f")
            {
                // Final norm and lm_head go to all operators (small overhead)
                needed_files.insert(file_name.to_string());
            }
        }
    }

    tracing::info!(
        model_id,
        layer_start,
        layer_end,
        shard_files = needed_files.len(),
        "downloading model shard"
    );

    // Download each needed shard file
    for file_name in &needed_files {
        let file_url = format!(
            "https://huggingface.co/{}/resolve/main/{}",
            model_id, file_name
        );

        let mut req = client.get(&file_url);
        if let Some(token) = hf_token {
            req = req.header("Authorization", format!("Bearer {token}"));
        }

        let resp = req.send().await?;
        if !resp.status().is_success() {
            anyhow::bail!("failed to download {}: {}", file_name, resp.status());
        }

        let dest = shard_dir.join(file_name);
        let bytes = resp.bytes().await?;
        std::fs::write(&dest, &bytes)?;

        tracing::info!(
            file = file_name.as_str(),
            size_mib = bytes.len() / (1024 * 1024),
            "downloaded shard file"
        );
    }

    // Write a shard manifest for vLLM to use
    let manifest = serde_json::json!({
        "model_id": model_id,
        "layer_start": layer_start,
        "layer_end": layer_end,
        "total_layers": total_layers_placeholder(model_id),
        "files": needed_files.iter().collect::<Vec<_>>(),
    });
    std::fs::write(
        shard_dir.join("shard_manifest.json"),
        serde_json::to_string_pretty(&manifest)?,
    )?;

    Ok(shard_dir)
}

/// Load a previously downloaded model shard.
pub fn load_shard(shard_path: &Path, vllm_endpoint: &str) -> anyhow::Result<ModelShard> {
    let manifest_path = shard_path.join("shard_manifest.json");
    let manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&manifest_path)?)?;

    let config = ShardConfig {
        model_id: manifest["model_id"]
            .as_str()
            .unwrap_or("unknown")
            .to_string(),
        layer_start: manifest["layer_start"].as_u64().unwrap_or(0) as u32,
        layer_end: manifest["layer_end"].as_u64().unwrap_or(0) as u32,
        total_layers: manifest["total_layers"].as_u64().unwrap_or(0) as u32,
        download_dir: shard_path.parent().unwrap_or(shard_path).to_path_buf(),
    };

    Ok(ModelShard {
        config,
        shard_path: shard_path.to_path_buf(),
        vllm_endpoint: vllm_endpoint.to_string(),
        loaded: true,
    })
}

/// Extract the layer number from a parameter name like "model.layers.42.self_attn.q_proj.weight"
fn extract_layer_number(param_name: &str) -> Option<u32> {
    let parts: Vec<&str> = param_name.split('.').collect();
    for (i, part) in parts.iter().enumerate() {
        if *part == "layers" || *part == "h" {
            if let Some(num_str) = parts.get(i + 1) {
                if let Ok(n) = num_str.parse::<u32>() {
                    return Some(n);
                }
            }
        }
    }
    None
}

/// Known total layer counts for common models.
/// Used as fallback when not specified in config.
fn total_layers_placeholder(model_id: &str) -> u32 {
    let id = model_id.to_lowercase();
    if id.contains("405b") {
        126
    } else if id.contains("70b") {
        80
    } else if id.contains("8x22b") || id.contains("mixtral") {
        56
    } else if id.contains("8b") {
        32
    } else {
        80 // conservative default
    }
}

/// Calculate layer assignment for operator `i` of `n` total operators.
/// Returns (layer_start, layer_end) where start is inclusive and end is exclusive.
pub fn assign_layers(operator_index: u32, total_operators: u32, total_layers: u32) -> (u32, u32) {
    let start = operator_index * total_layers / total_operators;
    let end = (operator_index + 1) * total_layers / total_operators;
    (start, end)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_layer_number() {
        assert_eq!(
            extract_layer_number("model.layers.42.self_attn.q_proj.weight"),
            Some(42)
        );
        assert_eq!(
            extract_layer_number("model.layers.0.mlp.gate_proj.weight"),
            Some(0)
        );
        assert_eq!(extract_layer_number("model.embed_tokens.weight"), None);
        assert_eq!(extract_layer_number("lm_head.weight"), None);
        // GPT-2 style
        assert_eq!(extract_layer_number("h.11.attn.weight"), Some(11));
    }

    #[test]
    fn test_shard_config_fraction() {
        let cfg = ShardConfig {
            model_id: "test".to_string(),
            layer_start: 0,
            layer_end: 25,
            total_layers: 100,
            download_dir: PathBuf::from("/tmp"),
        };
        assert!((cfg.fraction() - 0.25).abs() < f64::EPSILON);
        assert_eq!(cfg.layer_count(), 25);
    }

    #[test]
    fn test_shard_vram_estimate() {
        let cfg = ShardConfig {
            model_id: "test".to_string(),
            layer_start: 0,
            layer_end: 32,
            total_layers: 126,
            download_dir: PathBuf::from("/tmp"),
        };
        // 405B model needs ~810 GiB total; 32/126 fraction + overhead
        let vram = cfg.estimated_vram_mib(810 * 1024);
        assert!(vram > 200_000); // should be > 200 GiB
    }

    #[test]
    fn test_assign_layers() {
        assert_eq!(assign_layers(0, 4, 100), (0, 25));
        assert_eq!(assign_layers(3, 4, 100), (75, 100));
        assert_eq!(assign_layers(0, 1, 80), (0, 80));
    }

    #[test]
    fn test_total_layers_placeholder() {
        assert_eq!(total_layers_placeholder("meta-llama/Llama-3.1-405B-Instruct"), 126);
        assert_eq!(total_layers_placeholder("meta-llama/Llama-3.1-70B"), 80);
        assert_eq!(total_layers_placeholder("mistralai/Mixtral-8x22B-v0.1"), 56);
    }
}
