import Foundation

/// The Gemma 4 26B A4B target's transformed runtime `config.json` key-set
/// contract, shared between `Gemma4A4BConfig`
/// (`Sources/MLXFastModel/Gemma4A4BConfig.swift`, the participant worker's
/// config loader) and the trusted runtime-worker pinned-configuration gate
/// (`validateRuntimeWorkerPinnedConfigurationData` in both
/// `Sources/MLXFastHarness/Gemma4RuntimeWorker.swift` and its
/// `Sources/MLXFastTrustedHarness` twin).
///
/// This lives in `MLXFastCore` -- which carries no MLX/model dependency and
/// is linked by every target in this package, including the trusted
/// `mlxfast-swift` binary, which deliberately does NOT depend on
/// `MLXFastModel` -- specifically so the participant-side loader and the
/// trusted-side gate cannot silently diverge on WHICH keys this target's
/// config carries the way the pre-port gate diverged on architecture: it
/// still enforced the Qwen `qwen3_5_text` key set against a Gemma 4 artifact.
///
/// The frozen SCALAR invariants this target's config carries (`vocab_size`,
/// `hidden_size`, `num_hidden_layers`, ...) do not need a similar home: they
/// already live on `MLXFastConstants` in this same module, and both
/// `Gemma4A4BConfig` and the trusted gate already read them from there.
public enum Gemma4A4BConfigKeys {
    /// Every key the transformed config must carry, non-null. Read off the
    /// pinned revision's own `text_config`, which has exactly these 36 and no
    /// null values (`docs/gemma4-port-notes.md` section 1.1).
    public static let required: Set<String> = [
        "attention_bias", "attention_dropout", "attention_k_eq_v",
        "bos_token_id", "dtype", "enable_moe_block", "eos_token_id",
        "final_logit_softcapping", "global_head_dim", "head_dim",
        "hidden_activation", "hidden_size", "hidden_size_per_layer_input",
        "initializer_range", "intermediate_size", "layer_types",
        "max_position_embeddings", "model_type", "moe_intermediate_size",
        "num_attention_heads", "num_experts", "num_global_key_value_heads",
        "num_hidden_layers", "num_key_value_heads", "num_kv_shared_layers",
        "pad_token_id", "rms_norm_eps", "rope_parameters", "sliding_window",
        "tie_word_embeddings", "top_k_experts", "use_bidirectional_attention",
        "use_cache", "use_double_wide_mlp", "vocab_size",
        "vocab_size_per_layer_input",
    ]

    /// Keys that must be ABSENT or null. `moe_router_logit_softcapping` is
    /// the Gemma-family knob that would change router numerics if it ever
    /// appeared; `qkv_bias` and `query_pre_attn_scalar` likewise change
    /// attention. A key that silently appears is exactly as dangerous as one
    /// that disappears.
    public static let forbidden: [String] = [
        "moe_router_logit_softcapping", "qkv_bias", "query_pre_attn_scalar",
    ]

    /// The extra key the runtime `config.json` carries beyond `required`:
    /// the checkpoint-wide quantization block. Its own shape (three scalars
    /// plus 120 per-tensor overrides) is validated separately, not by exact
    /// key-set equality, so it is not part of `required`.
    public static let quantizationKey = "quantization"

    /// The four projection families the checkpoint promotes to 8 bits, on
    /// every layer -- mirrors
    /// `Gemma4A4BWeightNames.quantizationOverrideFamilies`
    /// (`Sources/MLXFastModel/Gemma4A4BWeights.swift`), which cannot itself
    /// live here because it also carries tensor-PATH helpers tied to
    /// `MLXFastModel`'s own naming conventions. This copy carries only the
    /// bare family names, which is all the trusted gate needs to build the
    /// expected override key set for a given layer count.
    public static let quantizationOverrideFamilies: [String] = [
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "router.proj",
    ]
}
