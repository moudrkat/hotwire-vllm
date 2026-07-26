
## unreleased — multimodal decoder-dim fix (2026-07-26)
FIX: `_install_into_model` read `hf_config.num_hidden_layers`/`hidden_size` from
the TOP-LEVEL config, which crashes on multimodal models (Gemma-4/3n) whose
decoder dims live in a nested `text_config` → "install failed; steering disabled".
Now uses `hf.get_text_config()` (falls back to `text_config`, then self), so
steering arms on multimodal text decoders too. Found while trying to steer
Gemma-4-E4B for Czech deployment; the layer-finding loop (`*DecoderLayer` + `.N`)
was already general — only the state dims were wrong. Add a Gemma smoke test.

## unreleased — multimodal decoder-dim fix (2026-07-26)
FIX: _install_into_model read num_hidden_layers/hidden_size from the top-level
hf_config, crashing on multimodal models (Gemma-4/3n) whose decoder dims live in
text_config -> "install failed; steering disabled". Now uses get_text_config().
Found steering Gemma-4-E4B for Czech deployment.
