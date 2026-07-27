
## unreleased — relative-dose guardrail (2026-07-27)
FEATURE: optional request-admission check against steering-mechanics'
relative-dose finding (`scale*||V[layer]||/||h[layer]||`; collapse ~1.9, safe
~0.7). `HOTWIRE_H_NORMS` loads a layer->mean-||h|| table;
`HOTWIRE_MAX_REL_DOSE` turns on the guardrail (unset = off, zero overhead);
`HOTWIRE_REL_DOSE_MODE=clamp` clamps instead of rejecting. Host-side only,
checked once per new (id, layer, scale) at first registration — never inside
the CUDA graphs. New `hotwire/_dose.py`.

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
