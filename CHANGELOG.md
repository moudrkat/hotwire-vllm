# Changelog

## 0.1.0 (2026-09-04)

- **Relative-dose guardrail** (optional, off by default): an admission check
  against steering-mechanics' relative-dose finding
  (`scale * ||V[layer]|| / ||h[layer]||`; coherence collapse ~1.9, safe
  working points ~0.7). `HOTWIRE_H_NORMS` loads a layer → mean `||h||` table
  (raw `{"20": 54.9, ...}` or the wrapped output of hidden-directions'
  `measure-h-norms`), `HOTWIRE_MAX_REL_DOSE` turns the check on (a number
  rejects entries over it, `warn` only logs), `HOTWIRE_REL_DOSE_MODE=clamp`
  clamps the scale instead of rejecting. Host-side only, evaluated once per
  new (id, layer, scale) combo at first registration — never inside the CUDA
  graphs. Unset = zero overhead, no behaviour change. New `hotwire/_dose.py`.
- **Fix: multimodal models.** Decoder dims (`num_hidden_layers`,
  `hidden_size`) are read via `get_text_config()`, so text decoders nested
  under `text_config` (Gemma-4 / Gemma-3n style) install instead of failing
  with "install failed; steering disabled".
- README: instrument status note, lab map.

## 0.0.2 (2026-07-23) — first PyPI release

- Triton kernel + `hotwire::steer` custom op, slot bank, decoder-layer patch
  for both vLLM 0.25 model runners, compile-cache salting, `decode_only`,
  `python -m hotwire.verify`, benchmarks (all on main since 2026-07-21).
- `AGENTS.md` shipped inside the wheel (guide for coding agents), module
  docstring, venv / IDE files excluded from the build.
