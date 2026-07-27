# Using hotwire-vllm (guide for coding agents)

You are helping someone add per-request activation steering to a vLLM server
with `hotwire-vllm`. It is a vLLM plugin — small surface, but the details
below are the ones agents get wrong.

## Install and load

    pip install hotwire-vllm

It auto-registers via the `vllm.general_plugins` entry point when vLLM
starts — nothing to import in user code. If steering silently does nothing,
the plugin did not load: confirm vLLM sees the entry point.

## Steer one request

Steering rides on each request as a `vllm_xargs` field. The `hotwire` value is
a JSON **string** (stringified spec), not a nested object:

    {"messages": [...],
     "vllm_xargs": {"hotwire": "{\"id\": \"NAME\", \"layer\": 20, \"scale\": 1.5}"}}

Multiple layers: pass a JSON list of such specs. Add `"decode_only": true` to
steer only generated tokens.

## The gotchas that actually bite

- **Slots are a fixed budget.** Each distinct steering config lives in a
  fixed-size GPU slot bank; distinct (layer, scale) combos each consume one
  persistent slot. Don't sweep hundreds of combos on a live server — you will
  exhaust the bank. Reuse configs; size the sweep to the slot budget.
- **Vectors must be registered** with the server (wire format is JSON /
  safetensors, no pickle). An unknown id degrades to *unsteered*, never a
  failed request — so "no effect" usually means "id not registered" or
  "plugin not loaded", not "steering broke".
- **Malformed specs degrade to unsteered**, silently. Validate your spec JSON.
- Unsteered requests (including batchmates of steered ones) are untouched, and
  idle steering is zero-cost — CUDA graphs stay intact.
- **Raw scale means nothing across layers/models.** If you're picking a
  scale, calibrate it (hidden-directions/brainscope) rather than guessing.
  Optionally, set `HOTWIRE_H_NORMS` + `HOTWIRE_MAX_REL_DOSE` to have hotwire
  reject or clamp requests whose *relative* dose (`scale*||V||/||h||`) is high
  enough to risk coherence collapse — see the README's "Relative-dose
  guardrail" section. Off by default.

Calibrate the vector first with
[hidden-directions](https://github.com/moudrkat/hidden-directions) (it produces
the `{id, layer, scale, decode_only}` spec you paste here).
