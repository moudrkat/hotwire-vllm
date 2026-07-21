# hotwire

**Activation steering for vLLM that doesn't turn off the engine.**

Every existing steering tool for vLLM ([vllm-lens](https://github.com/UKGovernmentBEIS/vllm-lens),
[EasySteer](https://arxiv.org/abs/2509.25175), IBM's vLLM Hook) forces
`enforce_eager=True`: PyTorch forward hooks don't survive CUDA graph capture, so
they disable CUDA graphs and torch.compile for the whole server — every request
pays, steered or not. Fine for research, a non-starter for production.

hotwire keeps the graphs. The steering addition is a custom torch op (Triton
kernel) that gets baked *into* the captured graph; per-request routing happens
by updating the contents of persistent GPU buffers between graph replays —
the graph reads fresh data at the same addresses.

The technique was proven viable in [RhizoNymph's vLLM fork](https://github.com/RhizoNymph/vllm)
(see [RFC #36998](https://github.com/vllm-project/vllm/issues/36998), where
in-flight steering is explicitly deferred to "Phase 2"). hotwire packages it as
an out-of-tree plugin: `pip install`, no fork, registered via vLLM's official
`general_plugins` entry point.

## Design

Three persistent GPU tensors, allocated at model-load time:

| buffer | shape | role |
|---|---|---|
| `bank` | `(n_slots, hidden)` | steering vectors, one per active slot |
| `scales` | `(n_slots,)` | per-slot multiplier |
| `slot_map` | `(n_layers, max_tokens)` | token → slot per layer, `-1` = untouched |

The op, called at the end of each decoder layer's forward:

```
hidden[tok] += scales[slot] * bank[slot]   where slot = slot_map[layer, tok] >= 0
```

- **Graph-safe:** the op is `torch.library.custom_op` with a fake impl —
  opaque to torch.compile, captured into CUDA graphs as a fixed kernel on
  fixed addresses. Steering on/off/vector changes are buffer *content*
  updates between replays (host-side copy), never a re-capture.
- **Per-request:** a pre-forward hook reads `forward_context`
  (`query_start_loc` + `req_ids`, same bookkeeping vllm-lens validated)
  and fills `slot_map` for the step.
- **No pickle:** vectors enter as safetensors files or base64 JSON via a
  registration endpoint (`POST /steer/vectors`), requests reference them by
  id + scale in `vllm_xargs`. Nothing executable crosses the wire.
- **Zero cost when idle:** `slot_map` all `-1` → kernel early-exits per token.
  (Benchmark target: unmeasurable vs baseline; RhizoNymph reported minimal
  overhead on H100.)

## Layout

- `hotwire/_kernel.py` — Triton kernel + `hotwire::steer` custom op (working)
- `hotwire/_bank.py` — slot allocation, vector registration (working)
- `hotwire/_patch.py` — decoder-layer wrapping + pre-forward slot fill (WIP:
  integration points against vLLM 0.25.x)
- `hotwire/wire.py` — JSON/safetensors vector wire format, no pickle (working)

## Status

Working end-to-end against vLLM 0.25.1's **V1 model runner** with CUDA graphs
captured (PIECEWISE + FULL) and torch.compile on: per-request steering via
`extra_args`/`vllm_xargs`, verified on Qwen3-0.6B and Qwen3-4B on a single
16 GB GPU. Run with `VLLM_USE_V2_MODEL_RUNNER=0` for now.

Known gaps:
- vLLM 0.25.1 ships a second runner (`vllm/v1/worker/gpu/model_runner.py`,
  `VLLM_USE_V2_MODEL_RUNNER`) that this setup selects by default; hotwire
  detects only the V1 class so far. V2 support is next.
- Vector registration is startup-time only (`$HOTWIRE_VECTORS` dir of .pt
  files); the HTTP registration endpoint is not built yet.
- No benchmark numbers yet (steered vs unsteered vs eager-mode vllm-lens).
