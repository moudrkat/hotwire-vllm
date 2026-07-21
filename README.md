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

## Quickstart

```bash
pip install -e .          # registers the vllm.general_plugins entry point
export HOTWIRE_VECTORS=/path/to/vectors   # dir of .pt files, (n_layers, hidden) each
vllm serve Qwen/Qwen3-4B-Instruct-2507    # CUDA graphs stay ON
```

Steer any request by id + layer + scale:

```python
# offline
SamplingParams(extra_args={"hotwire": '{"id": "tesla_car", "layer": 20, "scale": 1.5}'})
```
```bash
# OpenAI API
curl .../v1/chat/completions -d '{..., "vllm_xargs":
  {"hotwire": "{\"id\": \"tesla_car\", \"layer\": 20, \"scale\": 1.5}"}}'
```

Unsteered requests — including batchmates of steered ones — are untouched.
Malformed specs and unknown vector ids degrade to "unsteered", never to a
failed request.

## Status

Working end-to-end on vLLM 0.25.1, **both model runners** (the classic
`GPUModelRunner` and the new V2 runner that 0.25.1 selects by default for
dense generate models), with CUDA graphs captured (PIECEWISE + FULL) and
torch.compile on. Verified on Qwen3-0.6B / Qwen3-4B on a single 16 GB GPU:
solo steering, mixed batches, decode-phase graph replays.

hotwire also salts vLLM's torch.compile/AOT cache key (`VllmConfig.compute_hash`)
— the op is traced into the compiled model, and vLLM's cache doesn't know about
plugins, so without the salt a stale cache silently serves a model with no
steering op in it.

Tests: `pytest` (unit, CPU-safe), `pytest -m integration` (real engine, GPU).

Verified architectures (chaos-vector A/B + batchmate-isolation check, both
model runners exercised):

| model | steering works | unsteered untouched | TPOT idle → all-steered |
|---|---|---|---|
| Qwen3-4B-Instruct-2507 | ✓ | ✓ | 1.78 → 1.78 ms/tok |
| Qwen3-0.6B | ✓ | ✓ | — |
| Qwen2.5-1.5B-Instruct | ✓ | ✓ | 0.77 → 0.77 ms/tok |
| Phi-3.5-mini-instruct | ✓ | ✓ | 1.73 → 1.73 ms/tok |
| tiny-aya-water (Cohere) | ✓ | ✓ | 1.54 → 1.54 ms/tok |

Larger models (OLMo-2-7B, command-r7b, gpt-oss-20b, Qwen3.5-4B) OOM'd on the
16 GB test GPU before the plugin engaged — no architecture failure observed
yet; reports from bigger cards welcome. The layer patch targets any
`*DecoderLayer` module with the standard `(positions, hidden_states, residual)`
signature.

## Numbers

Qwen3-4B-Instruct-2507, bf16, RTX 4070 Ti SUPER 16 GB, 8 concurrent requests,
256 decode tokens each, medians of 3 (`benchmarks/bench_decode.py`):

| condition | TTFT | decode TPOT |
|---|---|---|
| vanilla vLLM (plugin not installed) | 4.9 ms | 1.78 ms/tok |
| hotwire installed, no request steered | 4.9 ms | 1.78 ms/tok |
| hotwire, **all 8 requests steered** | 4.6 ms | 1.78 ms/tok |
| vLLM `enforce_eager` (no plugin) | 5.0 ms | 1.88 ms/tok |

Idle and fully-steered are both within noise of vanilla. The eager row is what
hook-based steering tools pay *before* their Python hooks even run (~6% TPOT
here; the gap grows with model size and batch pressure).

Untested configurations (no known issues, but nobody has run them — treat as
unsupported until someone does): tensor parallel > 1, pipeline parallel,
speculative decoding, LoRA. Issues welcome.

Known limitation: one vector per (layer, token) — multiple spec entries
targeting the **same layer** don't stack; the last one wins. Different layers
compose fine. Workaround: pre-combine same-layer vectors into one .pt
(`a*v1 + b*v2`) and register the combo; native stacking is on the roadmap.

Roadmap:
- HTTP vector registration at runtime (via `vllm.endpoint_plugins`), replacing
  startup-only `$HOTWIRE_VECTORS`.
- Norm-matched and position-targeted steering modes.
- Tracking the RFC vllm-project/vllm#36998 Phase 2 interface as it lands.
