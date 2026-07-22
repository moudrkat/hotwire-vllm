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

## Verify on your hardware

Two commands, ~3 minutes on any CUDA box with vLLM installed:

```bash
pip install git+https://github.com/moudrkat/hotwire-vllm
python -m hotwire.verify --model Qwen/Qwen3-0.6B   # any HF model id works
```

It generates a throwaway steering vector for the model, checks that steering
fires, that unsteered requests (including batchmates) are untouched, and
compares decode cost idle vs all-steered — then prints a report block.
**Please paste the report into an issue**, especially from hardware, model
families, or configs (TP > 1, 7B+, H100s) the tables below don't cover yet —
that's currently the most useful contribution this project can receive.

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
| Qwen3-14B-AWQ (4-bit) | ✓ | ✓ | 1.91 → 1.91 ms/tok |
| Llama-3.1-8B-Instruct-AWQ | ✓ | ✓ | 1.10 → 1.10 ms/tok |
| Qwen3-8B-FP8 | ✓ | ✓ | 2.27 → 2.28 ms/tok |
| Mistral-7B-Instruct-v0.2-AWQ | ✓ | ✓ | 0.94 → 0.94 ms/tok |
| Qwen3-4B-Instruct-2507 | ✓ | ✓ | 1.78 → 1.78 ms/tok |
| Qwen3-0.6B | ✓ | ✓ | — |
| Qwen2.5-1.5B-Instruct | ✓ | ✓ | 0.77 → 0.77 ms/tok |
| Phi-3.5-mini-instruct | ✓ | ✓ | 1.73 → 1.73 ms/tok |
| tiny-aya-water (Cohere) | ✓ | ✓ | 1.54 → 1.54 ms/tok |

Quantized checkpoints work — steering touches the residual stream, not the
weights, and the AWQ / FP8 rows above confirm it end-to-end, CUDA graphs
captured (PIECEWISE + FULL).

Models that still OOM on the 16 GB test GPU before the plugin engages:
OLMo-2-7B, command-r7b, Qwen3.5-4B, gpt-oss-20b (13.8 GiB weight load
succeeds, engine init doesn't), gemma-4-E4B-it (vision tower). No
architecture failure observed yet; reports from bigger cards welcome. The
layer patch targets any `*DecoderLayer` module with the standard
`(positions, hidden_states, residual)` signature.

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
hook-based steering tools pay *before* their Python hooks even run.

Batch sweep (same model/GPU): the eager tax grows with batch pressure —
+2.3% at 1 request (batch-1 decode is weight-streaming-bound, which hides
launch overhead), +4.7% at 2, +5.6% at 8. hotwire's idle == steered holds at
every batch size, to the second decimal.

| batch | graphs idle | graphs all-steered | eager |
|---|---|---|---|
| 1 | 13.69 ms/tok | 13.69 | 14.00 |
| 2 | 6.97 ms/tok | 6.97 | 7.30 |
| 8 | 1.78 ms/tok | 1.78 | 1.88 |

Untested configurations (no known issues, but nobody has run them — treat as
unsupported until someone does): tensor parallel > 1, pipeline parallel,
speculative decoding, LoRA, GPTQ and MXFP4 quantization (AWQ and FP8 are
verified — see the table). Issues welcome.

Known limitation: one vector per (layer, token) — multiple spec entries
targeting the **same layer** don't stack; the last one wins. Different layers
compose fine. Workaround: pre-combine same-layer vectors into one .pt
(`a*v1 + b*v2`) and register the combo; native stacking is on the roadmap.

Known limitation: the slot budget. Steering configs live in a fixed-size GPU
table allocated before graph capture — CUDA graphs read fixed addresses, so
it can never grow at runtime. Size it with `HOTWIRE_SLOTS` (default 16;
a slot is one vector row, ~5 KB on a 4B model, so 256 costs ~1.3 MB and
nothing per token). Each distinct **(vector, layer, scale)** combo occupies
one slot **permanently** — nothing frees slots when requests finish. A fixed
catalog of vectors at fixed scales therefore runs forever, but continuously
varying scales (0.80, 0.83, 0.87, …) mint a fresh slot each and exhaust the
table; once full, requests with an unregistrable combo run unsteered (logged)
while already-registered combos keep working, batchmates included.
Workaround today: round scales to a small fixed
palette and set `HOTWIRE_SLOTS` generously at startup. The real fixes are on
the roadmap below — slots *can* recycle (the scale isn't baked into the
stored vector; the kernel reads it separately at replay), it's bookkeeping,
not graph physics.

Roadmap:
- HTTP vector registration at runtime (via `vllm.endpoint_plugins`), replacing
  startup-only `$HOTWIRE_VECTORS`.
- Slot eviction: refcount slots per in-flight request and `release()` when the
  last user of a combo finishes, so the table recycles instead of filling.
- Per-token scales: key slots by (vector, layer) only and move scale into a
  per-token buffer — continuous intensities without minting new slots.
- Norm-matched and position-targeted steering modes.
- Tracking the RFC vllm-project/vllm#36998 Phase 2 interface as it lands.
