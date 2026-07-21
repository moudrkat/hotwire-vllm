"""vLLM integration — the active construction site.

Plan (validated against vLLM 0.25.x layout, see README):

1. `install()` wraps model loading (gpu_model_runner.load_model) so that
   right after the model is built — and BEFORE torch.compile / CUDA graph
   capture — each decoder layer's forward is wrapped to call
   `torch.ops.hotwire.steer(hidden, bank, slot_map[layer_idx], scales)`
   on its residual output. The custom op is opaque to compile, so it is
   traced once and captured into the graphs.

2. A pre-forward hook (execute_model wrapper) reads the step's
   forward_context (`query_start_loc`, per-request ids — the same
   bookkeeping vllm-lens uses) plus each request's SteerSpec (from
   SamplingParams.extra_args["hotwire"]) and fills `slot_map` in-place.
   All -1 = kernel no-ops.

3. Buffers are sized at load: n_slots (HOTWIRE_SLOTS, default 16),
   slot_map (n_layers, max_num_batched_tokens) int32.

Open questions being resolved on aorus against vllm 0.25.1:
- exact wrap point so both eager and compiled paths share the wrapper
- decode-phase slot_map fill under padded CUDA-graph batch sizes
  (padding rows must be -1)
- preemption/abort lifecycle for releasing per-request slots
"""


def install() -> None:
    # TODO: implement per the plan above; keep a no-op until then so
    # installing hotwire never breaks a vanilla vLLM server.
    pass
