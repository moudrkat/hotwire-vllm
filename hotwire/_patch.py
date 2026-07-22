"""vLLM integration: splice the steering op into the model, feed it per step.

Two monkeypatches on GPUModelRunner (installed from the general_plugins entry
point, so they exist in every engine/worker process before anything runs):

* load_model — after the model is built and BEFORE the profile run /
  torch.compile / CUDA graph capture: allocate the persistent buffers and
  class-patch every *DecoderLayer's forward to call `torch.ops.hotwire.steer`
  on its hidden_states output. The op is compile-opaque, so it is traced once
  and captured into the graphs; from then on steering is pure buffer content.

  (vLLM keeps the residual stream split as hidden_states + residual between
  layers; adding the vector to hidden_states is equivalent to adding to their
  sum, i.e. the same thing as a residual hook at this layer's output.)

* execute_model — eager Python that runs every step OUTSIDE the graphs:
  reset slot_map to -1, then for each steered request in this batch fill its
  token span with its bank slot. Token spans follow input_batch.req_ids order
  with scheduler_output.num_scheduled_tokens[req_id] tokens each — the same
  layout _prepare_inputs uses for query_start_loc. Padded graph rows stay -1.

Request API: SamplingParams.extra_args["hotwire"] (or vllm_xargs over HTTP) =
'{"id": "v_pref_tesla_car", "layer": 20, "scale": 1.5}' (or a list of such).
Vectors are preloaded operator-side from $HOTWIRE_VECTORS. No pickle anywhere.
"""
import logging
import os
import re
import sys

import torch

from hotwire import _state
from hotwire._kernel import steer  # registers torch.ops.hotwire.steer  # noqa: F401

logger = logging.getLogger("hotwire")
_DEBUG = bool(os.environ.get("HOTWIRE_DEBUG"))


def _dbg(msg: str) -> None:
    if _DEBUG:
        line = f"[hotwire pid={os.getpid()}] {msg}"
        print(line, file=sys.stderr, flush=True)
        try:
            with open("/tmp/hotwire_dbg.log", "a") as f:
                f.write(line + "\n")
        except OSError:
            pass

_installed = False
_patched_layer_classes: set[type] = set()


def _steered_forward(orig_forward):
    def forward(self, *args, **kwargs):
        out = orig_forward(self, *args, **kwargs)
        st = _state.get()
        idx = getattr(self, "_hotwire_idx", -1)
        if st is not None and idx >= 0:
            hidden = out[0] if isinstance(out, tuple) else out
            torch.ops.hotwire.steer(
                hidden,
                st.bank.bank,
                st.slot_map[idx, : hidden.shape[0]],
                st.bank.scales,
            )
        return out

    return forward


def _install_into_model(runner) -> None:
    hf = runner.model_config.hf_config
    st = _state.init(
        n_layers=hf.num_hidden_layers,
        hidden_dim=hf.hidden_size,
        max_tokens=runner.max_num_tokens,
        device=runner.device,
        dtype=runner.model_config.dtype,
    )
    n_found = 0
    for name, module in runner.model.named_modules():
        if not type(module).__name__.endswith("DecoderLayer"):
            continue
        m = re.search(r"\.(\d+)$", name)
        if m is None:
            continue
        module._hotwire_idx = int(m.group(1))
        n_found += 1
        cls = type(module)
        if cls not in _patched_layer_classes:
            cls.forward = _steered_forward(cls.forward)
            _patched_layer_classes.add(cls)
    if n_found:
        logger.info("hotwire: armed %d decoder layers (%s)", n_found,
                    ", ".join(c.__name__ for c in _patched_layer_classes))
    else:
        logger.warning("hotwire: no decoder layers found; steering inactive")
    _dbg(f"armed {n_found} decoder layers "
         f"({', '.join(c.__name__ for c in _patched_layer_classes)}); "
         f"store={list(st.store)}")
    # runner-level cache: req_id -> parsed spec (avoids re-parsing every step)
    runner._hotwire_specs = {}
    _ = st


def _fill_slot_map(runner, scheduler_output) -> None:
    st = _state.get()
    if st is None:
        return
    num_scheduled = scheduler_output.num_scheduled_tokens
    specs = runner._hotwire_specs
    st.slot_map.fill_(-1)
    if not num_scheduled:
        return  # profile/dummy step — nothing scheduled, nothing to steer
    try:
        _walk_batch(runner, st, num_scheduled, specs)
    except Exception:
        st.slot_map.fill_(-1)  # never leave a partial fill behind
        raise


def _walk_batch(runner, st, num_scheduled, specs) -> None:
    start = 0
    for req_id in runner.input_batch.req_ids:
        n = num_scheduled.get(req_id)
        if n is None:
            # unknown span layout for this step — leave it unsteered
            st.slot_map.fill_(-1)
            _dbg(f"req {req_id} not in num_scheduled_tokens; step unsteered")
            return
        spec = specs.get(req_id)
        if spec is None:
            req = runner.requests.get(req_id)
            sp = req.sampling_params if req is not None else None
            raw = (sp.extra_args or {}).get("hotwire") if sp is not None else None
            try:
                spec = _state.parse_spec(raw) or False  # False = definitively none
            except Exception:
                logger.warning("hotwire: malformed spec for %s ignored: %r",
                               req_id, raw)
                spec = False
            specs[req_id] = spec
        if spec:
            for entry in spec:
                # decode_only: steer generated tokens, never the prompt.
                # Vectors calibrated on generation-only steering (brainscope
                # mutes prefill) are far too hot when applied to a long
                # prefill as well — skip multi-token (prefill) spans.
                if entry.get("decode_only") and n > 1:
                    continue
                slot = st.slot_for(entry["id"], int(entry["layer"]),
                                   float(entry.get("scale", 1.0)))
                if slot is not None:
                    st.slot_map[int(entry["layer"]), start : start + n] = slot
                    _dbg(f"steering req={req_id} tokens[{start}:{start + n}] "
                         f"layer={entry['layer']} slot={slot}")
        start += n
    # drop cache entries for finished requests
    if len(specs) > 4 * len(runner.requests) + 64:
        live = set(runner.requests)
        for rid in [r for r in specs if r not in live]:
            del specs[rid]


def _fill_slot_map_v2(runner, input_batch) -> None:
    """V2 runner: spans come pre-computed on the InputBatch (which is sorted
    decode-first — scheduler order does NOT apply here)."""
    st = _state.get()
    if st is None:
        return
    st.slot_map.fill_(-1)
    specs = runner._hotwire_specs
    if not specs:
        return
    qsl = input_batch.query_start_loc_np
    for i, req_id in enumerate(input_batch.req_ids):
        spec = specs.get(req_id)
        if not spec:
            continue
        start, end = int(qsl[i]), int(qsl[i + 1])
        for entry in spec:
            if entry.get("decode_only") and end - start > 1:
                continue  # prefill span; see _walk_batch
            slot = st.slot_for(entry["id"], int(entry["layer"]),
                               float(entry.get("scale", 1.0)))
            if slot is not None:
                st.slot_map[int(entry["layer"]), start:end] = slot
                _dbg(f"v2 steering req={req_id} tokens[{start}:{end}] "
                     f"layer={entry['layer']} slot={slot}")


def _install_v2(RunnerV2) -> None:
    """V2 (vllm.v1.worker.gpu.model_runner): SamplingParams is decomposed and
    discarded at add time, so extra_args must be captured in add_requests;
    prepare_inputs hands us exact per-request spans."""
    orig_load = RunnerV2.load_model
    orig_add = RunnerV2.add_requests
    orig_prep = RunnerV2.prepare_inputs
    orig_exec = RunnerV2.execute_model

    def load_model(self, *args, **kwargs):
        _dbg("v2 load_model wrapper entered")
        out = orig_load(self, *args, **kwargs)
        try:
            _install_into_model(self)
        except Exception:
            logger.exception("hotwire: v2 install failed; steering disabled")
            import traceback

            _dbg("v2 install failed:\n" + traceback.format_exc())
        return out

    def add_requests(self, scheduler_output, *args, **kwargs):
        out = orig_add(self, scheduler_output, *args, **kwargs)
        try:
            specs = getattr(self, "_hotwire_specs", None)
            if specs is not None:
                for req_data in scheduler_output.scheduled_new_reqs:
                    sp = req_data.sampling_params
                    raw = (sp.extra_args or {}).get("hotwire") if sp else None
                    if raw is None:
                        specs.pop(req_data.req_id, None)  # re-add without spec
                        continue
                    try:
                        parsed = _state.parse_spec(raw)
                    except Exception:
                        logger.warning("hotwire: malformed spec for %s ignored: %r",
                                       req_data.req_id, raw)
                        parsed = None
                    if parsed:
                        specs[req_data.req_id] = parsed
                if len(specs) > 4 * len(self.req_states.req_id_to_index) + 64:
                    live = set(self.req_states.req_id_to_index)
                    for rid in [r for r in specs if r not in live]:
                        del specs[rid]
        except Exception:
            logger.exception("hotwire: v2 spec capture failed")
        return out

    def prepare_inputs(self, *args, **kwargs):
        input_batch = orig_prep(self, *args, **kwargs)
        try:
            _fill_slot_map_v2(self, input_batch)
        except Exception:
            logger.exception("hotwire: v2 slot fill failed; step runs unsteered")
            st = _state.get()
            if st is not None:
                st.slot_map.fill_(-1)
        return input_batch

    def execute_model(self, scheduler_output, intermediate_tensors=None,
                      dummy_run=False, *args, **kwargs):
        if dummy_run or kwargs.get("is_profile"):
            st = _state.get()
            if st is not None:
                st.slot_map.fill_(-1)  # dummy batches must never be steered
        return orig_exec(self, scheduler_output, intermediate_tensors,
                         dummy_run, *args, **kwargs)

    RunnerV2.load_model = load_model
    RunnerV2.add_requests = add_requests
    RunnerV2.prepare_inputs = prepare_inputs
    RunnerV2.execute_model = execute_model
    logger.info("hotwire: installed on V2 model runner")
    _dbg("installed on V2 model runner")


def install() -> None:
    """Entry point hook — cheap, idempotent, safe on non-worker processes."""
    global _installed
    if _installed:
        return
    _dbg("install() called")

    # Salt vLLM's compile-cache key: our op is traced into the compiled model,
    # but vLLM's cache hash knows nothing about plugins. Without this, a cache
    # from a hotwire-less run silently serves a model with no steering op in it
    # (and vice versa after uninstall).
    try:
        import hashlib

        from importlib.metadata import version

        from vllm.config import VllmConfig

        salt = "hotwire-" + version("hotwire-vllm")
        orig_hash = VllmConfig.compute_hash

        def compute_hash(self):
            h = orig_hash(self)
            return hashlib.sha256((h + salt).encode()).hexdigest()[: len(h)]

        VllmConfig.compute_hash = compute_hash
        _dbg(f"compile cache salted with {salt!r}")
    except Exception as e:
        logger.warning("hotwire: could not salt compile cache key (%r); "
                       "clear ~/.cache/vllm/torch_compile_cache after "
                       "installing or removing hotwire", e)

    try:
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner as RunnerV2
    except ImportError as e:
        _dbg(f"v2 model_runner import failed (older vLLM?): {e!r}")
        RunnerV2 = None
    if RunnerV2 is not None:
        _install_v2(RunnerV2)
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except ImportError as e:
        _dbg(f"gpu_model_runner import failed: {e!r}")
        _installed = RunnerV2 is not None
        return  # not a vLLM process we can steer

    orig_load = GPUModelRunner.load_model
    orig_exec = GPUModelRunner.execute_model

    def load_model(self, *args, **kwargs):
        _dbg("load_model wrapper entered")
        out = orig_load(self, *args, **kwargs)
        try:
            _install_into_model(self)
        except Exception:
            logger.exception("hotwire: install failed; steering disabled")
            import traceback

            _dbg("install failed:\n" + traceback.format_exc())
        return out

    _exec_logged = [False]

    def execute_model(self, scheduler_output, *args, **kwargs):
        if not _exec_logged[0]:
            _exec_logged[0] = True
            _dbg("execute_model wrapper entered (first call)")
        try:
            _fill_slot_map(self, scheduler_output)
        except Exception:
            logger.exception("hotwire: slot fill failed; step runs unsteered")
            import traceback

            _dbg("slot fill failed:\n" + traceback.format_exc())
            st = _state.get()
            if st is not None:
                st.slot_map.fill_(-1)
        return orig_exec(self, scheduler_output, *args, **kwargs)

    GPUModelRunner.load_model = load_model
    GPUModelRunner.execute_model = execute_model
    _installed = True
    logger.info("hotwire: installed on GPUModelRunner")
    _dbg(f"installed on GPUModelRunner (module id={id(sys.modules['vllm.v1.worker.gpu_model_runner'])})")

    if _DEBUG:
        try:
            from vllm.v1.engine.core import EngineCore

            orig_core_init = EngineCore.__init__

            def core_init(self, *a, **kw):
                from vllm.v1.worker.gpu_model_runner import GPUModelRunner as G

                _dbg(f"EngineCore.__init__; load_model={G.load_model.__module__}."
                     f"{G.load_model.__qualname__} module_id={id(sys.modules['vllm.v1.worker.gpu_model_runner'])}")
                return orig_core_init(self, *a, **kw)

            EngineCore.__init__ = core_init
        except Exception as e:
            _dbg(f"could not probe EngineCore: {e!r}")
