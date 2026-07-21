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
            spec = _state.parse_spec(raw) or False  # False = definitively none
            specs[req_id] = spec
        if spec:
            for entry in spec:
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


def install() -> None:
    """Entry point hook — cheap, idempotent, safe on non-worker processes."""
    global _installed
    if _installed:
        return
    _dbg("install() called")
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except ImportError as e:
        _dbg(f"gpu_model_runner import failed: {e!r}")
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
