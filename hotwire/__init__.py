"""hotwire: CUDA-graph-safe activation steering plugin for vLLM.

Usage guide for humans and agents: see AGENTS.md in this package, or the
repo README. Steer a request with vllm_xargs={"hotwire": "<json spec
string>"}; mind the fixed slot budget (one per distinct vector,layer,scale).
"""


def flashinfer_sampler_fallback() -> str | None:
    """vLLM's default FlashInfer sampler JIT-compiles with nvcc at engine
    start (in its dummy sampler run — before any request, greedy or not).
    On a driver-only box that dies with "Could not find nvcc" before the
    plugin ever runs, plugin or no plugin. If there is no toolkit and the
    user hasn't chosen, fall back to vLLM's torch sampler: same
    distribution, a little slower. Returns the reason when it acted."""
    import os
    import shutil

    if os.environ.get("VLLM_USE_FLASHINFER_SAMPLER") is not None:
        return None
    if shutil.which("nvcc") or os.path.isdir(os.environ.get("CUDA_HOME", "/usr/local/cuda")):
        return None
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    return "no nvcc / CUDA toolkit found; VLLM_USE_FLASHINFER_SAMPLER=0 (torch sampler)"


def register() -> None:
    """vllm.general_plugins entry point — called by vLLM in every process.

    Import side effects only; must be cheap and idempotent. Model patching
    happens lazily at model-load time (see _patch.py), not here.
    """
    import os

    why = flashinfer_sampler_fallback()
    if why:
        import sys
        print(f"hotwire: {why}", file=sys.stderr, flush=True)

    if os.environ.get("HOTWIRE_DEBUG"):
        try:
            with open("/tmp/hotwire_dbg.log", "a") as f:
                f.write(f"[register pid={os.getpid()}]\n")
        except OSError:
            pass

    from hotwire import _patch

    _patch.install()
