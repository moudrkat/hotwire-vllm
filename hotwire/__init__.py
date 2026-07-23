"""hotwire: CUDA-graph-safe activation steering plugin for vLLM.

Usage guide for humans and agents: see AGENTS.md in this package, or the
repo README. Steer a request with vllm_xargs={"hotwire": "<json spec
string>"}; mind the fixed slot budget (one per distinct layer,scale).
"""


def register() -> None:
    """vllm.general_plugins entry point — called by vLLM in every process.

    Import side effects only; must be cheap and idempotent. Model patching
    happens lazily at model-load time (see _patch.py), not here.
    """
    import os

    if os.environ.get("HOTWIRE_DEBUG"):
        try:
            with open("/tmp/hotwire_dbg.log", "a") as f:
                f.write(f"[register pid={os.getpid()}]\n")
        except OSError:
            pass

    from hotwire import _patch

    _patch.install()
