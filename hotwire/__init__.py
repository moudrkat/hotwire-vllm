"""hotwire: CUDA-graph-safe activation steering plugin for vLLM."""


def register() -> None:
    """vllm.general_plugins entry point — called by vLLM in every process.

    Import side effects only; must be cheap and idempotent. Model patching
    happens lazily at model-load time (see _patch.py), not here.
    """
    from hotwire import _patch

    _patch.install()
