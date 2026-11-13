"""The compile-cache salt, on stand-ins for both VllmConfig APIs (no vLLM needed)."""
import pytest

from hotwire._patch import _salt_compile_key


def test_salts_compute_hash_up_to_033():
    class Cfg:  # <= 0.33: config part of the key is compute_hash()
        def compute_hash(self, include_version: bool = True):
            return "0123456789" if include_version else "abcdefghij"

    before = Cfg().compute_hash()
    assert _salt_compile_key(Cfg, "hotwire-x") == "compute_hash"
    after = Cfg().compute_hash()
    assert after != before and len(after) == len(before)
    # 0.33's flashinfer autotune cache calls compute_hash(include_version=False)
    assert Cfg().compute_hash(include_version=False) == Cfg().compute_hash(False) != after


def test_salts_compile_factors_from_034():
    class Cfg:  # >= 0.34 (vllm-project/vllm#65652): compile_factors()
        def compile_factors(self, include_version: bool = True):
            return {"model": {}, **({"version": "0.34"} if include_version else {})}

    assert _salt_compile_key(Cfg, "hotwire-x") == "compile_factors"
    assert Cfg().compile_factors() == {"model": {}, "version": "0.34", "hotwire": "hotwire-x"}
    assert Cfg().compile_factors(include_version=False) == {"model": {}, "hotwire": "hotwire-x"}


def test_salt_is_applied_once():
    class Cfg:
        def compile_factors(self):
            return {}

    _salt_compile_key(Cfg, "hotwire-x")
    assert _salt_compile_key(Cfg, "hotwire-x") == "already"
    assert Cfg().compile_factors() == {"hotwire": "hotwire-x"}


def test_neither_api_raises():
    class Cfg:
        pass

    with pytest.raises(AttributeError):
        _salt_compile_key(Cfg, "hotwire-x")
