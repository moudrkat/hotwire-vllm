"""Opt-in end-to-end test against a real vLLM engine.

Run on a GPU box:  pytest -m integration tests/test_integration_vllm.py
Needs: vllm installed, hotwire pip-installed (entry point!), a small model
(default Qwen/Qwen3-0.6B; override with HOTWIRE_TEST_MODEL).
"""
import json
import os

import pytest

pytestmark = pytest.mark.integration

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

MODEL = os.environ.get("HOTWIRE_TEST_MODEL", "Qwen/Qwen3-0.6B")


@pytest.fixture(scope="module")
def llm_and_spec(tmp_path_factory):
    vec_dir = tmp_path_factory.mktemp("vecs")
    torch.manual_seed(0)
    os.environ["HOTWIRE_VECTORS"] = str(vec_dir)

    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(MODEL)
    torch.save(torch.randn(cfg.num_hidden_layers, cfg.hidden_size),
               vec_dir / "chaos.pt")

    from vllm import LLM

    llm = LLM(model=MODEL, gpu_memory_utilization=0.4, max_model_len=2048,
              enable_prefix_caching=False)
    layer = cfg.num_hidden_layers // 2
    spec = json.dumps({"id": "chaos", "layer": layer, "scale": 40.0})
    return llm, spec


def _gen(llm, prompt, extra_args=None):
    from vllm import SamplingParams

    sp = SamplingParams(temperature=0.0, max_tokens=20, extra_args=extra_args or {})
    return llm.generate([prompt], sp, use_tqdm=False)[0].outputs[0].text


def test_steering_changes_output(llm_and_spec):
    llm, spec = llm_and_spec
    prompt = "The capital of France is"
    base = _gen(llm, prompt)
    steered = _gen(llm, prompt, {"hotwire": spec})
    assert steered != base


def test_unsteered_request_unaffected(llm_and_spec):
    llm, spec = llm_and_spec
    prompt = "The capital of France is"
    before = _gen(llm, prompt)
    _gen(llm, prompt, {"hotwire": spec})
    after = _gen(llm, prompt)
    assert before == after, "steering one request must not leak into others"


def test_mixed_batch_isolation(llm_and_spec):
    from vllm import SamplingParams

    llm, spec = llm_and_spec
    prompt = "The capital of France is"
    sps = [SamplingParams(temperature=0.0, max_tokens=20),
           SamplingParams(temperature=0.0, max_tokens=20,
                          extra_args={"hotwire": spec})]
    outs = llm.generate([prompt, prompt], sps, use_tqdm=False)
    plain, steered = outs[0].outputs[0].text, outs[1].outputs[0].text
    assert plain != steered
    assert plain == _gen(llm, prompt), "batchmate output matches solo baseline"


def test_malformed_spec_is_ignored(llm_and_spec):
    llm, _ = llm_and_spec
    prompt = "The capital of France is"
    base = _gen(llm, prompt)
    broken = _gen(llm, prompt, {"hotwire": "{not json"})
    assert broken == base
