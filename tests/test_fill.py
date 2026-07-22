"""_fill_slot_map against fake runner/scheduler objects (no vLLM needed)."""
import json
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("triton")  # _patch imports the kernel module

from hotwire import _patch, _state


@pytest.fixture
def state(tmp_path, monkeypatch):
    torch.save(torch.arange(4 * 8, dtype=torch.float32).reshape(4, 8),
               tmp_path / "vec.pt")
    st = _state.SteerState(4, 8, 32, torch.device("cpu"), torch.float32, 4)
    st.load_store(str(tmp_path))
    monkeypatch.setattr(_state, "_STATE", st)
    return st


def make_runner(reqs: dict[str, str | None]):
    """reqs: req_id -> hotwire extra_arg (raw json str) or None."""
    requests = {
        rid: SimpleNamespace(sampling_params=SimpleNamespace(
            extra_args={"hotwire": spec} if spec else {}))
        for rid, spec in reqs.items()
    }
    return SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=list(reqs)),
        requests=requests,
        _hotwire_specs={},
    )


def sched(counts: dict[str, int]):
    return SimpleNamespace(num_scheduled_tokens=counts)


SPEC = json.dumps({"id": "vec", "layer": 2, "scale": 1.5})


def test_spans_filled_per_request(state):
    runner = make_runner({"a": None, "b": SPEC, "c": None})
    _patch._fill_slot_map(runner, sched({"a": 3, "b": 4, "c": 2}))
    sm = state.slot_map
    slot = state.bank.slot_of("vec@2x1.5")
    assert sm[2, 3:7].eq(slot).all(), "steered request's span gets its slot"
    assert sm[2, :3].eq(-1).all() and sm[2, 7:].eq(-1).all()
    assert sm[[0, 1, 3]].eq(-1).all(), "other layers untouched"


def test_empty_schedule_is_noop(state):
    runner = make_runner({"a": SPEC})
    _patch._fill_slot_map(runner, sched({}))
    assert state.slot_map.eq(-1).all()


def test_unknown_req_bails_unsteered(state):
    runner = make_runner({"a": SPEC})
    _patch._fill_slot_map(runner, sched({"ghost": 5}))
    assert state.slot_map.eq(-1).all()


def test_spec_cache_reused_and_pruned(state):
    runner = make_runner({"a": SPEC})
    _patch._fill_slot_map(runner, sched({"a": 2}))
    assert runner._hotwire_specs["a"]
    # unparseable extra_args never crash the step
    runner.requests["a"].sampling_params.extra_args["hotwire"] = "{broken"
    runner._hotwire_specs.clear()
    _patch._fill_slot_map(runner, sched({"a": 2}))
    assert state.slot_map.eq(-1).all()


def test_bank_exhaustion_degrades_per_request(state):
    # 4-slot bank: request "a" grabs all slots, then "b" needs a 5th. Only
    # "b" runs unsteered; "a" keeps its vectors, nothing raises.
    a_spec = json.dumps([{"id": "vec", "layer": l, "scale": 1.0}
                         for l in range(4)])
    b_spec = json.dumps({"id": "vec", "layer": 2, "scale": 99.0})
    runner = make_runner({"a": a_spec, "b": b_spec})
    _patch._fill_slot_map(runner, sched({"a": 3, "b": 2}))
    sm = state.slot_map
    for layer in range(4):
        slot = state.bank.slot_of(f"vec@{layer}x1.0")
        assert sm[layer, :3].eq(slot).all(), "batchmate keeps its steering"
    assert sm[:, 3:].eq(-1).all(), "exhausted request's span stays unsteered"


def test_unexpected_error_never_leaves_partial_fill(state, monkeypatch):
    # If the walk dies mid-batch for any other reason, the already-written
    # spans must not survive into the step.
    a_spec = json.dumps({"id": "vec", "layer": 1, "scale": 1.0})
    runner = make_runner({"a": a_spec, "b": a_spec})
    calls = {"n": 0}
    orig = state.slot_for

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] > 1:
            raise ValueError("boom")
        return orig(*args, **kwargs)

    monkeypatch.setattr(state, "slot_for", flaky)
    with pytest.raises(ValueError):
        _patch._fill_slot_map(runner, sched({"a": 3, "b": 2}))
    assert state.slot_map.eq(-1).all(), "partial fill must be wiped on error"


def test_decode_only_skips_prefill_spans(state):
    spec = json.dumps({"id": "vec", "layer": 2, "scale": 1.5,
                       "decode_only": True})
    runner = make_runner({"a": spec})
    _patch._fill_slot_map(runner, sched({"a": 5}))  # prefill-like span
    assert state.slot_map.eq(-1).all(), "prefill span must stay unsteered"
    runner2 = make_runner({"a": spec})
    _patch._fill_slot_map(runner2, sched({"a": 1}))  # decode step
    assert not state.slot_map.eq(-1).all(), "decode token gets the vector"


def test_stale_slots_cleared_between_steps(state):
    runner = make_runner({"a": SPEC})
    _patch._fill_slot_map(runner, sched({"a": 5}))
    assert not state.slot_map.eq(-1).all()
    runner2 = make_runner({"b": None})
    _patch._fill_slot_map(runner2, sched({"b": 5}))
    assert state.slot_map.eq(-1).all()
