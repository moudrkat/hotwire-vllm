import json

import pytest
import torch

from hotwire import _dose, _state


def make_state(tmp_path=None, n_layers=4, dim=8, max_tokens=32, n_slots=4):
    return _state.SteerState(n_layers, dim, max_tokens,
                             torch.device("cpu"), torch.float32, n_slots)


def test_parse_spec_forms():
    d = {"id": "a", "layer": 2, "scale": 1.5}
    assert _state.parse_spec(json.dumps(d)) == [d]
    assert _state.parse_spec(json.dumps([d, d])) == [d, d]
    assert _state.parse_spec(d) == [d]
    assert _state.parse_spec([d]) == [d]
    assert _state.parse_spec(None) is None


def test_load_store_and_slot_for(tmp_path):
    t = torch.randn(4, 8)
    torch.save(t, tmp_path / "vec.pt")
    st = make_state()
    st.load_store(str(tmp_path))
    assert "vec" in st.store

    slot = st.slot_for("vec", layer=2, scale=1.5)
    assert slot is not None
    assert torch.allclose(st.bank.bank[slot], t[2])
    assert st.bank.scales[slot].item() == 1.5
    # same spec resolves to the same slot, no duplicate registration
    assert st.slot_for("vec", 2, 1.5) == slot
    # different scale is a distinct slot (scale lives bank-side)
    assert st.slot_for("vec", 2, 2.0) != slot


def test_slot_for_unknown_id_returns_none():
    st = make_state()
    assert st.slot_for("nope", 0, 1.0) is None


def test_load_store_single_file_and_dict_container(tmp_path):
    torch.save({"tensor": torch.randn(4, 8)}, tmp_path / "wrapped.pt")
    st = make_state()
    st.load_store(str(tmp_path / "wrapped.pt"))
    assert "wrapped" in st.store


def test_load_store_precomputes_per_layer_vector_norms(tmp_path):
    t = torch.randn(4, 8)
    torch.save(t, tmp_path / "vec.pt")
    st = make_state()
    st.load_store(str(tmp_path))
    assert torch.allclose(st.vector_norms["vec"], t.norm(dim=-1))


def test_load_h_norms(tmp_path):
    path = tmp_path / "h_norms.json"
    path.write_text(json.dumps({"0": 10.0, "2": 54.9}))
    st = make_state()
    st.load_h_norms(str(path))
    assert st.h_norms == {0: 10.0, 2: 54.9}


def test_slot_for_unconfigured_guardrail_is_backward_compatible(tmp_path):
    """dose_policy stays None unless something sets it — HOTWIRE_MAX_REL_DOSE
    unset must not change behavior at all vs. before this feature existed."""
    t = torch.randn(4, 8)
    torch.save(t, tmp_path / "vec.pt")
    st = make_state()
    st.load_store(str(tmp_path))
    assert st.dose_policy is None
    slot = st.slot_for("vec", 2, 999.0)  # absurd scale, no h_norms loaded either
    assert slot is not None
    assert st.bank.scales[slot].item() == 999.0


def test_slot_for_rejects_over_threshold(tmp_path):
    t = torch.ones(4, 8)  # ||row|| = sqrt(8) ~= 2.83
    torch.save(t, tmp_path / "vec.pt")
    st = make_state()
    st.load_store(str(tmp_path))
    st.h_norms = {2: 2.83}
    st.dose_policy = _dose.DosePolicy("1.0", clamp=False)
    # rel_dose = |10| * 2.83 / 2.83 = 10, way over 1.0
    assert st.slot_for("vec", 2, 10.0) is None
    assert st.bank.slot_of("vec@2x10.0") is None


def test_slot_for_clamps_over_threshold(tmp_path):
    t = torch.ones(4, 8)
    torch.save(t, tmp_path / "vec.pt")
    st = make_state()
    st.load_store(str(tmp_path))
    v_norm = t[2].norm().item()
    st.h_norms = {2: v_norm}  # h_norm == ||V|| so rel_dose == |scale|
    st.dose_policy = _dose.DosePolicy("1.0", clamp=True)
    slot = st.slot_for("vec", 2, 10.0)
    assert slot is not None
    # clamped down to threshold 1.0
    assert st.bank.scales[slot].item() == pytest.approx(1.0)


def test_slot_for_allows_under_threshold(tmp_path):
    t = torch.ones(4, 8)
    torch.save(t, tmp_path / "vec.pt")
    st = make_state()
    st.load_store(str(tmp_path))
    st.h_norms = {2: 1000.0}  # huge reference norm -> tiny rel_dose
    st.dose_policy = _dose.DosePolicy("1.0", clamp=False)
    slot = st.slot_for("vec", 2, 1.5)
    assert slot is not None
    assert st.bank.scales[slot].item() == 1.5


def test_slot_for_unmonitored_layer_still_registers(tmp_path):
    t = torch.ones(4, 8)
    torch.save(t, tmp_path / "vec.pt")
    st = make_state()
    st.load_store(str(tmp_path))
    st.h_norms = {}  # no entry for layer 2 -> guardrail can't judge it
    st.dose_policy = _dose.DosePolicy("1.0", clamp=False)
    slot = st.slot_for("vec", 2, 999.0)
    assert slot is not None
    assert st.bank.scales[slot].item() == 999.0
