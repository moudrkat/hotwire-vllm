import json

import torch

from hotwire import _state


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
