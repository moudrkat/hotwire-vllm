import pytest
import torch

from hotwire._bank import VectorBank


def make_bank(n_slots=4, dim=8):
    return VectorBank(dim, n_slots, torch.device("cpu"), dtype=torch.float32)


def test_register_and_lookup():
    b = make_bank()
    v = torch.ones(8)
    slot = b.register("a", v, 1.5)
    assert b.slot_of("a") == slot
    assert torch.equal(b.bank[slot], v)
    assert b.scales[slot].item() == 1.5


def test_register_same_id_reuses_slot():
    b = make_bank()
    s1 = b.register("a", torch.ones(8), 1.0)
    s2 = b.register("a", torch.full((8,), 2.0), 3.0)
    assert s1 == s2
    assert b.scales[s1].item() == 3.0
    assert torch.equal(b.bank[s1], torch.full((8,), 2.0))


def test_full_bank_raises():
    b = make_bank(n_slots=2)
    b.register("a", torch.ones(8), 1.0)
    b.register("b", torch.ones(8), 1.0)
    with pytest.raises(RuntimeError, match="full"):
        b.register("c", torch.ones(8), 1.0)


def test_release_frees_slot_and_zeroes_scale():
    b = make_bank(n_slots=1)
    slot = b.register("a", torch.ones(8), 2.0)
    b.release("a")
    assert b.slot_of("a") is None
    assert b.scales[slot].item() == 0.0
    b.register("b", torch.ones(8), 1.0)  # slot reusable


def test_wrong_shape_raises():
    b = make_bank()
    with pytest.raises(ValueError, match="expected"):
        b.register("a", torch.ones(7), 1.0)
