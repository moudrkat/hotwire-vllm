"""GPU unit tests for the steering op (run on aorus: pytest tests/)."""
import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")


@cuda
def test_steer_matches_reference():
    from hotwire._kernel import steer

    torch.manual_seed(0)
    n_tok, dim, n_slots = 7, 2560, 4
    hidden = torch.randn(n_tok, dim, device="cuda", dtype=torch.bfloat16)
    bank = torch.randn(n_slots, dim, device="cuda", dtype=torch.bfloat16)
    scales = torch.tensor([1.5, -1.0, 0.5, 2.0], device="cuda")
    slot_map = torch.tensor([0, -1, 1, 3, -1, 0, 2], device="cuda", dtype=torch.int32)

    ref = hidden.float().clone()
    for t in range(n_tok):
        s = slot_map[t].item()
        if s >= 0:
            ref[t] += scales[s] * bank[s].float()

    steer(hidden, bank, slot_map, scales)
    torch.testing.assert_close(hidden.float(), ref, atol=2e-2, rtol=2e-2)


@cuda
def test_steer_all_idle_is_noop():
    from hotwire._kernel import steer

    hidden = torch.randn(5, 512, device="cuda", dtype=torch.bfloat16)
    before = hidden.clone()
    steer(hidden,
          torch.randn(2, 512, device="cuda", dtype=torch.bfloat16),
          torch.full((5,), -1, device="cuda", dtype=torch.int32),
          torch.ones(2, device="cuda"))
    assert torch.equal(hidden, before)


@cuda
def test_steer_survives_cuda_graph_with_content_swap():
    """The core claim: capture once, change buffer contents, replay."""
    from hotwire._kernel import steer

    dim = 512
    hidden = torch.zeros(4, dim, device="cuda", dtype=torch.bfloat16)
    bank = torch.zeros(2, dim, device="cuda", dtype=torch.bfloat16)
    scales = torch.ones(2, device="cuda")
    slot_map = torch.full((4,), -1, device="cuda", dtype=torch.int32)

    steer(hidden, bank, slot_map, scales)  # warmup
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        steer(hidden, bank, slot_map, scales)

    # replay 1: no steering
    hidden.zero_(); g.replay()
    assert hidden.abs().max().item() == 0

    # replay 2: same graph, new buffer contents -> token 1 steered by slot 0
    bank[0] = 1.0
    slot_map.copy_(torch.tensor([-1, 0, -1, -1], device="cuda", dtype=torch.int32))
    hidden.zero_(); g.replay()
    assert hidden[1].float().sum().item() == pytest.approx(dim, rel=1e-2)
    assert hidden[0].abs().max().item() == 0
