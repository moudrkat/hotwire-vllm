"""The graph-safe steering op.

`hotwire::steer` adds `scales[slot] * bank[slot]` into each token's residual
row, where `slot = slot_map[tok]` and -1 means leave the token alone.

Registered as a torch custom op with a fake impl, so torch.compile treats it
as opaque and CUDA graph capture bakes it in as a fixed kernel on fixed
addresses. Changing which requests are steered (and by what) is a buffer
content update between graph replays — never a re-capture.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _steer_kernel(hidden_ptr, bank_ptr, slot_ptr, scale_ptr,
                  hidden_dim, BLOCK: tl.constexpr):
    tok = tl.program_id(0)
    slot = tl.load(slot_ptr + tok)
    if slot >= 0:
        scale = tl.load(scale_ptr + slot).to(tl.float32)
        for start in range(0, hidden_dim, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            mask = offs < hidden_dim
            h = tl.load(hidden_ptr + tok * hidden_dim + offs, mask=mask).to(tl.float32)
            v = tl.load(bank_ptr + slot * hidden_dim + offs, mask=mask).to(tl.float32)
            out = h + scale * v
            tl.store(hidden_ptr + tok * hidden_dim + offs,
                     out.to(hidden_ptr.dtype.element_ty), mask=mask)


@torch.library.custom_op("hotwire::steer", mutates_args=("hidden",))
def steer(hidden: torch.Tensor, bank: torch.Tensor,
          slot_map: torch.Tensor, scales: torch.Tensor) -> None:
    """In-place: hidden[t] += scales[slot_map[t]] * bank[slot_map[t]] where slot >= 0.

    hidden: (num_tokens, hidden_dim) — residual stream rows for this step
    bank: (n_slots, hidden_dim) — persistent vector bank
    slot_map: (num_tokens,) int32 — per-token slot, -1 = no steering
    scales: (n_slots,) float32 — per-slot multiplier
    """
    num_tokens, hidden_dim = hidden.shape
    if num_tokens == 0:
        return
    _steer_kernel[(num_tokens,)](
        hidden, bank, slot_map, scales, hidden_dim,
        BLOCK=min(1024, triton.next_power_of_2(hidden_dim)),
    )


@steer.register_fake
def _(hidden, bank, slot_map, scales) -> None:
    return None
