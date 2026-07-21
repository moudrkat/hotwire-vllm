"""Persistent vector bank + slot bookkeeping.

One bank per (device, hidden_dim). Buffers are allocated once and never
resized after graph capture; registration/eviction only rewrites contents.
"""
import threading

import torch


class VectorBank:
    def __init__(self, hidden_dim: int, n_slots: int, device: torch.device,
                 dtype: torch.dtype = torch.bfloat16):
        self.hidden_dim = hidden_dim
        self.n_slots = n_slots
        self.bank = torch.zeros(n_slots, hidden_dim, device=device, dtype=dtype)
        self.scales = torch.zeros(n_slots, device=device, dtype=torch.float32)
        self._free = list(range(n_slots - 1, -1, -1))
        self._by_id: dict[str, int] = {}
        self._lock = threading.Lock()

    def register(self, vector_id: str, vector: torch.Tensor, scale: float) -> int:
        """Copy a vector into a free slot; returns the slot index."""
        if vector.shape != (self.hidden_dim,):
            raise ValueError(f"expected ({self.hidden_dim},), got {tuple(vector.shape)}")
        with self._lock:
            if vector_id in self._by_id:
                slot = self._by_id[vector_id]
            else:
                if not self._free:
                    raise RuntimeError(f"vector bank full ({self.n_slots} slots)")
                slot = self._free.pop()
                self._by_id[vector_id] = slot
        self.bank[slot].copy_(vector.to(self.bank.dtype))
        self.scales[slot] = scale
        return slot

    def slot_of(self, vector_id: str) -> int | None:
        return self._by_id.get(vector_id)

    def release(self, vector_id: str) -> None:
        with self._lock:
            slot = self._by_id.pop(vector_id, None)
            if slot is not None:
                self.scales[slot] = 0.0
                self._free.append(slot)
