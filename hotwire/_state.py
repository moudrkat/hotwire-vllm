"""Per-process steering state: the persistent buffers the CUDA graphs read.

Allocated once in the worker process right after model load — before the
profile run, before torch.compile, before graph capture — so the compiled
code always sees the same tensors and no recompilation is ever triggered.
"""
import json
import logging
import os

import torch

from hotwire._bank import VectorBank

logger = logging.getLogger("hotwire")

_STATE: "SteerState | None" = None


class SteerState:
    def __init__(self, n_layers: int, hidden_dim: int, max_tokens: int,
                 device: torch.device, dtype: torch.dtype, n_slots: int):
        self.bank = VectorBank(hidden_dim, n_slots, device, dtype)
        # token -> slot, per layer; -1 = leave the token alone
        self.slot_map = torch.full((n_layers, max_tokens), -1,
                                   device=device, dtype=torch.int32)
        self.n_layers = n_layers
        self.max_tokens = max_tokens
        # host-side: vector_id -> (tensor on device, default layer, default scale)
        self.store: dict[str, tuple[torch.Tensor, int, float]] = {}

    def load_store(self, path: str) -> None:
        """Load a direction dict: .pt with a (n_layers, hidden) tensor per file,
        or a directory of such files (hidden-directions layout). Operator-side
        input — requests never upload tensors."""
        files = ([os.path.join(path, f) for f in sorted(os.listdir(path))
                  if f.endswith(".pt")] if os.path.isdir(path) else [path])
        for f in files:
            name = os.path.splitext(os.path.basename(f))[0]
            t = torch.load(f, map_location="cpu")
            if isinstance(t, dict):
                t = next(v for v in t.values() if isinstance(v, torch.Tensor))
            self.store[name] = (t.to(self.bank.bank.device), -1, 1.0)
            logger.info("hotwire: loaded vector %r %s", name, tuple(t.shape))

    def slot_for(self, vector_id: str, layer: int, scale: float) -> int | None:
        """Resolve request spec -> bank slot, registering lazily."""
        key = f"{vector_id}@{layer}x{scale}"
        slot = self.bank.slot_of(key)
        if slot is not None:
            return slot
        entry = self.store.get(vector_id)
        if entry is None:
            logger.warning("hotwire: unknown vector id %r", vector_id)
            return None
        t = entry[0]
        row = t[layer] if t.dim() == 2 else t
        return self.bank.register(key, row.to(torch.float32), scale)


def get() -> "SteerState | None":
    return _STATE


def init(n_layers: int, hidden_dim: int, max_tokens: int,
         device: torch.device, dtype: torch.dtype) -> "SteerState":
    global _STATE
    if _STATE is None:
        n_slots = int(os.environ.get("HOTWIRE_SLOTS", "16"))
        _STATE = SteerState(n_layers, hidden_dim, max_tokens, device, dtype, n_slots)
        vectors = os.environ.get("HOTWIRE_VECTORS")
        if vectors:
            _STATE.load_store(vectors)
        logger.info("hotwire: state ready (%d layers, %d slots, %d max tokens)",
                    n_layers, n_slots, max_tokens)
    return _STATE


def parse_spec(raw) -> list[dict] | None:
    """extra_args["hotwire"] -> [{"id", "layer", "scale"}, ...]; str or list."""
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = json.loads(raw)
    if isinstance(raw, dict):
        raw = [raw]
    return raw
