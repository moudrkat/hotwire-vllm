"""Wire format for vectors: JSON + base64 — data, not code. Never pickle.

Reserved for the roadmap's runtime HTTP registration; the shipped plugin
loads vectors operator-side from $HOTWIRE_VECTORS (see _state.py) and the
request spec it parses is the flat `{"id", "layer", "scale"}` form (or a
list of them) documented in the README — not the `{"vectors": [...]}` form
parsed here.
"""
import base64
import json
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SteerSpec:
    vector_id: str
    layer: int
    scale: float


def parse_request_spec(raw: str) -> list[SteerSpec]:
    data = json.loads(raw)
    return [SteerSpec(vector_id=v["id"], layer=int(v["layer"]), scale=float(v["scale"]))
            for v in data.get("vectors", [])]


def tensor_from_wire(obj: dict) -> torch.Tensor:
    """{"dtype": "bfloat16", "shape": [2560], "data_b64": "..."} -> tensor."""
    dtype = getattr(torch, obj["dtype"])
    buf = base64.b64decode(obj["data_b64"])
    return torch.frombuffer(bytearray(buf), dtype=dtype).reshape(obj["shape"]).clone()


def tensor_to_wire(t: torch.Tensor) -> dict:
    t = t.contiguous().cpu()
    return {"dtype": str(t.dtype).removeprefix("torch."),
            "shape": list(t.shape),
            "data_b64": base64.b64encode(t.view(torch.uint8).numpy().tobytes()
                                         if t.dtype == torch.bfloat16
                                         else t.numpy().tobytes()).decode()}
