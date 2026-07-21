import pytest
import torch

from hotwire.wire import SteerSpec, parse_request_spec, tensor_from_wire, tensor_to_wire


def test_tensor_roundtrip_float32():
    t = torch.randn(3, 5)
    assert torch.equal(tensor_from_wire(tensor_to_wire(t)), t)


def test_tensor_roundtrip_bfloat16():
    t = torch.randn(4, 8, dtype=torch.bfloat16)
    back = tensor_from_wire(tensor_to_wire(t))
    assert back.dtype == torch.bfloat16
    assert torch.equal(back.view(torch.uint8), t.view(torch.uint8))


def test_parse_request_spec():
    raw = '{"vectors": [{"id": "tesla", "layer": 20, "scale": 1.5}]}'
    assert parse_request_spec(raw) == [SteerSpec("tesla", 20, 1.5)]


def test_parse_request_spec_empty():
    assert parse_request_spec('{"vectors": []}') == []


def test_parse_request_spec_malformed_raises():
    with pytest.raises((KeyError, ValueError)):
        parse_request_spec('{"vectors": [{"layer": 20}]}')
