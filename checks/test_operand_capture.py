import pytest
import torch
from torch import nn

from mxq.nn.operand_capture import (
    CaptureDecision,
    MXOperandFakeQuantConfig,
    MXOperandLinear,
    dequant_mx_operand_for_graph,
    functional_contraction_operands,
    quantize_functional_contractions_,
    quantize_mx_operand,
    quantize_selected_linear_modules_,
)


def test_operand_codes_and_scales():
    x = torch.zeros(2, 32)
    x[0, 0] = 1
    x[0, 1] = -1
    for fmt in ("mxfp8", "mxfp6", "mxfp4"):
        dequant, codes, scales = quantize_mx_operand(x, fmt)
        assert dequant.dtype == torch.bfloat16
        assert codes.dtype == scales.dtype == torch.uint8
        assert scales.tolist() == [[127], [104]]
        assert dequant[0, 0] == 1 and dequant[0, 1] == -1
        assert torch.count_nonzero(codes[1]) == 0
        assert torch.equal(dequant.to(x.dtype), dequant_mx_operand_for_graph(x, fmt, -1))


def test_operand_rejects_partial_blocks_and_invalid_fp6_lut():
    with pytest.raises(ValueError, match="multiple of 32"):
        quantize_mx_operand(torch.ones(2, 31))
    with pytest.raises(ValueError, match="sixteen distinct"):
        quantize_mx_operand(torch.ones(2, 32), "mxfp6", codebook=tuple(range(15)) + (64,))
    book = tuple(range(16))
    _, codes, _ = quantize_mx_operand(torch.ones(2, 32), "mxfp6", codebook=book)
    assert set(codes.flatten().tolist()).issubset(set(book))


def test_torchao_selects_only_one_linear():
    model = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 32)).eval()
    host = model[2]
    quantize_selected_linear_modules_(model, {"0": MXOperandFakeQuantConfig()})
    assert isinstance(model[0], MXOperandLinear)
    assert model[2] is host
    assert model(torch.ones(32, 32)).shape == (32, 32)


def test_torchao_selection_rejects_missing_module_before_transform():
    model = nn.Sequential(nn.Linear(32, 32)).eval()
    with pytest.raises(ValueError, match="absent"):
        quantize_selected_linear_modules_(model, {
            "0": MXOperandFakeQuantConfig(), "missing": MXOperandFakeQuantConfig()
        })
    assert isinstance(model[0], nn.Linear)


def test_mixed_precision_functional_census_and_graph():
    class Net(nn.Module):
        def forward(self, x, a, b, c, d):
            x = torch.matmul(x, a)
            x = torch.matmul(x, b)
            x = torch.matmul(x, c)
            return torch.matmul(x, d)

    inputs = tuple(torch.ones(32, 32) for _ in range(5))
    exported = torch.export.export(Net().eval(), inputs)
    graph_module = exported.module()
    sites = [node for node in graph_module.graph.nodes if node.target == torch.ops.aten.matmul.default]
    assert len(sites) == 4
    choices = dict(zip((f"functional:{node.name}" for node in sites), (
        "mxfp8", "mxfp4", CaptureDecision("preserve", execution_route="simt_float"), "host"
    )))
    census = quantize_functional_contractions_(
        graph_module, choices.__getitem__, contract={},
        shape_reason=lambda _contract, _fmt, _m, _n, _k: None,
    )
    assert [(row["status"], row.get("format")) for row in census] == [
        ("quantized", "mxfp8"), ("quantized", "mxfp4"),
        ("preserved", None), ("host", None)
    ]
    assert torch.isfinite(graph_module(*inputs)).all()


def test_target_shape_refusal_leaves_graph_unmodified():
    class Net(nn.Module):
        def forward(self, lhs, rhs):
            return torch.matmul(lhs, rhs)

    inputs = (torch.ones(32, 32), torch.ones(32, 32))
    graph = torch.export.export(Net().eval(), inputs).module()
    before = str(graph.graph)
    census = quantize_functional_contractions_(
        graph, lambda _site: "mxfp8", contract={},
        shape_reason=lambda *_: "target refuses this shape",
    )
    assert census[0]["status"] == "skipped"
    assert census[0]["reason"] == "target refuses this shape"
    assert str(graph.graph) == before


def test_preserved_and_refused_sites_do_not_receive_mx_quantization():
    class Net(nn.Module):
        def forward(self, lhs, rhs):
            return torch.matmul(lhs, rhs)

    inputs = (torch.ones(32, 32), torch.ones(32, 32))
    graph = torch.export.export(Net().eval(), inputs).module()
    before = str(graph.graph)
    census = quantize_functional_contractions_(
        graph, lambda _site: CaptureDecision("preserve", execution_route="simt_float"),
        contract={}, shape_reason=lambda *_: None,
    )
    assert census[0]["status"] == "preserved"
    assert census[0]["execution_route"] == "simt_float"
    assert str(graph.graph) == before
    with pytest.raises(ValueError, match="operand scale encoding is unreviewed"):
        quantize_functional_contractions_(
            graph, lambda _site: CaptureDecision("refuse", reason="operand scale encoding is unreviewed"),
            contract={}, shape_reason=lambda *_: None,
        )
    assert str(graph.graph) == before


def test_handoff_keeps_k_axis_and_scale_layout():
    lhs = torch.ones(2, 32, 32)
    rhs = torch.ones(2, 32, 32)
    operands = functional_contraction_operands(lhs, rhs, "mxfp8")
    assert operands.activation_codes.shape == (2, 32, 32)
    assert operands.weight_codes.shape == (2, 32, 32)
    assert operands.activation_scales.shape == (2, 32, 1)
    assert operands.weight_scales.shape == (2, 32, 1)
