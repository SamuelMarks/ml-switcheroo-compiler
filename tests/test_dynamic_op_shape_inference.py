"""Tests for declarative dynamic operator shape inference and symbolic propagation."""

import pytest

from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.ops.registry import get_op
from ml_switcheroo_compiler.ops.shape_inference import (
    S,
    broadcast_shapes,
    compute_contiguous_strides,
    infer_concat,
    infer_conv,
    infer_dot,
    infer_expand_dims,
    infer_flatten,
    infer_matmul,
    infer_op_shape_declarative,
    infer_outer,
    infer_pool,
    infer_reduction,
    infer_reshape,
    infer_shape,
    infer_split,
    infer_squeeze,
    infer_stack,
    infer_transpose,
)
from ml_switcheroo_compiler.transforms.passes.shape_inference import shape_inference_pass


def test_symbolic_dimension():
    """Test symbolic dimension S behaviors."""
    s1 = S("batch")
    s2 = S("batch")
    s3 = S("seq")

    assert s1 == s2
    assert s1 != s3
    assert s1 != "batch"
    assert repr(s1) == "S('batch')"
    assert str(s1) == "batch"
    assert hash(s1) == hash(s2)


def test_compute_contiguous_strides():
    """Test contiguous stride calculation."""
    assert compute_contiguous_strides(()) == ()
    assert compute_contiguous_strides((3,)) == (1,)
    assert compute_contiguous_strides((2, 3, 4)) == (12, 4, 1)
    assert compute_contiguous_strides((S("batch"), 3, 4)) == (12, 4, 1)


def test_broadcast_shapes_success():
    """Test multi-operand broadcasting with static and symbolic dimensions."""
    assert broadcast_shapes() == ()
    assert broadcast_shapes((2, 3)) == (2, 3)
    assert broadcast_shapes((1, 3), (2, 1)) == (2, 3)
    assert broadcast_shapes((5, 1, 4), (3, 4), (1, 1, 4)) == (5, 3, 4)

    # Zero-sized dimensions
    assert broadcast_shapes((0, 3), (1, 3)) == (0, 3)
    assert broadcast_shapes((2, 0), (1, 1)) == (2, 0)

    # Symbolic dimensions
    assert broadcast_shapes((S("batch"), 1), (1, 10)) == (S("batch"), 10)
    assert broadcast_shapes((S("batch"), 3), (S("batch"), 1)) == (S("batch"), 3)


def test_broadcast_shapes_mismatch():
    """Test broadcast mismatch raises ValueError."""
    with pytest.raises(ValueError, match="Shape mismatch"):
        broadcast_shapes((2, 3), (2, 4))

    with pytest.raises(ValueError, match="Shape mismatch"):
        broadcast_shapes((S("batch"),), (S("seq"),))


def test_infer_matmul_variants():
    """Test 1D, 2D, and batched matrix multiplication with broadcasting."""
    # 1D @ 1D -> scalar
    assert infer_matmul((4,), (4,)) == ()
    with pytest.raises(ValueError, match="Contracting dimension mismatch"):
        infer_matmul((4,), (5,))

    # 2D @ 1D -> 1D
    assert infer_matmul((3, 4), (4,)) == (3,)
    with pytest.raises(ValueError, match="Contracting dimension mismatch"):
        infer_matmul((3, 4), (5,))

    # 1D @ 2D -> 1D
    assert infer_matmul((4,), (4, 5)) == (5,)
    with pytest.raises(ValueError, match="Contracting dimension mismatch"):
        infer_matmul((3,), (4, 5))

    # 2D @ 2D
    assert infer_matmul((3, 4), (4, 5)) == (3, 5)

    # Batched Matmul with broadcasting
    assert infer_matmul((2, 1, 3, 4), (1, 5, 4, 6)) == (2, 5, 3, 6)
    assert infer_matmul((S("batch"), 3, 4), (S("batch"), 4, 5)) == (S("batch"), 3, 5)

    with pytest.raises(ValueError, match="Contracting dimension mismatch"):
        infer_matmul((2, 3, 4), (2, 5, 6))

    with pytest.raises(ValueError, match="requires non-empty operands"):
        infer_matmul((), (2, 3))


def test_infer_dot_and_outer():
    """Test dot and outer product shape inference."""
    assert infer_dot((5,), (5,)) == ()
    with pytest.raises(ValueError, match="dimension mismatch"):
        infer_dot((5,), (6,))
    assert infer_dot((3, 4), (4, 5)) == (3, 5)

    assert infer_outer((3,), (4,)) == (3, 4)
    assert infer_outer((2, 3), (4, 5)) == (6, 20)


def test_infer_conv():
    """Test 1D, 2D, and 3D convolution shape inference."""
    # 2D Conv: (N, C_in, H, W) and (C_out, C_in, kH, kW)
    out2d = infer_conv((2, 3, 32, 32), (16, 3, 3, 3), stride=1, padding=1, dilation=1, spatial_dims=2)
    assert out2d == (2, 16, 32, 32)

    # Stride 2, no padding
    out2d_strided = infer_conv((2, 3, 32, 32), (16, 3, 3, 3), stride=2, padding=0, dilation=1, spatial_dims=2)
    assert out2d_strided == (2, 16, 15, 15)

    # 1D Conv: (N, C_in, L) and (C_out, C_in, K)
    out1d = infer_conv((4, 8, 100), (16, 8, 5), stride=2, padding=2, dilation=1, spatial_dims=1)
    assert out1d == (4, 16, 50)

    # Channel mismatch error
    with pytest.raises(ValueError, match="channel count"):
        infer_conv((2, 3, 32, 32), (16, 4, 3, 3), spatial_dims=2)

    # Negative output dimension error
    with pytest.raises(ValueError, match="Negative or zero"):
        infer_conv((1, 1, 2, 2), (1, 1, 5, 5), stride=1, padding=0, spatial_dims=2)


def test_infer_pool():
    """Test 1D, 2D, and 3D pooling shape inference."""
    # 2D Pool: (N, C, H, W)
    out2d = infer_pool((2, 16, 32, 32), kernel_size=2, stride=2, padding=0, spatial_dims=2)
    assert out2d == (2, 16, 16, 16)

    # 1D Pool: (N, C, L)
    out1d = infer_pool((2, 16, 30), kernel_size=3, stride=3, padding=0, spatial_dims=1)
    assert out1d == (2, 16, 10)

    with pytest.raises(ValueError, match="Pooling expects rank"):
        infer_pool((2, 16), kernel_size=2)


def test_infer_reshape():
    """Test reshape with dynamic -1 inference and compatibility checks."""
    assert infer_reshape((2, 3, 4), (6, -1)) == (6, 4)
    assert infer_reshape((2, 3, 4), (24,)) == (24,)
    assert infer_reshape((S("batch"), 12), (S("batch"), 3, 4)) == (S("batch"), 3, 4)

    # Multiple -1 error
    with pytest.raises(ValueError, match="Only one dimension can be -1"):
        infer_reshape((2, 3), (-1, -1))

    # Incompatible size
    with pytest.raises(ValueError, match="Cannot reshape"):
        infer_reshape((2, 3), (7,))


def test_infer_transpose():
    """Test transpose with axes permutation."""
    assert infer_transpose((2, 3, 4)) == (4, 3, 2)
    assert infer_transpose((2, 3, 4), axes=[1, 2, 0]) == (3, 4, 2)
    assert infer_transpose((2, 3, 4), axes=[-1, -2, -3]) == (4, 3, 2)

    with pytest.raises(ValueError, match="Invalid permutation axes"):
        infer_transpose((2, 3, 4), axes=[0, 1])


def test_infer_squeeze_and_expand_dims():
    """Test squeeze and expand_dims."""
    assert infer_squeeze((1, 2, 1, 3)) == (2, 3)
    assert infer_squeeze((1, 2, 1, 3), axis=0) == (2, 1, 3)
    assert infer_squeeze((1, 2, 1, 3), axis=[0, 2]) == (2, 3)

    with pytest.raises(ValueError, match="Cannot squeeze axis"):
        infer_squeeze((1, 2, 3), axis=1)

    assert infer_expand_dims((2, 3), axis=0) == (1, 2, 3)
    assert infer_expand_dims((2, 3), axis=1) == (2, 1, 3)
    assert infer_expand_dims((2, 3), axis=-1) == (2, 3, 1)


def test_infer_flatten():
    """Test flatten along dimension ranges."""
    assert infer_flatten((2, 3, 4, 5), start_dim=1, end_dim=2) == (2, 12, 5)
    assert infer_flatten((2, 3, 4), start_dim=0, end_dim=-1) == (24,)
    assert infer_flatten(()) == ()


def test_infer_split_concat_stack():
    """Test split, concat, and stack operations."""
    # Split
    splits = infer_split((6, 4), num_or_size_splits=3, axis=0)
    assert len(splits) == 3
    assert all(s == (2, 4) for s in splits)

    splits_sizes = infer_split((6, 4), num_or_size_splits=[1, 2, 3], axis=0)
    assert splits_sizes == [(1, 4), (2, 4), (3, 4)]

    with pytest.raises(ValueError, match="Cannot evenly divide"):
        infer_split((5, 4), num_or_size_splits=2, axis=0)

    # Concat
    assert infer_concat([(2, 3), (4, 3)], axis=0) == (6, 3)
    assert infer_concat([(2, 3), (2, 5)], axis=1) == (2, 8)

    with pytest.raises(ValueError, match="Concatenation rank mismatch"):
        infer_concat([(2, 3), (2, 3, 1)])

    with pytest.raises(ValueError, match="Dimension mismatch"):
        infer_concat([(2, 3), (2, 4)], axis=0)

    # Stack
    assert infer_stack([(2, 3), (2, 3)], axis=0) == (2, 2, 3)
    assert infer_stack([(2, 3), (2, 3)], axis=-1) == (2, 3, 2)

    with pytest.raises(ValueError, match="Stack requires identical shapes"):
        infer_stack([(2, 3), (2, 4)])


def test_infer_reduction():
    """Test reduction operations with axis and keepdims."""
    assert infer_reduction((2, 3, 4), axis=None, keepdims=False) == ()
    assert infer_reduction((2, 3, 4), axis=None, keepdims=True) == (1, 1, 1)
    assert infer_reduction((2, 3, 4), axis=1, keepdims=False) == (2, 4)
    assert infer_reduction((2, 3, 4), axis=1, keepdims=True) == (2, 1, 4)
    assert infer_reduction((2, 3, 4), axis=[0, 2], keepdims=False) == (3,)


def test_dynamic_op_def_declarative_inference():
    """Test that DynamicOpDef uses declarative signatures and raises explicit errors."""
    from ml_switcheroo_compiler.ops.registry import _YAML_REGISTRY, _load_yaml_registry

    _load_yaml_registry()
    _YAML_REGISTRY["CustomDynamicTestOp"] = {"category": "elementwise"}
    op_cls = get_op("CustomDynamicTestOp")
    op = op_cls()

    # Empty inputs
    assert op.infer_shape() == ()

    # Valid broadcast
    assert op.infer_shape((1, 3), (2, 1)) == (2, 3)

    # Mismatch raises ValueError
    with pytest.raises(ValueError, match="Shape mismatch"):
        op.infer_shape((2, 3), (2, 4))


def test_shape_inference_pass_with_strides_and_symbolic():
    """Test that shape_inference_pass correctly attaches shape_metadata and contiguous strides."""
    node_in1 = IRNode(id="in1", op_type="Input", inputs=[], attributes={}, shape_metadata=(S("batch"), 4))
    node_in2 = IRNode(id="in2", op_type="Input", inputs=[], attributes={}, shape_metadata=(S("batch"), 4))
    node_add = IRNode(id="add", op_type="Add", inputs=["in1", "in2"], attributes={}, shape_metadata=None)

    graph = IRGraph(name="test_graph", nodes={"in1": node_in1, "in2": node_in2, "add": node_add}, outputs=["add"])

    modified = shape_inference_pass(graph)
    assert modified is True
    assert graph.nodes["add"].shape_metadata == (S("batch"), 4)
    assert graph.nodes["add"].attributes["strides"] == (4, 1)


def test_load_shape_signatures_missing_file(monkeypatch):
    """Test loading shape signatures when YAML file does not exist."""
    import ml_switcheroo_compiler.ops.shape_inference as si

    monkeypatch.setattr(si, "_CACHED_SIGNATURES", None)
    monkeypatch.setattr("os.path.exists", lambda path: False)
    cfg = si.load_shape_signatures()
    assert isinstance(cfg.operations, dict)
    monkeypatch.setattr(si, "_CACHED_SIGNATURES", None)


def test_broadcast_shapes_none_and_edge_cases():
    """Test broadcasting with None inputs."""
    assert broadcast_shapes(None, None) == ()


def test_conv_3d_and_symbolic():
    """Test 3D convolution and symbolic dimension propagation."""
    out3d = infer_conv(
        (1, 4, 10, 10, 10),
        (8, 4, 3, 3, 3),
        stride=(1, 1, 1),
        padding="same",
        dilation=(1, 1, 1),
        spatial_dims=3,
    )
    assert out3d == (1, 8, 8, 8, 8)

    # Symbolic spatial dimension
    out_sym = infer_conv(
        (1, 4, S("H"), 10),
        (8, 4, 3, 3),
        stride=1,
        padding=1,
        dilation=1,
        spatial_dims=2,
    )
    assert out_sym[0] == 1
    assert out_sym[1] == 8
    assert isinstance(out_sym[2], S)
    assert out_sym[3] == 10


def test_pool_3d_and_symbolic():
    """Test 3D pooling and symbolic dimension propagation."""
    out3d = infer_pool(
        (1, 4, 16, 16, 16),
        kernel_size=(2, 2, 2),
        stride=(2, 2, 2),
        padding=(0, 0, 0),
        spatial_dims=3,
    )
    assert out3d == (1, 4, 8, 8, 8)

    # Symbolic pooling
    out_sym = infer_pool(
        (1, 4, S("H"), 16),
        kernel_size=2,
        stride=2,
        padding=0,
        spatial_dims=2,
    )
    assert out_sym[0] == 1
    assert out_sym[1] == 4
    assert isinstance(out_sym[2], S)
    assert out_sym[3] == 8


def test_reshape_other_prod_zero():
    """Test reshape error when non -1 dimensions product is 0."""
    with pytest.raises(ValueError, match="Cannot reshape"):
        infer_reshape((2, 3), (0, -1))


def test_squeeze_axis_out_of_range():
    """Test squeeze error when axis is out of range."""
    with pytest.raises(ValueError, match="out of range"):
        infer_squeeze((1, 2, 3), axis=5)


def test_flatten_with_symbolic():
    """Test flattening when tensor contains symbolic dimensions."""
    out = infer_flatten((2, S("H"), 4), start_dim=1, end_dim=2)
    assert out[0] == 2
    assert isinstance(out[1], S)


def test_split_symbolic():
    """Test splitting tensor with symbolic dimension."""
    splits = infer_split((S("batch"), 4), num_or_size_splits=2, axis=0)
    assert len(splits) == 2
    assert isinstance(splits[0][0], S)


def test_concat_symbolic():
    """Test concatenating tensors with symbolic dimension."""
    out = infer_concat([(S("batch"), 4), (S("batch"), 4)], axis=0)
    assert isinstance(out[0], S)
    assert out[1] == 4


def test_infer_op_shape_declarative_empty_and_branch_coverage():
    """Test empty input handling across all declarative op categories."""
    assert infer_op_shape_declarative("Add") == ()
    assert infer_op_shape_declarative("ReduceSum") == ()
    assert infer_op_shape_declarative("Reshape") == ()
    assert infer_op_shape_declarative("Transpose") == ()
    assert infer_op_shape_declarative("Squeeze") == ()
    assert infer_op_shape_declarative("ExpandDims") == ()
    assert infer_op_shape_declarative("Flatten") == ()
    assert infer_op_shape_declarative("Split") == ()

    # Reshape without target kwarg
    assert infer_op_shape_declarative("Reshape", (2, 3)) == (2, 3)

    # Transpose without axes kwarg
    assert infer_op_shape_declarative("Transpose", (2, 3)) == (3, 2)

    # Contracting with 1 shape and 2 shapes
    assert infer_op_shape_declarative("Matmul", (2, 3)) == (2, 3)
    assert infer_op_shape_declarative("Matmul", (2, 3), (3, 4)) == (2, 4)

    # Contracting with 0 shapes raises
    with pytest.raises(ValueError, match="requires 2 operands"):
        infer_op_shape_declarative("Matmul")

    # Spatial with 0 shapes raises
    with pytest.raises(ValueError, match="requires input and weight"):
        infer_op_shape_declarative("Conv2D")

    # Pooling with 0 shapes raises
    with pytest.raises(ValueError, match="requires an input shape"):
        infer_op_shape_declarative("MaxPool2D")

    # Unknown op fallback
    assert infer_op_shape_declarative("NonExistentOp") == ()


def test_infer_shape_input_extractors():
    """Test infer_shape with objects having shape_metadata, shape, or tuples."""

    class DummyMeta:
        def __init__(self, s):
            self.shape_metadata = s

    class DummyShape:
        def __init__(self, s):
            self.shape_metadata = None
            self.shape = s

    # Positional args
    assert infer_shape("Add", DummyMeta((2, 3)), DummyShape((1, 3))) == (2, 3)
    assert infer_shape("Add", (2, 3), [1, 3]) == (2, 3)

    # inputs kwarg
    assert infer_shape("Add", inputs=[DummyMeta((2, 3)), DummyShape((1, 3))]) == (2, 3)
    assert infer_shape("Add", inputs=[(2, 3), [1, 3]]) == (2, 3)


def test_coverage_gap_fillers():
    """Fill remaining branch coverage gaps in shape_inference.py."""
    # Line 129: compute_contiguous_strides with non-int in middle
    assert compute_contiguous_strides((3, S("batch"), 4)) == (4, 4, 1)

    # Line 290: infer_conv with sequence dilations and sequence padding
    out_conv = infer_conv((1, 2, 8, 8), (4, 2, 3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), spatial_dims=2)
    assert out_conv == (1, 4, 8, 8)

    # Line 362 & 378: infer_pool with sequence kernel_size and sequence padding
    out_pool = infer_pool((1, 2, 8, 8), kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), spatial_dims=2)
    assert out_pool == (1, 2, 4, 4)

    # Line 615: infer_stack with negative axis
    assert infer_stack([(2, 3), (2, 3)], axis=-1) == (2, 3, 2)

    # Dispatches via infer_op_shape_declarative:
    # Reduction
    assert infer_op_shape_declarative("ReduceSum", (2, 3, 4), axis=1, keepdims=True) == (2, 1, 4)

    # Squeeze & ExpandDims
    assert infer_op_shape_declarative("Squeeze", (1, 2, 1), axis=0) == (2, 1)
    assert infer_op_shape_declarative("ExpandDims", (2, 3), axis=1) == (2, 1, 3)

    # Flatten, Split, Concat, Stack
    assert infer_op_shape_declarative("Flatten", (2, 3, 4), start_dim=1, end_dim=2) == (2, 12)
    assert infer_op_shape_declarative("Split", (4, 6), num_or_size_splits=2, axis=0) == (2, 6)
    assert infer_op_shape_declarative("Concat", (2, 3), (2, 4), axis=1) == (2, 7)
    assert infer_op_shape_declarative("Stack", (2, 3), (2, 3), axis=0) == (2, 2, 3)

    # Spatial & Pooling 1D, 2D and 3D
    assert infer_op_shape_declarative("Conv1D", (1, 2, 10), (4, 2, 3)) == (1, 4, 8)
    assert infer_op_shape_declarative("Conv2D", (1, 2, 8, 8), (4, 2, 3, 3)) == (1, 4, 6, 6)
    assert infer_op_shape_declarative("Conv3D", (1, 2, 8, 8, 8), (4, 2, 3, 3, 3)) == (1, 4, 6, 6, 6)
    assert infer_op_shape_declarative("MaxPool1D", (1, 2, 10), kernel_size=2) == (1, 2, 5)
    assert infer_op_shape_declarative("MaxPool2D", (1, 2, 8, 8), kernel_size=2) == (1, 2, 4, 4)
    assert infer_op_shape_declarative("MaxPool3D", (1, 2, 8, 8, 8), kernel_size=2) == (1, 2, 4, 4, 4)

    # Empty shapes for concat and stack
    assert infer_concat([]) == ()
    assert infer_stack([]) == ()

    # Conv wrong rank error
    with pytest.raises(ValueError, match="Convolution expects rank"):
        infer_conv((1, 2), (1, 2), spatial_dims=2)

    # Pool negative dimension error
    with pytest.raises(ValueError, match="Negative or zero"):
        infer_pool((1, 1, 2, 2), kernel_size=5, spatial_dims=2)

    # Split sum of sizes mismatch error
    with pytest.raises(ValueError, match="Sum of split sizes"):
        infer_split((6, 4), num_or_size_splits=[1, 2], axis=0)

    # Spatial op with only 1 shape
    with pytest.raises(ValueError, match="requires input and weight"):
        infer_op_shape_declarative("Conv2D", (1, 2, 8, 8))

    # Fallback with valid shapes and custom category
    from ml_switcheroo_compiler.ops.shape_inference import ShapeOpSignature, load_shape_signatures

    cfg = load_shape_signatures()
    cfg.operations["TestCustomCat"] = ShapeOpSignature(category="custom_unknown")
    assert infer_op_shape_declarative("TestCustomCat", (2, 3), (1, 3)) == (2, 3)
