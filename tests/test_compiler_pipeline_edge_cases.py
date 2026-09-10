"""Tests for compiler pipeline edge cases, export archives, and autograd branches."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.registry import _LOADERS, BackendRegistry
from ml_switcheroo_compiler.core.device import Device, DeviceType
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.errors import CompilationError
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.core.utils.graph_utils import _TopologicalSorter, topological_sort
from ml_switcheroo_compiler.export.export_api import (
    ExportArchive,
    validate_onnx_binary,
    validate_saved_model_binary,
)
from ml_switcheroo_compiler.grad.api import (
    RegisterGradient,
    backward,
    grad,
    hook_gradient,
    ir_grad,
    overwrite_with_gradient,
    value_and_grad,
)
from ml_switcheroo_compiler.grad.utils import (
    GradOptions,
    _check_scalar,
    _compute_grad_and_value,
    _convert_to_tensors,
    _find_wrt_tensors,
    _get_concrete_val,
    _get_inputs_dict,
    _to_original_type,
)
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.passes.loop_tiling import _split_shape, loop_tiling_pass
from ml_switcheroo_compiler.transforms.passes.poly_lower import polyfill_lowering_pass
from ml_switcheroo_compiler.utils.generic_utils import (
    ArchiveConfig,
    CacheConfig,
    GetFileConfig,
    Progbar,
    ProgbarConfig,
    get_file,
)

device = Device(DeviceType.CPU, 0)


def test_export_api_empty_graph_outputs(tmp_path) -> None:
    """Test ExportArchive.write_out when graph has empty outputs list and validation edge cases."""
    archive = ExportArchive()
    g = IRGraph(name="empty_out_graph")
    g.inputs = ["x"]
    g.nodes["x"] = IRNode(id="x", op_type="Input", shape_metadata=(1,))
    g.outputs = []

    out_dir = str(tmp_path / "saved_model_empty_outs")
    archive.write_out(out_dir, graph=g)
    assert os.path.exists(os.path.join(out_dir, "saved_model.pb"))

    # 1. Fallback header reader when validate_onnx_model_bytes succeeds or raises exception
    short_file = tmp_path / "corrupt_short.onnx"
    with open(short_file, "wb") as f:
        f.write(b"xx")

    valid_file = tmp_path / "valid_len.onnx"
    with open(valid_file, "wb") as f:
        f.write(b"12345678")

    with patch(
        "ml_switcheroo_compiler.backends.edge.onnx.validate_onnx_model_bytes",
        return_value={"ir_version": 7, "opset_version": 14},
    ):
        assert validate_onnx_binary(str(short_file)) is True

    with patch(
        "ml_switcheroo_compiler.backends.edge.onnx.validate_onnx_model_bytes",
        side_effect=Exception("Corrupt"),
    ):
        assert validate_onnx_binary(str(valid_file)) is True
        assert validate_onnx_binary(str(short_file)) is False

    # 2. SavedModel where saved_model.pb is 0 bytes (covering line 276)
    zero_pb_sm = tmp_path / "zero_pb_sm"
    os.makedirs(zero_pb_sm, exist_ok=True)
    with open(zero_pb_sm / "saved_model.pb", "wb"):
        pass
    assert validate_saved_model_binary(str(zero_pb_sm)) is False

    # 3. SavedModel where variables directory is completely missing (covering line 279)
    missing_vars_sm = tmp_path / "missing_vars_sm"
    os.makedirs(missing_vars_sm, exist_ok=True)
    with open(missing_vars_sm / "saved_model.pb", "wb") as f:
        f.write(b"mock_pb")
    assert validate_saved_model_binary(str(missing_vars_sm)) is False


def test_grad_api_backward_branches() -> None:
    """Test backward() branches: missing outputs, present outputs, no tracing, no wrt_ids, missing loss_id."""
    from ml_switcheroo_compiler.core.errors import TracingError

    # 1. Non-tensor and not tracing raises TracingError
    dummy_obj = MagicMock()
    with pytest.raises(TracingError, match="tracing is not active"):
        backward(dummy_obj)

    t_outside = Tensor(np.array(1.0), TensorConfig((), DType.Float32, device))
    with pytest.raises(TracingError, match="tracing is not active"):
        backward(t_outside)

    # 2. In tracing, but no wrt_ids (lines 65-66)
    g_no_wrt = IRGraph()
    g_no_wrt.nodes["loss"] = IRNode(id="loss", op_type="Input")
    t_loss = Tensor(MagicMock(id="loss"), TensorConfig((), DType.Float32, device))

    with patch("ml_switcheroo_compiler.tracing.state.global_tracing_state.is_tracing", True):
        with patch("ml_switcheroo_compiler.tracing.state.global_tracing_state.active_graph", g_no_wrt):
            with patch("ml_switcheroo_compiler.grad.api._find_wrt_tensors", return_value=([], [])):
                backward(t_loss)
                assert t_loss.grad == 1.0

    # 3. loss_id is None (line 61) & multiple wrt_ids where one is present and one is absent (lines 84->82, 85-87)
    t_wrt1 = Tensor(MagicMock(id="inp1"), TensorConfig((), DType.Float32, device, requires_grad=True))
    t_wrt2 = Tensor(MagicMock(id="inp2"), TensorConfig((), DType.Float32, device, requires_grad=True))
    t_no_id = Tensor("literal_data", TensorConfig((), DType.Float32, device))

    with patch("ml_switcheroo_compiler.tracing.state.global_tracing_state.is_tracing", True):
        with patch("ml_switcheroo_compiler.tracing.state.global_tracing_state.active_graph", g_no_wrt):
            with patch("ml_switcheroo_compiler.grad.api._find_wrt_tensors", return_value=([t_wrt1, t_wrt2], ["inp1", "inp2"])):
                with patch("ml_switcheroo_compiler.transforms.autodiff.grad") as mock_grad:
                    # Return 2 outputs: g1 (present in outputs_dict) and g2 (missing in outputs_dict)
                    mock_grad.return_value = MagicMock(outputs=["g1", "g2"])
                    with patch("ml_switcheroo_compiler.interpreter.evaluator.evaluate_graph", return_value={"g1": 0.5}):
                        backward(t_no_id)
                        assert t_wrt1.grad == 0.5


def test_grad_api_features() -> None:
    """Test RegisterGradient, overwrite_with_gradient, ir_grad, hook_gradient, and value_and_grad."""
    # RegisterGradient decorator (line 99)
    decorator = RegisterGradient("MockOp")
    assert callable(decorator)

    # overwrite_with_gradient backward pass (lines 137, 149)
    t = Tensor(np.array(3.0), TensorConfig((), DType.Float32, device))
    g = Tensor(np.array(2.0), TensorConfig((), DType.Float32, device))
    grad_ow = grad(lambda x: overwrite_with_gradient(x, g))
    res_ow = grad_ow(t)
    assert res_ow is not None

    # ir_grad (lines 165-180)
    ig = ir_grad(lambda x: x)
    assert callable(ig)
    t_in = Tensor(np.array(1.0), TensorConfig((), DType.Float32, device))
    res_ig = ig(t_in)
    assert res_ig is not None

    # value_and_grad with has_aux=True (line 207)
    def aux_fn(x):
        return x, "aux_info"

    vg_fn = value_and_grad(aux_fn, options=GradOptions(has_aux=True))
    val_aux, g_val = vg_fn(t_in)
    assert val_aux[1] == "aux_info"

    g_aux_fn = grad(aux_fn, options=GradOptions(has_aux=True))
    grads_aux, aux_val = g_aux_fn(t_in)
    assert aux_val == "aux_info"

    # hook_gradient backward pass with hook returning None (line 293)
    grad_hook_none = grad(lambda x: hook_gradient(x, lambda g_in: None))
    res_hook_none = grad_hook_none(t)
    assert res_hook_none is not None


def test_grad_utils_branches() -> None:
    """Test branch coverage in grad/utils.py."""
    g = IRGraph()
    g.nodes["n1"] = IRNode(id="n1", op_type="Input")
    g.inputs = ["n1"]

    # 1. Tensor with data having no .id (line 128->127), tensor not in graph, and tensor requires_grad=False
    t_raw_numpy = Tensor(np.array(1.0), TensorConfig((), DType.Float32, device))
    mock_not_in_graph = MagicMock(id="unknown_node")
    t_outside_graph = Tensor(mock_not_in_graph, TensorConfig((), DType.Float32, device, requires_grad=True))
    mock_n1 = MagicMock(id="n1")
    t_no_grad = Tensor(mock_n1, TensorConfig((), DType.Float32, device, requires_grad=False))

    with patch("gc.get_objects", return_value=[t_raw_numpy, t_outside_graph, t_no_grad]):
        wrt_t, wrt_ids = _find_wrt_tensors(g)
        assert wrt_t == []
        assert wrt_ids == []

    # 2. _check_scalar with shape (1,) (line 209->211) and invalid shape
    from ml_switcheroo_compiler.core.errors import SwitcherooError

    t_scalar_1 = Tensor(np.array([1.0]), TensorConfig((1,), DType.Float32, device))
    _check_scalar(t_scalar_1)
    t_scalar_2 = Tensor(np.array([[1.0, 2.0]]), TensorConfig((1, 2), DType.Float32, device))
    with pytest.raises(SwitcherooError):
        _check_scalar(t_scalar_2)

    # 3. _convert_to_tensors with float32 and bool arrays (line 209->211)
    primals_f32 = _convert_to_tensors([np.array([1.0, 2.0], dtype=np.float32)])
    assert primals_f32[0].dtype == DType.Float32
    primals_bool = _convert_to_tensors([np.array([True], dtype=bool)])
    assert primals_bool[0].dtype == DType.Bool

    # 4. _get_concrete_val with _data not ProxyTensor
    t_dummy = Tensor(np.array(1.0), TensorConfig((), DType.Float32, device))
    t_dummy._data = 42
    assert _get_concrete_val(t_dummy) == 42

    # 5. _get_inputs_dict
    mock_out = MagicMock(id="outside_node", concrete_value=5.0)
    t_outside = Tensor(mock_out, TensorConfig((), DType.Float32, device))

    mock_none = MagicMock(id="n1", concrete_value=None)
    t_none_val = Tensor(mock_none, TensorConfig((), DType.Float32, device))
    t_none_val._data = None

    mock_valid = MagicMock(id="n1", concrete_value=2.5)
    t_valid = Tensor(mock_valid, TensorConfig((), DType.Float32, device))

    with patch("gc.get_objects", return_value=[t_outside, t_none_val, t_valid]):
        inp_dict = _get_inputs_dict(g)
        assert "n1" in inp_dict

    # 6. _to_original_type with boolean array (line 257)
    bool_arr = np.array([True, False], dtype=bool)
    orig_t = Tensor(np.array([1, 0]), TensorConfig((2,), DType.Int32, device))
    t_res_bool = _to_original_type(bool_arr, orig_t)
    assert t_res_bool.dtype == DType.Bool

    # 7. _compute_grad_and_value with active tracing (lines 314-319)
    with patch("ml_switcheroo_compiler.tracing.state.global_tracing_state.is_tracing", True):
        with patch("ml_switcheroo_compiler.tracing.state.global_tracing_state.active_graph", g):
            val, grads = _compute_grad_and_value(lambda x: x, GradOptions(), (t_valid,))
            assert val is not None


def test_graph_utils_topological_sort_branches() -> None:
    """Test graph_utils: cycle detection, non-dict/non-list nodes, and visited branch."""
    # 1. Cycle detection (lines 32-33)
    g_cycle = IRGraph()
    g_cycle.nodes["a"] = IRNode(id="a", op_type="Add", inputs=["b"])
    g_cycle.nodes["b"] = IRNode(id="b", op_type="Add", inputs=["a"])

    with pytest.raises(CompilationError) as exc:
        topological_sort(g_cycle)
    assert "Cycle detected in graph" in str(exc.value)

    # 2. list nodes where visited node is skipped and node not found in list (lines 43->48, 65-67, 69-67)
    class MockGraphList:
        def __init__(self):
            n1 = IRNode(id="n1", op_type="Input", inputs=[])
            n2 = IRNode(id="n2", op_type="Add", inputs=["n1"])
            self.nodes = [n2, n1]

    g_list = MockGraphList()
    sorter = _TopologicalSorter(g_list)
    sorter.visit("node_absent_from_list")  # triggers lines 42->48 and 43->48
    sorted_nodes = sorter.sort()
    assert len(sorted_nodes) == 2

    # 3. non-dict, non-list nodes iterable (lines 42->48)
    class MockGraphTuple:
        def __init__(self):
            n1 = IRNode(id="n1", op_type="Input", inputs=[])
            self.nodes = (n1,)

    g_tuple = MockGraphTuple()
    sorter2 = _TopologicalSorter(g_tuple)
    sorted_tuple = sorter2.sort()
    assert len(sorted_tuple) == 0


def test_generic_utils_branches(tmp_path) -> None:
    """Test generic_utils branches: custom cache_dir, untar=False, verbose!=1."""
    # 1. get_file with cache_dir not None and untar/extract False
    custom_cache = str(tmp_path / "custom_cache")
    config = GetFileConfig(
        cache_config=CacheConfig(cache_dir=custom_cache, cache_subdir="test_sub"),
        archive_config=ArchiveConfig(untar=False, extract=False),
    )

    with patch("ml_switcheroo_compiler.utils.generic_utils._download_remote_file") as mock_down:
        mock_down.side_effect = lambda origin, fpath: open(fpath, "w").write("hello")
        res_path = get_file("dest.txt", "http://example.com/dest.txt", config=config)
        assert os.path.exists(res_path)

    # 2. Progbar with verbose != 1 (e.g. verbose = 0)
    pb = Progbar(target=10, config=ProgbarConfig(verbose=0))
    pb.update(5)
    assert pb._seen_so_far == 5


def test_backends_registry_lazy_loader_branches() -> None:
    """Test BackendRegistry lazy loading branches when backend not in _LOADERS (lines 146, 197)."""
    dummy_key = "dummy_unregistered_lazy_backend"
    BackendRegistry._LAZY_MODULES[dummy_key] = "some.dummy.module"

    try:
        # Line 146: name in _LAZY_MODULES but NOT in _LOADERS
        assert dummy_key not in _LOADERS
        BackendRegistry._try_load_lazy(dummy_key)

        # Line 197: inside get_all() where name not in _LOADERS
        all_b = BackendRegistry.get_all()
        assert isinstance(all_b, dict)
    finally:
        BackendRegistry._LAZY_MODULES.pop(dummy_key, None)


def test_poly_lower_unknown_rule_branches() -> None:
    """Test polyfill_lowering_pass when rule has unknown python func or other rule type (lines 186, 189)."""
    g = IRGraph()
    g.nodes["node_unknown_py"] = IRNode(id="node_unknown_py", op_type="UnknownPyOp")
    g.nodes["node_other_type"] = IRNode(id="node_other_type", op_type="OtherTypeOp")

    custom_rules = {
        "UnknownPyOp": {"type": "python", "func": "_nonexistent_handler"},
        "OtherTypeOp": {"type": "unsupported_type"},
    }

    with patch("ml_switcheroo_compiler.transforms.passes.poly_lower._load_poly_rules", return_value=custom_rules):
        modified = polyfill_lowering_pass(g)
        assert not modified


def test_loop_tiling_symbolic_shapes_and_no_split() -> None:
    """Test loop_tiling with symbolic non-integer shapes and no-split branch (lines 93, 105, 144)."""
    # 1. Matmul with non-integer symbolic dimensions (line 93)
    shape_sym_matmul = (4, "M", "N")
    op_config_matmul = {"TILE_M": 16, "TILE_N": 16}
    assert _split_shape("matmul", shape_sym_matmul, op_config_matmul) == shape_sym_matmul

    # 2. Conv2D with non-integer symbolic dimensions (line 105)
    shape_sym_conv = (1, "H", "W", 3)
    op_config_conv = {"TILE_H": 16, "TILE_W": 16}
    assert _split_shape("conv2d", shape_sym_conv, op_config_conv) == shape_sym_conv

    # 3. loop_tiling_pass where new_shape == shape (line 144)
    g = IRGraph()
    g.nodes["small_mat"] = IRNode(id="small_mat", op_type="matmul", shape_metadata=(4, "M", "N"))
    custom_tiling_config = {
        "matmul": {"TILE_M": 16, "TILE_N": 16, "MIN_M": 1, "MIN_N": 1},
    }
    with patch("ml_switcheroo_compiler.transforms.passes.loop_tiling._get_tiling_config", return_value=custom_tiling_config):
        modified = loop_tiling_pass(g)
        assert not modified
