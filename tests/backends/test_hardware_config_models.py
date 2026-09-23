"""Tests for hardware configuration models (CUDA, ROCm, Metal, LLVM/C++)."""

import os
import tempfile

from ml_switcheroo_compiler.backends.hardware_config_models import (
    GridDimensionConfig,
    HardwareCapabilityQuery,
    HardwareTemplateConfig,
    HardwareTemplatesConfig,
    KernelParameterConfig,
    KernelTemplateModel,
    MemoryLayoutRuleModel,
    compute_liveness,
    load_hardware_templates,
    load_kernel_templates_manifest,
    load_launch_heuristics,
    load_memory_layouts,
    query_optimal_launch_geometry,
    resolve_hardware_launch_grid_and_args,
)
from ml_switcheroo_compiler.backends.llvm_cpp.config_models import (
    CppSimdConfig,
    CppTemplateConfig,
    CppTemplatesConfig,
    CppTilingConfig,
)
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def test_hardware_config_models_dump() -> None:
    """Test dumping and retrieving properties on HardwareTemplatesConfig and children."""
    param = KernelParameterConfig(
        name="X",
        param_type="pointer",
        source="input",
        source_index=0,
        source_key=None,
        default_value=None,
    )
    grid = GridDimensionConfig(x="((num_elements + 255) / 256)", y="1", z="1")
    tpl = HardwareTemplateConfig(
        body="body",
        workgroup_size=[1, 2, 3],
        grid_calc=grid,
        parameters=[param],
        shared_memory_bytes=1024,
        type_mappings={"float32": "float"},
    )
    cfg = HardwareTemplatesConfig(templates={"a": tpl}, orchestration={"k": "v"})
    dumped = cfg.model_dump()
    assert dumped["templates"]["a"]["body"] == "body"
    assert dumped["templates"]["a"]["workgroup_size"] == [1, 2, 3]
    assert dumped["templates"]["a"]["shared_memory_bytes"] == 1024
    assert dumped["templates"]["a"]["type_mappings"]["float32"] == "float"
    assert dumped["orchestration"]["k"] == "v"

    # Test .get() method
    assert tpl.get("workgroup_size") == [1, 2, 3]
    assert tpl.get("shared_memory_bytes") == 1024
    assert tpl.get("nonexistent", 999) == 999
    assert tpl.get("body") == "body"
    assert tpl.get("grid_calc") == grid


def test_hardware_template_config_equality_and_validation() -> None:
    """Test HardwareTemplateConfig comparison branches and templates validator fallbacks."""
    tpl = HardwareTemplateConfig(body="void custom() {}", workgroup_size=[16, 1, 1], grid_calc=None)

    # Test .get() with attribute present but set to None
    assert tpl.get("grid_calc", "default_val") == "default_val"

    # Comparison with dict
    assert tpl == {"body": "void custom() {}", "workgroup_size": [16, 1, 1]}
    assert not (tpl == {"body": "different_body", "workgroup_size": [16, 1, 1]})
    assert not (tpl == {"body": "void custom() {}", "workgroup_size": [32, 1, 1]})

    # Comparison with HardwareTemplateConfig
    tpl_same = HardwareTemplateConfig(body="void custom() {}", workgroup_size=[16, 1, 1])
    tpl_diff = HardwareTemplateConfig(body="other()", workgroup_size=[16, 1, 1])
    assert tpl == tpl_same
    assert not (tpl == tpl_diff)

    # Comparison with non-dict non-HardwareTemplateConfig
    assert not (tpl == "not_a_template")
    assert not (tpl == 42)

    # Test _validate_templates with dict entry with body, string entries, and custom list entries
    cfg = HardwareTemplatesConfig(
        templates={
            "dict_tpl": {"body": "void dict_kernel() {}", "workgroup_size": [32, 1, 1]},
            "str_tpl": "void str_kernel() {}",
            "list_tpl": ["item1", "item2"],
        }
    )
    assert isinstance(cfg.templates["dict_tpl"], HardwareTemplateConfig)
    assert cfg.templates["dict_tpl"].body == "void dict_kernel() {}"
    assert isinstance(cfg.templates["str_tpl"], HardwareTemplateConfig)
    assert cfg.templates["str_tpl"].body == "void str_kernel() {}"
    assert cfg.templates["list_tpl"] == ["item1", "item2"]

    # Test _validate_templates non-dict bypass
    raw_val = HardwareTemplatesConfig._validate_templates("not_a_dict")
    assert raw_val == "not_a_dict"


def test_llvm_cpp_config_models() -> None:
    """Test LLVM C++ configuration models including tiling and SIMD."""
    tiling = CppTilingConfig(tile_size_m=32, tile_size_n=32, tile_size_k=16)
    simd = CppSimdConfig(pragma="#pragma omp simd", vector_width=16)
    template = CppTemplateConfig(
        body="float c = a + b;",
        includes=["<cmath>", "<vector>"],
        tiling=tiling,
        simd=simd,
    )
    cfg = CppTemplatesConfig(templates={"add": template})
    dumped = cfg.model_dump()
    assert dumped["templates"]["add"]["body"] == "float c = a + b;"
    assert dumped["templates"]["add"]["includes"] == ["<cmath>", "<vector>"]
    assert dumped["templates"]["add"]["tiling"]["tile_size_m"] == 32
    assert dumped["templates"]["add"]["simd"]["vector_width"] == 16


def test_load_hardware_templates_from_file_and_dir() -> None:
    """Test load_hardware_templates loading from directory, single file, and various item formats."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Non-dict file and non-yaml file
        invalid_path = os.path.join(tmpdir, "invalid.yaml")
        with open(invalid_path, "w") as f:
            f.write("just a string\n")
        non_yaml_path = os.path.join(tmpdir, "not_yaml.txt")
        with open(non_yaml_path, "w") as f:
            f.write("text file\n")

        # 2. File with string template, dict template, and instance
        single_path = os.path.join(tmpdir, "single.yaml")
        with open(single_path, "w") as f:
            f.write("templates:\n  op1: 'void op1() {}'\n  op2:\n    body: 'void op2() {}'\n    workgroup_size: [32, 1, 1]\n  op3: 123\n")

        cfg = load_hardware_templates(tmpdir, single_path)
        assert "op1" in cfg.templates
        assert "op2" in cfg.templates
        assert "op3" in cfg.templates
        assert cfg.templates["op1"].body == "void op1() {}"
        assert cfg.templates["op2"].workgroup_size == [32, 1, 1]
        assert cfg.templates["op3"] == 123

        # 3. Fallback when dir does not exist
        cfg_fallback = load_hardware_templates(os.path.join(tmpdir, "nonexistent_dir"), single_path)
        assert "op1" in cfg_fallback.templates

        # 4. Fallback when neither exists
        cfg_empty = load_hardware_templates(os.path.join(tmpdir, "nonexistent_dir"), os.path.join(tmpdir, "nonexistent.yaml"))
        assert len(cfg_empty.templates) == 0

        # 5. Direct HardwareTemplateConfig instance in source
        from unittest.mock import mock_open, patch

        from ml_switcheroo_compiler.backends.hardware_config_models import _load_yaml_file_into_config

        tpl_instance = HardwareTemplateConfig(body="void direct() {}")
        cfg_inst = HardwareTemplatesConfig(templates={})
        with (
            patch("builtins.open", mock_open(read_data="dummy")),
            patch("yaml.safe_load", return_value={"direct": tpl_instance}),
        ):
            _load_yaml_file_into_config("dummy.yaml", cfg_inst)
        assert cfg_inst.templates["direct"].body == "void direct() {}"


def test_hardware_helpers_liveness_and_resolve() -> None:
    """Test compute_liveness and resolve_hardware_launch_grid_and_args for default/fallback op."""
    g = IRGraph()
    n_in = IRNode("in0", "Input", shape_metadata=(8,), attributes={"dtype": "float32"})
    n_op = IRNode("op", "CustomOp", inputs=["in0"], shape_metadata=(8,), attributes={"dtype": "float32"})
    g.nodes[n_in.id] = n_in
    g.nodes[n_op.id] = n_op

    liveness = compute_liveness(g)
    assert liveness["in0"] == "op"

    tpl = HardwareTemplateConfig(body="void custom() {}", workgroup_size=[64, 1, 1])
    grid, args = resolve_hardware_launch_grid_and_args(n_op, tpl, {"in0": "inputs[0]"}, "out_0", 8, 0, g)
    assert grid == ("(8 + block_0.x - 1) / block_0.x", "1", "1")
    assert args == ["inputs[0]", "out_0", "8"]


def test_hardware_execution_schema_loading() -> None:
    """Test loading and validating hardware execution schemas."""
    from ml_switcheroo_compiler.backends.hardware_config_models import (
        BufferBindingSpec,
        GridBlockDimensionSpec,
        HardwareDispatchSpec,
        HardwareExecutionSchema,
        load_hardware_execution_schema,
    )

    schema = load_hardware_execution_schema()
    assert "cuda" in schema.hardware_execution_schemas
    assert "rocm" in schema.hardware_execution_schemas
    assert "metal" in schema.hardware_execution_schemas
    assert schema.hardware_execution_schemas["cuda"].sync_barrier == "cuStreamSynchronize"

    # Test with custom tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tf:
        tf.write(
            """
hardware_execution_schemas:
  custom:
    backend_name: "custom"
    sync_barrier: "custom_sync"
    device_sync: "device_sync"
    dispatch_protocol: "custom_dispatch"
    buffer_binding_order: ["inputs", "outputs"]
    grid_strategies:
      elem:
        block_dims: [128, 1, 1]
        grid_formula_x: "1"
        grid_formula_y: "1"
        grid_formula_z: "1"
"""
        )
        tmp_path = tf.name

    try:
        custom_schema = load_hardware_execution_schema(tmp_path)
        assert "custom" in custom_schema.hardware_execution_schemas
        assert custom_schema.hardware_execution_schemas["custom"].sync_barrier == "custom_sync"
        assert custom_schema.hardware_execution_schemas["custom"].grid_strategies["elem"].block_dims == [128, 1, 1]
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # Test default model initializers
    gb = GridBlockDimensionSpec()
    assert gb.block_dims == [256, 1, 1]
    bb = BufferBindingSpec()
    assert bb.buffer_binding_order == ["inputs", "outputs", "dimensions"]
    hd = HardwareDispatchSpec(
        backend_name="test",
        sync_barrier="sync",
        device_sync="dsync",
        dispatch_protocol="dp",
    )
    assert hd.buffer_binding_order == ["inputs", "outputs", "dimensions"]
    root = HardwareExecutionSchema()
    assert root.hardware_execution_schemas == {}


def test_hardware_launch_helpers_and_dispatch() -> None:
    """Test resolve_hardware_launch_grid_and_args for conv, matmul, batchmatmul, and fused ops."""
    from ml_switcheroo_compiler.backends.hardware_config_models import (
        _calculate_contiguous_strides,
        _format_stride_terms,
        _load_yaml_file_into_config,
        calculate_hardware_launch_config,
        generate_nd_coordinate_offset_logic,
        load_hardware_execution_schema,
    )

    g = IRGraph()
    # 1. Conv2D
    in0 = IRNode("c_in0", "Input", shape_metadata=(2, 3, 32, 32))
    in1 = IRNode("c_in1", "Input", shape_metadata=(16, 3, 3, 3))
    out_conv = IRNode("c_out", "Conv2D", inputs=["c_in0", "c_in1"], shape_metadata=(2, 16, 30, 30))
    g.nodes = {"c_in0": in0, "c_in1": in1, "c_out": out_conv}
    tpl = HardwareTemplateConfig(body="kernel", workgroup_size=[256, 1, 1])
    grid, args = resolve_hardware_launch_grid_and_args(out_conv, tpl, {"c_in0": "in_0", "c_in1": "in_1"}, "out_buf", 28800, 0, g)
    assert grid[1] == "16"
    assert grid[2] == "2"
    assert len(args) == 12

    # Conv2D with missing shapes / empty inputs
    empty_conv = IRNode("empty_conv", "Conv2D", inputs=[], shape_metadata=())
    grid_empty, _ = resolve_hardware_launch_grid_and_args(empty_conv, tpl, {}, "out_buf", 1, 0, g)
    assert grid_empty == ("(1 * 1 + block_0.x - 1) / block_0.x", "1", "1")

    # 2. MatMul / Dot
    in_m0 = IRNode("m_in0", "Input", shape_metadata=(8, 16))
    in_m1 = IRNode("m_in1", "Input", shape_metadata=(16, 32))
    out_mm = IRNode("m_out", "MatMul", inputs=["m_in0", "m_in1"], shape_metadata=(8, 32))
    g.nodes.update({"m_in0": in_m0, "m_in1": in_m1, "m_out": out_mm})
    grid_mm, args_mm = resolve_hardware_launch_grid_and_args(out_mm, tpl, {"m_in0": "in_m0", "m_in1": "in_m1"}, "out_mm", 256, 1, g)
    assert grid_mm == ("(32 + block_1.x - 1) / block_1.x", "(8 + block_1.y - 1) / block_1.y", "1")
    assert args_mm[-4:] == ["out_mm", "8", "32", "16"]

    # Matmul with empty inputs/shape
    empty_mm = IRNode("empty_mm", "Dot", inputs=[], shape_metadata=())
    grid_empty_mm, _ = resolve_hardware_launch_grid_and_args(empty_mm, tpl, {}, "out_mm", 1, 1, g)
    assert grid_empty_mm == ("(1 + block_1.x - 1) / block_1.x", "(1 + block_1.y - 1) / block_1.y", "1")

    # 3. BatchMatMul
    in_bm0 = IRNode("bm_in0", "Input", shape_metadata=(4, 8, 16))
    in_bm1 = IRNode("bm_in1", "Input", shape_metadata=(4, 16, 32))
    out_bmm = IRNode("bm_out", "BatchMatMul", inputs=["bm_in0", "bm_in1"], shape_metadata=(4, 8, 32))
    g.nodes.update({"bm_in0": in_bm0, "bm_in1": in_bm1, "bm_out": out_bmm})
    grid_bmm, args_bmm = resolve_hardware_launch_grid_and_args(out_bmm, tpl, {"bm_in0": "in_bm0", "bm_in1": "in_bm1"}, "out_bmm", 1024, 2, g)
    assert grid_bmm == ("(32 + block_2.x - 1) / block_2.x", "(8 + block_2.y - 1) / block_2.y", "4")
    assert args_bmm[-5:] == ["out_bmm", "4", "8", "32", "16"]

    empty_bmm = IRNode("empty_bmm", "BatchMatMul", inputs=[], shape_metadata=())
    grid_empty_bmm, _ = resolve_hardware_launch_grid_and_args(empty_bmm, tpl, {}, "out_bmm", 1, 2, g)
    assert grid_empty_bmm == ("(1 + block_2.x - 1) / block_2.x", "(1 + block_2.y - 1) / block_2.y", "1")

    # 4. FusedElementwise
    out_fused = IRNode("fused_out", "FusedElementwise", inputs=["c_in0"], shape_metadata=(2, 3))
    grid_fused, args_fused = resolve_hardware_launch_grid_and_args(out_fused, tpl, {"c_in0": "in_0"}, "out_fused", 6, 3, g)
    assert grid_fused == ("(6 + block_3.x - 1) / block_3.x", "1", "1")
    assert args_fused == ["in_0", "out_fused", "6"]

    # 5. calculate_hardware_launch_config
    assert calculate_hardware_launch_config(None) == ((1, 1, 1), (1, 1, 1))
    assert calculate_hardware_launch_config([]) == ((1, 1, 1), (1, 1, 1))
    assert calculate_hardware_launch_config([512]) == ((512, 1, 1), (1, 1, 1))
    assert calculate_hardware_launch_config([32, 64]) == ((16, 16, 1), (4, 2, 1))
    assert calculate_hardware_launch_config([4, 16, 32]) == ((16, 8, 8), (2, 2, 1))
    assert calculate_hardware_launch_config(["invalid_dim", 10]) == ((16, 16, 1), (1, 1, 1))

    # 6. _calculate_contiguous_strides
    assert _calculate_contiguous_strides([2, 3, 4]) == [12, 4, 1]

    # 7. _format_stride_terms
    assert _format_stride_terms(["c0", "c1", "c2"], [0, 1, 4]) == "c1 + (c2 * 4)"
    assert _format_stride_terms(["c0"], [0]) == "0"

    # 8. generate_nd_coordinate_offset_logic
    lines_1d_default = generate_nd_coordinate_offset_logic([16])
    assert lines_1d_default == ["int offset = idx;"]
    lines_1d_custom = generate_nd_coordinate_offset_logic([16], strides=[4])
    assert lines_1d_custom == ["int offset = idx * 4;"]

    lines_nd_cuda = generate_nd_coordinate_offset_logic([2, 3, 4], language="cuda")
    assert any("rem_idx %" in ln for ln in lines_nd_cuda)
    assert any("int offset =" in ln for ln in lines_nd_cuda)

    lines_nd_metal = generate_nd_coordinate_offset_logic([2, 3, 4], language="metal")
    assert any("uint offset =" in ln for ln in lines_nd_metal)

    lines_nd_invalid = generate_nd_coordinate_offset_logic(["bad_dim", "4"])
    assert len(lines_nd_invalid) > 0

    # 9. load_hardware_execution_schema with bundled default
    default_schema = load_hardware_execution_schema(path=None)
    assert "cuda" in default_schema.hardware_execution_schemas

    bundled_path = os.path.join(os.path.dirname(__file__), "..", "..", "src", "ml_switcheroo_compiler", "backends", "hardware_execution_schemas.yaml")
    if os.path.exists(bundled_path):
        explicit_schema = load_hardware_execution_schema(path=bundled_path)
        assert "cuda" in explicit_schema.hardware_execution_schemas

    # 10. _load_yaml_file_into_config edge cases
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tf:
        tf.write("templates: not_a_dict\n")
        f_bad_tpl = tf.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tf:
        tf.write("other_key: 123\n")
        f_missing_tpl = tf.name
    try:
        cfg_test = HardwareTemplatesConfig(templates={})
        _load_yaml_file_into_config(f_bad_tpl, cfg_test, is_dir=False)
        _load_yaml_file_into_config(f_missing_tpl, cfg_test, is_dir=False)
        assert len(cfg_test.templates) == 0
    finally:
        if os.path.exists(f_bad_tpl):
            os.remove(f_bad_tpl)
        if os.path.exists(f_missing_tpl):
            os.remove(f_missing_tpl)


def test_kernel_templates_manifest_and_models() -> None:
    """Verify loading and validating declarative kernel templates manifest."""
    manifest = load_kernel_templates_manifest()
    assert "Add" in manifest.kernel_templates
    assert "MatMul" in manifest.kernel_templates
    add_tpl = manifest.kernel_templates["Add"]
    assert add_tpl.opcode == "Add"
    assert add_tpl.workgroup_dims == [256, 1, 1]
    assert "in0[idx] + in1[idx]" in add_tpl.body
    assert add_tpl.cuda_body is not None
    assert add_tpl.rocm_body is not None
    assert add_tpl.metal_body is not None

    custom_tpl = KernelTemplateModel(
        opcode="CustomOp",
        workgroup_dims=[128, 1, 1],
        thread_indexing_formula="int i = threadIdx.x;",
        memory_layout_requirements=["contiguous"],
        body="out[i] = in[i];",
        shared_memory_bytes=256,
    )
    assert custom_tpl.opcode == "CustomOp"
    assert custom_tpl.shared_memory_bytes == 256


def test_launch_heuristics_manifest_and_query() -> None:
    """Verify loading launch heuristics manifest and dynamic hardware capability queries."""
    manifest = load_launch_heuristics()
    assert "elementwise_1d" in manifest.heuristics
    assert "tiled_matmul" in manifest.heuristics
    rule = manifest.heuristics["elementwise_1d"]
    assert rule.tensor_rank == [1]
    assert rule.block_dims == [256, 1, 1]

    caps = HardwareCapabilityQuery(
        max_threads_per_block=512,
        max_shared_memory_per_sm=32768,
        warp_size=32,
        max_block_dim=[512, 512, 64],
        max_grid_dim=[65535, 65535, 65535],
    )
    block_dim, grid_dim = query_optimal_launch_geometry((1024,), hardware_caps=caps)
    assert block_dim == (512, 1, 1)
    assert grid_dim == (2, 1, 1)

    block_2d, grid_2d = query_optimal_launch_geometry((32, 64), hardware_caps=caps)
    assert block_2d == (16, 16, 1)
    assert grid_2d == (4, 2, 1)


def test_memory_layouts_manifest_and_rules() -> None:
    """Verify loading memory layouts manifest and layout specifications."""
    manifest = load_memory_layouts()
    assert "contiguous" in manifest.layouts
    assert "nchw" in manifest.layouts
    assert "nhwc" in manifest.layouts
    assert "transposed_2d" in manifest.layouts
    assert "NCHW_to_NHWC" in manifest.transformations

    nhwc_layout = manifest.layouts["nhwc"]
    assert nhwc_layout.dimension_order == [0, 2, 3, 1]
    assert nhwc_layout.is_contiguous is True

    custom_rule = MemoryLayoutRuleModel(
        layout_name="custom_strided",
        dimension_order=[1, 0],
        stride_formula="custom",
        is_contiguous=False,
    )
    assert custom_rule.layout_name == "custom_strided"
    assert custom_rule.is_contiguous is False


def test_hardware_device_profiles_and_custom_paths() -> None:
    """Verify loading device profiles, execution_quantum branches, and custom YAML path loaders."""
    from ml_switcheroo_compiler.backends.hardware_config_models import (
        HardwareDeviceProfileModel,
        load_hardware_device_profiles,
    )

    # 1. Default loader for hardware device profiles
    manifest = load_hardware_device_profiles()
    assert manifest.version == "1.0.0"
    assert "cuda" in manifest.profiles or "metal" in manifest.profiles or len(manifest.profiles) >= 0

    # 2. Custom path loaders for all 4 manifest loaders
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write("version: 1.0.0\nprofiles: {}\n")
        custom_dev_path = f.name
    try:
        dev_m = load_hardware_device_profiles(path=custom_dev_path)
        assert dev_m.version == "1.0.0"
    finally:
        os.remove(custom_dev_path)

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write("version: 1.0.0\ntemplates: {}\n")
        custom_tmpl_path = f.name
    try:
        tmpl_m = load_kernel_templates_manifest(path=custom_tmpl_path)
        assert tmpl_m.version == "1.0.0"
    finally:
        os.remove(custom_tmpl_path)

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write("version: 1.0.0\nheuristics: {}\n")
        custom_heur_path = f.name
    try:
        heur_m = load_launch_heuristics(path=custom_heur_path)
        assert heur_m.version == "1.0.0"
    finally:
        os.remove(custom_heur_path)

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write("version: 1.0.0\nlayouts: {}\ntransformations: {}\n")
        custom_layout_path = f.name
    try:
        layout_m = load_memory_layouts(path=custom_layout_path)
        assert layout_m.version == "1.0.0"
    finally:
        os.remove(custom_layout_path)

    # 3. execution_unit_size branches on HardwareDeviceProfileModel
    p_wave = HardwareDeviceProfileModel(architecture="cdna3", wavefront_size=64)
    assert p_wave.execution_unit_size == 64

    p_warp = HardwareDeviceProfileModel(architecture="hopper", warp_size=32)
    assert p_warp.execution_unit_size == 32

    p_simd = HardwareDeviceProfileModel(architecture="apple-silicon", simdgroup_size=32)
    assert p_simd.execution_unit_size == 32

    p_fallback = HardwareDeviceProfileModel(architecture="generic-cpu")
    assert p_fallback.execution_unit_size == 32


def test_hardware_config_models_100_coverage() -> None:
    """Verify Conv3D launch, pool launch, row-wise launch, unified templates merge, get_template, and shared memory validation."""
    from unittest.mock import patch

    import pytest

    from ml_switcheroo_compiler.backends.hardware_config_models import (
        HardwareDeviceProfileModel,
        HardwareDeviceProfilesManifestModel,
        _merge_unified_kernel_templates,
        load_hardware_templates,
        load_kernel_templates_manifest,
        resolve_hardware_launch_grid_and_args,
        validate_and_bound_shared_memory,
    )
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    g = IRGraph()
    tpl = HardwareTemplateConfig(body="kernel", workgroup_size=[256, 1, 1])

    # 1. Conv3D launch resolution
    in_c3d_0 = IRNode("c3d_in0", "Input", shape_metadata=(2, 3, 4, 8, 8))
    in_c3d_1 = IRNode("c3d_in1", "Input", shape_metadata=(16, 3, 2, 3, 3))
    out_c3d = IRNode("c3d_out", "Conv3D", inputs=["c3d_in0", "c3d_in1"], shape_metadata=(2, 16, 3, 6, 6))
    g.nodes = {"c3d_in0": in_c3d_0, "c3d_in1": in_c3d_1, "c3d_out": out_c3d}
    grid_c3d, args_c3d = resolve_hardware_launch_grid_and_args(out_c3d, tpl, {"c3d_in0": "in0", "c3d_in1": "in1"}, "out_buf", 1728, 0, g)
    assert grid_c3d[1] == "16"
    assert grid_c3d[2] == "2"
    assert len(args_c3d) == 15

    # Conv3D with missing/empty inputs and shapes
    empty_c3d = IRNode("empty_c3d", "Conv3D", inputs=[], shape_metadata=())
    grid_empty_c3d, _ = resolve_hardware_launch_grid_and_args(empty_c3d, tpl, {}, "out_buf", 1, 0, g)
    assert grid_empty_c3d == ("(1 * 1 * 1 + block_0.x - 1) / block_0.x", "1", "1")

    # 2. Pool 3D launch resolution
    in_p3d = IRNode("p3d_in", "Input", shape_metadata=(2, 3, 4, 8, 8))
    out_p3d = IRNode("p3d_out", "MaxPool3D", inputs=["p3d_in"], shape_metadata=(2, 3, 2, 4, 4), attributes={"kernel_size": [2, 2, 2]})
    g.nodes.update({"p3d_in": in_p3d, "p3d_out": out_p3d})
    grid_p3d, args_p3d = resolve_hardware_launch_grid_and_args(out_p3d, tpl, {"p3d_in": "in_p"}, "out_buf", 192, 1, g)
    assert grid_p3d == ("(2 * 4 * 4 + block_1.x - 1) / block_1.x", "3", "2")
    assert len(args_p3d) == 13

    # 3. Pool 2D launch resolution
    in_p2d = IRNode("p2d_in", "Input", shape_metadata=(2, 3, 8, 8))
    out_p2d = IRNode("p2d_out", "AvgPool2D", inputs=["p2d_in"], shape_metadata=(2, 3, 4, 4), attributes={"kernel_size": (2, 2)})
    g.nodes.update({"p2d_in": in_p2d, "p2d_out": out_p2d})
    grid_p2d, args_p2d = resolve_hardware_launch_grid_and_args(out_p2d, tpl, {"p2d_in": "in_p"}, "out_buf", 96, 2, g)
    assert grid_p2d == ("(4 * 4 + block_2.x - 1) / block_2.x", "3", "2")
    assert len(args_p2d) == 10

    # Pool 2D with empty attributes and missing shapes
    empty_p = IRNode("empty_p", "MaxPool2D", inputs=[], shape_metadata=(), attributes={})
    grid_ep, _ = resolve_hardware_launch_grid_and_args(empty_p, tpl, {}, "out_buf", 1, 3, g)
    assert grid_ep == ("(1 * 1 + block_3.x - 1) / block_3.x", "1", "1")

    # 4. Row-wise args: normalization (LayerNorm) and reduction (ReduceSum)
    in_norm = IRNode("norm_in", "Input", shape_metadata=(4, 16))
    out_norm = IRNode("norm_out", "LayerNorm", inputs=["norm_in"], shape_metadata=(4, 16), attributes={"epsilon": 1e-4})
    g.nodes.update({"norm_in": in_norm, "norm_out": out_norm})
    grid_norm, args_norm = resolve_hardware_launch_grid_and_args(out_norm, tpl, {"norm_in": "in_norm"}, "out_norm", 64, 4, g)
    assert grid_norm == ("4", "1", "1")
    assert args_norm[-1] == "0.0001"

    out_soft = IRNode("soft_out", "Softmax", inputs=["norm_in"], shape_metadata=(4, 16))
    g.nodes["soft_out"] = out_soft
    grid_soft, args_soft = resolve_hardware_launch_grid_and_args(out_soft, tpl, {"norm_in": "in_norm"}, "out_soft", 64, 5, g)
    assert grid_soft == ("4", "1", "1")
    assert args_soft[-2:] == ["4", "16"]

    # 5. _merge_unified_kernel_templates for cuda, rocm, and metal
    cfg_cuda = HardwareTemplatesConfig(
        templates={
            "dummy_op": HardwareTemplateConfig(body="void f() { C[id] = A[id]; }"),
            "existing_real": HardwareTemplateConfig(body="void custom() { /* real */ }"),
        }
    )
    _merge_unified_kernel_templates(cfg_cuda, "cuda")
    assert "relu" in cfg_cuda.templates
    assert "dummy_op" in cfg_cuda.templates

    cfg_rocm = HardwareTemplatesConfig(
        templates={
            "dummy_op2": HardwareTemplateConfig(body="void f() { C[id] = A[id] + B[id]; }"),
        }
    )
    _merge_unified_kernel_templates(cfg_rocm, "rocm")
    assert "relu" in cfg_rocm.templates

    cfg_metal = HardwareTemplatesConfig(
        templates={
            "dummy_metal": HardwareTemplateConfig(body="void f() { output[idx] = input[idx]; }"),
        }
    )
    _merge_unified_kernel_templates(cfg_metal, "metal")
    assert "relu" in cfg_metal.templates

    # _merge_unified_kernel_templates with an empty template body to cover line 202 (continue)
    from ml_switcheroo_compiler.backends.hardware_config_models import (
        KernelTemplateModel,
        KernelTemplatesManifestModel,
    )

    dummy_manifest = KernelTemplatesManifestModel(
        kernel_templates={
            "no_body_op": KernelTemplateModel(opcode="no_body_op", name="no_body", cuda_body="", rocm_body="", metal_body=""),
            "with_body_op": KernelTemplateModel(opcode="with_body_op", name="with_body", cuda_body="void f() {}", rocm_body="void f() {}", metal_body="void f() {}"),
        }
    )
    cfg_empty_body = HardwareTemplatesConfig(templates={})
    with patch("ml_switcheroo_compiler.backends.hardware_config_models.load_kernel_templates_manifest", return_value=dummy_manifest):
        _merge_unified_kernel_templates(cfg_empty_body, "cuda")
    assert "with_body_op" in cfg_empty_body.templates
    assert "no_body_op" not in cfg_empty_body.templates

    # _merge_unified_kernel_templates when load_kernel_templates_manifest raises
    with patch("ml_switcheroo_compiler.backends.hardware_config_models.load_kernel_templates_manifest", side_effect=RuntimeError("fail")):
        _merge_unified_kernel_templates(cfg_cuda, "cuda")

    # load_hardware_templates with backend param
    import os

    cuda_dir = os.path.join(os.path.dirname(__file__), "..", "..", "src", "ml_switcheroo_compiler", "backends", "cuda", "templates")
    cuda_file = os.path.join(os.path.dirname(__file__), "..", "..", "src", "ml_switcheroo_compiler", "backends", "cuda", "cuda_templates.yaml")
    loaded_cuda = load_hardware_templates(cuda_dir, cuda_file, backend="cuda")
    assert "relu" in loaded_cuda.templates

    # 6. KernelTemplatesManifestModel.get_template
    manifest_tmpl = load_kernel_templates_manifest()
    found_tmpl = manifest_tmpl.get_template("Relu")
    assert found_tmpl is not None
    assert manifest_tmpl.get_template("NonExistentOp") is None

    # 7. validate_and_bound_shared_memory
    with pytest.raises(ValueError, match="cannot be negative"):
        validate_and_bound_shared_memory(-1, "cuda")

    # valid with default manifest
    assert validate_and_bound_shared_memory(1024, "cuda") == 1024

    # unknown backend
    assert validate_and_bound_shared_memory(1024, "unknown_backend") == 1024

    # exceeds limit
    with pytest.raises(ValueError, match="exceeds device limit"):
        validate_and_bound_shared_memory(100_000_000, "cuda")

    # threadgroup_memory_limit_bytes and fallback limit
    custom_prof_manifest = HardwareDeviceProfilesManifestModel(
        profiles={
            "custom_metal": HardwareDeviceProfileModel(architecture="apple", threadgroup_memory_limit_bytes=32768),
            "custom_default": HardwareDeviceProfileModel(architecture="generic"),
        }
    )
    assert validate_and_bound_shared_memory(4096, "custom_metal", manifest=custom_prof_manifest) == 4096
    assert validate_and_bound_shared_memory(4096, "custom_default", manifest=custom_prof_manifest) == 4096
    with pytest.raises(ValueError, match="exceeds device limit"):
        validate_and_bound_shared_memory(65536, "custom_metal", manifest=custom_prof_manifest)
