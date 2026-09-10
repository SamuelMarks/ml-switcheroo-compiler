"""Pydantic models and helpers for hardware generators (CUDA, ROCm, Metal, LLVM/C++)."""

import os
from typing import Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator

from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


class KernelParameterConfig(BaseModel):
    """Declarative specification for a kernel argument binding.

    Attributes:
        name (str): The name of the parameter in the kernel signature.
        param_type (str): Type category ('pointer', 'int', 'float', 'shape_dim').
        source (str): Source for binding ('input', 'output', 'shape', 'attribute', 'num_elements').
        source_index (Optional[int]): Input or output index if applicable.
        source_key (Optional[str]): Key for attribute or shape dimension.
        default_value (Optional[Union[int, float, str]]): Fallback value.
    """

    name: str
    param_type: str = Field(default="pointer", description="Type category: pointer, int, float, shape_dim")
    source: str = Field(default="input", description="Source: input, output, shape, attribute, num_elements")
    source_index: Optional[int] = Field(default=None, description="Input or output index")
    source_key: Optional[str] = Field(default=None, description="Key for attribute or shape dimension")
    default_value: Optional[Union[int, float, str]] = Field(default=None, description="Fallback value")


class GridDimensionConfig(BaseModel):
    """Declarative specification for 3D grid layout computation.

    Attributes:
        x (str): Expression to compute grid.x.
        y (str): Expression to compute grid.y.
        z (str): Expression to compute grid.z.
    """

    x: str = Field(default="((num_elements + block_x - 1) / block_x)", description="grid.x formula")
    y: str = Field(default="1", description="grid.y formula")
    z: str = Field(default="1", description="grid.z formula")


class HardwareTemplateConfig(BaseModel):
    """Configuration for a hardware template.

    Attributes:
        body (str): The kernel source code template.
        workgroup_size (list[int]): Block / threadgroup dimensions [x, y, z].
        grid_calc (Optional[GridDimensionConfig]): Grid sizing calculation formulas.
        parameters (Optional[list[KernelParameterConfig]]): Explicit parameter signature list.
        shared_memory_bytes (int): Allocated shared memory size in bytes.
        type_mappings (dict[str, str]): Type promotion and primitive mappings.
    """

    body: str
    workgroup_size: list[int] = Field(default_factory=lambda: [256, 1, 1])
    grid_calc: Optional[GridDimensionConfig] = None
    parameters: Optional[list[KernelParameterConfig]] = None
    shared_memory_bytes: int = 0
    type_mappings: dict[str, str] = Field(default_factory=dict)

    def get(
        self,
        key: str,
        default: Optional[Union[int, float, str, list[int], dict[str, str], GridDimensionConfig, list[KernelParameterConfig]]] = None,
    ) -> Optional[Union[int, float, str, list[int], dict[str, str], GridDimensionConfig, list[KernelParameterConfig]]]:
        """Retrieve an attribute by string key with fallback default.

        Args:
            key (str): The attribute name to look up.
            default (Optional[Union[int, float, str, list[int], dict[str, str], GridDimensionConfig, list[KernelParameterConfig]]]): Fallback value if attribute not found.

        Returns:
            Optional[Union[int, float, str, list[int], dict[str, str], GridDimensionConfig, list[KernelParameterConfig]]]: The attribute value or default.
        """
        if hasattr(self, key):
            val = getattr(self, key)
            if val is not None:
                return val
        return default

    def __eq__(self, other: object) -> bool:
        """Compare template configuration against another instance or dict.

        Args:
            other (object): Other object to compare against.

        Returns:
            bool: True if contents match.
        """
        if isinstance(other, dict):
            if "body" in other and self.body != other["body"]:
                return False
            if "workgroup_size" in other and self.workgroup_size != other["workgroup_size"]:
                return False
            return True
        if isinstance(other, HardwareTemplateConfig):
            return self.body == other.body and self.workgroup_size == other.workgroup_size
        return False


TemplateEntryType = Union[
    HardwareTemplateConfig,
    dict[str, Union[str, int, float, list[int], None]],
    list[str],
    str,
]


class HardwareTemplatesConfig(BaseModel):
    """Configuration for all hardware templates.

    Attributes:
        templates (dict[str, TemplateEntryType]): Map of operation names to templates.
        orchestration (dict[str, str]): Target runtime orchestration logic.
    """

    templates: dict[str, TemplateEntryType]
    orchestration: dict[str, str] = Field(default_factory=dict)

    @field_validator("templates", mode="before")
    @classmethod
    def _validate_templates(
        cls,
        v: Union[dict[str, TemplateEntryType], dict[str, Union[str, int, float, list[int], None]]],
    ) -> dict[str, TemplateEntryType]:
        """Validate templates and convert dict entries with body to HardwareTemplateConfig.

        Args:
            v (Union[dict[str, TemplateEntryType], dict[str, Union[str, int, float, list[int], None]]]): Raw template mapping.

        Returns:
            dict[str, TemplateEntryType]: Normalized template mapping.
        """
        if not isinstance(v, dict):
            return v
        out: dict[str, TemplateEntryType] = {}
        for key, val in v.items():
            if isinstance(val, dict) and "body" in val:
                out[str(key)] = HardwareTemplateConfig(**val)
            elif isinstance(val, str):
                out[str(key)] = HardwareTemplateConfig(body=val)
            else:
                out[str(key)] = val
        return out


def load_hardware_templates(yaml_dir: str, yaml_path: str) -> HardwareTemplatesConfig:
    """Load hardware template configurations from a directory or single YAML file.

    Args:
        yaml_dir (str): Directory containing op-specific YAML files.
        yaml_path (str): Fallback path to a monolithic YAML template file.

    Returns:
        HardwareTemplatesConfig: Aggregated templates configuration.
    """
    cfg = HardwareTemplatesConfig(templates={})
    if os.path.exists(yaml_path):
        _load_yaml_file_into_config(yaml_path, cfg, is_dir=False)
    if os.path.isdir(yaml_dir):
        for filename in sorted(os.listdir(yaml_dir)):
            if filename.endswith(".yaml"):
                _load_yaml_file_into_config(os.path.join(yaml_dir, filename), cfg, is_dir=True)
    return cfg


def _load_yaml_file_into_config(file_path: str, cfg: HardwareTemplatesConfig, is_dir: bool = True) -> None:
    """Load a YAML file and populate template entries into a HardwareTemplatesConfig.

    Args:
        file_path (str): Path to the YAML file.
        cfg (HardwareTemplatesConfig): Target configuration to populate.
        is_dir (bool): True if loaded from an individual op YAML in a directory.
    """
    with open(file_path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        return
    if is_dir:
        source = data.get("templates", data)
    else:
        if "templates" not in data or not isinstance(data["templates"], dict):
            return
        source = data["templates"]
    for k, v in source.items():
        if isinstance(v, dict) and "body" in v:
            cfg.templates[k] = HardwareTemplateConfig(**v)
        elif isinstance(v, str):
            cfg.templates[k] = HardwareTemplateConfig(body=v)
        elif isinstance(v, HardwareTemplateConfig):
            cfg.templates[k] = v
        else:
            cfg.templates[k] = v


def compute_liveness(graph: IRGraph) -> dict[str, str]:
    """Compute the last consumer node ID for each input and intermediate buffer.

    Args:
        graph (IRGraph): The IR computation graph.

    Returns:
        dict[str, str]: Mapping from producer node ID to last consumer node ID.
    """
    consumers: dict[str, list[str]] = {}
    for n in getattr(graph, "nodes", {}).values():
        for inp in getattr(n, "inputs", []):
            consumers.setdefault(inp, []).append(n.id)
    return {inp: user_ids[-1] for inp, user_ids in consumers.items()}


def _resolve_conv2d_args(
    node: IRNode,
    node_buffers: dict[str, str],
    out_name: str,
    node_idx: int,
    graph: IRGraph,
) -> tuple[tuple[str, str, str], list[str]]:
    """Resolve 3D grid and arguments for Conv2D.

    Args:
        node (IRNode): Conv2D node.
        node_buffers (dict[str, str]): Variable map for allocated device buffers.
        out_name (str): Output buffer name.
        node_idx (int): Unique node index.
        graph (IRGraph): IR graph context.

    Returns:
        tuple[tuple[str, str, str], list[str]]: ((grid_x, grid_y, grid_z), launch_args).
    """
    in0_node = graph.nodes.get(node.inputs[0]) if node.inputs else None
    in1_node = graph.nodes.get(node.inputs[1]) if len(node.inputs) > 1 else None
    in0_shape = getattr(in0_node, "shape_metadata", None) or (1, 1, 1, 1)
    in1_shape = getattr(in1_node, "shape_metadata", None) or (1, 1, 1, 1)
    out_shape = getattr(node, "shape_metadata", None) or (1, 1, 1, 1)
    n_dim = in0_shape[0] if len(in0_shape) > 0 else 1
    c_in = in0_shape[1] if len(in0_shape) > 1 else 1
    h_dim = in0_shape[2] if len(in0_shape) > 2 else 1
    w_in = in0_shape[3] if len(in0_shape) > 3 else 1
    c_out = out_shape[1] if len(out_shape) > 1 else 1
    k_h = in1_shape[2] if len(in1_shape) > 2 else 1
    k_w = in1_shape[3] if len(in1_shape) > 3 else 1
    h_out = out_shape[2] if len(out_shape) > 2 else 1
    w_out = out_shape[3] if len(out_shape) > 3 else 1

    grid_x = f"({h_out} * {w_out} + block_{node_idx}.x - 1) / block_{node_idx}.x"
    grid_y = str(c_out)
    grid_z = str(n_dim)
    in_args = [node_buffers.get(inp, "inputs[0]") for inp in getattr(node, "inputs", [])]
    launch_args = in_args + [out_name, str(n_dim), str(c_in), str(h_dim), str(w_in), str(c_out), str(k_h), str(k_w), str(h_out), str(w_out)]
    return (grid_x, grid_y, grid_z), launch_args


def _resolve_matmul_args(
    node: IRNode,
    node_buffers: dict[str, str],
    out_name: str,
    node_idx: int,
    graph: IRGraph,
) -> tuple[tuple[str, str, str], list[str]]:
    """Resolve 3D grid and arguments for MatMul.

    Args:
        node (IRNode): MatMul node.
        node_buffers (dict[str, str]): Variable map for allocated device buffers.
        out_name (str): Output buffer name.
        node_idx (int): Unique node index.
        graph (IRGraph): IR graph context.

    Returns:
        tuple[tuple[str, str, str], list[str]]: ((grid_x, grid_y, grid_z), launch_args).
    """
    in0_node = graph.nodes.get(node.inputs[0]) if node.inputs else None
    in1_node = graph.nodes.get(node.inputs[1]) if len(node.inputs) > 1 else None
    in0_shape = getattr(in0_node, "shape_metadata", None) or (1, 1)
    in1_shape = getattr(in1_node, "shape_metadata", None) or (1, 1)
    m_dim = in0_shape[-2] if len(in0_shape) >= 2 else 1
    k_dim = in0_shape[-1] if len(in0_shape) >= 1 else 1
    n_dim = in1_shape[-1] if len(in1_shape) >= 1 else 1

    grid_x = f"({n_dim} + block_{node_idx}.x - 1) / block_{node_idx}.x"
    grid_y = f"({m_dim} + block_{node_idx}.y - 1) / block_{node_idx}.y"
    grid_z = "1"
    in_args = [node_buffers.get(inp, "inputs[0]") for inp in getattr(node, "inputs", [])]
    launch_args = in_args + [out_name, str(m_dim), str(n_dim), str(k_dim)]
    return (grid_x, grid_y, grid_z), launch_args


def _resolve_batchmatmul_args(
    node: IRNode,
    node_buffers: dict[str, str],
    out_name: str,
    node_idx: int,
    graph: IRGraph,
) -> tuple[tuple[str, str, str], list[str]]:
    """Resolve 3D grid and arguments for BatchMatMul.

    Args:
        node (IRNode): BatchMatMul node.
        node_buffers (dict[str, str]): Variable map for allocated device buffers.
        out_name (str): Output buffer name.
        node_idx (int): Unique node index.
        graph (IRGraph): IR graph context.

    Returns:
        tuple[tuple[str, str, str], list[str]]: ((grid_x, grid_y, grid_z), launch_args).
    """
    in0_node = graph.nodes.get(node.inputs[0]) if node.inputs else None
    in1_node = graph.nodes.get(node.inputs[1]) if len(node.inputs) > 1 else None
    in0_shape = getattr(in0_node, "shape_metadata", None) or (1, 1, 1)
    in1_shape = getattr(in1_node, "shape_metadata", None) or (1, 1, 1)
    b_sz = in0_shape[0] if len(in0_shape) > 2 else 1
    m_dim = in0_shape[-2] if len(in0_shape) >= 2 else 1
    k_dim = in0_shape[-1] if len(in0_shape) >= 1 else 1
    n_dim = in1_shape[-1] if len(in1_shape) >= 1 else 1

    grid_x = f"({n_dim} + block_{node_idx}.x - 1) / block_{node_idx}.x"
    grid_y = f"({m_dim} + block_{node_idx}.y - 1) / block_{node_idx}.y"
    grid_z = str(b_sz)
    in_args = [node_buffers.get(inp, "inputs[0]") for inp in getattr(node, "inputs", [])]
    launch_args = in_args + [out_name, str(b_sz), str(m_dim), str(n_dim), str(k_dim)]
    return (grid_x, grid_y, grid_z), launch_args


def resolve_hardware_launch_grid_and_args(
    node: IRNode,
    tpl: HardwareTemplateConfig,
    node_buffers: dict[str, str],
    out_name: str,
    num_elements: int,
    node_idx: int,
    graph: IRGraph,
) -> tuple[tuple[str, str, str], list[str]]:
    """Resolve 3D grid dimensions and kernel launch arguments dynamically.

    Args:
        node (IRNode): Computation node.
        tpl (HardwareTemplateConfig): Op hardware template.
        node_buffers (dict[str, str]): Variable map for allocated device buffers.
        out_name (str): Destination buffer name.
        num_elements (int): Total elements in the output tensor.
        node_idx (int): Unique node index for variable naming.
        graph (IRGraph): IR graph containing node context.

    Returns:
        tuple[tuple[str, str, str], list[str]]: ((grid_x, grid_y, grid_z), launch_args).
    """
    op_type_lower = getattr(node, "op_type", "").lower()

    if op_type_lower == "conv2d":
        return _resolve_conv2d_args(node, node_buffers, out_name, node_idx, graph)

    if op_type_lower in ("matmul", "dot"):
        return _resolve_matmul_args(node, node_buffers, out_name, node_idx, graph)

    if op_type_lower == "batchmatmul":
        return _resolve_batchmatmul_args(node, node_buffers, out_name, node_idx, graph)

    if op_type_lower == "fusedelementwise":
        grid_x = f"({num_elements} + block_{node_idx}.x - 1) / block_{node_idx}.x"
        in_args = [node_buffers.get(inp, "inputs[0]") for inp in getattr(node, "inputs", [])]
        launch_args = in_args + [out_name, str(num_elements)]
        return (grid_x, "1", "1"), launch_args

    grid_x = f"({num_elements} + block_{node_idx}.x - 1) / block_{node_idx}.x"
    in_args = [node_buffers.get(inp, "inputs[0]") for inp in getattr(node, "inputs", [])]
    launch_args = in_args + [out_name, str(num_elements)]
    return (grid_x, "1", "1"), launch_args


def calculate_hardware_launch_config(
    shape: Union[tuple[Union[int, str], ...], list[Union[int, str]], tuple[int, ...], list[int], None],
    max_threads_per_block: int = 1024,
    max_block_dim: tuple[int, int, int] = (1024, 1024, 64),
    max_grid_dim: tuple[int, int, int] = (2147483647, 65535, 65535),
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Compute optimal hardware thread block and launch grid for N-dimensional tensor.

    Args:
        shape (Union[tuple[Union[int, str], ...], list[Union[int, str]], tuple[int, ...], list[int], None]): N-dimensional tensor shape.
        max_threads_per_block (int): Maximum threads allowed in a single thread block.
        max_block_dim (tuple[int, int, int]): Maximum sizing for (x, y, z) block dims.
        max_grid_dim (tuple[int, int, int]): Maximum sizing for (x, y, z) grid dims.

    Returns:
        tuple[tuple[int, int, int], tuple[int, int, int]]: ((block_x, block_y, block_z), (grid_x, grid_y, grid_z)).
    """
    int_shape: list[int] = []
    if shape is not None:
        for d in shape:
            try:
                int_shape.append(max(1, int(d)))
            except (ValueError, TypeError):
                int_shape.append(1)

    ndim = len(int_shape)
    if ndim == 0:
        return (1, 1, 1), (1, 1, 1)
    if ndim == 1:
        elem = int_shape[0]
        bx = min(elem, max_threads_per_block, max_block_dim[0])
        gx = min((elem + bx - 1) // bx, max_grid_dim[0])
        return (bx, 1, 1), (gx, 1, 1)
    if ndim == 2:
        h, w = int_shape[0], int_shape[1]
        bx = min(16, max_block_dim[0])
        by = min(16, max_block_dim[1])
        gx = min((w + bx - 1) // bx, max_grid_dim[0])
        gy = min((h + by - 1) // by, max_grid_dim[1])
        return (bx, by, 1), (gx, gy, 1)

    inner = int_shape[-1]
    middle = int_shape[-2]
    outer = 1
    for d in int_shape[:-2]:
        outer *= d
    bx = min(16, max_block_dim[0])
    by = min(8, max_block_dim[1])
    bz = min(max(1, max_threads_per_block // (bx * by)), max_block_dim[2])
    gx = min((inner + bx - 1) // bx, max_grid_dim[0])
    gy = min((middle + by - 1) // by, max_grid_dim[1])
    gz = min((outer + bz - 1) // bz, max_grid_dim[2])
    return (bx, by, bz), (gx, gy, gz)


def _calculate_contiguous_strides(shape: list[int]) -> list[int]:
    """Compute standard row-major contiguous strides for a shape.

    Args:
        shape (list[int]): Tensor shape.

    Returns:
        list[int]: Strides.
    """
    cur = 1
    rev_strides: list[int] = []
    for d in reversed(shape):
        rev_strides.append(cur)
        cur *= d
    return list(reversed(rev_strides))


def _format_stride_terms(coord_vars: list[str], strides: list[int]) -> str:
    """Format polynomial sum for coordinate strides.

    Args:
        coord_vars (list[str]): Coordinate variable names.
        strides (list[int]): Dimension strides.

    Returns:
        str: Linear offset expression.
    """
    terms: list[str] = []
    for d, s in enumerate(strides):
        if s == 0:
            continue
        if s == 1:
            terms.append(coord_vars[d])
        else:
            terms.append(f"({coord_vars[d]} * {s})")
    return " + ".join(terms) if terms else "0"


def generate_nd_coordinate_offset_logic(
    shape: Union[tuple[Union[int, str], ...], list[Union[int, str]], tuple[int, ...], list[int]],
    strides: Optional[Union[tuple[int, ...], list[int]]] = None,
    linear_var: str = "idx",
    offset_var: str = "offset",
    language: str = "cuda",
) -> list[str]:
    """Generate C++/CUDA/MSL unrolled statements to compute strided offset from flat index.

    Args:
        shape (Union[tuple[Union[int, str], ...], list[Union[int, str]], tuple[int, ...], list[int]]): Tensor shape.
        strides (Optional[Union[tuple[int, ...], list[int]]]): Optional explicit strides.
        linear_var (str): Name of linear input index variable.
        offset_var (str): Destination variable name for strided offset.
        language (str): Target dialect ('cuda', 'hip', 'metal').

    Returns:
        list[str]: Unrolled source code lines computing multidimensional offset.
    """
    int_shape: list[int] = []
    for d in shape:
        try:
            int_shape.append(max(1, int(d)))
        except (ValueError, TypeError):
            int_shape.append(1)

    ndim = len(int_shape)
    type_decl = "uint" if language == "metal" else "int"
    if ndim <= 1:
        s0 = strides[0] if strides else 1
        mult = f" * {s0}" if s0 != 1 else ""
        return [f"{type_decl} {offset_var} = {linear_var}{mult};"]

    calc_strides: list[int] = [int(s) for s in strides] if strides else _calculate_contiguous_strides(int_shape)

    lines: list[str] = [f"{type_decl} rem_{linear_var} = {linear_var};"]
    coord_vars: list[str] = []
    for d in reversed(range(ndim)):
        dim_sz = int_shape[d]
        c_var = f"c_{linear_var}_{d}"
        coord_vars.append(c_var)
        lines.append(f"{type_decl} {c_var} = rem_{linear_var} % {dim_sz};")
        if d > 0:
            lines.append(f"rem_{linear_var} /= {dim_sz};")

    coord_vars.reverse()
    expr = _format_stride_terms(coord_vars, calc_strides)
    lines.append(f"{type_decl} {offset_var} = {expr};")
    return lines


class GridBlockDimensionSpec(BaseModel):
    """Declarative specification of grid block dimensions and sizing formulas.

    Attributes:
        block_dims (list[int]): Block or threadgroup dimensions [x, y, z].
        grid_formula_x (str): Mathematical expression for grid.x dimension.
        grid_formula_y (str): Mathematical expression for grid.y dimension.
        grid_formula_z (str): Mathematical expression for grid.z dimension.
    """

    block_dims: list[int] = Field(default_factory=lambda: [256, 1, 1], description="Block or threadgroup dimensions")
    grid_formula_x: str = Field(default="(num_elements + block_x - 1) / block_x", description="grid.x formula")
    grid_formula_y: str = Field(default="1", description="grid.y formula")
    grid_formula_z: str = Field(default="1", description="grid.z formula")


class BufferBindingSpec(BaseModel):
    """Declarative binding configuration for device buffers.

    Attributes:
        buffer_binding_order (list[str]): Binding priority order (e.g. ['inputs', 'outputs', 'dimensions']).
    """

    buffer_binding_order: list[str] = Field(
        default_factory=lambda: ["inputs", "outputs", "dimensions"],
        description="Buffer binding ordering",
    )


class HardwareDispatchSpec(BaseModel):
    """Declarative specification for hardware backend execution and synchronization.

    Attributes:
        backend_name (str): Target hardware accelerator identifier ('cuda', 'rocm', 'metal').
        sync_barrier (str): Stream or command buffer synchronization call.
        device_sync (str): Device-wide synchronization barrier.
        dispatch_protocol (str): Kernel execution invocation API name.
        buffer_binding_order (list[str]): Order for buffer parameters.
        grid_strategies (dict[str, GridBlockDimensionSpec]): Mapping of op categories to grid dimension specs.
    """

    backend_name: str = Field(description="Hardware backend name")
    sync_barrier: str = Field(description="Stream/queue synchronization primitive")
    device_sync: str = Field(description="Device synchronization primitive")
    dispatch_protocol: str = Field(description="Kernel launch protocol")
    buffer_binding_order: list[str] = Field(
        default_factory=lambda: ["inputs", "outputs", "dimensions"],
        description="Parameter ordering",
    )
    grid_strategies: dict[str, GridBlockDimensionSpec] = Field(
        default_factory=dict,
        description="Strategy specifications",
    )


class HardwareExecutionSchema(BaseModel):
    """Root configuration holding execution specifications for all hardware backends.

    Attributes:
        hardware_execution_schemas (dict[str, HardwareDispatchSpec]): Mapping from backend name to dispatch spec.
    """

    hardware_execution_schemas: dict[str, HardwareDispatchSpec] = Field(
        default_factory=dict,
        description="Mapping of backend names to dispatch specs",
    )


def load_hardware_execution_schema(path: Optional[str] = None) -> HardwareExecutionSchema:
    """Load hardware execution specification schemas from YAML configuration.

    Args:
        path (Optional[str]): Path to hardware_execution_schemas.yaml file. Defaults to bundled YAML.

    Returns:
        HardwareExecutionSchema: Validated hardware dispatch schema configuration.
    """
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "hardware_execution_schemas.yaml")
    with open(path, encoding="utf-8") as f:
        raw_data = yaml.safe_load(f)
    return HardwareExecutionSchema.model_validate(raw_data)
