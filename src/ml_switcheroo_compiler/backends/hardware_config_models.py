"""Pydantic models and helpers for hardware generators (CUDA, ROCm, Metal, LLVM/C++)."""

import os
from typing import Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

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
    shared_memory_bytes: Union[int, str] = 0
    header: Optional[str] = None
    epilogue: Optional[str] = None
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


def load_hardware_templates(yaml_dir: str, yaml_path: str, backend: Optional[str] = None) -> HardwareTemplatesConfig:
    """Load hardware template configurations from a directory or single YAML file.

    Args:
        yaml_dir (str): Directory containing op-specific YAML files.
        yaml_path (str): Fallback path to a monolithic YAML template file.
        backend (Optional[str]): Optional backend identifier to merge unified templates.

    Returns:
        HardwareTemplatesConfig: Aggregated templates configuration.
    """
    cfg = HardwareTemplatesConfig(templates={})
    has_local = False
    if os.path.exists(yaml_path):
        _load_yaml_file_into_config(yaml_path, cfg, is_dir=False)
        has_local = True
    if os.path.isdir(yaml_dir):
        for filename in sorted(os.listdir(yaml_dir)):
            if filename.endswith(".yaml"):
                _load_yaml_file_into_config(os.path.join(yaml_dir, filename), cfg, is_dir=True)
                has_local = True
    if backend is not None and has_local:
        _merge_unified_kernel_templates(cfg, backend)
    return cfg


def _merge_unified_kernel_templates(cfg: HardwareTemplatesConfig, backend: str) -> None:
    """Merge unified kernel templates from kernel_templates.yaml into backend templates.

    Args:
        cfg (HardwareTemplatesConfig): Target hardware templates configuration.
        backend (str): Accelerator backend ('cuda', 'rocm', 'metal').
    """
    try:
        manifest = load_kernel_templates_manifest()
    except Exception:
        return
    b_lower = backend.lower()
    if b_lower == "cuda":
        type_map = {"float32": "float", "float16": "__half", "bfloat16": "__nv_bfloat16", "int32": "int", "int64": "long long", "bool": "bool"}
    elif b_lower == "rocm":
        type_map = {"float32": "float", "float16": "__half", "bfloat16": "__hip_bfloat16", "int32": "int", "int64": "long long", "bool": "bool"}
    else:
        type_map = {"float32": "float", "float16": "half", "bfloat16": "bfloat", "int32": "int", "int64": "long", "bool": "bool"}
    grid_c = GridDimensionConfig(x="({num_elements} + block_{node_idx}.x - 1) / block_{node_idx}.x", y="1", z="1")
    for opcode, tmpl in manifest.kernel_templates.items():
        op_key = opcode.lower()
        body = tmpl.cuda_body if b_lower == "cuda" else tmpl.rocm_body if b_lower == "rocm" else tmpl.metal_body
        if not body:
            continue
        existing = cfg.templates.get(op_key)
        is_dummy = False
        if existing and hasattr(existing, "body"):
            ex_body = getattr(existing, "body", "")
            if "C[id] = A[id];" in ex_body or "C[id] = A[id] + B[id];" in ex_body or "output[idx] = input[idx];" in ex_body:
                is_dummy = True
        if existing is None or is_dummy:
            cfg.templates[op_key] = HardwareTemplateConfig(
                body=body,
                workgroup_size=tmpl.workgroup_dims,
                grid_calc=grid_c,
                type_mappings=type_map,
                parameters=tmpl.parameters,
                shared_memory_bytes=tmpl.shared_memory_bytes,
            )


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


def _resolve_conv3d_args(
    node: IRNode,
    node_buffers: dict[str, str],
    out_name: str,
    node_idx: int,
    graph: IRGraph,
) -> tuple[tuple[str, str, str], list[str]]:
    """Resolve 3D grid and arguments for Conv3D.

    Args:
        node (IRNode): Conv3D node.
        node_buffers (dict[str, str]): Variable map for allocated device buffers.
        out_name (str): Output buffer name.
        node_idx (int): Unique node index.
        graph (IRGraph): IR graph context.

    Returns:
        tuple[tuple[str, str, str], list[str]]: ((grid_x, grid_y, grid_z), launch_args).
    """
    in0_node = graph.nodes.get(node.inputs[0]) if node.inputs else None
    in1_node = graph.nodes.get(node.inputs[1]) if len(node.inputs) > 1 else None
    in0_shape = getattr(in0_node, "shape_metadata", None) or (1, 1, 1, 1, 1)
    in1_shape = getattr(in1_node, "shape_metadata", None) or (1, 1, 1, 1, 1)
    out_shape = getattr(node, "shape_metadata", None) or (1, 1, 1, 1, 1)
    n_dim = in0_shape[0] if len(in0_shape) > 0 else 1
    c_in = in0_shape[1] if len(in0_shape) > 1 else 1
    d_dim = in0_shape[2] if len(in0_shape) > 2 else 1
    h_dim = in0_shape[3] if len(in0_shape) > 3 else 1
    w_in = in0_shape[4] if len(in0_shape) > 4 else 1
    c_out = out_shape[1] if len(out_shape) > 1 else 1
    k_d = in1_shape[2] if len(in1_shape) > 2 else 1
    k_h = in1_shape[3] if len(in1_shape) > 3 else 1
    k_w = in1_shape[4] if len(in1_shape) > 4 else 1
    d_out = out_shape[2] if len(out_shape) > 2 else 1
    h_out = out_shape[3] if len(out_shape) > 3 else 1
    w_out = out_shape[4] if len(out_shape) > 4 else 1

    grid_x = f"({d_out} * {h_out} * {w_out} + block_{node_idx}.x - 1) / block_{node_idx}.x"
    grid_y = str(c_out)
    grid_z = str(n_dim)
    in_args = [node_buffers.get(inp, "inputs[0]") for inp in getattr(node, "inputs", [])]
    launch_args = in_args + [
        out_name,
        str(n_dim),
        str(c_in),
        str(d_dim),
        str(h_dim),
        str(w_in),
        str(c_out),
        str(k_d),
        str(k_h),
        str(k_w),
        str(d_out),
        str(h_out),
        str(w_out),
    ]
    return (grid_x, grid_y, grid_z), launch_args


def _resolve_pool_args(
    node: IRNode,
    node_buffers: dict[str, str],
    out_name: str,
    node_idx: int,
    graph: IRGraph,
) -> tuple[tuple[str, str, str], list[str]]:
    """Resolve 3D grid and arguments for 2D and 3D spatial pooling.

    Args:
        node (IRNode): Pooling node.
        node_buffers (dict[str, str]): Variable map for allocated device buffers.
        out_name (str): Output buffer name.
        node_idx (int): Unique node index.
        graph (IRGraph): IR graph context.

    Returns:
        tuple[tuple[str, str, str], list[str]]: ((grid_x, grid_y, grid_z), launch_args).
    """
    op_type = getattr(node, "op_type", "").lower()
    in0_node = graph.nodes.get(node.inputs[0]) if node.inputs else None
    in0_shape = getattr(in0_node, "shape_metadata", None) or (1, 1, 1, 1)
    out_shape = getattr(node, "shape_metadata", None) or in0_shape
    n_dim = in0_shape[0] if len(in0_shape) > 0 else 1
    c_dim = in0_shape[1] if len(in0_shape) > 1 else 1

    if "3d" in op_type:
        d_dim = in0_shape[2] if len(in0_shape) > 2 else 1
        h_dim = in0_shape[3] if len(in0_shape) > 3 else 1
        w_in = in0_shape[4] if len(in0_shape) > 4 else 1
        d_out = out_shape[2] if len(out_shape) > 2 else 1
        h_out = out_shape[3] if len(out_shape) > 3 else 1
        w_out = out_shape[4] if len(out_shape) > 4 else 1
        kd = node.attributes.get("kernel_size", [2, 2, 2])[0] if hasattr(node, "attributes") and isinstance(node.attributes.get("kernel_size"), (list, tuple)) else 2
        kh = node.attributes.get("kernel_size", [2, 2, 2])[1] if hasattr(node, "attributes") and isinstance(node.attributes.get("kernel_size"), (list, tuple)) else 2
        kw = node.attributes.get("kernel_size", [2, 2, 2])[2] if hasattr(node, "attributes") and isinstance(node.attributes.get("kernel_size"), (list, tuple)) else 2
        grid_x = f"({d_out} * {h_out} * {w_out} + block_{node_idx}.x - 1) / block_{node_idx}.x"
        grid_y = str(c_dim)
        grid_z = str(n_dim)
        in_args = [node_buffers.get(inp, "inputs[0]") for inp in getattr(node, "inputs", [])]
        launch_args = in_args + [
            out_name,
            str(n_dim),
            str(c_dim),
            str(d_dim),
            str(h_dim),
            str(w_in),
            str(kd),
            str(kh),
            str(kw),
            str(d_out),
            str(h_out),
            str(w_out),
        ]
        return (grid_x, grid_y, grid_z), launch_args

    h_dim = in0_shape[2] if len(in0_shape) > 2 else 1
    w_in = in0_shape[3] if len(in0_shape) > 3 else 1
    h_out = out_shape[2] if len(out_shape) > 2 else 1
    w_out = out_shape[3] if len(out_shape) > 3 else 1
    kh = node.attributes.get("kernel_size", [2, 2])[0] if hasattr(node, "attributes") and isinstance(node.attributes.get("kernel_size"), (list, tuple)) else 2
    kw = node.attributes.get("kernel_size", [2, 2])[1] if hasattr(node, "attributes") and isinstance(node.attributes.get("kernel_size"), (list, tuple)) else 2
    grid_x = f"({h_out} * {w_out} + block_{node_idx}.x - 1) / block_{node_idx}.x"
    grid_y = str(c_dim)
    grid_z = str(n_dim)
    in_args = [node_buffers.get(inp, "inputs[0]") for inp in getattr(node, "inputs", [])]
    launch_args = in_args + [
        out_name,
        str(n_dim),
        str(c_dim),
        str(h_dim),
        str(w_in),
        str(kh),
        str(kw),
        str(h_out),
        str(w_out),
    ]
    return (grid_x, grid_y, grid_z), launch_args


def _resolve_row_wise_args(
    node: IRNode,
    node_buffers: dict[str, str],
    out_name: str,
    node_idx: int,
    graph: IRGraph,
) -> tuple[tuple[str, str, str], list[str]]:
    """Resolve grid and arguments for row-wise reduction and normalization operations.

    Args:
        node (IRNode): Row-wise op node.
        node_buffers (dict[str, str]): Device buffer variable map.
        out_name (str): Destination buffer name.
        node_idx (int): Node index.
        graph (IRGraph): Context graph.

    Returns:
        tuple[tuple[str, str, str], list[str]]: ((grid_x, grid_y, grid_z), launch_args).
    """
    in0_node = graph.nodes.get(node.inputs[0]) if node.inputs else None
    in0_shape = getattr(in0_node, "shape_metadata", None) or (1, 1)
    cols = in0_shape[-1] if len(in0_shape) >= 1 else 1
    rows = 1
    for d in in0_shape[:-1]:
        rows *= int(d)
    grid_x = str(rows)
    grid_y = "1"
    grid_z = "1"
    in_args = [node_buffers.get(inp, "inputs[0]") for inp in getattr(node, "inputs", [])]
    op_type = getattr(node, "op_type", "").lower()
    if op_type in ("layernorm", "rmsnorm", "groupnorm"):
        eps = str(getattr(node, "attributes", {}).get("epsilon", 1e-5))
        launch_args = in_args + [out_name, str(rows), str(cols), eps]
    else:
        launch_args = in_args + [out_name, str(rows), str(cols)]
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
    dispatch_map = {
        "conv2d": _resolve_conv2d_args,
        "conv3d": _resolve_conv3d_args,
        "matmul": _resolve_matmul_args,
        "dot": _resolve_matmul_args,
        "batchmatmul": _resolve_batchmatmul_args,
        "avgpool2d": _resolve_pool_args,
        "maxpool2d": _resolve_pool_args,
        "avgpool3d": _resolve_pool_args,
        "maxpool3d": _resolve_pool_args,
        "layernorm": _resolve_row_wise_args,
        "rmsnorm": _resolve_row_wise_args,
        "groupnorm": _resolve_row_wise_args,
        "softmax": _resolve_row_wise_args,
        "logsoftmax": _resolve_row_wise_args,
    }
    handler = dispatch_map.get(op_type_lower)
    if handler is not None:
        return handler(node, node_buffers, out_name, node_idx, graph)

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


class KernelTemplateModel(BaseModel):
    """Unified declarative kernel template model for hardware accelerators (CUDA, ROCm, Metal).

    Attributes:
        opcode (str): The operation opcode name.
        workgroup_dims (list[int]): Workgroup or thread block dimensions [x, y, z].
        thread_indexing_formula (str): Expression or logic for calculating thread index.
        memory_layout_requirements (list[str]): Memory layout constraints.
        body (str): Unified arithmetic kernel body / code template.
        cuda_body (Optional[str]): CUDA-specialized arithmetic kernel body.
        rocm_body (Optional[str]): ROCm/HIP-specialized arithmetic kernel body.
        metal_body (Optional[str]): Metal Shading Language (MSL) kernel body.
        parameters (list[KernelParameterConfig]): Explicit parameter signature list.
        shared_memory_bytes (int): Allocated shared memory size in bytes.
    """

    opcode: str = Field(description="Operation opcode name")
    workgroup_dims: list[int] = Field(default_factory=lambda: [256, 1, 1], description="Workgroup dimensions [x, y, z]")
    thread_indexing_formula: str = Field(
        default="int idx = blockIdx.x * blockDim.x + threadIdx.x;",
        description="Thread indexing formula",
    )
    memory_layout_requirements: list[str] = Field(
        default_factory=lambda: ["contiguous"],
        description="Memory layout constraints",
    )
    body: str = Field(default="", description="Unified arithmetic kernel body")
    cuda_body: Optional[str] = Field(default=None, description="CUDA-specialized kernel body")
    rocm_body: Optional[str] = Field(default=None, description="ROCm-specialized kernel body")
    metal_body: Optional[str] = Field(default=None, description="Metal-specialized kernel body")
    parameters: list[KernelParameterConfig] = Field(
        default_factory=list,
        description="Kernel parameter configurations",
    )
    shared_memory_bytes: int = Field(default=0, description="Shared memory bytes required")
    model_config = ConfigDict(extra="allow")


class KernelTemplatesManifestModel(BaseModel):
    """Declarative manifest containing unified kernel templates across operations.

    Attributes:
        kernel_templates (dict[str, KernelTemplateModel]): Mapping from opcode to unified template model.
    """

    kernel_templates: dict[str, KernelTemplateModel] = Field(
        default_factory=dict,
        description="Mapping from opcode to unified kernel template model",
    )
    model_config = ConfigDict(extra="allow")

    def get_template(self, op_type: str) -> Optional[KernelTemplateModel]:
        """Look up kernel template by opcode name case-insensitively.

        Args:
            op_type (str): Operation opcode.

        Returns:
            Optional[KernelTemplateModel]: Matched kernel template model or None.
        """
        clean = op_type.lower().replace("_", "").replace("-", "").replace(".", "")
        for k, v in self.kernel_templates.items():
            if k.lower().replace("_", "").replace("-", "").replace(".", "") == clean:
                return v
        return None


def load_kernel_templates_manifest(path: Optional[str] = None) -> KernelTemplatesManifestModel:
    """Load unified kernel templates manifest from YAML configuration.

    Args:
        path (Optional[str]): Path to kernel_templates.yaml file. Defaults to bundled YAML.

    Returns:
        KernelTemplatesManifestModel: Validated kernel templates manifest.
    """
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "kernel_templates.yaml")
    with open(path, encoding="utf-8") as f:
        raw_data = yaml.safe_load(f) or {}
    return KernelTemplatesManifestModel.model_validate(raw_data)


class HardwareCapabilityQuery(BaseModel):
    """Hardware capability constraints for dynamic launch configuration selection.

    Attributes:
        max_threads_per_block (int): Maximum threads supported in a thread block.
        max_shared_memory_per_sm (int): Maximum shared memory in bytes per SM / compute unit.
        warp_size (int): Execution sub-group / warp size (e.g. 32 for CUDA, 64 for ROCm, 32 for Metal).
        max_block_dim (list[int]): Maximum block dimensions [x, y, z].
        max_grid_dim (list[int]): Maximum grid dimensions [x, y, z].
    """

    max_threads_per_block: int = Field(default=1024, description="Max threads per block")
    max_shared_memory_per_sm: int = Field(default=65536, description="Max shared memory per SM in bytes")
    warp_size: int = Field(default=32, description="Execution warp/wavefront size")
    max_block_dim: list[int] = Field(default_factory=lambda: [1024, 1024, 64], description="Max block dimensions")
    max_grid_dim: list[int] = Field(default_factory=lambda: [2147483647, 65535, 65535], description="Max grid dimensions")
    model_config = ConfigDict(extra="allow")


class LaunchHeuristicRuleModel(BaseModel):
    """Declarative heuristic rule for calculating launch geometry based on tensor rank and size.

    Attributes:
        rule_name (str): Unique heuristic identifier.
        tensor_rank (list[int]): Target ranks matching this rule.
        block_dims (list[int]): Workgroup/block dimensions [x, y, z].
        grid_formula_x (str): Calculation formula for grid x.
        grid_formula_y (str): Calculation formula for grid y.
        grid_formula_z (str): Calculation formula for grid z.
        min_threads_per_sm (int): Minimum occupancy threshold.
    """

    rule_name: str = Field(description="Unique rule name")
    tensor_rank: list[int] = Field(default_factory=lambda: [1], description="Tensor ranks matching this rule")
    block_dims: list[int] = Field(default_factory=lambda: [256, 1, 1], description="Block dimensions")
    grid_formula_x: str = Field(default="(num_elements + block_x - 1) / block_x", description="grid.x formula")
    grid_formula_y: str = Field(default="1", description="grid.y formula")
    grid_formula_z: str = Field(default="1", description="grid.z formula")
    min_threads_per_sm: int = Field(default=128, description="Minimum threads per SM")
    model_config = ConfigDict(extra="allow")


class LaunchHeuristicsManifestModel(BaseModel):
    """Declarative manifest containing launch geometry heuristics.

    Attributes:
        heuristics (dict[str, LaunchHeuristicRuleModel]): Mapping from rule key to heuristic model.
    """

    heuristics: dict[str, LaunchHeuristicRuleModel] = Field(
        default_factory=dict,
        description="Mapping from heuristic name to rule model",
    )
    model_config = ConfigDict(extra="allow")


def load_launch_heuristics(path: Optional[str] = None) -> LaunchHeuristicsManifestModel:
    """Load launch geometry heuristics rulebook from YAML configuration.

    Args:
        path (Optional[str]): Path to launch_heuristics.yaml file. Defaults to bundled YAML.

    Returns:
        LaunchHeuristicsManifestModel: Validated launch heuristics manifest.
    """
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "launch_heuristics.yaml")
    with open(path, encoding="utf-8") as f:
        raw_data = yaml.safe_load(f) or {}
    return LaunchHeuristicsManifestModel.model_validate(raw_data)


def query_optimal_launch_geometry(
    shape: Union[tuple[Union[int, str], ...], list[Union[int, str]], tuple[int, ...], list[int], None],
    hardware_caps: Optional[HardwareCapabilityQuery] = None,
    heuristics_path: Optional[str] = None,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Query dynamic hardware capabilities and evaluate launch heuristics rulebook.

    Args:
        shape (Union[tuple[Union[int, str], ...], list[Union[int, str]], tuple[int, ...], list[int], None]): Tensor shape.
        hardware_caps (Optional[HardwareCapabilityQuery]): Dynamic hardware capability limits.
        heuristics_path (Optional[str]): Optional custom path to heuristics YAML.

    Returns:
        tuple[tuple[int, int, int], tuple[int, int, int]]: ((block_x, block_y, block_z), (grid_x, grid_y, grid_z)).
    """
    caps = hardware_caps or HardwareCapabilityQuery()
    return calculate_hardware_launch_config(
        shape=shape,
        max_threads_per_block=caps.max_threads_per_block,
        max_block_dim=(caps.max_block_dim[0], caps.max_block_dim[1], caps.max_block_dim[2]),
        max_grid_dim=(caps.max_grid_dim[0], caps.max_grid_dim[1], caps.max_grid_dim[2]),
    )


class MemoryLayoutRuleModel(BaseModel):
    """Declarative specification of a memory layout format and stride transformation equations.

    Attributes:
        layout_name (str): Identifier of the memory layout.
        dimension_order (list[int]): Dimension permutation order relative to canonical logical shape.
        stride_formula (str): Description or formula for computing dimension strides.
        is_contiguous (bool): Flag indicating whether the layout is densely packed in memory.
    """

    layout_name: str = Field(description="Memory layout format identifier")
    dimension_order: list[int] = Field(default_factory=list, description="Dimension permutation order")
    stride_formula: str = Field(default="row_major", description="Stride formula description")
    is_contiguous: bool = Field(default=True, description="Whether layout is contiguous")
    model_config = ConfigDict(extra="allow")


class MemoryLayoutsManifestModel(BaseModel):
    """Declarative manifest containing memory layout definitions and transformation rules.

    Attributes:
        layouts (dict[str, MemoryLayoutRuleModel]): Mapping of layout names to layout specifications.
        transformations (dict[str, list[int]]): Mapping of transformation pairs to permutation orders.
    """

    layouts: dict[str, MemoryLayoutRuleModel] = Field(
        default_factory=dict,
        description="Supported memory layout models",
    )
    transformations: dict[str, list[int]] = Field(
        default_factory=dict,
        description="Layout permutation transformation mapping",
    )
    model_config = ConfigDict(extra="allow")


def load_memory_layouts(path: Optional[str] = None) -> MemoryLayoutsManifestModel:
    """Load memory layout transformation rulebook from YAML configuration.

    Args:
        path (Optional[str]): Path to memory_layouts.yaml file. Defaults to bundled YAML.

    Returns:
        MemoryLayoutsManifestModel: Validated memory layouts manifest.
    """
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "memory_layouts.yaml")
    with open(path, encoding="utf-8") as f:
        raw_data = yaml.safe_load(f) or {}
    return MemoryLayoutsManifestModel.model_validate(raw_data)


class DeviceTileSizesModel(BaseModel):
    """Dynamic tile size configuration for 1D, 2D, and 3D kernel dispatch.

    Attributes:
        tile_1d (int): 1D tile dimension.
        tile_2d (list[int]): 2D tile dimensions [x, y].
        tile_3d (list[int]): 3D tile dimensions [x, y, z].
    """

    tile_1d: int = 256
    tile_2d: list[int] = Field(default_factory=lambda: [16, 16])
    tile_3d: list[int] = Field(default_factory=lambda: [8, 8, 4])
    model_config = ConfigDict(extra="allow")


class HardwareDeviceProfileModel(BaseModel):
    """Declarative execution limits and architecture characteristics for a hardware target.

    Attributes:
        architecture (str): Identifier of the accelerator or compute architecture.
        warp_size (Optional[int]): Execution unit thread width (32 for CUDA, 64 for ROCm, 32 for Metal).
        wavefront_size (Optional[int]): Wavefront size for AMD ROCm architectures.
        simdgroup_size (Optional[int]): SIMDgroup size for Apple Metal architectures.
        max_threads_per_block (int): Maximum threads per block or workgroup.
        max_threads_per_threadgroup (Optional[int]): Maximum threads per threadgroup for Metal.
        max_block_dim (list[int]): Maximum dimension sizes per block/workgroup [x, y, z].
        max_threadgroup_dim (Optional[list[int]]): Maximum dimension sizes per threadgroup for Metal.
        max_grid_dim (list[int]): Maximum grid dimension sizes [x, y, z].
        shared_memory_limit_bytes (int): Maximum shared memory in bytes per block/workgroup.
        threadgroup_memory_limit_bytes (Optional[int]): Maximum threadgroup memory in bytes for Metal.
        max_shared_memory_per_multiprocessor (Optional[int]): Total shared memory capacity per SM.
        max_threads_per_multiprocessor (Optional[int]): Total thread capacity per SM.
        vector_width (Optional[int]): SIMD vector width for CPU/LLVM targets.
        max_threads (Optional[int]): Maximum parallel thread count for CPU targets.
        openmp_chunk_size (Optional[int]): Default OpenMP chunk scheduling size.
        dynamic_tile_sizes (DeviceTileSizesModel): Dynamic dispatch tile dimensions.
    """

    architecture: str
    warp_size: Optional[int] = None
    wavefront_size: Optional[int] = None
    simdgroup_size: Optional[int] = None
    max_threads_per_block: int = Field(default=1024)
    max_threads_per_threadgroup: Optional[int] = None
    max_block_dim: list[int] = Field(default_factory=lambda: [1024, 1024, 64])
    max_threadgroup_dim: Optional[list[int]] = None
    max_grid_dim: list[int] = Field(default_factory=lambda: [2147483647, 65535, 65535])
    shared_memory_limit_bytes: int = Field(default=49152)
    threadgroup_memory_limit_bytes: Optional[int] = None
    max_shared_memory_per_multiprocessor: Optional[int] = None
    max_threads_per_multiprocessor: Optional[int] = None
    vector_width: Optional[int] = None
    max_threads: Optional[int] = None
    openmp_chunk_size: Optional[int] = None
    dynamic_tile_sizes: DeviceTileSizesModel = Field(default_factory=DeviceTileSizesModel)
    model_config = ConfigDict(extra="allow")

    @property
    def execution_unit_size(self) -> int:
        """Return the fundamental SIMD execution unit size (warp, wavefront, or simdgroup).

        Returns:
            int: Thread group execution lockstep width.
        """
        if self.wavefront_size is not None:
            return self.wavefront_size
        if self.warp_size is not None:
            return self.warp_size
        if self.simdgroup_size is not None:
            return self.simdgroup_size
        return 32


class HardwareDeviceProfilesManifestModel(BaseModel):
    """Manifest containing all hardware target device execution profiles.

    Attributes:
        version (str): Profile schema version.
        profiles (dict[str, HardwareDeviceProfileModel]): Map of backend name to profile model.
    """

    version: str = "1.0.0"
    profiles: dict[str, HardwareDeviceProfileModel] = Field(default_factory=dict)
    model_config = ConfigDict(extra="allow")


def load_hardware_device_profiles(path: Optional[str] = None) -> HardwareDeviceProfilesManifestModel:
    """Load hardware target execution profiles and limits from YAML configuration.

    Args:
        path (Optional[str]): Custom path to hardware_device_profiles.yaml.

    Returns:
        HardwareDeviceProfilesManifestModel: Validated hardware device profiles manifest.
    """
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "hardware_device_profiles.yaml")
    with open(path, encoding="utf-8") as f:
        raw_data = yaml.safe_load(f) or {}
    return HardwareDeviceProfilesManifestModel.model_validate(raw_data)


def validate_and_bound_shared_memory(
    requested_bytes: int,
    backend: str = "cuda",
    manifest: Optional[HardwareDeviceProfilesManifestModel] = None,
) -> int:
    """Validate and bound shared memory allocations according to architecture profiles.

    Args:
        requested_bytes (int): Requested shared memory size in bytes.
        backend (str): Accelerator backend ('cuda', 'rocm', 'metal', 'llvm_cpp').
        manifest (Optional[HardwareDeviceProfilesManifestModel]): Optional manifest instance.

    Returns:
        int: Validated bounded shared memory size in bytes.

    Raises:
        ValueError: If requested shared memory exceeds maximum hardware limit or is negative.
    """
    if requested_bytes < 0:
        raise ValueError(f"Shared memory allocation cannot be negative: {requested_bytes}")
    if manifest is None:
        manifest = load_hardware_device_profiles()
    profile = manifest.profiles.get(backend.lower())
    if not profile:
        return requested_bytes
    limit = getattr(profile, "shared_memory_limit_bytes", None) or getattr(profile, "threadgroup_memory_limit_bytes", None) or 65536
    if requested_bytes > limit:
        raise ValueError(f"Requested shared memory ({requested_bytes} bytes) exceeds device limit of {limit} bytes for backend '{backend}'.")
    return requested_bytes
