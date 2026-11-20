"""Normalization, slicing, padding, and scatter/gather visitors for CppGenerator."""

from __future__ import annotations

from collections.abc import Callable, Sequence  # noqa: F401

from ml_switcheroo_ir import IRGraph, IRNode

from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    coerce_float,
    coerce_int_or_tuple,
)
from ml_switcheroo_compiler.backends.llvm_cpp._cpp_gen_base import (
    AttrType,
    _node_attrs_dict,
    _node_inputs_list,
)
from ml_switcheroo_compiler.backends.llvm_cpp._cpp_gen_conv3d import _CppGenConv3D


class _CppGenNormIndex(_CppGenConv3D):
    """C++17 emission layer for LayerNorm, BatchNorm, RMSNorm, Slice, Pad, GatherND, and ScatterND."""

    def visit_LayerNorm(self, node: IRNode, graph_to_use: IRGraph | None = None) -> None:
        """Generate LayerNorm LLVM CPP.

        Args:
            node (IRNode): The IRNode.
            graph_to_use (IRGraph | None): Input `graph_to_use` argument (graph to use).
        """
        from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_template

        template: dict[str, str | BackendArg] = get_cpp_template("layernorm")
        inputs_list: list[str] = _node_inputs_list(node)
        shape: list[int] = self._get_shape(node)
        d_val: int = shape[-1] if shape else 1
        has_gamma: str = "true" if len(inputs_list) > 1 else "false"
        has_beta: str = "true" if len(inputs_list) > 2 else "false"
        gamma_var: str = inputs_list[1] if len(inputs_list) > 1 else inputs_list[0]
        beta_var: str = inputs_list[2] if len(inputs_list) > 2 else inputs_list[0]

        expr_args: dict[str, AttrType] = {
            "clean_id": node.id.replace("-", "_"),
            "in0": inputs_list[0] if len(inputs_list) > 0 else "dummy",
            "out_shape_str": "{" + ", ".join(map(str, shape)) + "}",
            "D": d_val,
            "has_gamma": has_gamma,
            "has_beta": has_beta,
            "gamma": gamma_var,
            "beta": beta_var,
        }
        body: str = str(template["body"]).format(**expr_args)
        for line in body.split("\n"):
            self.lines.append(f"    {line}")

    def visit_BatchNorm(self, node: IRNode, graph_to_use: IRGraph | None = None) -> None:
        """Generate BatchNorm LLVM CPP.

        Args:
            node (IRNode): The IRNode.
            graph_to_use (IRGraph | None): Input `graph_to_use` argument (graph to use).
        """
        from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_template

        template: dict[str, str | BackendArg] = get_cpp_template("batchnorm")
        inputs_list: list[str] = _node_inputs_list(node)
        shape: list[int] = self._get_shape(node)
        channels: int = shape[1] if len(shape) > 1 else 1
        spatial: int = 1
        for dim in shape[2:]:
            spatial *= dim

        has_mean: str = "true" if len(inputs_list) > 1 else "false"
        has_var: str = "true" if len(inputs_list) > 2 else "false"
        has_gamma: str = "true" if len(inputs_list) > 3 else "false"
        has_beta: str = "true" if len(inputs_list) > 4 else "false"

        mean_var: str = inputs_list[1] if len(inputs_list) > 1 else inputs_list[0]
        var_var: str = inputs_list[2] if len(inputs_list) > 2 else inputs_list[0]
        gamma_var: str = inputs_list[3] if len(inputs_list) > 3 else inputs_list[0]
        beta_var: str = inputs_list[4] if len(inputs_list) > 4 else inputs_list[0]

        expr_args: dict[str, AttrType] = {
            "clean_id": node.id.replace("-", "_"),
            "in0": inputs_list[0] if len(inputs_list) > 0 else "dummy",
            "out_shape_str": "{" + ", ".join(map(str, shape)) + "}",
            "channels": channels,
            "spatial": spatial,
            "has_mean": has_mean,
            "has_var": has_var,
            "has_gamma": has_gamma,
            "has_beta": has_beta,
            "mean": mean_var,
            "var": var_var,
            "gamma": gamma_var,
            "beta": beta_var,
        }
        body: str = str(template["body"]).format(**expr_args)
        for line in body.split("\n"):
            self.lines.append(f"    {line}")

    def visit_RMSNorm(self, node: IRNode, graph_to_use: IRGraph | None = None) -> None:
        """Generate RMSNorm LLVM CPP.

        Args:
            node (IRNode): The IRNode.
            graph_to_use (IRGraph | None): Input `graph_to_use` argument (graph to use).
        """
        from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_template

        template: dict[str, str | BackendArg] = get_cpp_template("rmsnorm")
        inputs_list: list[str] = _node_inputs_list(node)
        shape: list[int] = self._get_shape(node)
        d_val: int = shape[-1] if shape else 1
        has_gamma: str = "true" if len(inputs_list) > 1 else "false"
        gamma_var: str = inputs_list[1] if len(inputs_list) > 1 else inputs_list[0]

        expr_args: dict[str, AttrType] = {
            "clean_id": node.id.replace("-", "_"),
            "in0": inputs_list[0] if len(inputs_list) > 0 else "dummy",
            "out_shape_str": "{" + ", ".join(map(str, shape)) + "}",
            "D": d_val,
            "has_gamma": has_gamma,
            "gamma": gamma_var,
        }
        body: str = str(template["body"]).format(**expr_args)
        for line in body.split("\n"):
            self.lines.append(f"    {line}")

    def visit_Slice(self, node: IRNode, graph_to_use: IRGraph | None = None) -> None:
        """Generate Slice LLVM CPP.

        Args:
            node (IRNode): The IRNode.
            graph_to_use (IRGraph | None): Input `graph_to_use` argument (graph to use).
        """
        from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_template

        template: dict[str, str | BackendArg] = get_cpp_template("slice")
        inputs_list: list[str] = _node_inputs_list(node)
        shape: list[int] = self._get_shape(node)
        rank: int = len(shape)
        attrs: dict[str, AttrType] = _node_attrs_dict(node)
        starts: int | tuple[int, ...] = coerce_int_or_tuple(attrs.get("starts", [0] * rank))
        strides: int | tuple[int, ...] = coerce_int_or_tuple(attrs.get("strides", [1] * rank))
        starts_list: list[int] = [starts] * rank if isinstance(starts, int) else list(starts[:rank])
        strides_list: list[int] = [strides] * rank if isinstance(strides, int) else list(strides[:rank])

        starts_str: str = "{" + ", ".join(map(str, starts_list)) + "}"
        strides_str: str = "{" + ", ".join(map(str, strides_list)) + "}"

        expr_args: dict[str, AttrType] = {
            "clean_id": node.id.replace("-", "_"),
            "in0": inputs_list[0] if len(inputs_list) > 0 else "dummy",
            "out_shape_str": "{" + ", ".join(map(str, shape)) + "}",
            "rank": rank,
            "starts": starts_str,
            "strides": strides_str,
        }
        body: str = str(template["body"]).format(**expr_args)
        for line in body.split("\n"):
            self.lines.append(f"    {line}")

    def visit_Pad(self, node: IRNode, graph_to_use: IRGraph | None = None) -> None:
        """Generate Pad LLVM CPP.

        Args:
            node (IRNode): The IRNode.
            graph_to_use (IRGraph | None): Input `graph_to_use` argument (graph to use).
        """
        from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_template

        template: dict[str, str | BackendArg] = get_cpp_template("pad")
        inputs_list: list[str] = _node_inputs_list(node)
        shape: list[int] = self._get_shape(node)
        rank: int = len(shape)
        attrs: dict[str, AttrType] = _node_attrs_dict(node)
        paddings: int | tuple[int, ...] = coerce_int_or_tuple(attrs.get("paddings", [0] * rank))
        pad_val: float = float(coerce_float(attrs.get("constant_value", 0.0)))
        pad_low: list[int] = []
        if isinstance(paddings, int):
            pad_low = [paddings] * rank
        elif len(paddings) >= rank * 2:
            pad_low = [int(paddings[2 * i]) for i in range(rank)]
        else:
            pad_low = [int(p) for p in paddings[:rank]]
        pad_low_str: str = "{" + ", ".join(map(str, pad_low)) + "}"

        expr_args: dict[str, AttrType] = {
            "clean_id": node.id.replace("-", "_"),
            "in0": inputs_list[0] if len(inputs_list) > 0 else "dummy",
            "out_shape_str": "{" + ", ".join(map(str, shape)) + "}",
            "rank": rank,
            "pad_low": pad_low_str,
            "pad_val": f"{pad_val}f",
        }
        body: str = str(template["body"]).format(**expr_args)
        for line in body.split("\n"):
            self.lines.append(f"    {line}")

    def visit_GatherND(self, node: IRNode, graph_to_use: IRGraph | None = None) -> None:
        """Generate GatherND LLVM CPP.

        Args:
            node (IRNode): The IRNode.
            graph_to_use (IRGraph | None): Input `graph_to_use` argument (graph to use).
        """
        from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_template

        template: dict[str, str | BackendArg] = get_cpp_template("gather_nd")
        inputs_list: list[str] = _node_inputs_list(node)
        input_nodes: list[IRNode | None] = [graph_to_use.nodes.get(inp) for inp in inputs_list] if graph_to_use else []
        idx_shape: list[int] = self._get_shape(input_nodes[1]) if len(input_nodes) > 1 and input_nodes[1] else [1, 1]
        index_depth: int = idx_shape[-1] if idx_shape else 1
        shape: list[int] = self._get_shape(node)

        expr_args: dict[str, AttrType] = {
            "clean_id": node.id.replace("-", "_"),
            "in0": inputs_list[0] if len(inputs_list) > 0 else "dummy",
            "in1": inputs_list[1] if len(inputs_list) > 1 else "dummy",
            "out_shape_str": "{" + ", ".join(map(str, shape)) + "}",
            "index_depth": index_depth,
        }
        body: str = str(template["body"]).format(**expr_args)
        for line in body.split("\n"):
            self.lines.append(f"    {line}")

    def visit_ScatterND(self, node: IRNode, graph_to_use: IRGraph | None = None) -> None:
        """Generate ScatterND LLVM CPP.

        Args:
            node (IRNode): The IRNode.
            graph_to_use (IRGraph | None): Input `graph_to_use` argument (graph to use).
        """
        from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_template

        template: dict[str, str | BackendArg] = get_cpp_template("scatter_nd")
        inputs_list: list[str] = _node_inputs_list(node)
        input_nodes: list[IRNode | None] = [graph_to_use.nodes.get(inp) for inp in inputs_list] if graph_to_use else []
        idx_shape: list[int] = self._get_shape(input_nodes[1]) if len(input_nodes) > 1 and input_nodes[1] else [1, 1]
        index_depth: int = idx_shape[-1] if idx_shape else 1
        shape: list[int] = self._get_shape(node)

        expr_args: dict[str, AttrType] = {
            "clean_id": node.id.replace("-", "_"),
            "in0": inputs_list[0] if len(inputs_list) > 0 else "dummy",
            "in1": inputs_list[1] if len(inputs_list) > 1 else "dummy",
            "in2": inputs_list[2] if len(inputs_list) > 2 else "dummy",
            "out_shape_str": "{" + ", ".join(map(str, shape)) + "}",
            "index_depth": index_depth,
        }
        body: str = str(template["body"]).format(**expr_args)
        for line in body.split("\n"):
            self.lines.append(f"    {line}")
