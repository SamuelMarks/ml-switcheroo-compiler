"""Pooling, Linear, Attention, and LayerNorm visitors for WasmCodeGenerator."""

from __future__ import annotations

from collections.abc import Sequence

from ml_switcheroo_ir import IRNode

from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    coerce_float,
    coerce_int_list,
    coerce_str,
)
from ml_switcheroo_compiler.backends.edge._wasm_base import (
    _node_attrs,
)
from ml_switcheroo_compiler.backends.edge._wasm_control_matmul import _WasmControlMatMul
from ml_switcheroo_compiler.backends.protocols import adapt_backend_module


class _WasmNNPooling(_WasmControlMatMul):
    """WASM emission layer for 2D/1D pooling, Linear, Attention, and LayerNorm."""

    def visit_MaxPool2D(
        self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int
    ) -> None:
        """Generate MaxPool2D WASM.

        Args:
            node: Input `node` argument (node).
            op_type: Input `op_type` argument (op type).
            clean_id: Input `clean_id` argument (clean id).
            inputs: Input `inputs` argument (inputs).
            shape: Input `shape` argument (shape).
            nelem: Input `nelem` argument (nelem).
        """
        self._generate_pooling2d(node, clean_id, inputs, shape, "max_pool2d")

    def visit_AvgPool2D(
        self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int
    ) -> None:
        """Generate AvgPool2D WASM.

        Args:
            node: Input `node` argument (node).
            op_type: Input `op_type` argument (op type).
            clean_id: Input `clean_id` argument (clean id).
            inputs: Input `inputs` argument (inputs).
            shape: Input `shape` argument (shape).
            nelem: Input `nelem` argument (nelem).
        """
        self._generate_pooling2d(node, clean_id, inputs, shape, "avg_pool2d")

    def _generate_pooling2d(
        self, node: IRNode, clean_id: str, inputs: list[str], shape: Sequence[int] | int | float, template_name: str
    ) -> None:
        """Generate Pooling 2D.

        Args:
            node: Input `node` argument (node).
            clean_id: Input `clean_id` argument (clean id).
            inputs: Input `inputs` argument (inputs).
            shape: Input `shape` argument (shape).
            template_name: Input `template_name` argument (template name).
        """
        from ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider import get_wasm_template

        template = get_wasm_template(template_name)
        attrs = _node_attrs(node)

        in0_node = (
            next(
                (n for n in self.sorted_nodes if coerce_str(adapt_backend_module(n).get_symbol("id", "")) == inputs[0]),
                None,
            )
            if len(inputs) > 0
            else None
        )

        raw_in0_sh: BackendArg = (
            adapt_backend_module(in0_node).get_symbol("shape_metadata", [1, 1, 1, 1]) if in0_node else [1, 1, 1, 1]
        )

        if not raw_in0_sh:
            in0_shape: list[int] = [1, 1, 1, 1]
        elif isinstance(raw_in0_sh, (int, float)):
            in0_shape = [1, 1, 1, int(raw_in0_sh)]
        else:
            in0_shape = [int(x) for x in coerce_int_list(raw_in0_sh)]
        if len(in0_shape) < 4:
            in0_shape = [1] * (4 - len(in0_shape)) + in0_shape
        if not shape:
            shape_list = [1, 1, 1, 1]
        elif isinstance(shape, (int, float)):
            shape_list = [1, 1, 1, int(shape)]
        else:
            shape_list = list(shape)
        if len(shape_list) < 4:
            shape_list = [1] * (4 - len(shape_list)) + shape_list

        kernel_size = attrs.get("kernel_size", attrs.get("pool_size", 1))
        kernel_h = kernel_size[0] if isinstance(kernel_size, (list, tuple)) else kernel_size
        kernel_w = kernel_size[1] if isinstance(kernel_size, (list, tuple)) else kernel_size

        stride = attrs.get("stride", attrs.get("strides", 1))
        stride_h = stride[0] if isinstance(stride, (list, tuple)) else stride
        stride_w = stride[1] if isinstance(stride, (list, tuple)) else stride

        expr_args = {
            "B": shape_list[0],
            "C": shape_list[1],
            "out_height": shape_list[2],
            "out_width": shape_list[3],
            "in_height": in0_shape[2],
            "in_width": in0_shape[3],
            "kernel_h": kernel_h,
            "kernel_w": kernel_w,
            "stride_h": stride_h,
            "stride_w": stride_w,
            "clean_id": clean_id,
            "in0": inputs[0] if len(inputs) > 0 else "dummy",
        }
        body = template.get("body", "").format(**expr_args)
        for line in body.split("\\n"):
            self.add_line(f"    {line}")

    def visit_Linear(
        self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int
    ) -> None:
        """Generate Linear (Dense) WASM.

        Args:
            node: Input `node` argument (node).
            op_type: Input `op_type` argument (op type).
            clean_id: Input `clean_id` argument (clean id).
            inputs: Input `inputs` argument (inputs).
            shape: Input `shape` argument (shape).
            nelem: Input `nelem` argument (nelem).
        """
        from ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider import get_wasm_template

        template: dict[str, str] = get_wasm_template("Linear")

        in0 = inputs[0] if len(inputs) > 0 else "dummy"
        in1 = inputs[1] if len(inputs) > 1 else "dummy"
        in2 = inputs[2] if len(inputs) > 2 else "dummy"

        M = shape[0] if len(shape) > 0 else 1
        N = shape[1] if len(shape) > 1 else 1

        in0_node = (
            next(
                (
                    n
                    for n in self.sorted_nodes
                    if coerce_str(adapt_backend_module(n).get_symbol("id", "")).replace("-", "_") == inputs[0]
                ),
                None,
            )
            if len(inputs) > 0
            else None
        )
        in0_shape = adapt_backend_module(in0_node).get_symbol("shape_metadata", None) if in0_node else [1, 1]
        K = in0_shape[1] if isinstance(in0_shape, (list, tuple)) and len(in0_shape) > 1 else 1

        has_bias = "true" if in2 != "dummy" else "false"
        body = template.get("body", "").format(
            M=M, N=N, K=K, in0=in0, in1=in1, in2=in2, clean_id=clean_id, nelem=nelem, has_bias=has_bias
        )
        for line in body.split("\n"):
            if line.strip() or line == "":
                self.add_line(f"  {line}")

    def visit_Attention(
        self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int
    ) -> None:
        """Generate Attention WASM.

        Args:
            node: Input `node` argument (node).
            op_type: Input `op_type` argument (op type).
            clean_id: Input `clean_id` argument (clean id).
            inputs: Input `inputs` argument (inputs).
            shape: Input `shape` argument (shape).
            nelem: Input `nelem` argument (nelem).
        """
        from ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider import get_wasm_template

        template: dict[str, str] = get_wasm_template("Attention")
        in0 = inputs[0] if len(inputs) > 0 else "dummy"
        in1 = inputs[1] if len(inputs) > 1 else "dummy"
        in2 = inputs[2] if len(inputs) > 2 else "dummy"

        in0_node = (
            next(
                (
                    n
                    for n in self.sorted_nodes
                    if coerce_str(adapt_backend_module(n).get_symbol("id", "")).replace("-", "_") == inputs[0]
                ),
                None,
            )
            if len(inputs) > 0
            else None
        )
        in0_shape = adapt_backend_module(in0_node).get_symbol("shape_metadata", None) if in0_node else [1, 1]
        seq_len = in0_shape[0] if isinstance(in0_shape, (list, tuple)) and len(in0_shape) > 0 else 1
        embed_dim = in0_shape[1] if isinstance(in0_shape, (list, tuple)) and len(in0_shape) > 1 else 1

        body = template.get("body", "").format(
            clean_id=clean_id, nelem=nelem, in0=in0, in1=in1, in2=in2, seq_len=seq_len, embed_dim=embed_dim
        )
        for line in body.split("\n"):
            if line.strip() or line == "":
                self.add_line(f"  {line}")

    def visit_MaxPool(
        self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int
    ) -> None:
        """Generate MaxPool WASM.

        Args:
            node: Input `node` argument (node).
            op_type: Input `op_type` argument (op type).
            clean_id: Input `clean_id` argument (clean id).
            inputs: Input `inputs` argument (inputs).
            shape: Input `shape` argument (shape).
            nelem: Input `nelem` argument (nelem).
        """
        self.visit_MaxPool2D(node, op_type, clean_id, inputs, shape, nelem)

    def visit_LayerNorm(
        self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int
    ) -> None:
        """Generate LayerNorm WASM.

        Args:
            node: Input `node` argument (node).
            op_type: Input `op_type` argument (op type).
            clean_id: Input `clean_id` argument (clean id).
            inputs: Input `inputs` argument (inputs).
            shape: Input `shape` argument (shape).
            nelem: Input `nelem` argument (nelem).
        """
        from ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider import get_wasm_template

        template: dict[str, str] = get_wasm_template("LayerNorm")
        in0 = inputs[0] if len(inputs) > 0 else "dummy"
        in1 = inputs[1] if len(inputs) > 1 else "dummy"  # scale
        in2 = inputs[2] if len(inputs) > 2 else "dummy"  # bias

        attrs = _node_attrs(node)
        eps = float(coerce_float(attrs.get("epsilon", 1e-5)))

        in0_node = (
            next(
                (
                    n
                    for n in self.sorted_nodes
                    if coerce_str(adapt_backend_module(n).get_symbol("id", "")).replace("-", "_") == inputs[0]
                ),
                None,
            )
            if len(inputs) > 0
            else None
        )
        in0_shape = adapt_backend_module(in0_node).get_symbol("shape_metadata", None) if in0_node else [1, 1]
        rows = in0_shape[0] if isinstance(in0_shape, (list, tuple)) and len(in0_shape) > 0 else 1
        cols = in0_shape[1] if isinstance(in0_shape, (list, tuple)) and len(in0_shape) > 1 else 1

        body = template.get("body", "").format(
            clean_id=clean_id, nelem=nelem, in0=in0, in1=in1, in2=in2, rows=rows, cols=cols, eps=eps
        )
        for line in body.split("\n"):
            if line.strip() or line == "":
                self.add_line(f"  {line}")
