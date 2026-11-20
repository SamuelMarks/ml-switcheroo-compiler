"""MatMul, Attention, 1D/2D pooling, shape telemetry, LayerNorm, and reduction visitors."""

from __future__ import annotations

import typing  # noqa: F401

from ml_switcheroo_ir import IRNode

from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    coerce_int,
    coerce_int_list,
    coerce_str,
)
from ml_switcheroo_compiler.backends.edge._webgpu_js_orchestrator import _WebGPUJsOrchestrator
from ml_switcheroo_compiler.backends.protocols import adapt_backend_module


class _WebGPUMatMulPoolTelem(_WebGPUJsOrchestrator):
    """WebGPU WGSL emission layer for matmul, attention, 2D pooling, and telemetry."""

    def visit_Linear(
        self, node: IRNode, input_vars: list[str], **kwargs: BackendArg
    ) -> tuple[list[str], str, str, str]:
        """Emit WGSL for Linear.

        Args:
            node (IRNode): The IR node.
            input_vars (list[str]): Names of the input variables.
            **kwargs (BackendArg): Additional generation kwargs.

        Returns:
            tuple[list[str], str, str, str]: Generated WGSL lines and dispatch dimensions.
        """
        return self._get_wgsl_for_op(
            node,
            coerce_int_list(kwargs.get("shape", [])),
            int(coerce_int(kwargs.get("nelem", 1))),
            str(kwargs.get("clean_id", "")),
        )

    def visit_MatMul(
        self, node: IRNode, input_vars: list[str], **kwargs: BackendArg
    ) -> tuple[list[str], str, str, str]:
        """Emit WGSL for MatMul with tiling and shared memory caching.

        Args:
            node (IRNode): The IR node.
            input_vars (list[str]): Names of the input variables.
            **kwargs (BackendArg): Additional generation kwargs.

        Returns:
            tuple[list[str], str, str, str]: Generated WGSL lines and dispatch dimensions.
        """
        return self._get_wgsl_for_op(
            node,
            coerce_int_list(kwargs.get("shape", [])),
            int(coerce_int(kwargs.get("nelem", 1))),
            str(kwargs.get("clean_id", "")),
        )

    def visit_BatchMatMul(
        self, node: IRNode, input_vars: list[str], **kwargs: BackendArg
    ) -> tuple[list[str], str, str, str]:
        """Emit WGSL for BatchMatMul with tiling and shared memory caching.

        Args:
            node (IRNode): The IR node.
            input_vars (list[str]): Names of the input variables.
            **kwargs (BackendArg): Additional generation kwargs.

        Returns:
            tuple[list[str], str, str, str]: Generated WGSL lines and dispatch dimensions.
        """
        return self._get_wgsl_for_op(
            node,
            coerce_int_list(kwargs.get("shape", [])),
            int(coerce_int(kwargs.get("nelem", 1))),
            str(kwargs.get("clean_id", "")),
        )

    def visit_Attention(
        self, node: IRNode, input_vars: list[str], **kwargs: BackendArg
    ) -> tuple[list[str], str, str, str]:
        """Emit WGSL for Multi-Head Attention with workgroup shared memory reduction.

        Args:
            node (IRNode): The IR node representing the attention op.
            input_vars (list[str]): Names of the input variables.
            **kwargs (BackendArg): Additional generation kwargs (shape, nelem, clean_id).

        Returns:
            tuple[list[str], str, str, str]: Generated WGSL lines and dispatch (x, y, z) expressions.
        """
        shape = coerce_int_list(kwargs.get("shape", []))
        nelem = int(coerce_int(kwargs.get("nelem", 1)))
        clean_id = str(kwargs.get("clean_id", ""))

        from ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider import get_wgsl_template
        from ml_switcheroo_compiler.backends.edge.wgsl_ast import WGSLFunction, WGSLRaw

        template = get_wgsl_template("attention") or {"body": "buf_out_f32[global_id.x] = 0.0;"}

        head_dim: int = shape[-1] if len(shape) >= 2 else (shape[0] if shape else 1)
        seq_len: int = shape[-2] if len(shape) >= 2 else (nelem // head_dim if head_dim > 0 else 1)
        if seq_len < 1:
            seq_len = 1
        if head_dim < 1:
            head_dim = 1
        scale: float = 1.0 / (head_dim**0.5)

        q_buf: str = "buf_in0_f32"
        k_buf: str = "buf_in1_f32" if len(input_vars) > 1 else "buf_in0_f32"
        v_buf: str = "buf_in2_f32" if len(input_vars) > 2 else "buf_in0_f32"

        body: str = str(template["body"]).format(
            nelem=nelem,
            seq_len=seq_len,
            head_dim=head_dim,
            scale=scale,
            q_buf=q_buf,
            k_buf=k_buf,
            v_buf=v_buf,
        )

        func: WGSLFunction = WGSLFunction(
            f"compute_{clean_id}",
            [
                "@builtin(global_invocation_id) global_id: vec3<u32>",
                "@builtin(local_invocation_id) local_id: vec3<u32>",
            ],
            [WGSLRaw(body)],
            ["@compute @workgroup_size(64, 1, 1)"],
        )
        return self.emitter.emit(func).split("\n"), f"Math.ceil({seq_len} / 64)", "1", "1"

    def visit_MaxPool(
        self, node: IRNode, input_vars: list[str], **kwargs: BackendArg
    ) -> tuple[list[str], str, str, str]:
        """Emit WGSL for MaxPool.

        Args:
            node (IRNode): The IR node.
            input_vars (list[str]): Names of the input variables.
            **kwargs (BackendArg): Additional generation kwargs.

        Returns:
            tuple[list[str], str, str, str]: Generated WGSL lines and dispatch dimensions.
        """
        return self._get_wgsl_for_op(
            node,
            coerce_int_list(kwargs.get("shape", [])),
            int(coerce_int(kwargs.get("nelem", 1))),
            str(kwargs.get("clean_id", "")),
        )

    def visit_MaxPool2D(
        self, node: IRNode, input_vars: list[str], **kwargs: BackendArg
    ) -> tuple[list[str], str, str, str]:
        """Emit WGSL for MaxPool2D compute shader.

        Args:
            node (IRNode): The IR node.
            input_vars (list[str]): Names of the input variables.
            **kwargs (BackendArg): Additional generation kwargs.

        Returns:
            tuple[list[str], str, str, str]: Generated WGSL lines and dispatch dimensions.
        """
        return self._get_wgsl_for_op(
            node,
            coerce_int_list(kwargs.get("shape", [])),
            int(coerce_int(kwargs.get("nelem", 1))),
            str(kwargs.get("clean_id", "")),
        )

    def visit_AvgPool(
        self, node: IRNode, input_vars: list[str], **kwargs: BackendArg
    ) -> tuple[list[str], str, str, str]:
        """Emit WGSL for AvgPool compute shader.

        Args:
            node (IRNode): The IR node.
            input_vars (list[str]): Names of the input variables.
            **kwargs (BackendArg): Additional generation kwargs.

        Returns:
            tuple[list[str], str, str, str]: Generated WGSL lines and dispatch dimensions.
        """
        return self._get_wgsl_for_op(
            node,
            coerce_int_list(kwargs.get("shape", [])),
            int(coerce_int(kwargs.get("nelem", 1))),
            str(kwargs.get("clean_id", "")),
        )

    def visit_AvgPool2D(
        self, node: IRNode, input_vars: list[str], **kwargs: BackendArg
    ) -> tuple[list[str], str, str, str]:
        """Emit WGSL for AvgPool2D compute shader.

        Args:
            node (IRNode): The IR node.
            input_vars (list[str]): Names of the input variables.
            **kwargs (BackendArg): Additional generation kwargs.

        Returns:
            tuple[list[str], str, str, str]: Generated WGSL lines and dispatch dimensions.
        """
        return self._get_wgsl_for_op(
            node,
            coerce_int_list(kwargs.get("shape", [])),
            int(coerce_int(kwargs.get("nelem", 1))),
            str(kwargs.get("clean_id", "")),
        )

    def _record_node_concrete_shape(
        self,
        node: IRNode,
        concrete: list[BackendArg] | tuple[BackendArg, ...],
        solved_env: dict[str, int],
    ) -> None:
        """Bind symbolic dimensions from a node's existing metadata to concrete telemetry values.

        Args:
            node (IRNode): The IR node being updated.
            concrete (list[BackendArg] | tuple[BackendArg, ...]): Concrete dimension sequence from telemetry.
            solved_env (dict[str, int]): Mapping of symbolic dimension names to resolved integer sizes.

        Returns:
            None: Updates solved_env and node.shape_metadata in-place.
        """
        meta = adapt_backend_module(node).get_symbol("shape_metadata", None)
        if meta and isinstance(meta, (list, tuple)) and len(meta) == len(concrete):
            for m_dim, c_dim in zip(meta, concrete):
                if adapt_backend_module(m_dim).has_symbol("node"):
                    solved_env[str(adapt_backend_module(m_dim).require_symbol("node"))] = int(coerce_int(c_dim))
                elif isinstance(m_dim, str):
                    solved_env[m_dim] = int(coerce_int(c_dim))
        node.shape_metadata = tuple(int(x) for x in concrete if isinstance(x, (int, float)))

    def _eval_symbolic_dim(self, dim: BackendArg, solved_env: dict[str, int]) -> int | None:
        """Evaluate a single symbolic or numeric shape dimension using the solved environment.

        Args:
            dim (BackendArg): Symbolic dimension object, variable name, or numeric value.
            solved_env (dict[str, int]): Mapping of symbolic dimension names to resolved integer sizes.

        Returns:
            int | None: Resolved integer dimension size, or None if unresolvable.
        """
        if adapt_backend_module(dim).has_symbol("node") and adapt_backend_module(
            adapt_backend_module(dim).require_symbol("node")
        ).has_symbol("eval"):
            try:
                return int(coerce_int(adapt_backend_module(dim).require_symbol("node").eval(solved_env)))
            except (
                AttributeError,
                ImportError,
                LookupError,
                OSError,
                ArithmeticError,
                RuntimeError,
                SyntaxError,
                TypeError,
                ValueError,
            ):
                return int(dim) if isinstance(dim, (int, float)) else 1
        if isinstance(dim, str) and dim in solved_env:
            return solved_env[dim]
        if isinstance(dim, (int, float)):
            return int(dim)
        return None

    def apply_shape_telemetry(self, telemetry: dict[str, BackendArg] | BackendArg) -> None:
        """Update IRGraph node shape metadata and refine symbolic dimensions from client telemetry.

        Args:
            telemetry (dict[str, BackendArg] | BackendArg): Mapping or payload containing learned concrete shapes.

        Returns:
            None: Updates node shape_metadata across the graph in-place.
        """
        shape_data = telemetry.get("shapes", telemetry) if isinstance(telemetry, dict) else {}
        if not isinstance(shape_data, dict):
            return

        solved_env: dict[str, int] = {}
        for node in self.sorted_nodes:
            nid: str = coerce_str(adapt_backend_module(node).get_symbol("id", ""))
            concrete = shape_data.get(nid)
            if isinstance(concrete, (list, tuple)):
                self._record_node_concrete_shape(node, concrete, solved_env)

        for node in self.sorted_nodes:
            meta = adapt_backend_module(node).get_symbol("shape_metadata", None)
            if meta and isinstance(meta, (list, tuple)):
                refined: list[int] = [
                    val for dim in meta if (val := self._eval_symbolic_dim(dim, solved_env)) is not None
                ]
                if refined:
                    node.shape_metadata = tuple(refined)

    def visit_LayerNorm(
        self, node: IRNode, input_vars: list[str], **kwargs: BackendArg
    ) -> tuple[list[str], str, str, str]:
        """Emit WGSL for LayerNorm.

        Args:
            node: Input `node` argument (node).
            input_vars: Input `input_vars` argument (input vars).
            **kwargs: Input `kwargs` argument (kwargs).

        Returns:
            tuple[list[str], str, str, str]: Computed result.
        """
        return self._get_wgsl_for_op(
            node,
            coerce_int_list(kwargs.get("shape", [])),
            int(coerce_int(kwargs.get("nelem", 1))),
            str(kwargs.get("clean_id", "")),
        )

    def visit_Trig(self, node: IRNode, input_vars: list[str], **kwargs: BackendArg) -> tuple[list[str], str, str, str]:
        """Emit WGSL for Trig (Sin/Cos).

        Args:
            node: Input `node` argument (node).
            input_vars: Input `input_vars` argument (input vars).
            **kwargs: Input `kwargs` argument (kwargs).

        Returns:
            tuple[list[str], str, str, str]: Computed result.
        """
        return self._get_wgsl_for_op(
            node,
            coerce_int_list(kwargs.get("shape", [])),
            int(coerce_int(kwargs.get("nelem", 1))),
            str(kwargs.get("clean_id", "")),
        )

    def visit_ReduceSum(
        self, node: IRNode, input_vars: list[str], **kwargs: BackendArg
    ) -> tuple[list[str], str, str, str]:
        """Emit WGSL for ReduceSum.

        Args:
            node: Input `node` argument (node).
            input_vars: Input `input_vars` argument (input vars).
            **kwargs: Input `kwargs` argument (kwargs).

        Returns:
            tuple[list[str], str, str, str]: Computed result.
        """
        return self._get_wgsl_for_op(
            node,
            coerce_int_list(kwargs.get("shape", [])),
            int(coerce_int(kwargs.get("nelem", 1))),
            str(kwargs.get("clean_id", "")),
        )
