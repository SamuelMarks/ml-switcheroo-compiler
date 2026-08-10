from __future__ import annotations

# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Provide mixin module."""
from typing import Any

from .common import CommonASTVisitor


class LinearAlgebraASTVisitor(CommonASTVisitor):
    """LinearAlgebra AST visitor."""

    def visit_CholeskyVjp(self, node: Any, input_vars: list[str], **kwargs: Any) -> str:
        """Evaluate visit_CholeskyVjp operation.

        Args:
        node (object): The node parameter.
        input_vars (object): The input_vars parameter.
        **kwargs (object): Keyword args.

        Returns:
        str: Result.
        """
        if not hasattr(self.generator, "_cholesky_vjp_imported"):
            self.generator.add_line("    from ml_switcheroo_compiler.backends.eager.linalg import _cholesky_vjp_eager")
            self.generator._cholesky_vjp_imported = True
        pfx = self.generator._get_backend_prefix()
        backend_name = pfx[:-1] if pfx.endswith(".") else pfx
        return f"_cholesky_vjp_eager({backend_name}, {input_vars[0]}, {input_vars[1]})"

    def visit_LuVjp(self, node: Any, input_vars: list[str], **kwargs: Any) -> str:
        """Evaluate visit_LuVjp operation.

        Args:
        node (object): The node parameter.
        input_vars (object): The input_vars parameter.
        **kwargs (object): Keyword args.

        Returns:
        str: Result.
        """
        if not hasattr(self.generator, "_lu_vjp_imported"):
            self.generator.add_line("    from ml_switcheroo_compiler.backends.eager.linalg import _lu_vjp_eager")
            self.generator._lu_vjp_imported = True
        pfx = self.generator._get_backend_prefix()
        backend_name = pfx[:-1] if pfx.endswith(".") else pfx
        return f"_lu_vjp_eager({backend_name}, {input_vars[0]}, {input_vars[1]}, {input_vars[2]}, {input_vars[3]})"

    # pylint: disable=abstract-method
    """Linear algebra AST generator mixin."""

    def visit_BandPart(self, node: Any, input_vars: list[str], **kwargs: Any) -> str:
        """Evaluate visit_BandPart operation.

        Args:
        node (object): The node parameter.
        input_vars (object): The input_vars parameter.
        **kwargs (object): Keyword args.

        Returns:
        str: Result.
        """
        pfx = self.generator._get_backend_prefix()
        num_lower = kwargs.get("num_lower", -1)
        num_upper = kwargs.get("num_upper", -1)
        return f"{pfx}_band_part({input_vars[0]}, {num_lower}, {num_upper})"

    def visit_BandedTriangularSolve(self, node: Any, input_vars: list[str], **kwargs: Any) -> str:
        """Evaluate visit_BandedTriangularSolve operation.

        Args:
        node (object): The node parameter.
        input_vars (object): The input_vars parameter.
        **kwargs (object): Keyword args.

        Returns:
        str: Result.
        """
        pfx = self.generator._get_backend_prefix()
        lower = kwargs.get("lower", False)
        adjoint = kwargs.get("adjoint", False)
        return f"{pfx}_banded_triangular_solve({input_vars[0]}, {input_vars[1]}, lower={lower}, adjoint={adjoint})"

    def visit_GatherMm(self, node: Any, input_vars: list[str], **kwargs: Any) -> str:
        """Evaluate visit_GatherMm operation.

        Args:
        node (object): The node parameter.
        input_vars (object): The input_vars parameter.
        **kwargs (object): Keyword args.

        Returns:
        str: Result.
        """
        args_str = f"{input_vars[0]}, {input_vars[1]}"
        if "lhs_indices" in node.attributes:
            args_str += f", lhs_indices={input_vars[node.attributes['lhs_indices']]}"
        if "rhs_indices" in node.attributes:
            args_str += f", rhs_indices={input_vars[node.attributes['rhs_indices']]}"
        return f"{self.generator._get_backend_prefix()}.gather_mm({args_str})"

    def visit_SegmentedMm(self, node: Any, input_vars: list[str], **kwargs: Any) -> str:
        """Evaluate visit_SegmentedMm operation.

        Args:
        node (object): The node parameter.
        input_vars (object): The input_vars parameter.
        **kwargs (object): Keyword args.

        Returns:
        str: Result.
        """
        return f"{self.generator._get_backend_prefix()}.segmented_mm({input_vars[0]}, {input_vars[1]}, {input_vars[node.attributes.get('segments', 2)]})"

    def visit_BlockMaskedMm(self, node: Any, input_vars: list[str], **kwargs: Any) -> str:
        """Evaluate visit_BlockMaskedMm operation.

        Args:
        node (object): The node parameter.
        input_vars (object): The input_vars parameter.
        **kwargs (object): Keyword args.

        Returns:
        str: Result.
        """
        a = input_vars[0]
        b = input_vars[1]
        out = f"{self.generator._get_backend_prefix()}.matmul({a}, {b})"
        return out

    def visit_Quantize(self, node: Any, input_vars: list[str], **kwargs: Any) -> str:
        """Evaluate visit_Quantize operation.

        Args:
        node (object): The node parameter.
        input_vars (object): The input_vars parameter.
        **kwargs (object): Keyword args.

        Returns:
        str: Result.
        """
        group_size = node.attributes.get("group_size", 64)
        bits = node.attributes.get("bits", 4)
        idx = node.attributes.get("return_idx", 0)
        pfx = self.generator._get_backend_prefix()

        if pfx in ("mlx", "mx"):
            return f"mx.quantize({input_vars[0]}, group_size={group_size}, bits={bits})[{idx}]"
        return f"{input_vars[0]}"

    def visit_QuantizedMatmul(self, node: Any, input_vars: list[str], **kwargs: Any) -> str:
        """Evaluate visit_QuantizedMatmul operation.

        Args:
        node (object): The node parameter.
        input_vars (object): The input_vars parameter.
        **kwargs (object): Keyword args.

        Returns:
        str: Result.
        """
        transpose = node.attributes.get("transpose", True)
        group_size = node.attributes.get("group_size", 64)
        bits = node.attributes.get("bits", 4)
        pfx = self.generator._get_backend_prefix()

        x, w, scales, biases = input_vars[0], input_vars[1], input_vars[2], input_vars[3]
        if pfx in ("mlx", "mx"):
            return f"mx.quantized_matmul({x}, {w}, {scales}, {biases}, transpose={transpose}, group_size={group_size}, bits={bits})"
        return f"{pfx}.matmul({x}, {w}.T if {transpose} else {w})"

    def visit_GatherQMM(self, node: Any, input_vars: list[str], **kwargs: Any) -> str:
        """Evaluate visit_GatherQMM operation.

        Args:
        node (object): The node parameter.
        input_vars (object): The input_vars parameter.
        **kwargs (object): Keyword args.

        Returns:
        str: Result.
        """
        transpose = node.attributes.get("transpose", True)
        group_size = node.attributes.get("group_size", 64)
        bits = node.attributes.get("bits", 4)
        pfx = self.generator._get_backend_prefix()

        x, w, scales, biases, indices = (
            input_vars[0],
            input_vars[1],
            input_vars[2],
            input_vars[3],
            input_vars[4],
        )
        if pfx in ("mlx", "mx"):
            return f"mx.gather_qmm({x}, {w}, {scales}, {biases}, {indices}, transpose={transpose}, group_size={group_size}, bits={bits})"
        return f"{pfx}.matmul({x}, {w}[{indices}].T if {transpose} else {w}[{indices}])"

    def visit_QrVjp(self, node: Any, input_vars: list[str], **kwargs: Any) -> str:
        """Evaluate visit_QrVjp operation.

        Args:
        node (object): The node parameter.
        input_vars (object): The input_vars parameter.
        **kwargs (object): Keyword args.

        Returns:
        str: Result.
        """
        if not hasattr(self.generator, "_qr_vjp_imported"):
            self.generator.add_line("    from ml_switcheroo_compiler.backends.eager.linalg import _qr_vjp_eager")
            self.generator._qr_vjp_imported = True
        pfx = self.generator._get_backend_prefix()
        return f"_qr_vjp_eager({pfx}, {input_vars[0]}, {input_vars[1]}, {input_vars[2]})"

    def visit_SvdVjp(self, node: Any, input_vars: list[str], **kwargs: Any) -> str:
        """Evaluate visit_SvdVjp operation.

        Args:
        node (object): The node parameter.
        input_vars (object): The input_vars parameter.
        **kwargs (object): Keyword args.

        Returns:
        str: Result.
        """
        if not hasattr(self.generator, "_svd_vjp_imported"):
            self.generator.add_line("    from ml_switcheroo_compiler.backends.eager.linalg import _svd_vjp_eager")
            self.generator._svd_vjp_imported = True
        pfx = self.generator._get_backend_prefix()
        compute_uv = kwargs.get("compute_uv", True)
        if not compute_uv:
            return f"_svd_vjp_eager({pfx}, {input_vars[0]}, {input_vars[1]}, compute_uv=False)"
        return f"_svd_vjp_eager({pfx}, {input_vars[0]}, {input_vars[1]}, {input_vars[2]}, {input_vars[3]}, compute_uv=True)"
