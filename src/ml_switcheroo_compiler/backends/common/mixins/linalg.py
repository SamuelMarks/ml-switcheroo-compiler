"""Implementation of linalg operations and utilities for the backends subsystem (`ml_switcheroo_compiler.backends.common.mixins.linalg`)."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.backends.eager_registry import EagerValue, ensure_int
from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.ir.core import IRNode

from .common import CommonASTVisitor, FallbackPrefixProtocol


@typing.runtime_checkable
class _CholeskyVjpImportedProtocol(typing.Protocol):
    """Protocol for generators tracking `_cholesky_vjp_imported`."""

    _cholesky_vjp_imported: bool


@typing.runtime_checkable
class _LuVjpImportedProtocol(typing.Protocol):
    """Protocol for generators tracking `_lu_vjp_imported`."""

    _lu_vjp_imported: bool


@typing.runtime_checkable
class _QrVjpImportedProtocol(typing.Protocol):
    """Protocol for generators tracking `_qr_vjp_imported`."""

    _qr_vjp_imported: bool


@typing.runtime_checkable
class _SvdVjpImportedProtocol(typing.Protocol):
    """Protocol for generators tracking `_svd_vjp_imported`."""

    _svd_vjp_imported: bool


class LinearAlgebraASTVisitor(CommonASTVisitor):
    """LinearAlgebra AST visitor."""

    def visit_CholeskyVjp(self, node: IRNode, input_vars: list[str], **kwargs: EagerValue) -> str:
        """Emit target backend code or IR transformation for the CholeskyVjp operation.

        Args:
            node (IRNode): Input `node` argument (node).
            input_vars (list[str]): Input `input_vars` argument (input vars).
            **kwargs: Keyword args.

        Returns:
            str: Computed `str` output.
        """
        raw_gen: TensorOrScalar | FallbackPrefixProtocol = self.generator
        if not (isinstance(raw_gen, _CholeskyVjpImportedProtocol) and raw_gen._cholesky_vjp_imported):
            self.generator.add_line("    from ml_switcheroo_compiler.backends.eager.linalg import _cholesky_vjp_eager")
            self.generator._cholesky_vjp_imported = True
        pfx: str = self.generator.get_fallback_prefix()
        backend_name: str = pfx[:-1] if pfx.endswith(".") else pfx
        return f"_cholesky_vjp_eager({backend_name}, {input_vars[0]}, {input_vars[1]})"

    def visit_LuVjp(self, node: IRNode, input_vars: list[str], **kwargs: EagerValue) -> str:
        """Emit target backend code or IR transformation for the LuVjp operation.

        Args:
            node (IRNode): Input `node` argument (node).
            input_vars (list[str]): Input `input_vars` argument (input vars).
            **kwargs: Keyword args.

        Returns:
            str: Computed `str` output.
        """
        raw_gen: TensorOrScalar | FallbackPrefixProtocol = self.generator
        if not (isinstance(raw_gen, _LuVjpImportedProtocol) and raw_gen._lu_vjp_imported):
            self.generator.add_line("    from ml_switcheroo_compiler.backends.eager.linalg import _lu_vjp_eager")
            self.generator._lu_vjp_imported = True
        pfx: str = self.generator.get_fallback_prefix()
        backend_name: str = pfx[:-1] if pfx.endswith(".") else pfx
        return f"_lu_vjp_eager({backend_name}, {input_vars[0]}, {input_vars[1]}, {input_vars[2]}, {input_vars[3]})"

    def visit_BandPart(self, node: IRNode, input_vars: list[str], **kwargs: int) -> str:
        """Emit target backend code or IR transformation for the BandPart operation.

        Args:
            node (IRNode): Input `node` argument (node).
            input_vars (list[str]): Input `input_vars` argument (input vars).
            **kwargs: Keyword args.

        Returns:
            str: Computed `str` output.
        """
        pfx: str = self.generator.get_fallback_prefix()
        num_lower: int = kwargs.get("num_lower", -1)
        num_upper: int = kwargs.get("num_upper", -1)
        return f"{pfx}_band_part({input_vars[0]}, {num_lower}, {num_upper})"

    def visit_BandedTriangularSolve(self, node: IRNode, input_vars: list[str], **kwargs: bool) -> str:
        """Emit target backend code or IR transformation for the BandedTriangularSolve operation.

        Args:
            node (IRNode): Input `node` argument (node).
            input_vars (list[str]): Input `input_vars` argument (input vars).
            **kwargs: Keyword args.

        Returns:
            str: Computed `str` output.
        """
        pfx: str = self.generator.get_fallback_prefix()
        lower: bool = kwargs.get("lower", False)
        adjoint: bool = kwargs.get("adjoint", False)
        return f"{pfx}_banded_triangular_solve({input_vars[0]}, {input_vars[1]}, lower={lower}, adjoint={adjoint})"

    def visit_GatherMm(self, node: IRNode, input_vars: list[str], **kwargs: EagerValue) -> str:
        """Emit target backend code or IR transformation for the GatherMm operation.

        Args:
            node (IRNode): Input `node` argument (node).
            input_vars (list[str]): Input `input_vars` argument (input vars).
            **kwargs: Keyword args.

        Returns:
            str: Computed `str` output.
        """
        args_str: str = f"{input_vars[0]}, {input_vars[1]}"
        if "lhs_indices" in node.attributes:
            args_str += f", lhs_indices={input_vars[int(ensure_int(node.attributes['lhs_indices']))]}"
        if "rhs_indices" in node.attributes:
            args_str += f", rhs_indices={input_vars[int(ensure_int(node.attributes['rhs_indices']))]}"
        return f"{self.generator.get_fallback_prefix()}.gather_mm({args_str})"

    def visit_SegmentedMm(self, node: IRNode, input_vars: list[str], **kwargs: EagerValue) -> str:
        """Emit target backend code or IR transformation for the SegmentedMm operation.

        Args:
            node (IRNode): Input `node` argument (node).
            input_vars (list[str]): Input `input_vars` argument (input vars).
            **kwargs: Keyword args.

        Returns:
            str: Computed `str` output.
        """
        return f"{self.generator.get_fallback_prefix()}.segmented_mm({input_vars[0]}, {input_vars[1]}, {input_vars[int(ensure_int(node.attributes.get('segments', 2)))]})"

    def visit_BlockMaskedMm(self, node: IRNode, input_vars: list[str], **kwargs: EagerValue) -> str:
        """Emit target backend code or IR transformation for the BlockMaskedMm operation.

        Args:
            node (IRNode): Input `node` argument (node).
            input_vars (list[str]): Input `input_vars` argument (input vars).
            **kwargs: Keyword args.

        Returns:
            str: Computed `str` output.
        """
        a: str = input_vars[0]
        b: str = input_vars[1]
        out: str = f"{self.generator.get_fallback_prefix()}.matmul({a}, {b})"
        return out

    def visit_Quantize(self, node: IRNode, input_vars: list[str], **kwargs: EagerValue) -> str:
        """Emit target backend code or IR transformation for the Quantize operation.

        Args:
            node (IRNode): Input `node` argument (node).
            input_vars (list[str]): Input `input_vars` argument (input vars).
            **kwargs: Keyword args.

        Returns:
            str: Computed `str` output.
        """
        group_size: int = int(ensure_int(node.attributes.get("group_size", 64)))
        bits: int = int(ensure_int(node.attributes.get("bits", 4)))
        idx: int = int(ensure_int(node.attributes.get("return_idx", 0)))
        pfx: str = self.generator.get_fallback_prefix()

        if pfx in ("mlx", "mx"):
            return f"mx.quantize({input_vars[0]}, group_size={group_size}, bits={bits})[{idx}]"
        return f"{input_vars[0]}"

    def visit_QuantizedMatmul(self, node: IRNode, input_vars: list[str], **kwargs: EagerValue) -> str:
        """Emit target backend code or IR transformation for the QuantizedMatmul operation.

        Args:
            node (IRNode): Input `node` argument (node).
            input_vars (list[str]): Input `input_vars` argument (input vars).
            **kwargs: Keyword args.

        Returns:
            str: Computed `str` output.
        """
        transpose: bool = bool(node.attributes.get("transpose", True))
        group_size: int = int(ensure_int(node.attributes.get("group_size", 64)))
        bits: int = int(ensure_int(node.attributes.get("bits", 4)))
        pfx: str = self.generator.get_fallback_prefix()

        x: str = input_vars[0]
        w: str = input_vars[1]
        scales: str = input_vars[2]
        biases: str = input_vars[3]
        if pfx in ("mlx", "mx"):
            return f"mx.quantized_matmul({x}, {w}, {scales}, {biases}, transpose={transpose}, group_size={group_size}, bits={bits})"
        return f"{pfx}.matmul({x}, {w}.T if {transpose} else {w})"

    def visit_GatherQMM(self, node: IRNode, input_vars: list[str], **kwargs: EagerValue) -> str:
        """Emit target backend code or IR transformation for the GatherQMM operation.

        Args:
            node (IRNode): Input `node` argument (node).
            input_vars (list[str]): Input `input_vars` argument (input vars).
            **kwargs: Keyword args.

        Returns:
            str: Computed `str` output.
        """
        transpose: bool = bool(node.attributes.get("transpose", True))
        group_size: int = int(ensure_int(node.attributes.get("group_size", 64)))
        bits: int = int(ensure_int(node.attributes.get("bits", 4)))
        pfx: str = self.generator.get_fallback_prefix()

        x: str = input_vars[0]
        w: str = input_vars[1]
        scales: str = input_vars[2]
        biases: str = input_vars[3]
        indices: str = input_vars[4]
        if pfx in ("mlx", "mx"):
            return f"mx.gather_qmm({x}, {w}, {scales}, {biases}, {indices}, transpose={transpose}, group_size={group_size}, bits={bits})"
        return f"{pfx}.matmul({x}, {w}[{indices}].T if {transpose} else {w}[{indices}])"

    def visit_QrVjp(self, node: IRNode, input_vars: list[str], **kwargs: EagerValue) -> str:
        """Emit target backend code or IR transformation for the QrVjp operation.

        Args:
            node (IRNode): Input `node` argument (node).
            input_vars (list[str]): Input `input_vars` argument (input vars).
            **kwargs: Keyword args.

        Returns:
            str: Computed `str` output.
        """
        raw_gen: TensorOrScalar | FallbackPrefixProtocol = self.generator
        if not (isinstance(raw_gen, _QrVjpImportedProtocol) and raw_gen._qr_vjp_imported):
            self.generator.add_line("    from ml_switcheroo_compiler.backends.eager.linalg import _qr_vjp_eager")
            self.generator._qr_vjp_imported = True
        pfx: str = self.generator.get_fallback_prefix()
        return f"_qr_vjp_eager({pfx}, {input_vars[0]}, {input_vars[1]}, {input_vars[2]})"

    def visit_SvdVjp(self, node: IRNode, input_vars: list[str], **kwargs: bool) -> str:
        """Emit target backend code or IR transformation for the SvdVjp operation.

        Args:
            node (IRNode): Input `node` argument (node).
            input_vars (list[str]): Input `input_vars` argument (input vars).
            **kwargs: Keyword args.

        Returns:
            str: Computed `str` output.
        """
        raw_gen: TensorOrScalar | FallbackPrefixProtocol = self.generator
        if not (isinstance(raw_gen, _SvdVjpImportedProtocol) and raw_gen._svd_vjp_imported):
            self.generator.add_line("    from ml_switcheroo_compiler.backends.eager.linalg import _svd_vjp_eager")
            self.generator._svd_vjp_imported = True
        pfx: str = self.generator.get_fallback_prefix()
        compute_uv: bool = kwargs.get("compute_uv", True)
        if not compute_uv:
            return f"_svd_vjp_eager({pfx}, {input_vars[0]}, {input_vars[1]}, compute_uv=False)"
        return f"_svd_vjp_eager({pfx}, {input_vars[0]}, {input_vars[1]}, {input_vars[2]}, {input_vars[3]}, compute_uv=True)"
