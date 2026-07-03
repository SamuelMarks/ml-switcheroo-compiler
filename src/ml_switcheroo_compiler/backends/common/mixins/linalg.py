"""Mixin module."""

from __future__ import annotations

from .common import CommonASTVisitor


class LinearAlgebraASTVisitor(CommonASTVisitor):
    # pylint: disable=abstract-method
    """Linear algebra AST generator mixin."""

    def visit_BandPart(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate BandPart."""
        pfx = self.generator._get_backend_prefix()
        num_lower = kwargs.get("num_lower", -1)
        num_upper = kwargs.get("num_upper", -1)
        return f"{pfx}_band_part({input_vars[0]}, {num_lower}, {num_upper})"

    def visit_BandedTriangularSolve(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate BandedTriangularSolve."""
        pfx = self.generator._get_backend_prefix()
        lower = kwargs.get("lower", False)
        adjoint = kwargs.get("adjoint", False)
        return f"{pfx}_banded_triangular_solve({input_vars[0]}, {input_vars[1]}, lower={lower}, adjoint={adjoint})"

    def visit_GatherMm(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate GatherMm."""
        args_str = f"{input_vars[0]}, {input_vars[1]}"
        if "lhs_indices" in node.attributes:
            args_str += f", lhs_indices={input_vars[node.attributes['lhs_indices']]}"
        if "rhs_indices" in node.attributes:
            args_str += f", rhs_indices={input_vars[node.attributes['rhs_indices']]}"
        return f"{self.generator._get_backend_prefix()}.gather_mm({args_str})"

    def visit_SegmentedMm(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate SegmentedMm."""
        return f"{self.generator._get_backend_prefix()}.segmented_mm({input_vars[0]}, {input_vars[1]}, {input_vars[node.attributes.get('segments', 2)]})"

    def visit_BlockMaskedMm(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate BlockMaskedMm."""
        a = input_vars[0]
        b = input_vars[1]
        out = f"{self.generator._get_backend_prefix()}.matmul({a}, {b})"
        return out

    def visit_Quantize(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate Quantize."""
        group_size = node.attributes.get("group_size", 64)
        bits = node.attributes.get("bits", 4)
        idx = node.attributes.get("return_idx", 0)
        pfx = self.generator._get_backend_prefix()

        if pfx in ("mlx", "mx"):
            return f"mx.quantize({input_vars[0]}, group_size={group_size}, bits={bits})[{idx}]"
        return f"{input_vars[0]}"

    def visit_QuantizedMatmul(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate QuantizedMatmul."""
        transpose = node.attributes.get("transpose", True)
        group_size = node.attributes.get("group_size", 64)
        bits = node.attributes.get("bits", 4)
        pfx = self.generator._get_backend_prefix()

        x, w, scales, biases = input_vars[0], input_vars[1], input_vars[2], input_vars[3]
        if pfx in ("mlx", "mx"):
            return f"mx.quantized_matmul({x}, {w}, {scales}, {biases}, transpose={transpose}, group_size={group_size}, bits={bits})"
        return f"{pfx}.matmul({x}, {w}.T if {transpose} else {w})"

    def visit_GatherQMM(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate GatherQMM."""
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
