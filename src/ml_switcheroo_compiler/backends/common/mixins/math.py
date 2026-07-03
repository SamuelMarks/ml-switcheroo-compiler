"""Mixin module."""

from __future__ import annotations

from .common import CommonASTVisitor


class MathASTVisitor(CommonASTVisitor):
    # pylint: disable=abstract-method
    # pylint: disable=too-many-public-methods
    """Math AST generator mixin."""

    def visit_AddN(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate AddN."""
        if not input_vars:
            return "0.0"
        return " + ".join(input_vars)

    def visit_AccumulateN(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate AccumulateN."""
        node.op_type = "AddN"
        return self.generator.visit(node, input_vars, **kwargs)

    def visit_Angle(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate Angle."""
        pfx = self.generator._get_backend_prefix()
        return f"{pfx}_angle({input_vars[0]})"

    def visit_Ball(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate Ball."""
        pfx = self.generator._get_backend_prefix()
        return f"{pfx}_ball()"

    def visit_Bartlett(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate Bartlett."""
        pfx = self.generator._get_backend_prefix()
        return f"{pfx}_bartlett({input_vars[0]})"

    def visit_BesselI0(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate BesselI0."""
        pfx = self.generator._get_backend_prefix()
        return f"{pfx}_bessel_i0({input_vars[0]})"

    def visit_BesselI0e(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate BesselI0e."""
        pfx = self.generator._get_backend_prefix()
        return f"{pfx}_bessel_i0e({input_vars[0]})"

    def visit_BesselI1(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate BesselI1."""
        pfx = self.generator._get_backend_prefix()
        return f"{pfx}_bessel_i1({input_vars[0]})"

    def visit_BesselI1e(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate BesselI1e."""
        pfx = self.generator._get_backend_prefix()
        return f"{pfx}_bessel_i1e({input_vars[0]})"

    def visit_BesselJ0(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate BesselJ0."""
        pfx = self.generator._get_backend_prefix()
        return f"{pfx}_bessel_j0({input_vars[0]})"

    def visit_BesselJ1(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate BesselJ1."""
        pfx = self.generator._get_backend_prefix()
        return f"{pfx}_bessel_j1({input_vars[0]})"

    def visit_BesselJn(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate BesselJn."""
        pfx = self.generator._get_backend_prefix()
        return f"{pfx}_bessel_jn({input_vars[0]}, {input_vars[1]})"

    def visit_BesselK0(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate BesselK0."""
        pfx = self.generator._get_backend_prefix()
        return f"{pfx}_bessel_k0({input_vars[0]})"

    def visit_BesselK0e(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate BesselK0e."""
        pfx = self.generator._get_backend_prefix()
        return f"{pfx}_bessel_k0e({input_vars[0]})"

    def visit_BesselK1(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate BesselK1."""
        pfx = self.generator._get_backend_prefix()
        return f"{pfx}_bessel_k1({input_vars[0]})"

    def visit_BesselK1e(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate BesselK1e."""
        pfx = self.generator._get_backend_prefix()
        return f"{pfx}_bessel_k1e({input_vars[0]})"

    def visit_BesselY0(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate BesselY0."""
        pfx = self.generator._get_backend_prefix()
        return f"{pfx}_bessel_y0({input_vars[0]})"

    def visit_BesselY1(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate BesselY1."""
        pfx = self.generator._get_backend_prefix()
        return f"{pfx}_bessel_y1({input_vars[0]})"

    def visit_Beta(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate Beta."""
        pfx = self.generator._get_backend_prefix()
        return f"{pfx}_beta({input_vars[0]}, {input_vars[1]})"

    def visit_Betainc(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate Betainc."""
        pfx = self.generator._get_backend_prefix()
        return f"{pfx}_betainc({input_vars[0]}, {input_vars[1]}, {input_vars[2]})"

    def visit_Logcumsumexp(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate Logcumsumexp."""
        return f"{self.generator._get_backend_prefix()}.logcumsumexp({input_vars[0]}, axis={node.attributes.get('axis', None)})"
