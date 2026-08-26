"""Module image.py."""

from __future__ import annotations

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Provide mixin module."""

from typing import Any

from ml_switcheroo_compiler.ir.core import IRNode

from .common import CommonASTVisitor


class ImageASTVisitor(CommonASTVisitor):
    # pylint: disable=abstract-method
    """Image processing AST generator mixin."""

    def visit_AdjustBrightness(self, node: IRNode, input_vars: list[str], **kwargs: Any) -> str:
        """Evaluate visit_AdjustBrightness operation.

        Args:
            node (IRNode): The node parameter.
            input_vars (list[str]): The input_vars parameter.
            **kwargs (Any): Keyword args.

        Returns:
            str: Result.
        """
        pfx = self.generator.get_fallback_prefix()
        delta = kwargs.get("delta", 0.0)
        return f"{pfx}_adjust_brightness({input_vars[0]}, {delta})"

    def visit_AdjustContrast(self, node: IRNode, input_vars: list[str], **kwargs: Any) -> str:
        """Evaluate visit_AdjustContrast operation.

        Args:
            node (IRNode): The node parameter.
            input_vars (list[str]): The input_vars parameter.
            **kwargs (Any): Keyword args.

        Returns:
            str: Result.
        """
        pfx = self.generator.get_fallback_prefix()
        contrast_factor = kwargs.get("contrast_factor", 1.0)
        return f"{pfx}_adjust_contrast({input_vars[0]}, {contrast_factor})"

    def visit_AdjustHue(self, node: IRNode, input_vars: list[str], **kwargs: Any) -> str:
        """Evaluate visit_AdjustHue operation.

        Args:
            node (IRNode): The node parameter.
            input_vars (list[str]): The input_vars parameter.
            **kwargs (Any): Keyword args.

        Returns:
            str: Result.
        """
        pfx = self.generator.get_fallback_prefix()
        delta = kwargs.get("delta", 0.0)
        return f"{pfx}_adjust_hue({input_vars[0]}, {delta})"

    def visit_AdjustSaturation(self, node: IRNode, input_vars: list[str], **kwargs: Any) -> str:
        """Evaluate visit_AdjustSaturation operation.

        Args:
            node (IRNode): The node parameter.
            input_vars (list[str]): The input_vars parameter.
            **kwargs (Any): Keyword args.

        Returns:
            str: Result.
        """
        pfx = self.generator.get_fallback_prefix()
        saturation_factor = kwargs.get("saturation_factor", 1.0)
        return f"{pfx}_adjust_saturation({input_vars[0]}, {saturation_factor})"

    def visit_AffineGenerator(self, node: IRNode, input_vars: list[str], **kwargs: Any) -> str:
        """Evaluate visit_AffineGenerator operation.

        Args:
            node (IRNode): The node parameter.
            input_vars (list[str]): The input_vars parameter.
            **kwargs (Any): Keyword args.

        Returns:
            str: Result.
        """
        pfx = self.generator.get_fallback_prefix()
        batch_size = kwargs.get("batch_size", 1)
        return f"{pfx}_affine_generator({batch_size}, {', '.join(input_vars)})"

    def visit_AffineGrid(self, node: IRNode, input_vars: list[str], **kwargs: Any) -> str:
        """Evaluate visit_AffineGrid operation.

        Args:
            node (IRNode): The node parameter.
            input_vars (list[str]): The input_vars parameter.
            **kwargs (Any): Keyword args.

        Returns:
            str: Result.
        """
        pfx = self.generator.get_fallback_prefix()
        size = kwargs.get("size", ())
        align_corners = kwargs.get("align_corners", False)
        return f"{pfx}_affine_grid({input_vars[0]}, size={size}, align_corners={align_corners})"

    def visit_AffineTransform(self, node: IRNode, input_vars: list[str], **kwargs: Any) -> str:
        """Evaluate visit_AffineTransform operation.

        Args:
            node (IRNode): The node parameter.
            input_vars (list[str]): The input_vars parameter.
            **kwargs (Any): Keyword args.

        Returns:
            str: Result.
        """
        pfx = self.generator.get_fallback_prefix()
        interpolation = kwargs.get("interpolation", "nearest")
        return f"{pfx}_affine_transform({input_vars[0]}, {input_vars[1]}, interpolation='{interpolation}')"

    def visit_AugMix(self, node: IRNode, input_vars: list[str], **kwargs: Any) -> str:
        """Evaluate visit_AugMix operation.

        Args:
            node (IRNode): The node parameter.
            input_vars (list[str]): The input_vars parameter.
            **kwargs (Any): Keyword args.

        Returns:
            str: Result.
        """
        pfx = self.generator.get_fallback_prefix()
        return f"{pfx}_augmix({input_vars[0]})"

    def visit_AutoContrast(self, node: IRNode, input_vars: list[str], **kwargs: Any) -> str:
        """Evaluate visit_AutoContrast operation.

        Args:
            node (IRNode): The node parameter.
            input_vars (list[str]): The input_vars parameter.
            **kwargs (Any): Keyword args.

        Returns:
            str: Result.
        """
        pfx = self.generator.get_fallback_prefix()
        return f"{pfx}_auto_contrast({input_vars[0]})"
