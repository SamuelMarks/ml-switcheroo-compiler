# ruff: noqa: E501
"""Provide mixin module."""

from __future__ import annotations

from .common import CommonASTVisitor


class ImageASTVisitor(CommonASTVisitor):
    # pylint: disable=abstract-method
    """Image processing AST generator mixin."""

    def visit_AdjustBrightness(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate visit_AdjustBrightness operation.

        Args:
        node (object): The node parameter.
        input_vars (object): The input_vars parameter.
        **kwargs (object): Keyword args.

        Returns:
        str: Result.
        """
        pfx = self.generator._get_backend_prefix()
        delta = kwargs.get("delta", 0.0)
        return f"{pfx}_adjust_brightness({input_vars[0]}, {delta})"

    def visit_AdjustContrast(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate visit_AdjustContrast operation.

        Args:
        node (object): The node parameter.
        input_vars (object): The input_vars parameter.
        **kwargs (object): Keyword args.

        Returns:
        str: Result.
        """
        pfx = self.generator._get_backend_prefix()
        contrast_factor = kwargs.get("contrast_factor", 1.0)
        return f"{pfx}_adjust_contrast({input_vars[0]}, {contrast_factor})"

    def visit_AdjustHue(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate visit_AdjustHue operation.

        Args:
        node (object): The node parameter.
        input_vars (object): The input_vars parameter.
        **kwargs (object): Keyword args.

        Returns:
        str: Result.
        """
        pfx = self.generator._get_backend_prefix()
        delta = kwargs.get("delta", 0.0)
        return f"{pfx}_adjust_hue({input_vars[0]}, {delta})"

    def visit_AdjustSaturation(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate visit_AdjustSaturation operation.

        Args:
        node (object): The node parameter.
        input_vars (object): The input_vars parameter.
        **kwargs (object): Keyword args.

        Returns:
        str: Result.
        """
        pfx = self.generator._get_backend_prefix()
        saturation_factor = kwargs.get("saturation_factor", 1.0)
        return f"{pfx}_adjust_saturation({input_vars[0]}, {saturation_factor})"

    def visit_AffineGenerator(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate visit_AffineGenerator operation.

        Args:
        node (object): The node parameter.
        input_vars (object): The input_vars parameter.
        **kwargs (object): Keyword args.

        Returns:
        str: Result.
        """
        pfx = self.generator._get_backend_prefix()
        batch_size = kwargs.get("batch_size", 1)
        return f"{pfx}_affine_generator({batch_size}, {', '.join(input_vars)})"

    def visit_AffineGrid(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate visit_AffineGrid operation.

        Args:
        node (object): The node parameter.
        input_vars (object): The input_vars parameter.
        **kwargs (object): Keyword args.

        Returns:
        str: Result.
        """
        pfx = self.generator._get_backend_prefix()
        size = kwargs.get("size", ())
        align_corners = kwargs.get("align_corners", False)
        return f"{pfx}_affine_grid({input_vars[0]}, size={size}, align_corners={align_corners})"

    def visit_AffineTransform(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate visit_AffineTransform operation.

        Args:
        node (object): The node parameter.
        input_vars (object): The input_vars parameter.
        **kwargs (object): Keyword args.

        Returns:
        str: Result.
        """
        pfx = self.generator._get_backend_prefix()
        interpolation = kwargs.get("interpolation", "nearest")
        return f"{pfx}_affine_transform({input_vars[0]}, {input_vars[1]}, interpolation='{interpolation}')"

    def visit_AugMix(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate visit_AugMix operation.

        Args:
        node (object): The node parameter.
        input_vars (object): The input_vars parameter.
        **kwargs (object): Keyword args.

        Returns:
        str: Result.
        """
        pfx = self.generator._get_backend_prefix()
        return f"{pfx}_augmix({input_vars[0]})"

    def visit_AutoContrast(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate visit_AutoContrast operation.

        Args:
        node (object): The node parameter.
        input_vars (object): The input_vars parameter.
        **kwargs (object): Keyword args.

        Returns:
        str: Result.
        """
        pfx = self.generator._get_backend_prefix()
        return f"{pfx}_auto_contrast({input_vars[0]})"
