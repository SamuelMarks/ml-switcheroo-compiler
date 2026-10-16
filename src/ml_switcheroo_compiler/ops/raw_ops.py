"""Raw Operations mapping strategy."""

from __future__ import annotations

import math

from ml_switcheroo_compiler.backends.registry import EagerTensorProtocol, EagerValue
from ml_switcheroo_compiler.ops.base import OpDef, register_op


class RawOp(OpDef):
    """Define base class for all Raw operations."""

    def infer_shape(
        self,
        *args: EagerTensorProtocol,
        **kwargs: EagerValue,
    ) -> EagerValue:
        """Infer shape dynamically or fallback to unknown.

        Args:
            *args: The args parameter.
            **kwargs: The kwargs parameter.
        """
        if not args and "a" in kwargs:
            a = kwargs["a"]
        elif args:
            a = args[0]
        else:
            return ()
        return getattr(a, "shape", getattr(a, "shape_metadata", ()))


@register_op("RawSwitch")
class RawSwitch(RawOp):
    """Dynamic control flow RawSwitch."""

    op_name = "RawSwitch"

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: EagerValue) -> EagerValue:
        """Infer shape.

        Args:
            *args: The args parameter.
            **kwargs: The kwargs parameter.
        """
        shape = super().infer_shape(*args, **kwargs)
        return (shape, shape)


@register_op("RawMerge")
class RawMerge(RawOp):
    """Dynamic control flow RawMerge."""

    op_name = "RawMerge"

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: EagerValue) -> EagerValue:
        """Infer shape.

        Args:
            *args: The args parameter.
            **kwargs: The kwargs parameter.
        """
        shape = super().infer_shape(*args, **kwargs)
        return (shape, ())


@register_op("RawConv2D")
class RawConv2D(RawOp):
    """Raw Conv2D mapping."""

    op_name = "RawConv2D"

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: EagerValue) -> EagerValue:
        """Infer shape.

        Args:
            *args: The args parameter.
            **kwargs: The kwargs parameter.
        """
        inp: tuple[int, ...] = getattr(args[0], "shape", getattr(args[0], "shape_metadata", ()))
        filters: tuple[int, ...] = getattr(args[1], "shape", getattr(args[1], "shape_metadata", ()))
        padding = kwargs.get("padding", "VALID")
        strides = kwargs.get("strides", [1, 1, 1, 1])

        # inp: N, H, W, C
        # filters: FH, FW, C, K
        fh, fw = filters[0], filters[1]
        sh, sw = getattr(strides, "__getitem__")(1), getattr(strides, "__getitem__")(2)

        out_shape = [inp[0], 0, 0, filters[3]]

        for i, (dim, f, s) in enumerate(zip([inp[1], inp[2]], [fh, fw], [sh, sw])):
            if padding == "SAME":
                out_shape[i + 1] = math.ceil(dim / s)
            else:
                out_shape[i + 1] = math.ceil((dim - f + 1) / s)

        return tuple(out_shape)


@register_op("RawMatMul")
class RawMatMul(RawOp):
    """Raw MatMul mapping."""

    op_name = "RawMatMul"

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: EagerValue) -> EagerValue:
        """Infer shape.

        Args:
            *args: The args parameter.
            **kwargs: The kwargs parameter.
        """
        a: tuple[int, ...] = getattr(args[0], "shape", getattr(args[0], "shape_metadata", ()))
        b: tuple[int, ...] = getattr(args[1], "shape", getattr(args[1], "shape_metadata", ()))
        transpose_a = kwargs.get("transpose_a", False)
        transpose_b = kwargs.get("transpose_b", False)

        a_m = a[-1] if transpose_a else a[-2]
        b_p = b[-2] if transpose_b else b[-1]

        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        batch_shape = broadcast_shapes(a[:-2], b[:-2])
        return batch_shape + (a_m, b_p)
