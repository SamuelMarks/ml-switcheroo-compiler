"""Raw Operations mapping strategy."""

from __future__ import annotations

import math

from ml_switcheroo_compiler.backends.registry import EagerTensorProtocol, ensure_int
from ml_switcheroo_compiler.core.shape import TensorOrScalar, extract_shape
from ml_switcheroo_compiler.ops.base import OpDef, register_op


class RawOp(OpDef):
    """Define base class for all Raw operations."""

    def infer_shape(
        self,
        *args: EagerTensorProtocol,
        **kwargs: TensorOrScalar,
    ) -> TensorOrScalar:
        """Infer shape dynamically or fallback to unknown.

        Args:
            *args: Input `args` argument (args).
            **kwargs: Input `kwargs` argument (kwargs).

        Returns:
            TensorOrScalar: Computed result.
        """
        if not args and "a" in kwargs:
            a = kwargs["a"]
        elif args:
            a = args[0]
        else:
            return ()
        return extract_shape(a)


@register_op("RawSwitch")
class RawSwitch(RawOp):
    """Dynamic control flow RawSwitch."""

    op_name = "RawSwitch"

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer shape.

        Args:
            *args: Input `args` argument (args).
            **kwargs: Input `kwargs` argument (kwargs).

        Returns:
            TensorOrScalar: Computed result.
        """
        shape = super().infer_shape(*args, **kwargs)
        return (shape, shape)


@register_op("RawMerge")
class RawMerge(RawOp):
    """Dynamic control flow RawMerge."""

    op_name = "RawMerge"

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer shape.

        Args:
            *args: Input `args` argument (args).
            **kwargs: Input `kwargs` argument (kwargs).

        Returns:
            TensorOrScalar: Computed result.
        """
        shape = super().infer_shape(*args, **kwargs)
        return (shape, ())


@register_op("RawConv2D")
class RawConv2D(RawOp):
    """Raw Conv2D mapping."""

    op_name = "RawConv2D"

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer shape.

        Args:
            *args: Input `args` argument (args).
            **kwargs: Input `kwargs` argument (kwargs).

        Returns:
            TensorOrScalar: Computed result.
        """
        inp: tuple[int, ...] = extract_shape(args[0])
        filters: tuple[int, ...] = extract_shape(args[1])
        padding = kwargs.get("padding", "VALID")
        raw_strides = kwargs.get("strides", [1, 1, 1, 1])
        strides_seq = raw_strides if isinstance(raw_strides, (list, tuple)) else [1, 1, 1, 1]

        # inp: N, H, W, C
        # filters: FH, FW, C, K
        fh, fw = filters[0], filters[1]
        sh, sw = int(ensure_int(strides_seq[1])), int(ensure_int(strides_seq[2]))

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

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer shape.

        Args:
            *args: Input `args` argument (args).
            **kwargs: Input `kwargs` argument (kwargs).

        Returns:
            TensorOrScalar: Computed result.
        """
        a: tuple[int, ...] = extract_shape(args[0])
        b: tuple[int, ...] = extract_shape(args[1])
        transpose_a = kwargs.get("transpose_a", False)
        transpose_b = kwargs.get("transpose_b", False)

        a_m = a[-1] if transpose_a else a[-2]
        b_p = b[-2] if transpose_b else b[-1]

        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        batch_shape = broadcast_shapes(a[:-2], b[:-2])
        return batch_shape + (a_m, b_p)
