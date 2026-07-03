"""Module docstring."""

from ml_switcheroo_compiler.ops.base import OpDef, register_op


@register_op("Fft")
class Fft(OpDef):
    """FFT operation."""

    op_name = "Fft"

    def infer_shape(
        self,
        a: object,
        n: object = None,
        axis: object = -1,
        **kwargs: object,
    ) -> object:
        """Infer shape.

        Args:
            a (object): The input a tensor.
            n (object): The n parameter for the operation.
            axis (object): The axis along which to perform the operation.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The evaluated output resulting from this operation.
        """
        if not hasattr(a, "shape") or not a.shape:
            return ()
        out_shape = list(a.shape)
        if n is not None:
            out_shape[axis] = n
        return tuple(out_shape)

    def emit_jax(self, *args: object, **kwargs: object) -> object:
        """Emit jax code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The evaluated output resulting from this operation.
        """
        return "Not implemented Fft"

    def emit_keras(self, *args: object, **kwargs: object) -> object:
        """Emit keras code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The evaluated output resulting from this operation.
        """
        return "Not implemented Fft"

    def emit_mlx(self, *args: object, **kwargs: object) -> object:
        """Emit mlx code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The evaluated output resulting from this operation.
        """
        return "Not implemented Fft"

    def emit_pytorch(self, *args: object, **kwargs: object) -> object:
        """Emit pytorch code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The evaluated output resulting from this operation.
        """
        return "Not implemented Fft"

    def emit_tensorflow(self, *args: object, **kwargs: object) -> object:
        """Emit tensorflow code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The evaluated output resulting from this operation.
        """
        return "Not implemented Fft"


@register_op("Rfft")
class Rfft(OpDef):
    """RFFT operation."""

    op_name = "Rfft"

    def infer_shape(
        self,
        a: object,
        n: object = None,
        axis: object = -1,
        **kwargs: object,
    ) -> object:
        """Infer shape.

        Args:
            a (object): The input a tensor.
            n (object): The n parameter for the operation.
            axis (object): The axis along which to perform the operation.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The evaluated output resulting from this operation.
        """
        if not hasattr(a, "shape") or not a.shape:
            return ()
        out_shape = list(a.shape)
        if n is None:
            n = out_shape[axis]
        out_shape[axis] = n // 2 + 1
        return tuple(out_shape)

    def emit_jax(self, *args: object, **kwargs: object) -> object:
        """Emit jax code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The evaluated output resulting from this operation.
        """
        return "Not implemented Rfft"

    def emit_keras(self, *args: object, **kwargs: object) -> object:
        """Emit keras code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The evaluated output resulting from this operation.
        """
        return "Not implemented Rfft"

    def emit_mlx(self, *args: object, **kwargs: object) -> object:
        """Emit mlx code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The evaluated output resulting from this operation.
        """
        return "Not implemented Rfft"

    def emit_pytorch(self, *args: object, **kwargs: object) -> object:
        """Emit pytorch code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The evaluated output resulting from this operation.
        """
        return "Not implemented Rfft"

    def emit_tensorflow(self, *args: object, **kwargs: object) -> object:
        """Emit tensorflow code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The evaluated output resulting from this operation.
        """
        return "Not implemented Rfft"


@register_op("Ifft")
class Ifft(OpDef):
    """Ifft operator.

    Computes the IFFT.
    """

    def infer_shape(self, a: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            a: Arg.
            **kwargs: Kwargs.
        """
        """Infer shape.

        Args:
            a: Arg.
            **kwargs: Kwargs.
        """
        """Infer shape.

        Args:
            a: arg.
            **kwargs: kwargs.
        """
        return a.shape


@register_op("Fft2d")
class Fft2d(OpDef):
    """Fft2d operator.

    Computes the Fft2d.
    """

    def infer_shape(self, a: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            a: Arg.
            **kwargs: Kwargs.
        """
        """Infer shape.

        Args:
            a: Arg.
            **kwargs: Kwargs.
        """
        """Infer shape.

        Args:
            a: arg.
            **kwargs: kwargs.
        """
        return a.shape


@register_op("Ifft2d")
class Ifft2d(OpDef):
    """Ifft2d operator.

    Computes the Ifft2d.
    """

    def infer_shape(self, a: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            a: Arg.
            **kwargs: Kwargs.
        """
        """Infer shape.

        Args:
            a: Arg.
            **kwargs: Kwargs.
        """
        """Infer shape.

        Args:
            a: arg.
            **kwargs: kwargs.
        """
        return a.shape


@register_op("Irfft")
class Irfft(OpDef):
    """Irfft operator.

    Computes the Irfft.
    """

    def infer_shape(self, a: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            a: Arg.
            **kwargs: Kwargs.
        """
        """Infer shape.

        Args:
            a: Arg.
            **kwargs: Kwargs.
        """
        """Infer shape.

        Args:
            a: arg.
            **kwargs: kwargs.
        """
        return a.shape


@register_op("Fft3d")
class Fft3d(OpDef):
    """Fft3d operator.

    Computes the Fft3d.
    """

    def infer_shape(self, a: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            a: Arg.
            **kwargs: Kwargs.
        """
        """Infer shape.

        Args:
            a: Arg.
            **kwargs: Kwargs.
        """
        """Infer shape.

        Args:
            a: arg.
            **kwargs: kwargs.
        """
        return a.shape


@register_op("Ifft3d")
class Ifft3d(OpDef):
    """Ifft3d operator.

    Computes the Ifft3d.
    """

    def infer_shape(self, a: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            a: Arg.
            **kwargs: Kwargs.
        """
        """Infer shape.

        Args:
            a: Arg.
            **kwargs: Kwargs.
        """
        """Infer shape.

        Args:
            a: arg.
            **kwargs: kwargs.
        """
        return a.shape


@register_op("Rfft2d")
class Rfft2d(OpDef):
    """Rfft2d operator.

    Computes the Rfft2d.
    """

    def infer_shape(self, a: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            a: Arg.
            **kwargs: Kwargs.
        """
        """Infer shape.

        Args:
            a: Arg.
            **kwargs: Kwargs.
        """
        """Infer shape.

        Args:
            a: arg.
            **kwargs: kwargs.
        """
        return a.shape


@register_op("Rfft3d")
class Rfft3d(OpDef):
    """Rfft3d operator.

    Computes the Rfft3d.
    """

    def infer_shape(self, a: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            a: Arg.
            **kwargs: Kwargs.
        """
        """Infer shape.

        Args:
            a: Arg.
            **kwargs: Kwargs.
        """
        """Infer shape.

        Args:
            a: arg.
            **kwargs: kwargs.
        """
        return a.shape


@register_op("Irfft2d")
class Irfft2d(OpDef):
    """Irfft2d operator.

    Computes the Irfft2d.
    """

    def infer_shape(self, a: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            a: Arg.
            **kwargs: Kwargs.
        """
        """Infer shape.

        Args:
            a: Arg.
            **kwargs: Kwargs.
        """
        """Infer shape.

        Args:
            a: arg.
            **kwargs: kwargs.
        """
        return a.shape


@register_op("Irfft3d")
class Irfft3d(OpDef):
    """Irfft3d operator.

    Computes the Irfft3d.
    """

    def infer_shape(self, a: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            a: Arg.
            **kwargs: Kwargs.
        """
        """Infer shape.

        Args:
            a: Arg.
            **kwargs: Kwargs.
        """
        """Infer shape.

        Args:
            a: arg.
            **kwargs: kwargs.
        """
        return a.shape


@register_op("Fftnd")
class Fftnd(OpDef):
    """Fftnd operator."""

    op_name = "Fftnd"

    def infer_shape(self, a: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            a: Arg.
            **kwargs: Kwargs.
        """
        """Infer shape.

        Args:
            a: Arg.
            **kwargs: Kwargs.
        """
        shape = list(a.shape)
        s = kwargs.get("s", None)
        axes = kwargs.get("axes", None)
        if s is not None and axes is not None:
            for sz, ax in zip(s, axes):
                shape[ax] = sz
        return tuple(shape)


@register_op("Ifftnd")
class Ifftnd(OpDef):
    """Ifftnd operator."""

    op_name = "Ifftnd"

    def infer_shape(self, a: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            a: Arg.
            **kwargs: Kwargs.
        """
        """Infer shape.

        Args:
            a: Arg.
            **kwargs: Kwargs.
        """
        shape = list(a.shape)
        s = kwargs.get("s", None)
        axes = kwargs.get("axes", None)
        if s is not None and axes is not None:
            for sz, ax in zip(s, axes):
                shape[ax] = sz
        return tuple(shape)


@register_op("Rfftnd")
class Rfftnd(OpDef):
    """Rfftnd operator."""

    op_name = "Rfftnd"

    def infer_shape(self, a: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            a: Arg.
            **kwargs: Kwargs.
        """
        """Infer shape.

        Args:
            a: Arg.
            **kwargs: Kwargs.
        """
        shape = list(a.shape)
        s = kwargs.get("s", None)
        axes = kwargs.get("axes", None)
        if s is not None and axes is not None:
            for sz, ax in zip(s, axes):
                shape[ax] = sz
        else:
            ax = axes[-1] if axes is not None else -1
            shape[ax] = shape[ax] // 2 + 1
        return tuple(shape)


@register_op("Irfftnd")
class Irfftnd(OpDef):
    """Irfftnd operator."""

    op_name = "Irfftnd"

    def infer_shape(self, a: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            a: Arg.
            **kwargs: Kwargs.
        """
        """Infer shape.

        Args:
            a: Arg.
            **kwargs: Kwargs.
        """
        shape = list(a.shape)
        s = kwargs.get("s", None)
        axes = kwargs.get("axes", None)
        if s is not None and axes is not None:
            for sz, ax in zip(s, axes):
                shape[ax] = sz
        else:
            ax = axes[-1] if axes is not None else -1
            shape[ax] = (shape[ax] - 1) * 2
        return tuple(shape)


@register_op("Fftshift")
class Fftshift(OpDef):
    """Fftshift operator."""

    op_name = "Fftshift"

    def infer_shape(self, a: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            a: Arg.
            **kwargs: Kwargs.
        """
        """Infer shape.

        Args:
            a: Arg.
            **kwargs: Kwargs.
        """
        return a.shape


@register_op("Ifftshift")
class Ifftshift(OpDef):
    """Ifftshift operator."""

    op_name = "Ifftshift"

    def infer_shape(self, a: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            a: Arg.
            **kwargs: Kwargs.
        """
        """Infer shape.

        Args:
            a: Arg.
            **kwargs: Kwargs.
        """
        return a.shape
