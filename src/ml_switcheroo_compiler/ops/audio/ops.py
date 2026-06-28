"""Audio operations class definitions."""

from ml_switcheroo_compiler.ops.base import OpDef, register_op


@register_op("Stft")
class Stft(OpDef):
    """Stft op."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape."""
        return ()


@register_op("MelSpectrogram")
class MelSpectrogram(OpDef):
    """MelSpectrogram op."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape."""
        return ()


@register_op("Istft")
class Istft(OpDef):
    """Istft op."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape."""
        return ()


@register_op("MelFilterbank")
class MelFilterbank(OpDef):
    """MelFilterbank op."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape."""
        return ()


@register_op("Mfcc")
class Mfcc(OpDef):
    """Mfcc op."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape."""
        return ()


@register_op("MfccsFromLogMelSpectrograms")
class MfccsFromLogMelSpectrograms(OpDef):
    """MfccsFromLogMelSpectrograms op."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape."""
        return ()


@register_op("HannWindow")
class HannWindow(OpDef):
    """HannWindow op."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape."""
        return ()


@register_op("HammingWindow")
class HammingWindow(OpDef):
    """HammingWindow op."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape."""
        return ()


@register_op("KaiserWindow")
class KaiserWindow(OpDef):
    """KaiserWindow op."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape."""
        return ()


@register_op("Dct")
class Dct(OpDef):
    """Dct operator."""

    op_name = "Dct"

    def infer_shape(self, a: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            a: Arg.
            **kwargs: Kwargs.
        """
        return a.shape


@register_op("Idct")
class Idct(OpDef):
    """Idct operator."""

    op_name = "Idct"

    def infer_shape(self, a: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            a: Arg.
            **kwargs: Kwargs.
        """
        return a.shape


@register_op("Mdct")
class Mdct(OpDef):
    """Mdct operator."""

    op_name = "Mdct"

    def infer_shape(self, a: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            a: Arg.
            **kwargs: Kwargs.
        """
        # Dummy infer_shape for now
        shape = list(a.shape)
        if len(shape) > 0:  # pragma: no cover
            shape[-1] = shape[-1] // 2  # pragma: no cover
        return tuple(shape)  # pragma: no cover


@register_op("InverseMdct")
class InverseMdct(OpDef):
    """InverseMdct operator."""

    op_name = "InverseMdct"

    def infer_shape(self, a: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            a: Arg.
            **kwargs: Kwargs.
        """
        shape = list(a.shape)
        if len(shape) > 0:  # pragma: no cover
            shape[-1] = shape[-1] * 2  # pragma: no cover
        return tuple(shape)  # pragma: no cover


@register_op("Frame")
class Frame(OpDef):
    """Frame operator."""

    op_name = "Frame"

    def infer_shape(self, a: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            a: Arg.
            **kwargs: Kwargs.
        """
        # Dummy infer_shape for now
        shape = list(a.shape)
        frame_length = kwargs.get("frame_length", 1)  # pragma: no cover
        frame_step = kwargs.get("frame_step", 1)  # pragma: no cover
        if len(shape) > 0:  # pragma: no cover
            num_frames = max(0, (shape[-1] - frame_length) // frame_step + 1)  # pragma: no cover
            shape[-1] = num_frames  # pragma: no cover
            shape.append(frame_length)  # pragma: no cover
        return tuple(shape)  # pragma: no cover


@register_op("OverlapAndAdd")
class OverlapAndAdd(OpDef):
    """OverlapAndAdd operator."""

    op_name = "OverlapAndAdd"

    def infer_shape(self, a: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            a: Arg.
            **kwargs: Kwargs.
        """
        shape = list(a.shape)
        frame_step = kwargs.get("frame_step", 1)  # pragma: no cover
        if len(shape) >= 2:  # noqa: PLR2004  # pragma: no cover
            num_frames = shape[-2]  # pragma: no cover
            frame_length = shape[-1]  # pragma: no cover
            shape.pop()  # pragma: no cover
            shape[-1] = (num_frames - 1) * frame_step + frame_length  # pragma: no cover
        return tuple(shape)  # pragma: no cover
