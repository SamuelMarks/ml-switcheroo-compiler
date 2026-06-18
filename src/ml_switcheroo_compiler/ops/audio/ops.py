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
