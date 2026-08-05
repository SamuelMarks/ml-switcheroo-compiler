"""Audio operations class definitions."""

from ml_switcheroo_compiler.ops.base import OpDef, register_op


@register_op("Stft")
class Stft(OpDef):
    """Short-time Fourier transform operator."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infers the shape of the STFT output.

        Args:
            *args: Positional arguments including the input signal.
            **kwargs: Keyword arguments including STFT parameters.

        Returns:
            The output shape as a tuple of integers.
        """
        if len(args) > 0 and hasattr(args[0], "shape"):
            shape = list(getattr(args[0], "shape", ()))
            if len(shape) > 0:
                signal_length = shape[-1]
                frame_length = kwargs.get("frame_length", 256)
                frame_step = kwargs.get("frame_step", 128)
                fft_length = kwargs.get("fft_length", kwargs.get("fft_size", frame_length))

                num_frames = max(1, (signal_length - frame_length) // frame_step + 1)
                fft_bins = fft_length // 2 + 1
                shape[-1] = num_frames
                shape.append(fft_bins)
            return tuple(shape)
        return ()


@register_op("MelSpectrogram")
class MelSpectrogram(OpDef):
    """Mel spectrogram computation operator."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infers the shape of the mel spectrogram output.

        Args:
            *args: Positional arguments including the input audio signal.
            **kwargs: Keyword arguments for mel spectrogram configuration.

        Returns:
            The output shape as a tuple of integers.
        """
        if len(args) > 0 and hasattr(args[0], "shape"):
            shape = list(getattr(args[0], "shape", ()))
            if len(shape) > 0:
                signal_length = shape[-1]
                frame_length = kwargs.get("frame_length", 256)
                frame_step = kwargs.get("frame_step", 128)
                num_mel_bins = kwargs.get("num_mel_bins", 128)
                num_frames = max(1, (signal_length - frame_length) // frame_step + 1)
                shape[-1] = num_frames
                shape.append(num_mel_bins)
            return tuple(shape)
        return ()


@register_op("Istft")
class Istft(OpDef):
    """Inverse short-time Fourier transform operator."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infers the shape of the inverse STFT output.

        Args:
            *args: Positional arguments including the STFT representation.
            **kwargs: Keyword arguments including ISTFT parameters.

        Returns:
            The output shape as a tuple of integers.
        """
        if len(args) > 0 and hasattr(args[0], "shape"):
            shape = list(getattr(args[0], "shape", ()))
            if len(shape) >= 2:
                num_frames = shape[-2]
                frame_length = kwargs.get("frame_length", 256)
                frame_step = kwargs.get("frame_step", 128)
                signal_length = (num_frames - 1) * frame_step + frame_length
                shape = shape[:-2]
                shape.append(signal_length)
            return tuple(shape)
        return ()


@register_op("MelFilterbank")
class MelFilterbank(OpDef):
    """Mel filterbank generation operator."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infers the shape of the mel filterbank matrix.

        Args:
            *args: Positional arguments for filterbank generation.
            **kwargs: Keyword arguments for filterbank configuration.

        Returns:
            The output shape as a tuple of integers.
        """
        fft_length = kwargs.get("fft_length", kwargs.get("fft_size", 256))
        num_mel_bins = kwargs.get("num_mel_bins", 128)
        fft_bins = fft_length // 2 + 1
        return (fft_bins, num_mel_bins)


@register_op("Mfcc")
class Mfcc(OpDef):
    """Mel-frequency cepstral coefficients (MFCC) extraction operator."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infers the shape of the MFCC output.

        Args:
            *args: Positional arguments including the input signal.
            **kwargs: Keyword arguments for MFCC configuration.

        Returns:
            The output shape as a tuple of integers.
        """
        if len(args) > 0 and hasattr(args[0], "shape"):
            shape = list(getattr(args[0], "shape", ()))
            if len(shape) > 0:
                signal_length = shape[-1]
                frame_length = kwargs.get("frame_length", 256)
                frame_step = kwargs.get("frame_step", 128)
                num_mfccs = kwargs.get("num_mfccs", 13)
                num_frames = max(1, (signal_length - frame_length) // frame_step + 1)
                shape[-1] = num_frames
                shape.append(num_mfccs)
            return tuple(shape)
        return ()


@register_op("MfccsFromLogMelSpectrograms")
class MfccsFromLogMelSpectrograms(OpDef):
    """Operator to compute MFCCs directly from log mel spectrograms."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infers the shape of the resulting MFCC output.

        Args:
            *args: Positional arguments including the log mel spectrogram.
            **kwargs: Keyword arguments for MFCC conversion.

        Returns:
            The output shape as a tuple of integers.
        """
        if len(args) > 0 and hasattr(args[0], "shape"):
            shape = list(getattr(args[0], "shape", ()))
            if len(shape) > 0:
                num_mfccs = kwargs.get("num_mfccs", 13)
                shape[-1] = num_mfccs
            return tuple(shape)
        return ()


@register_op("HannWindow")
class HannWindow(OpDef):
    """Hann window function generation operator."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infers the shape of the generated Hann window.

        Args:
            *args: Positional arguments including the window size.
            **kwargs: Keyword arguments for the window function.

        Returns:
            The output shape as a tuple of integers.
        """
        window_length = kwargs.get("window_length", 256)
        if len(args) > 0:
            val = getattr(args[0], "value", args[0])
            if isinstance(val, int):
                window_length = val
        return (window_length,)


@register_op("HammingWindow")
class HammingWindow(OpDef):
    """Hamming window function generation operator."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infers the shape of the generated Hamming window.

        Args:
            *args: Positional arguments including the window size.
            **kwargs: Keyword arguments for the window function.

        Returns:
            The output shape as a tuple of integers.
        """
        window_length = kwargs.get("window_length", 256)
        if len(args) > 0:
            val = getattr(args[0], "value", args[0])
            if isinstance(val, int):
                window_length = val
        return (window_length,)


@register_op("KaiserWindow")
class KaiserWindow(OpDef):
    """Kaiser window function generation operator."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infers the shape of the generated Kaiser window.

        Args:
            *args: Positional arguments including the window size and beta parameter.
            **kwargs: Keyword arguments for the window function.

        Returns:
            The output shape as a tuple of integers.
        """
        window_length = kwargs.get("window_length", 256)
        if len(args) > 0:
            val = getattr(args[0], "value", args[0])
            if isinstance(val, int):
                window_length = val
        return (window_length,)


@register_op("Dct")
class Dct(OpDef):
    """Discrete cosine transform (DCT) operator."""

    op_name = "Dct"

    def infer_shape(self, a: object, **kwargs: object) -> object:
        """Infers the shape of the DCT output based on the input signal.

        Args:
            a: The input signal tensor for the DCT.
            **kwargs: Keyword arguments for DCT configuration (e.g., type, normalization).

        Returns:
            The output shape of the transformed signal.
        """
        return a.shape


@register_op("Idct")
class Idct(OpDef):
    """Inverse discrete cosine transform (IDCT) operator."""

    op_name = "Idct"

    def infer_shape(self, a: object, **kwargs: object) -> object:
        """Infers the shape of the IDCT output based on the input transformed signal.

        Args:
            a: The input transformed signal tensor.
            **kwargs: Keyword arguments for IDCT configuration.

        Returns:
            The output shape of the inverse transformed signal.
        """
        return a.shape


@register_op("Mdct")
class Mdct(OpDef):
    """Modify discrete cosine transform (MDCT) operator."""

    op_name = "Mdct"

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infers the shape of the MDCT output based on the input signal.

        Args:
            *args: Positional arguments containing the input signal.
            **kwargs: Keyword arguments for MDCT configuration.

        Returns:
            tuple[int, ...]: The output shape of the transformed signal.
        """
        if len(args) == 0 or not hasattr(args[0], "shape"):
            return ()
        a = args[0]
        shape = list(getattr(a, "shape", ()) or ())
        if not shape:
            return ()

        frame_length = int(kwargs.get("frame_length", 512))
        frame_step = int(kwargs.get("frame_step", 256))
        pad_end = bool(kwargs.get("pad_end", False))

        # If the input represents a pre-framed block (last dim is frame_length)
        if shape[-1] == frame_length:
            shape[-1] = frame_length // 2
            return tuple(shape)

        # Otherwise, the input is a continuous 1D signal (..., signal_length)
        signal_length = shape[-1]
        if signal_length < frame_length:
            shape[-1] = signal_length // 2
            return tuple(shape)

        if pad_end:
            num_frames = (signal_length + frame_step - 1) // frame_step if signal_length > 0 else 0
        else:
            num_frames = max(0, (signal_length - frame_length) // frame_step + 1)

        shape[-1] = num_frames
        shape.append(frame_length // 2)
        return tuple(shape)


@register_op("InverseMdct")
class InverseMdct(OpDef):
    """Inverse modified discrete cosine transform (IMDCT) operator."""

    op_name = "InverseMdct"

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infers the shape of the IMDCT output based on the input transformed signal.

        Args:
            *args: Positional arguments containing the input transformed signal.
            **kwargs: Keyword arguments for IMDCT configuration.

        Returns:
            tuple[int, ...]: The output shape of the inverse transformed signal.
        """
        if not args or not hasattr(args[0], "shape"):
            return ()
        shape = list(getattr(args[0], "shape", ()) or ())
        if not shape:
            return ()

        fl = int(kwargs.get("frame_length", 512))
        fs = int(kwargs.get("frame_step", 256))
        last = shape[-1]

        if last != fl // 2:
            shape[-1] = last * 2
            return tuple(shape)

        if "frame_step" in kwargs and len(shape) >= 2:
            num_frames = shape[-2]
            shape.pop()
            shape[-1] = (num_frames - 1) * fs + fl if num_frames > 0 else 0
            return tuple(shape)

        shape[-1] = fl
        return tuple(shape)


@register_op("Frame")
class Frame(OpDef):
    """Operator to split a signal into overlapping frames."""

    op_name = "Frame"

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infers the shape of the framed output tensor.

        Args:
            *args: Positional arguments containing the input signal.
            **kwargs: Keyword arguments including 'frame_length' and 'frame_step'.

        Returns:
            tuple[int, ...]: The output shape containing the newly added frame dimension.
        """
        if len(args) == 0 or not hasattr(args[0], "shape"):
            return ()
        a = args[0]
        shape = list(getattr(a, "shape", ()) or ())
        if not shape:
            return ()

        frame_length = int(kwargs.get("frame_length", 1))
        frame_step = int(kwargs.get("frame_step", 1))
        pad_end = bool(kwargs.get("pad_end", False))

        signal_length = shape[-1]
        if pad_end:
            num_frames = (signal_length + frame_step - 1) // frame_step if signal_length > 0 else 0
        else:
            num_frames = max(0, (signal_length - frame_length) // frame_step + 1)

        shape[-1] = num_frames
        shape.append(frame_length)
        return tuple(shape)


@register_op("OverlapAndAdd")
class OverlapAndAdd(OpDef):
    """Operator to reconstruct a signal from overlapping frames."""

    op_name = "OverlapAndAdd"

    def infer_shape(self, a: object, **kwargs: object) -> object:
        """Infers the shape of the reconstructed signal output.

        Args:
            a: The input tensor containing overlapping frames.
            **kwargs: Keyword arguments including 'frame_step'.

        Returns:
            The output shape of the reconstructed signal.
        """
        shape = list(a.shape)
        frame_step = kwargs.get("frame_step", 1)
        if len(shape) >= 2:
            num_frames = shape[-2]
            frame_length = shape[-1]
            shape.pop()
            shape[-1] = (num_frames - 1) * frame_step + frame_length
        return tuple(shape)
