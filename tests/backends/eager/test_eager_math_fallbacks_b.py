"""Direct behavioral and fallback-branch tests for `core_math_ops` matrix, NN, reduction, IO, FFT, and vision."""

from __future__ import annotations

import pathlib
import types

import numpy as np

from ml_switcheroo_compiler.backends.eager.audio import MelFilterbankConfig, MFCCConfig
from ml_switcheroo_compiler.backends.eager.core_math_ops import (
    math_matrix as mmat,
)
from ml_switcheroo_compiler.backends.eager.core_math_ops import (
    math_matrix_utils as mmu,
)
from ml_switcheroo_compiler.backends.eager.core_math_ops import (
    math_nn as mnn,
)
from ml_switcheroo_compiler.backends.eager.core_math_ops import (
    math_reduction as mred,
)
from ml_switcheroo_compiler.backends.eager.core_math_ops import (
    math_string_io as msio,
)
from ml_switcheroo_compiler.backends.eager.nn_ops.activations import _np_relu
from ml_switcheroo_compiler.backends.eager.reference_math.fft_ops import (
    FFTConfig,
    _np_fftnd,
    _np_ifftnd,
    _np_irfftnd,
    _np_rfftnd,
)
from ml_switcheroo_compiler.backends.eager.reference_math.math_auxiliary import _np_reduce_window
from ml_switcheroo_compiler.backends.eager.reference_math.math_string_io import _coerce_io_filename
from ml_switcheroo_compiler.backends.eager.vision_ops.common import _np_mel_filterbank, _np_mfcc
from ml_switcheroo_compiler.backends.eager.vision_ops.filtering import _np_gaussian_blur, _np_median_filter
from ml_switcheroo_compiler.backends.eager_registry import BackendArg
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from ml_switcheroo_compiler.core.shape import VariadicCallableProtocol
from tests.testing_utils import (
    coerce_operand_list,
    extract_test_shape,
    make_mock_module,
)


def _mod(name: str, attrs: dict[str, BackendArg]) -> types.ModuleType:
    """Create a mock module with `attrs` bound via `make_mock_module`.

    Args:
        name (str): Module name.
        attrs (dict[str, BackendArg]): Symbol table to attach.

    Returns:
        types.ModuleType: Constructed module stub.
    """
    del name
    return make_mock_module(**attrs)


def _tag_fn(tag: str) -> VariadicCallableProtocol:
    """Build a deterministic callable returning `tag` for custom backend-symbol dispatch tests.

    Args:
        tag (str): Expected marker string.

    Returns:
        VariadicCallableProtocol: Variadic marker function.
    """

    def _fn(*args: BackendArg, **kwargs: BackendArg) -> str:
        """Return `tag` marker string.

        Args:
            *args (BackendArg): Positional arguments.
            **kwargs (BackendArg): Keyword arguments.

        Returns:
            str: Marker string `tag`.
        """
        del args, kwargs
        return tag

    return _fn


class _ReduceWindowConfig:
    """Window configuration object satisfying `ShapeProtocol` (`BackendArg`) for `_np_reduce_window`."""

    def __init__(self) -> None:
        """Initialize 1D window dimensions, strides, dilations, and padding."""
        self.shape: tuple[int, ...] = ()
        self.window_dimensions: list[int] = [2]
        self.window_strides: list[int] = [1]
        self.window_dilation: list[int] = [1]
        self.base_dilation: list[int] = [1]
        self.padding: str = "VALID"


def test_math_matrix_and_matrix_utils_branches() -> None:
    """Verify top-level symbol hits and NumPy fallbacks in `math_matrix.py` and `math_matrix_utils.py`."""
    eye2 = np.eye(2)
    vec2 = np.array([1.0, 2.0])
    top_mat_mod = _mod(
        "top_mat_mod",
        {
            "matrix_power": _tag_fn("mpow_top"),
            "matrix_rank": _tag_fn("mrank_top"),
            "solve": _tag_fn("solve_top"),
            "tensorsolve": _tag_fn("tsolve_top"),
            "vecdot": _tag_fn("vecdot_top"),
        },
    )
    assert mmat._matrix_power(top_mat_mod, eye2, 2) == "mpow_top"
    assert mmat._matrix_rank(top_mat_mod, eye2) == "mrank_top"
    assert mmat._solve(top_mat_mod, eye2, vec2) == "solve_top"
    assert mmat._tensorsolve(top_mat_mod, eye2, vec2) == "tsolve_top"
    assert mmat._np_tensorsolve(top_mat_mod, eye2, vec2) == "tsolve_top"
    assert mmat._np_vecdot(top_mat_mod, vec2, vec2) == "vecdot_top"

    empty_mod = types.ModuleType("empty_mod")
    np.testing.assert_allclose(ensure_ndarray(mmat._np_tensorsolve(empty_mod, eye2, vec2)), vec2)
    assert float(ensure_ndarray(mmat._np_vecdot(empty_mod, vec2, vec2))) == 5.0

    geom_sample = ensure_ndarray(mmu._geometric(np, 0.5, size=(3,)))
    assert geom_sample.shape == (3,)
    assert ensure_ndarray(mmu._indices(np, (2, 2))).shape == (2, 2, 2)
    mask_idx = mmu._maskindices(np, 3, np.triu)
    assert isinstance(mask_idx, tuple) and len(mask_idx) == 2
    assert ensure_ndarray(mmu._tri(np, 2)).shape == (2, 2)
    assert ensure_ndarray(mmu._tril(np, np.ones((2, 2)))).shape == (2, 2)
    assert ensure_ndarray(mmu._trimzeros(np, np.array([0, 1, 2, 0]))).tolist() == [1, 2]
    assert ensure_ndarray(mmu._triu(np, np.ones((2, 2)))).shape == (2, 2)

    tri_custom = _mod(
        "tri_custom",
        {
            "bandpart": _tag_fn("bandpart_hit"),
            "tridiagonal": _tag_fn("tridiag_hit"),
            "trilindices": _tag_fn("trilidx_hit"),
            "trilindicesfrom": _tag_fn("trilidxfrom_hit"),
            "trimzeros": _tag_fn("trimzeros_hit"),
            "triuindices": _tag_fn("triuidx_hit"),
            "triuindicesfrom": _tag_fn("triuidxfrom_hit"),
        },
    )
    assert mmu._np_bandpart(tri_custom, eye2) == "bandpart_hit"
    assert mmu._np_tridiagonal(tri_custom, vec2) == "tridiag_hit"
    assert mmu._np_trilindices(tri_custom, 2) == "trilidx_hit"
    assert mmu._np_trilindicesfrom(tri_custom, eye2) == "trilidxfrom_hit"
    assert mmu._np_trimzeros(tri_custom, vec2) == "trimzeros_hit"
    assert mmu._np_triuindices(tri_custom, 2) == "triuidx_hit"
    assert mmu._np_triuindicesfrom(tri_custom, eye2) == "triuidxfrom_hit"

    assert len(coerce_operand_list(mmu._np_trilindices(empty_mod, 2))) == 2
    assert len(coerce_operand_list(mmu._np_trilindicesfrom(empty_mod, eye2))) == 2
    assert ensure_ndarray(mmu._np_trimzeros(empty_mod, np.array([0, 3, 0]))).tolist() == [3]
    assert len(coerce_operand_list(mmu._np_triuindices(empty_mod, 2))) == 2
    assert len(coerce_operand_list(mmu._np_triuindicesfrom(empty_mod, eye2))) == 2


def test_math_nn_reduction_and_string_io_branches(tmp_path: pathlib.Path) -> None:
    """Verify NN, reduction, and string/file IO helpers across `core_math_ops/`.

    Args:
        tmp_path (pathlib.Path): Temporary directory fixture.
    """
    arr4d = np.ones((1, 2, 4, 4))
    arr5d = np.ones((1, 2, 4, 4, 4))
    assert mnn._activity_regularization(np, 7.0) == 7.0
    assert extract_test_shape(mnn._adaptive_avg_pool2d(np, arr4d, (1, 1))) == (1, 2, 1, 1)
    assert extract_test_shape(mnn._adaptive_avg_pool3d(np, arr5d, (1, 1, 1))) == (1, 2, 1, 1, 1)
    assert mnn._alpha_dropout(np, 9.0) == 9.0

    empty_mod = types.ModuleType("empty_mod")
    fap_mod = _mod("fap_mod", {"fractionalavgpool": _tag_fn("fap_hit")})
    assert mnn._np_fractionalavgpool(fap_mod, arr4d) == "fap_hit"
    assert mnn._np_fractionalavgpool(empty_mod, 11.0) == 11.0

    assert extract_test_shape(mred._adaptive_max_pool2d(np, arr4d, (1, 1))) == (1, 2, 1, 1)
    assert extract_test_shape(mred._adaptive_max_pool3d(np, arr5d, (1, 1, 1))) == (1, 2, 1, 1, 1)
    clse_mod = _mod("clse_mod", {"cumlogsumexp": _tag_fn("clse_hit")})
    assert mred._cumlogsumexp(clse_mod, np.array([1.0, 2.0])) == "clse_hit"

    wh_mod = _mod("wh_mod", {"windowhamming": _tag_fn("wh_hit")})
    assert mred._np_windowhamming(wh_mod, 4) == "wh_hit"
    assert ensure_ndarray(mred._np_windowhamming(empty_mod, 4)).shape == (4,)

    assert msio._as_string(np, 123) == "123"
    bin_path = tmp_path / "nums.bin"
    np.array([1.0, 2.0], dtype=np.float64).tofile(bin_path)
    np.testing.assert_allclose(ensure_ndarray(msio._fromfile(np, str(bin_path), dtype=np.float64)), [1.0, 2.0])

    sio_custom = _mod(
        "sio_custom",
        {
            "decodeimage": _tag_fn("dec_hit"),
            "fromstring": _tag_fn("fstr_hit"),
            "parsetensor": _tag_fn("pt_hit"),
            "stringtohash": _tag_fn("sth_hit"),
            "textvectorization": _tag_fn("tv_hit"),
        },
    )
    assert msio._np_decodeimage(sio_custom, b"img") == "dec_hit"
    assert ensure_ndarray(msio._np_decodeimage(empty_mod, b"img", shape=(4, 4, 3))).shape == (4, 4, 3)
    assert msio._np_fromstring(sio_custom, "1 2") == "fstr_hit"
    np.testing.assert_allclose(ensure_ndarray(msio._np_fromstring(empty_mod, "1 2", sep=" ")), [1.0, 2.0])
    assert msio._np_parsetensor(sio_custom, [1, 2]) == "pt_hit"
    assert ensure_ndarray(msio._np_parsetensor(empty_mod, [1, 2])).tolist() == [1, 2]
    assert msio._np_stringtohash(sio_custom, ["a"]) == "sth_hit"
    assert ensure_ndarray(msio._np_stringtohash(empty_mod, np.array(["abc", "def"]))).shape == (2,)
    assert msio._np_textvectorization(sio_custom, "hello") == "tv_hit"
    assert msio._np_textvectorization(empty_mod, "hello") == "hello"


def test_activations_fft_auxiliary_io_and_vision_branches() -> None:
    """Verify uncovered branches in `activations`, `fft_ops`, `math_auxiliary`, `math_string_io`, and `vision_ops`."""
    np.testing.assert_allclose(ensure_ndarray(_np_relu(np, np.array([-2.0, 3.0]))), np.array([0.0, 3.0]))

    sig2d = np.ones((2, 4), dtype=np.float32)
    fft_cfg = FFTConfig(axes=(0, 1))
    assert ensure_ndarray(_np_fftnd(np, sig2d, config=None)).shape == (2, 4)
    assert ensure_ndarray(_np_ifftnd(np, sig2d, config=fft_cfg)).shape == (2, 4)
    rfft_out = ensure_ndarray(_np_rfftnd(np, sig2d, config=None))
    assert rfft_out.shape == (2, 3)
    assert ensure_ndarray(_np_irfftnd(np, rfft_out, config=None)).shape == (2, 4)

    rw_out = ensure_ndarray(_np_reduce_window(np, np.array([1.0, 5.0, 2.0]), 0.0, "max", _ReduceWindowConfig()))
    np.testing.assert_allclose(rw_out, np.array([5.0, 5.0]))

    assert _coerce_io_filename(b"weights.bin") == "weights.bin"

    mel_cfg = MelFilterbankConfig(num_mel_bins=8, num_spectrogram_bins=17, sample_rate=8000)
    assert ensure_ndarray(_np_mel_filterbank(np, None, config=mel_cfg)).shape == (17, 8)
    assert ensure_ndarray(_np_mel_filterbank(np, None, config=None)).shape == (129, 40)

    spec = np.ones((2, 17), dtype=np.float32)
    mfcc_cfg = MFCCConfig(num_mfccs=4, num_mel_bins=8, num_spectrogram_bins=17, sample_rate=8000)
    assert ensure_ndarray(_np_mfcc(np, spec, config=mfcc_cfg)).shape == (2, 4)
    spec_default = np.ones((2, 129), dtype=np.float32)
    assert ensure_ndarray(_np_mfcc(np, spec_default, config=None)).shape == (2, 13)

    img4d = np.ones((1, 1, 5, 5), dtype=np.float32)
    assert ensure_ndarray(_np_gaussian_blur(np, img4d, sigma=1.5)).shape == (1, 1, 5, 5)
    assert ensure_ndarray(_np_gaussian_blur(np, img4d, sigma=[1.0])).shape == (1, 1, 5, 5)
    assert ensure_ndarray(_np_gaussian_blur(np, img4d, config=None)).shape == (1, 1, 5, 5)

    assert ensure_ndarray(_np_median_filter(np, img4d, kernel_size=3)).shape == (1, 1, 5, 5)
    assert ensure_ndarray(_np_median_filter(np, img4d, kernel_size=3.0)).shape == (1, 1, 5, 5)
