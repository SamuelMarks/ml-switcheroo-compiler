"""Tests for Keras prefix image filtering and transform operations."""

import numpy as np
import pytest

pytest.importorskip("keras")
import keras.ops as kops

from ml_switcheroo_compiler.backends.keras.generator import KerasCodeGenerator
from ml_switcheroo_compiler.ir.core import IRGraph


def _load_prefix_functions() -> dict[str, callable]:
    """Load prefix functions from keras_prefix.yaml for direct execution.

    Returns:
        dict[str, callable]: Dictionary mapping function names to callable functions.
    """
    gen = KerasCodeGenerator(IRGraph())
    imports = gen._resolve_imports()
    code_str = "\n".join(imports)
    namespace: dict[str, object] = {}
    exec(code_str, namespace)
    return {
        "keras_median_filter": namespace["keras_median_filter"],
        "keras_gaussian_blur": namespace["keras_gaussian_blur"],
        "keras_elastic_transform": namespace["keras_elastic_transform"],
    }


def test_keras_median_filter_channels_last() -> None:
    """Verify keras_median_filter on channels_last image tensors."""
    funcs = _load_prefix_functions()
    med_fn = funcs["keras_median_filter"]
    img_np = np.ones((1, 8, 8, 2), dtype=np.float32)
    # inject impulse noise
    img_np[0, 4, 4, :] = 100.0
    img_tensor = kops.convert_to_tensor(img_np)
    filtered = med_fn(img_tensor, kernel_size=3, padding="same", data_format="channels_last")
    out_np = kops.convert_to_numpy(filtered)
    assert out_np.shape == (1, 8, 8, 2)
    # Median of 3x3 neighborhood around impulse should be 1.0
    assert float(out_np[0, 4, 4, 0]) == 1.0


def test_keras_median_filter_channels_first() -> None:
    """Verify keras_median_filter on channels_first image tensors."""
    funcs = _load_prefix_functions()
    med_fn = funcs["keras_median_filter"]
    img_np = np.ones((1, 2, 8, 8), dtype=np.float32)
    img_np[0, :, 4, 4] = 50.0
    img_tensor = kops.convert_to_tensor(img_np)
    filtered = med_fn(img_tensor, kernel_size=(3, 3), padding="same", data_format="channels_first")
    out_np = kops.convert_to_numpy(filtered)
    assert out_np.shape == (1, 2, 8, 8)
    assert float(out_np[0, 0, 4, 4]) == 1.0


def test_keras_gaussian_blur_filtering() -> None:
    """Verify keras_gaussian_blur produces smoothed output."""
    funcs = _load_prefix_functions()
    blur_fn = funcs["keras_gaussian_blur"]
    img_np = np.zeros((1, 9, 9, 1), dtype=np.float32)
    img_np[0, 4, 4, 0] = 1.0
    img_tensor = kops.convert_to_tensor(img_np)
    blurred = blur_fn(img_tensor, kernel_size=3, sigma=1.0, padding="same", data_format="channels_last")
    out_np = kops.convert_to_numpy(blurred)
    assert out_np.shape == (1, 9, 9, 1)
    # Center should be positive and neighbors should also be positive due to smoothing
    assert float(out_np[0, 4, 4, 0]) > 0.0
    assert float(out_np[0, 4, 5, 0]) > 0.0


def test_keras_elastic_transform_identity() -> None:
    """Verify keras_elastic_transform produces identity when displacement is zero."""
    funcs = _load_prefix_functions()
    elastic_fn = funcs["keras_elastic_transform"]
    img_np = np.arange(16, dtype=np.float32).reshape((1, 4, 4, 1))
    disp_np = np.zeros((4, 4, 2), dtype=np.float32)
    img_tensor = kops.convert_to_tensor(img_np)
    disp_tensor = kops.convert_to_tensor(disp_np)
    res = elastic_fn(img_tensor, disp_tensor, interpolation="bilinear", fill_value=0.0, data_format="channels_last")
    out_np = kops.convert_to_numpy(res)
    assert out_np.shape == (1, 4, 4, 1)
    np.testing.assert_allclose(out_np, img_np, atol=1e-5)


def test_keras_elastic_transform_none_displacement() -> None:
    """Verify keras_elastic_transform passes through if displacement is None."""
    funcs = _load_prefix_functions()
    elastic_fn = funcs["keras_elastic_transform"]
    img_np = np.ones((1, 4, 4, 1), dtype=np.float32)
    img_tensor = kops.convert_to_tensor(img_np)
    res = elastic_fn(img_tensor, None, data_format="channels_last")
    assert res is img_tensor
