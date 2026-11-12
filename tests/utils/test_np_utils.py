"""Unit tests for categorical encoding and L-p tensor normalization utilities."""

from __future__ import annotations

import typing

from tests.testing_utils import ensure_eager_tensor

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


def test_np_utils() -> None:
    """test_np_utils."""
    from unittest.mock import patch

    from ml_switcheroo_compiler.utils.np_utils import normalize, to_categorical

    with (
        patch("ml_switcheroo_compiler.ops.max") as mock_max,
        patch("ml_switcheroo_compiler.ops.cast") as mock_cast,
        patch("ml_switcheroo_compiler.ops.arange") as mock_arange,
        patch("ml_switcheroo_compiler.ops.expand_dims") as mock_expand,
        patch("ml_switcheroo_compiler.ops.equal") as mock_equal,
        patch("ml_switcheroo_compiler.ops.sum") as mock_sum,
        patch("ml_switcheroo_compiler.ops.square") as mock_square,
        patch("ml_switcheroo_compiler.ops.maximum") as mock_max_fn,
        patch("ml_switcheroo_compiler.ops.sqrt") as mock_sqrt,
        patch("ml_switcheroo_compiler.ops.divide") as mock_div,
    ):
        mock_max.return_value = 5.0
        mock_cast.return_value = "casted"
        mock_arange.return_value = "aranged"
        mock_expand.return_value = "expanded"
        mock_equal.return_value = "equaled"

        assert to_categorical(1) == "casted"
        assert to_categorical(1, num_classes=10) == "casted"

        mock_sum.return_value = "summed"
        mock_square.return_value = "squared"
        mock_max_fn.return_value = "maxed"
        mock_sqrt.return_value = "sqrted"
        mock_div.return_value = "divided"

        assert normalize(1) == "divided"

        class Dummy:
            """Dummy."""

            def item(self) -> EagerValue:
                """Item.

                Returns:
                    EagerValue: Computed output value from `item`.
                """
                return 5.0

        mock_max.return_value = Dummy()
        assert to_categorical(1) == "casted"

    import numpy as np

    from ml_switcheroo_compiler.core.config import ConfigContext
    from ml_switcheroo_compiler.core.device import Device
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    with ConfigContext(eager_mode=True):
        vec = Tensor(np.array([3.0, 4.0], dtype=np.float32), TensorConfig((2,), DType.Float32, Device("cpu")))
        out_l2 = normalize(vec, axis=-1, order=2)
        assert np.allclose(np.asarray(ensure_eager_tensor(out_l2).eval()), np.array([0.6, 0.8], dtype=np.float32), atol=1e-5)

        out_l1 = normalize(vec, axis=-1, order=1)
        assert np.allclose(np.asarray(ensure_eager_tensor(out_l1).eval()), np.array([3.0 / 7.0, 4.0 / 7.0], dtype=np.float32), atol=1e-5)

        out_l0 = normalize(vec, axis=-1, order=0)
        assert np.allclose(np.asarray(ensure_eager_tensor(out_l0).eval()), np.array([1.5, 2.0], dtype=np.float32), atol=1e-5)

        out_inf = normalize(vec, axis=-1, order=float("inf"))
        assert np.allclose(np.asarray(ensure_eager_tensor(out_inf).eval()), np.array([0.75, 1.0], dtype=np.float32), atol=1e-5)

        out_l3 = normalize(vec, axis=-1, order=3)
        expected_l3 = np.array([3.0, 4.0]) / ((27.0 + 64.0) ** (1.0 / 3.0))
        assert np.allclose(np.asarray(ensure_eager_tensor(out_l3).eval()), expected_l3, atol=1e-5)
