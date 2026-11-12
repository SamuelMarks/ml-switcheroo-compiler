"""Test module."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.backends.eager.utils import _from_channels_last, _from_numpy_array, _jax_from_numpy, _mlx_from_numpy, _to_channels_last, _to_numpy_array, _torch_from_numpy
from tests.testing_utils import ensure_callable, ensure_eager_tensor

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


class DummyNp:
    """Test class DummyNp."""

    def asarray(self, x: EagerValue) -> EagerValue:
        """Test function asarray.

        Args:
            x: Test parameter `x`.

        Returns:
            EagerValue: Computed output value from `asarray`.
        """
        return "np_asarray_res"

    def array(self, out: EagerValue, dtype: EagerValue = None) -> EagerValue:
        """Test function array.

        Args:
            out: Test parameter `out`.
            dtype: Test parameter `dtype`.

        Returns:
            EagerValue: Computed output value from `array`.
        """
        if dtype:
            return f"np_array_res_{dtype}"
        return "np_array_res"


class DummyTensorWithNumpy:
    """Test class DummyTensorWithNumpy."""

    def numpy(self) -> EagerValue:
        """Test function numpy.

        Returns:
            EagerValue: Computed output value from `numpy`.
        """
        return "numpy_res"


class DummyTensorTorch:
    """Test class DummyTensorTorch."""

    def detach(self) -> EagerValue:
        """Test function detach.

        Returns:
            EagerValue: Computed output value from `detach`.
        """

        class Cpu:
            """Test class Cpu."""

            def cpu(self) -> EagerValue:
                """Test function cpu.

                Returns:
                    EagerValue: Computed output value from `cpu`.
                """

                class Numpy:
                    """Test class Numpy."""

                    def numpy(self) -> EagerValue:
                        """Test function numpy.

                        Returns:
                            EagerValue: Computed output value from `numpy`.
                        """
                        return "torch_numpy_res"

                return ensure_eager_tensor(Numpy())

        return ensure_eager_tensor(Cpu())


class DummyOriginal:
    """Test class DummyOriginal."""

    dtype = "float32"


def test_to_numpy_array() -> None:
    """Test function test_to_numpy_array."""
    np_mod = DummyNp()

    t1 = DummyTensorWithNumpy()
    assert ensure_callable(_to_numpy_array)(np_mod, t1, "test") == "numpy_res"

    t2 = DummyTensorTorch()
    assert ensure_callable(_to_numpy_array)(np_mod, t2, "torch") == "torch_numpy_res"

    assert ensure_callable(_to_numpy_array)(np_mod, "other", "test") == "np_asarray_res"


def test_from_numpy_array() -> None:
    """Test function test_from_numpy_array."""
    backend = DummyNp()

    assert ensure_callable(_from_numpy_array)(backend, "out", "torch") == "out"
    assert ensure_callable(_from_numpy_array)(backend, "out", "mlx.core") == "out"
    assert ensure_callable(_from_numpy_array)(backend, "out", "jax.numpy") == "out"

    assert ensure_callable(_from_numpy_array)(backend, "out", "other") == "np_array_res"

    orig = DummyOriginal()
    assert ensure_callable(_from_numpy_array)(backend, "out", "other", orig) == "np_array_res_float32"


def test_other_utils() -> None:
    """Test function test_other_utils."""
    assert _torch_from_numpy("out") == "out"
    assert _mlx_from_numpy("out") == "out"
    assert _jax_from_numpy("out") == "out"

    assert _to_channels_last(None, "imgs", None) == "imgs"
    assert _from_channels_last(None, "out", None) == "out"
