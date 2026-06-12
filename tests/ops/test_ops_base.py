"""Tests for the base operation registry."""

import pytest
from ml_switcheroo.ops.base import OpDef, register_op, get_op, _OP_REGISTRY


def test_register_and_get_op() -> None:
    """Test registering and retrieving an operation."""
    # Clear registry for clean test
    original_registry = _OP_REGISTRY.copy()
    _OP_REGISTRY.clear()

    try:

        @register_op("TestOp")
        class TestOp(OpDef):
            """Docstring."""

            def infer_shape(self, *args: object, **kwargs: object) -> object:
                """Docstring."""
                pass

            def numpy_eval(self, *args: object, **kwargs: object) -> object:
                """Docstring."""
                pass

            def vjp(
                self, cotangent: object, *args: object, **kwargs: object
            ) -> tuple[object, ...]:
                """Docstring."""
                return ()

            def jvp(self, tangent: object, *args: object, **kwargs: object) -> object:
                """Docstring."""
                pass

            def emit_jax(self, *args: object, **kwargs: object) -> str:
                """Docstring."""
                return "jax"

            def emit_pytorch(self, *args: object, **kwargs: object) -> str:
                """Docstring."""
                return "torch"

            def emit_mlx(self, *args: object, **kwargs: object) -> str:
                """Docstring."""
                return "mlx"

            def emit_keras(self, *args: object, **kwargs: object) -> str:
                """Docstring."""
                return "keras"

            def emit_tensorflow(self, *args: object, **kwargs: object) -> str:
                """Docstring."""
                return "tf"

        retrieved_op = get_op("TestOp")
        assert retrieved_op is TestOp

        with pytest.raises(ValueError, match="already registered"):

            @register_op("TestOp")
            class DuplicateOp(OpDef):
                """Docstring."""

                pass

        with pytest.raises(KeyError, match="not found in registry"):
            get_op("UnknownOp")

    finally:
        # Restore registry
        _OP_REGISTRY.clear()
        _OP_REGISTRY.update(original_registry)
