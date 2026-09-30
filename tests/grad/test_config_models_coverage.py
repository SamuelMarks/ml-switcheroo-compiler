"""Tests for test_config_models_coverage."""

from __future__ import annotations

import ml_switcheroo_compiler.grad.config_models as grad_config_models


class _MockBackendModule:
    """Mock backend module implementing zeros, array, and asarray."""

    def zeros(self, shape: tuple[int, ...]) -> tuple[str, tuple[int, ...]]:
        """Mock zeros.

        Args:
            shape (tuple[int, ...]): Output shape.

        Returns:
            tuple[str, tuple[int, ...]]: Mock zeros descriptor.
        """
        return ("zeros", shape)

    def array(self, data: object, dtype: object = None) -> tuple[str, object, object]:
        """Mock array.

        Args:
            data (object): Input data.
            dtype (object): Optional data type.

        Returns:
            tuple[str, object, object]: Mock array descriptor.
        """
        return ("array", data, dtype)

    def asarray(self, data: object) -> tuple[str, object]:
        """Mock asarray.

        Args:
            data (object): Input data.

        Returns:
            tuple[str, object]: Mock asarray descriptor.
        """
        return ("asarray", data)


class _MockItemObject:
    """Mock object implementing item()."""

    def item(self) -> float:
        """Return scalar item value.

        Returns:
            float: Scalar float value.
        """
        return 42.5


class _MockDaskDtypeWrapper:
    """Mock dtype object with value attribute."""

    def __init__(self, val: str) -> None:
        """Initialize mock dtype wrapper.

        Args:
            val (str): Type name.
        """
        self.value = val


def _sample_add_fn(x: int, y: int = 10) -> int:
    """Add two integers.

    Args:
        x (int): First int.
        y (int): Second int.

    Returns:
        int: Sum of x and y.
    """
    return x + y


def _sample_shape_fn(a: int, b: int) -> tuple[int, int]:
    """Sample shape function returning tuple.

    Args:
        a (int): First dimension.
        b (int): Second dimension.

    Returns:
        tuple[int, int]: Pair (a, b).
    """
    return (a, b)


def test_grad_config_models() -> None:
    """Verify all Pydantic models in grad/config_models.py."""
    f32_dtype = grad_config_models.FiniteDifferenceDtypeConfig(epsilon=1e-4)
    f64_dtype = grad_config_models.FiniteDifferenceDtypeConfig(epsilon=1e-7)
    assert f32_dtype.epsilon == 1e-4

    fd_cfg = grad_config_models.FiniteDifferenceConfig(float32=f32_dtype, float64=f64_dtype)
    assert fd_cfg.float32.epsilon == 1e-4
    assert fd_cfg.float64.epsilon == 1e-7

    rule = grad_config_models.AutodiffRuleModel(
        opcode="Mul",
        jvp="x * y_dot + x_dot * y",
        vjp="cotangent * y, cotangent * x",
        description="Product rule",
    )
    assert rule.opcode == "Mul"
    assert rule.jvp == "x * y_dot + x_dot * y"
    assert rule.vjp == "cotangent * y, cotangent * x"
    assert rule.description == "Product rule"
    assert rule.cotangent_inputs == ["$cotangent"]
    assert rule.primal_outputs == ["$output"]

    ho_model = grad_config_models.HigherOrderAutodiffModel(
        opcode="Sin",
        hvp="cotangent * (-sin(x))",
        jvp_order=2,
        vjp_order=2,
        rewrite_rules=[{"rule": "sin_to_cos"}],
    )
    assert ho_model.opcode == "Sin"
    assert ho_model.hvp == "cotangent * (-sin(x))"
    assert ho_model.jvp_order == 2
    assert ho_model.vjp_order == 2
    assert ho_model.rewrite_rules == [{"rule": "sin_to_cos"}]

    manifest = grad_config_models.AutodiffRulesManifestModel(
        jvp_rules={"Mul": rule},
        vjp_rules={"Mul": rule},
        higher_order_rules={"Sin": ho_model},
    )
    assert "Mul" in manifest.jvp_rules
    assert "Mul" in manifest.vjp_rules
    assert "Sin" in manifest.higher_order_rules
