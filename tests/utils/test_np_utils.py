"""Module test_np_utils.py."""


def test_np_utils():
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

            def item(self):
                """item."""
                return 5.0

        mock_max.return_value = Dummy()
        assert to_categorical(1) == "casted"
