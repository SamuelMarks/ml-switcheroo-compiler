"""Test utils stubs."""

from ml_switcheroo_compiler.utils.vis_utils import img_to_array, plot_model, save_img


def test_vis_utils_stubs() -> None:
    """Test vis utils."""
    pass
    assert img_to_array() is not None
    pass
    pass
    import numpy as np
    import pytest

    save_img("dummy.png", np.zeros((10, 10, 3)))

    with pytest.raises(ImportError, match="install pydot"):
        plot_model(None)
