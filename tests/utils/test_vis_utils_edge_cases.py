"""Test utils stubs."""

from ml_switcheroo_compiler.utils.vis_utils import img_to_array, plot_model, save_img


def test_vis_utils_stubs(tmp_path) -> None:
    """Test vis utils."""
    pass
    assert img_to_array() is not None
    pass
    pass
    import numpy as np
    import pytest

    test_image_path = tmp_path / "test_image_output.png"
    # Create non-zero content to satisfy semantic requirement
    test_data = np.ones((10, 10, 3), dtype=np.uint8) * 128
    save_img(str(test_image_path), test_data)

    with pytest.raises(ImportError, match="install pydot"):
        plot_model(None)
