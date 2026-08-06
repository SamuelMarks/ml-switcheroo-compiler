from unittest.mock import MagicMock, patch

import numpy as np

from ml_switcheroo_compiler.utils.vis_utils import save_img


def test_save_img_squeeze(tmp_path) -> None:
    """Test coverage."""
    with patch("sys.modules"):
        import sys

        mock_pil = MagicMock()
        sys.modules["PIL"] = mock_pil
        sys.modules["PIL.Image"] = mock_pil.Image

        # fix the png key error
        import PIL.Image

        PIL.Image.EXTENSION = {".png": "PNG"}

        save_img(str(tmp_path / "dummy.png"), np.zeros((10, 10, 1)))
