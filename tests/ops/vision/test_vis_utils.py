# ruff: noqa: E501
import tempfile

import numpy as np
import pytest

from ml_switcheroo_compiler.utils.vis_utils import PlotModelConfig, img_to_array, model_to_dot, plot_model, save_img


def test_plot_model(mocker):
    mocker.patch.dict("sys.modules", {"pydot": None})
    with pytest.raises(ImportError):
        plot_model(None)
    mock_pydot = mocker.MagicMock()
    mock_dot = mocker.MagicMock()
    mock_pydot.Dot.return_value = mock_dot
    mock_pydot.Node = mocker.MagicMock()
    mocker.patch.dict("sys.modules", {"pydot": mock_pydot})
    assert plot_model(None) == mock_dot
    plot_model(None, PlotModelConfig(to_file=""))
    plot_model(None, PlotModelConfig(to_file="test.png"))
    mock_dot.write_png.assert_called_with("test.png")
    plot_model(None, PlotModelConfig(to_file="test.svg"))
    mock_dot.write_svg.assert_called_with("test.svg")
    plot_model(None, PlotModelConfig(to_file="test.pdf"))
    mock_dot.write.assert_called_with("test.pdf", format="pdf")


def test_dummy_functions():
    pass
    assert img_to_array() is not None
    pass
    assert model_to_dot() is None


def test_save_img(mocker):
    mocker.patch.dict("sys.modules", {"PIL": None, "PIL.Image": None})
    with pytest.raises(ImportError):
        save_img("test", None)
    mock_pil = mocker.MagicMock()
    mock_img = mocker.MagicMock()
    mock_pil.Image.fromarray.return_value = mock_img
    mocker.patch.dict("sys.modules", {"PIL": mock_pil, "PIL.Image": mock_pil.Image})
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        arr = np.zeros((10, 10, 1))
        save_img(f.name, arr)
        arr2 = np.zeros((10, 10, 3))
        save_img(f.name, arr2)
        mock_img.save.assert_called_with(f.name)
        mock_img2 = mocker.MagicMock()
        del mock_img2.numpy
        save_img(f.name, mock_img2)
        mock_img2.save.assert_called_with(f.name)
