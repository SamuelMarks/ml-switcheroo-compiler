# ruff: noqa: E501
from ml_switcheroo_compiler.utils.np_utils import normalize, to_categorical


def test_np_utils(mocker):
    mocker.patch("ml_switcheroo_compiler.ops.max", return_value=1)
    mocker.patch("ml_switcheroo_compiler.ops.cast", return_value="cast")
    mocker.patch("ml_switcheroo_compiler.ops.arange", return_value="arange")
    mocker.patch("ml_switcheroo_compiler.ops.expand_dims", return_value="expand")
    mocker.patch("ml_switcheroo_compiler.ops.equal", return_value="equal")
    assert to_categorical("x") == "cast"
    assert to_categorical("x", num_classes=5) == "cast"
    mocker.patch("ml_switcheroo_compiler.ops.sum", return_value="sum")
    mocker.patch("ml_switcheroo_compiler.ops.square", return_value="square")
    mocker.patch("ml_switcheroo_compiler.ops.sqrt", return_value="sqrt")
    mocker.patch("ml_switcheroo_compiler.ops.maximum", return_value="max")
    mocker.patch("ml_switcheroo_compiler.ops.divide", return_value="div")
    assert normalize("x") == "div"


def test_to_categorical_item(mocker):

    class M:
        def item(self):
            return 2

    mocker.patch("ml_switcheroo_compiler.ops.max", return_value=M())
    mocker.patch("ml_switcheroo_compiler.ops.cast", return_value="cast")
    mocker.patch("ml_switcheroo_compiler.ops.arange")
    mocker.patch("ml_switcheroo_compiler.ops.expand_dims")
    mocker.patch("ml_switcheroo_compiler.ops.equal")
    assert to_categorical("x") == "cast"
