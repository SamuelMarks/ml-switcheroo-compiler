# ruff: noqa: E501
from ml_switcheroo_compiler.serialization.formats.npz import NpzWeightFormat


def test_npz_load(mocker):
    fmt = NpzWeightFormat()
    mock_backend = mocker.patch("ml_switcheroo_compiler.backends.registry.get_active_backend").return_value
    mock_backend.load_npz.return_value = {"a": 1}
    assert fmt.load("test.npz") == {"a": 1}
    mock_backend.load_npz.side_effect = NotImplementedError
    mocker.patch("ml_switcheroo_compiler.serialization.formats.npz.parse_npz", return_value={"b": 2})
    assert fmt.load("test.npz") == {"b": 2}
    del mock_backend.load_npz
    assert fmt.load("test.npz") == {"b": 2}
