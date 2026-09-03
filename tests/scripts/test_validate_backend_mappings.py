import json
import os
import tempfile
from unittest.mock import mock_open, patch

import yaml

from scripts.validate_backend_mappings import get_snapshot_api_dict, main, validate_mappings


def test_get_snapshot_api_dict_invalid_prefix():
    assert get_snapshot_api_dict("invalid") == {}


def test_get_snapshot_api_dict_exception_in_listdir():
    with patch("os.listdir", side_effect=Exception("error")):
        assert get_snapshot_api_dict("np") == {}


def test_get_snapshot_api_dict_no_files():
    with patch("os.listdir", return_value=[]):
        assert get_snapshot_api_dict("np") == {}


def test_get_snapshot_api_dict_exception_in_read():
    with patch("os.listdir", return_value=["numpy_v1.json"]), patch("builtins.open", side_effect=Exception("Read error")):
        assert get_snapshot_api_dict("np") == {}


def test_get_snapshot_api_dict_success():
    mock_data = {"categories": {"math": [{"name": "Add", "kwargs": ["keepdims"], "params": [{"name": "axis"}]}]}}
    m_open = mock_open(read_data=json.dumps(mock_data))

    with patch("os.listdir", return_value=["numpy_v1.0.json", "numpy_v2.0.json"]), patch("builtins.open", m_open):
        api_dict = get_snapshot_api_dict("np")

    assert "add" in api_dict
    assert "keepdims" in api_dict["add"]
    assert "axis" in api_dict["add"]


def test_validate_mappings():
    with tempfile.TemporaryDirectory() as d:
        backend_dir = os.path.join(d, "backends", "keras")
        os.makedirs(backend_dir)

        mock_yaml = {
            "backend_name": "keras",
            "operations": {
                "valid_op": {"target_api": "keras.add"},
                "valid_op_kwargs": {"target_api": "keras.sum", "kwarg_translations": {"dim": "axis", "keepdim": {"target_name": "keepdims"}, "unmapped": {"target_name": "unmapped_kwarg"}, "ignore_me": "ignored"}},
                "invalid_op": {"target_api": "keras.hallucinated_op"},
                "custom_op": {"target_api": "custom_op"},
                "ast_template_op": {"ast_template": "keras.sub(a, b)"},
                "ast_template_invalid": {"ast_template": "keras.hallucinated_sub(a, b)"},
                "ast_template_no_match": {"ast_template": "!invalid"},
                "lambda_op": {"target_api": "lambda x: x"},
                "ignored_op": {"target_api": "keras.abs"},  # explicitly ignored in script built-ins
                "tf_sparse_op": {"target_api": "tf.sparse.something"},  # explicitly ignored in script
            },
        }

        filepath = os.path.join(backend_dir, "mappings.yaml")
        with open(filepath, "w") as f:
            yaml.dump(mock_yaml, f)

        mock_api_dict = {"add": set(), "sum": {"axis", "keepdims", "ignored"}, "sub": set()}

        with patch("glob.glob", return_value=[filepath]), patch("scripts.validate_backend_mappings.get_snapshot_api_dict", return_value=mock_api_dict):
            errors = validate_mappings()

        err_str = " ".join(errors)
        assert "unmapped_kwarg" in err_str
        assert "hallucinated_op" in err_str
        assert "hallucinated_sub" in err_str
        # 'valid_op', 'sum', 'sub' should not cause errors
        assert "'valid_op'" not in err_str
        assert "lambda" not in err_str
        assert "tf.sparse.something" not in err_str
        assert "keras.abs" not in err_str


def test_validate_mappings_no_api_dict():
    with tempfile.TemporaryDirectory() as d:
        backend_dir = os.path.join(d, "backends", "torch")
        os.makedirs(backend_dir)

        filepath = os.path.join(backend_dir, "mappings.yaml")
        with open(filepath, "w") as f:
            yaml.dump({"backend_name": "torch"}, f)

        with patch("glob.glob", return_value=[filepath]), patch("scripts.validate_backend_mappings.get_snapshot_api_dict", return_value={}):
            errors = validate_mappings()

        assert not errors


def test_validate_mappings_ignored_backends():
    with tempfile.TemporaryDirectory() as d:
        backend_dir = os.path.join(d, "backends", "numpy")
        os.makedirs(backend_dir)

        filepath = os.path.join(backend_dir, "mappings.yaml")
        with open(filepath, "w") as f:
            yaml.dump({"backend_name": "numpy", "operations": {"op": {"target_api": "numpy.fake"}}}, f)

        with patch("glob.glob", return_value=[filepath]), patch("scripts.validate_backend_mappings.get_snapshot_api_dict", return_value={"real": set()}):
            errors = validate_mappings()

        assert not errors


def test_main(capsys):
    with patch("scripts.validate_backend_mappings.validate_mappings", return_value=["err1"]):
        assert main() == 1
        assert "Hallucinations detected" in capsys.readouterr().out

    with patch("scripts.validate_backend_mappings.validate_mappings", return_value=[]):
        assert main() == 0
        assert "Passed" in capsys.readouterr().out


def test_main_block(capsys):
    import runpy
    import sys

    with patch.object(sys, "argv", ["validate_backend_mappings.py"]):
        with patch("scripts.validate_backend_mappings.validate_mappings", return_value=[]):
            try:
                runpy.run_path("scripts/validate_backend_mappings.py", run_name="__main__")
            except SystemExit as e:
                assert e.code == 0
    assert "Passed" in capsys.readouterr().out
