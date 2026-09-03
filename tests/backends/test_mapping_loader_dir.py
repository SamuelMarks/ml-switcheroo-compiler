def test_mapping_loader_dir_yaml(monkeypatch):
    """Test loading mappings from a directory of yaml files."""
    import os
    from unittest.mock import MagicMock, patch

    from ml_switcheroo_compiler.backends.mapping_loader import load_backend_mappings

    def mock_isdir(path):
        return True

    def mock_listdir(path):
        return ["test.yaml"]

    mock_data = {"TestOp": {"target_api": "test_api"}}

    with patch("builtins.open", new_callable=MagicMock) as mock_open_func:
        mock_open_func.return_value.__enter__.return_value = "mock_file"
        with patch("yaml.safe_load", return_value=mock_data):
            monkeypatch.setattr(os.path, "isdir", mock_isdir)
            monkeypatch.setattr(os, "listdir", mock_listdir)

            schema = load_backend_mappings("mock_backend")
            assert schema.operations["TestOp"].target_api == "test_api"
