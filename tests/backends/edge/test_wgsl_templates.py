from unittest import mock
from unittest.mock import MagicMock, patch

import ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider as provider
from ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider import (
    _load_templates,
    get_js_orchestration_template,
    get_webgpu_ops,
    get_wgsl_global_bindings,
    get_wgsl_kernels_config,
    get_wgsl_op_mapping,
    get_wgsl_template,
)


def test_wgsl_provider_functions():
    """Test WGSL template and config retrieval functions."""
    tpl = get_wgsl_template("add")
    assert isinstance(tpl, (dict, str))

    js_orch = get_js_orchestration_template("init")
    assert isinstance(js_orch, str)

    bindings = get_wgsl_global_bindings()
    assert isinstance(bindings, str)

    ops = get_webgpu_ops()
    assert isinstance(ops, dict)

    # Test cache hit on _load_webgpu_ops (branch 48->exit)
    ops2 = get_webgpu_ops()
    assert ops2 is ops


def test_wgsl_provider_missing_files():
    """Test behavior when YAML template files do not exist."""
    with patch("os.path.exists", return_value=False):
        provider._WGSL_TEMPLATES = {}
        provider._WEBGPU_OPS = {}

        assert get_wgsl_template("add") == {}
        assert get_js_orchestration_template("init") == ""
        assert get_wgsl_global_bindings() == ""
        assert get_webgpu_ops() == {}


def test_wgsl_provider_non_dict_fallback():
    """Test non-dict fallbacks for templates and orchestration."""
    provider._WGSL_TEMPLATES = {"templates": "not_a_dict", "js_orchestration": "not_a_dict"}
    assert get_wgsl_template("add") == {}
    assert get_js_orchestration_template("init") == ""

    # Test when tpl itself is not a dict
    provider._WGSL_TEMPLATES = {"templates": {"test": "string_template"}}
    assert get_wgsl_template("test") == {}

    # Test when modular_path returns non-dict
    with patch("os.path.exists", return_value=True):
        with patch("yaml.safe_load", return_value="not_a_dict"):
            provider._merge_modular_templates({})

    # Test non-yaml files and corrupted template files in modular_dir
    with patch("os.path.exists", side_effect=lambda p: not p.endswith("templates.yaml")):
        with patch("os.listdir", return_value=["ignore.txt", "broken.yaml"]):
            with patch("builtins.open", side_effect=OSError("corrupted file")):
                d: dict = {}
                provider._merge_modular_templates(d)

    # Reset
    provider._WGSL_TEMPLATES = {}


def test_wgsl_provider_branches() -> None:
    """Test wgsl_provider modular template loading, kernels loading, and fallback paths."""
    import ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider as wp

    orig_templates = dict(wp._WGSL_TEMPLATES)
    orig_kernels = dict(wp._WGSL_KERNELS)
    try:
        wp._WGSL_KERNELS.clear()
        k1 = wp._load_kernels()
        assert isinstance(k1, dict)
        wp._WGSL_KERNELS.clear()
        with mock.patch("yaml.safe_load", return_value=[1, 2, 3]):
            k2 = wp._load_kernels()
            assert isinstance(k2, dict)
        wp._WGSL_KERNELS = {"op_mappings": {"Add": {"template": "add_template"}}}
        assert wp.get_wgsl_op_mapping("Add") == {"template": "add_template"}
        wp._WGSL_KERNELS = {"op_mappings": {"Bad": "not_a_dict"}}
        assert wp.get_wgsl_op_mapping("Bad") == {}
        wp._WGSL_TEMPLATES.clear()
        with mock.patch("ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider._load_kernels", return_value="not_a_dict"):
            _load_templates()
            assert "templates" in wp._WGSL_TEMPLATES
        wp._WGSL_TEMPLATES.clear()
        with mock.patch("ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider._load_kernels", return_value={"templates": None, "bindings": None}):
            _load_templates()
            assert isinstance(wp._WGSL_TEMPLATES, dict)
        wp._WGSL_TEMPLATES.clear()
        with mock.patch("os.path.exists", return_value=True), mock.patch("builtins.open", mock.mock_open(read_data="[1, 2, 3]")), mock.patch("yaml.safe_load", return_value=[1, 2, 3]):
            _load_templates()
            assert isinstance(wp._WGSL_TEMPLATES, dict)
    finally:
        wp._WGSL_TEMPLATES = orig_templates
        wp._WGSL_KERNELS = orig_kernels


def test_wgsl_provider_kernels_config_and_global_bindings() -> None:
    """Test get_wgsl_kernels_config, get_wgsl_template_for_op, and _load_templates global_bindings merge branch.

    Returns:
        None
    """
    cfg = get_wgsl_kernels_config()
    assert isinstance(cfg, dict)
    with patch("ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider._load_kernels", return_value={"op_mappings": {"invalid_op": "not_a_dict"}}):
        assert get_wgsl_op_mapping("invalid_op") == {}
    mock_base = {"templates": {}}
    mock_kernels = {"templates": {"mock_k": {}}, "bindings": {"global_bindings": "// global bindings test"}}
    import ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider as wp

    orig_templates = dict(wp._WGSL_TEMPLATES)
    try:
        with patch("os.path.exists", return_value=True), patch("builtins.open", MagicMock()), patch("yaml.safe_load", return_value=mock_base):
            with patch("ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider._load_kernels", return_value=mock_kernels):
                wp._WGSL_TEMPLATES = {}
                _load_templates()
                assert wp._WGSL_TEMPLATES.get("global_bindings") == "// global bindings test"
    finally:
        wp._WGSL_TEMPLATES = orig_templates


def test_backends_wgsl_empty_kernels() -> None:
    """Test get_wgsl_kernels_config with empty dict when file returns empty or non-dict.

    Returns:
        None
    """
    import ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider as wp

    old_kernels = wp._WGSL_KERNELS
    try:
        wp._WGSL_KERNELS = {}
        with patch("builtins.open", MagicMock()):
            with patch("yaml.safe_load", return_value="not-a-dict"):
                conf = wp.get_wgsl_kernels_config()
                assert conf == {"bindings": {}, "op_mappings": {}, "templates": {}}
    finally:
        wp._WGSL_KERNELS = old_kernels


def test_wgsl_grounding_schema_missing_and_validate_edges() -> None:
    """Test get_wgsl_grounding_schema when json file is missing and validate_wgsl_statement branches."""
    from unittest import mock

    import ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider as wp

    # 1. get_wgsl_grounding_schema when schema_path does not exist
    wp._WGSL_GROUNDING_SCHEMA = None
    with mock.patch("os.path.exists", return_value=False):
        schema = wp.get_wgsl_grounding_schema()
        assert schema == {"ops": []}

    # Reset cache
    wp._WGSL_GROUNDING_SCHEMA = None

    # 2. validate_wgsl_statement where ops is not a list (or op_entry is not a dict)
    with mock.patch("ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider.get_wgsl_grounding_schema", return_value={"ops": "not_a_list"}):
        with mock.patch("ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider.get_wgsl_op_mapping", return_value=None):
            with mock.patch("ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider._WGSL_TEMPLATES", {"templates": "not_a_dict"}):
                res = wp.validate_wgsl_statement("non_existent_op")
                assert res is False
