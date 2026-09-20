"""Tests for YAML operation registry models and schema validation."""

import glob
import os
from typing import Union

import yaml

from ml_switcheroo_compiler.ops.config_models import (
    BackendMappingRuleModel,
    MathematicalSemanticsModel,
    OpArgConfig,
    OperationDefinitionModel,
    OpRegistryConfig,
    OpSignatureModel,
    OpsRegistry,
    VariantConfig,
)


def test_variant_config_defaults() -> None:
    """Test default values and instantiation of VariantConfig."""
    cfg = VariantConfig()
    assert cfg.generator is None
    assert cfg.eager is None
    assert cfg.expr is None


def test_backend_mapping_rule_model() -> None:
    """Test BackendMappingRuleModel field assignment and validation."""
    rule = BackendMappingRuleModel(
        generator="np.add",
        eager="numpy.add",
        macro_template="np.add({a}, {b})",
        args={"a": "x1", "b": "x2"},
        import_requirements=["numpy"],
        min_version="1.20.0",
        target_api="numpy.add",
    )
    assert rule.generator == "np.add"
    assert rule.eager == "numpy.add"
    assert rule.macro_template == "np.add({a}, {b})"
    assert rule.args == {"a": "x1", "b": "x2"}
    assert rule.import_requirements == ["numpy"]
    assert rule.min_version == "1.20.0"
    assert rule.target_api == "numpy.add"


def test_mathematical_semantics_model() -> None:
    """Test MathematicalSemanticsModel invariants and properties."""
    math_props = MathematicalSemanticsModel(
        is_commutative=True,
        is_associative=True,
        is_idempotent=False,
        identity_element=0,
        pure_math_equivalent="a + b",
        differentiable=True,
    )
    assert math_props.is_commutative is True
    assert math_props.is_associative is True
    assert math_props.is_idempotent is False
    assert math_props.identity_element == 0
    assert math_props.pure_math_equivalent == "a + b"
    assert math_props.differentiable is True


def test_op_signature_model() -> None:
    """Test OpSignatureModel instantiation and default field values."""
    sig = OpSignatureModel(
        args=[
            OpArgConfig(name="x1", type="Tensor", is_variadic=False),
            OpArgConfig(name="x2", type="Tensor", is_variadic=False),
        ],
        keyword_defaults={"axis": 0},
        has_variadic_args=False,
        has_variadic_kwargs=False,
        output_shape_formula="broadcast(x1.shape, x2.shape)",
        output_dtype_formula="promote(x1.dtype, x2.dtype)",
    )
    assert len(sig.args) == 2
    assert sig.args[0].name == "x1"
    assert sig.keyword_defaults["axis"] == 0
    assert sig.output_shape_formula == "broadcast(x1.shape, x2.shape)"
    assert sig.output_dtype_formula == "promote(x1.dtype, x2.dtype)"


def test_operation_definition_model_properties() -> None:
    """Test property accessors on OperationDefinitionModel."""
    op_def = OperationDefinitionModel(
        operation="Add",
        description="Elementwise addition",
        domain="core",
        input_tensors=["x1", "x2"],
        output_tensors=["out"],
        math_semantics=MathematicalSemanticsModel(is_commutative=True),
    )
    assert op_def.canonical_name == "Add"
    assert op_def.documentation == "Elementwise addition"
    assert op_def.domain == "core"
    assert op_def.input_tensors == ["x1", "x2"]

    op_def_alt = OperationDefinitionModel(
        opcode="Sub",
        docstring="Elementwise subtraction",
    )
    assert op_def_alt.canonical_name == "Sub"
    assert op_def_alt.documentation == "Elementwise subtraction"

    op_def_empty = OperationDefinitionModel()
    assert op_def_empty.canonical_name == ""
    assert op_def_empty.documentation == ""


def test_ops_registry_items_and_get() -> None:
    """Test OpsRegistry items() and get() methods."""
    data: dict[str, OpRegistryConfig] = {
        "Add": OpRegistryConfig(operation="Add", description="Addition"),
        "Mul": OpRegistryConfig(operation="Mul", description="Multiplication"),
    }
    registry = OpsRegistry(root=data)
    items = list(registry.items())
    assert len(items) == 2
    assert registry.get("Add") is not None
    assert registry.get("NonExistent") is None
    default_cfg = OpRegistryConfig(operation="Fallback")
    assert registry.get("Missing", default=default_cfg) == default_cfg


def test_validate_sample_op_definitions_from_yaml() -> None:
    """Validate sample YAML operation definitions against OperationDefinitionModel."""
    definitions_dir = os.path.join("src", "ml_switcheroo_compiler", "ops", "definitions")
    yaml_files: list[str] = sorted(glob.glob(os.path.join(definitions_dir, "*.yaml")))
    assert len(yaml_files) > 0, "Definitions directory must contain YAML files"

    # Validate first 30 op definitions to verify schema compatibility
    for path in yaml_files[:30]:
        with open(path, encoding="utf-8") as f:
            raw_data: Union[dict[str, object], None] = yaml.safe_load(f)
        if raw_data is not None and isinstance(raw_data, dict):
            if "operation" in raw_data or "opcode" in raw_data:
                model = OperationDefinitionModel.model_validate(raw_data)
                assert isinstance(model.canonical_name, str)
            else:
                from ml_switcheroo_compiler.ops.config_models import DomainOperationsModel

                domain_model = DomainOperationsModel.model_validate(raw_data)
                assert len(domain_model.root) > 0


def test_all_thirteen_domain_yaml_registries() -> None:
    """Exhaustively validate all 13 domain-specific YAML files against DomainOperationsModel."""
    from ml_switcheroo_compiler.ops.config_models import DomainOperationsModel

    domain_files: list[str] = [
        "binary_ops.yaml",
        "unary_ops.yaml",
        "activations.yaml",
        "reductions.yaml",
        "linalg.yaml",
        "shape_ops.yaml",
        "normalization.yaml",
        "loss_ops.yaml",
        "creation_ops.yaml",
        "random_ops.yaml",
        "control_flow_ops.yaml",
        "vision_ops.yaml",
        "signal_ops.yaml",
    ]
    definitions_dir: str = os.path.join("src", "ml_switcheroo_compiler", "ops", "definitions")

    for fname in domain_files:
        fpath: str = os.path.join(definitions_dir, fname)
        if not os.path.exists(fpath):
            cap_path = os.path.join(definitions_dir, fname.capitalize())
            if os.path.exists(cap_path):
                fpath = cap_path
        assert os.path.exists(fpath), f"Domain YAML file {fname} must exist"

        with open(fpath, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert isinstance(data, dict), f"{fname} must contain a mapping of operations"
        domain_model = DomainOperationsModel.model_validate(data)
        assert len(domain_model.root) > 0, f"{fname} must contain at least one operation"

        # Check domain model items and get methods
        items = list(domain_model.items())
        assert len(items) == len(domain_model.root)

        for op_name, op_def in items:
            assert op_def.canonical_name == op_name
            assert op_def.domain != ""
            assert op_def.math_semantics is not None
            assert op_def.signature is not None
            assert len(op_def.output_tensors) > 0
            assert domain_model.get(op_name) is op_def

        assert domain_model.get("NonExistentOp") is None
