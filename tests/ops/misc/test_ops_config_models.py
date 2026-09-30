"""Tests for test_ops_config_models."""

from __future__ import annotations

from ml_switcheroo_ir import LogicalGraph, LogicalNode

import ml_switcheroo_compiler.backends.generator_mixins as gen_mixins
import ml_switcheroo_compiler.ops.config_models as ops_cfg


class _MockDummyGenerator(gen_mixins.GeneratorLifecycleMixin):
    """Mock generator class using GeneratorLifecycleMixin."""

    def __init__(self, graph: LogicalGraph) -> None:
        """Initialize mock generator.

        Args:
            graph (LogicalGraph): Logical graph to simulate.
        """
        self.graph = graph
        self.header = "# Header comment"
        self.code: list[str] = []
        self.indent_level: int = 0
        self.sorted_nodes: list[LogicalNode] = list(graph.nodes.values())
        self.input_idx: int = 0
        self._output_returns: list[str] = []
        self.var_names: dict[str, str] = {"in1": "args[0]"}

    def assign_var_name(self, node_id: str) -> str:
        """Assign variable name.

        Args:
            node_id (str): Node id.

        Returns:
            str: Assigned variable name.
        """
        name = f"var_{node_id}"
        self.var_names[node_id] = name
        return name

    def visit(self, node: LogicalNode, input_vars: list[str]) -> str:
        """Generate code for node.

        Args:
            node (LogicalNode): The node.
            input_vars (list[str]): Input variables.

        Returns:
            str: Code string.
        """
        return f"Identity({', '.join(input_vars)})"

    def _emit_body_return(self, returns: list[str]) -> None:
        """Emit return statement for body.

        Args:
            returns (list[str]): Return variables.
        """
        self.add_line(f"return {', '.join(returns) if returns else 'None'}")

    def add_line(self, line: str) -> None:
        """Add line to code buffer.

        Args:
            line (str): Line string.
        """
        self.code.append("    " * self.indent_level + line)


class _MockCustomModule:
    """Mock backend module exposing zeros, array, asarray, and item."""

    def zeros(self, shape: tuple[int, ...]) -> tuple[str, tuple[int, ...]]:
        """Mock zeros.

        Args:
            shape (tuple[int, ...]): Target shape.

        Returns:
            tuple[str, tuple[int, ...]]: Descriptor.
        """
        return ("custom_zeros", shape)

    def array(self, data: object, dtype: object | None = None) -> tuple[str, object, object | None]:
        """Mock array.

        Args:
            data (object): Input data.
            dtype (object | None): Data type.

        Returns:
            tuple[str, object, object | None]: Descriptor.
        """
        return ("custom_array", data, dtype)

    def asarray(self, data: object) -> tuple[str, object]:
        """Mock asarray.

        Args:
            data (object): Input data.

        Returns:
            tuple[str, object]: Descriptor.
        """
        return ("custom_asarray", data)

    def item(self, data: object) -> float:
        """Mock item.

        Args:
            data (object): Input object.

        Returns:
            float: 99.9.
        """
        return 99.9


class _MockEagerBackendWithModule(gen_mixins.EagerExecutionMixin):
    """Eager backend class providing custom module."""

    @classmethod
    def get_module(cls: type) -> _MockCustomModule:
        """Return custom backend module.

        Returns:
            _MockCustomModule: Mock module.
        """
        return _MockCustomModule()


class _MockEagerBackendDefault(gen_mixins.EagerExecutionMixin):
    """Eager backend class using default numpy module."""

    pass


class _MockScalarHolder:
    """Mock scalar container providing item()."""

    def item(self) -> float:
        """Return scalar value.

        Returns:
            float: Scalar value.
        """
        return 33.3


def test_ops_config_models() -> None:
    """Verify all Pydantic models and methods in ops/config_models.py."""
    var_cfg = ops_cfg.VariantConfig(
        generator="gen_fn",
        eager="eager_fn",
        expr="a + b",
        scalar_expr="a + b",
        simd_expr="simd_add",
        template="tmpl",
    )
    assert var_cfg.generator == "gen_fn"
    assert var_cfg.expr == "a + b"

    rule_model = ops_cfg.BackendMappingRuleModel(
        generator="ir_gen",
        eager="eager_eval",
        expr="x * y",
        scalar_expr="x * y",
        simd_expr="simd_mul",
        template="mul_template",
        macro_template="MUL(%1, %2)",
        args={"scale": 2.0},
        import_requirements=["math"],
        min_version="1.0",
        max_version="2.5",
        custom_code="custom_emit()",
        target_api="torch.mul",
        supported=True,
    )
    assert rule_model.generator == "ir_gen"
    assert rule_model.target_api == "torch.mul"

    math_sem = ops_cfg.MathematicalSemanticsModel(
        is_commutative=True,
        is_associative=True,
        is_idempotent=False,
        identity_element=0.0,
        pure_math_equivalent="a + b",
        differentiable=True,
    )
    assert math_sem.is_commutative is True
    assert math_sem.identity_element == 0.0

    arg_cfg = ops_cfg.OpArgConfig(
        name="input_tensor",
        type="Tensor",
        is_variadic=False,
        kind="positional",
        default=None,
    )
    assert arg_cfg.name == "input_tensor"
    assert arg_cfg.type == "Tensor"

    sig_model = ops_cfg.OpSignatureModel(
        args=[arg_cfg],
        input_tensors=["x"],
        output_tensors=["y"],
        keyword_constraints={"axis": 0},
        keyword_defaults={"axis": -1},
        has_variadic_args=False,
        has_variadic_kwargs=False,
        output_shape_formula="x.shape",
        output_dtype_formula="x.dtype",
    )
    assert len(sig_model.args) == 1
    assert sig_model.output_shape_formula == "x.shape"

    dtype_rules = ops_cfg.DtypeRulesModel(
        promotion_matrix="standard",
        allowed_input_dtypes=["float32", "float64"],
        output_dtype_inference="promote(in0, in1)",
    )
    assert dtype_rules.allowed_input_dtypes == ["float32", "float64"]

    shape_sig = ops_cfg.ShapeSignatureModel(
        pattern="broadcast(in0, in1)",
        symbolic_expression="[B, M], [B, N] -> [B, M, N]",
        category="binary",
    )
    assert shape_sig.pattern == "broadcast(in0, in1)"

    invariants = ops_cfg.InvariantsModel(
        is_commutative=True,
        is_associative=True,
        is_idempotent=False,
        identity_value=1.0,
        absorbing_value=0.0,
    )
    assert invariants.identity_value == 1.0

    ad_cfg = ops_cfg.AutodiffConfig(
        vjp_rule_id="vjp_add",
        jvp_rule_id="jvp_add",
        jvp="x_dot + y_dot",
        vjp=["cotangent", "cotangent"],
    )
    assert ad_cfg.vjp_rule_id == "vjp_add"

    op_def1 = ops_cfg.OperationDefinitionModel(
        opcode="Add",
        description="Addition op",
        domain="core",
        input_tensors=["x", "y"],
        output_tensors=["out"],
        attribute_specs={"alpha": "float"},
        broadcast_semantics="numpy",
        type_promotion="standard",
        std_args=[arg_cfg],
        signature=sig_model,
        dtype_rules=dtype_rules,
        shape_signature=shape_sig,
        invariants=invariants,
        math_semantics=math_sem,
        autodiff=ad_cfg,
        variants={"llvm_cpp": rule_model},
    )
    assert op_def1.canonical_name == "Add"
    assert op_def1.documentation == "Addition op"

    op_def2 = ops_cfg.OperationDefinitionModel(
        operation="Sub",
        docstring="Subtraction op",
    )
    assert op_def2.canonical_name == "Sub"
    assert op_def2.documentation == "Subtraction op"

    op_def3 = ops_cfg.OperationDefinitionModel()
    assert op_def3.canonical_name == ""
    assert op_def3.documentation == ""

    # OpRegistryConfig & OpsRegistry
    op_reg_cfg = ops_cfg.OpRegistryConfig(
        description="Add config",
        operation="Add",
        std_args=[arg_cfg],
        autodiff=ad_cfg,
        variants={"default": var_cfg},
    )
    ops_registry = ops_cfg.OpsRegistry(root={"Add": op_reg_cfg})
    assert dict(ops_registry.items()) == {"Add": op_reg_cfg}
    assert ops_registry.get("Add") == op_reg_cfg
    assert ops_registry.get("Unknown", None) is None

    # DomainOperationsModel
    domain_ops = ops_cfg.DomainOperationsModel(root={"Add": op_def1})
    assert dict(domain_ops.items()) == {"Add": op_def1}
    assert domain_ops.get("Add") == op_def1
    assert domain_ops.get("Unknown", None) is None
