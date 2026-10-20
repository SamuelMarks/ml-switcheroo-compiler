"""Pydantic models for ops registry configuration files."""

from __future__ import annotations

from collections.abc import ItemsView

from pydantic import BaseModel, ConfigDict, Field, RootModel


class VariantConfig(BaseModel):
    """Configuration for an op variant on a specific backend."""

    generator: str | None = None
    eager: str | None = None
    expr: str | None = None
    scalar_expr: str | None = None
    simd_expr: str | None = None
    template: str | None = None
    model_config = ConfigDict(extra="allow")


class BackendMappingRuleModel(BaseModel):
    """Declarative backend mapping rule and emission templates for an operation."""

    generator: str | None = Field(default=None, description="IR AST generator target or function.")
    eager: str | None = Field(default=None, description="Eager evaluation callable function path.")
    expr: str | None = Field(default=None, description="Inline code generation expression.")
    scalar_expr: str | None = Field(default=None, description="C++ or host scalar expression template.")
    simd_expr: str | None = Field(default=None, description="SIMD vectorized expression template.")
    template: str | None = Field(default=None, description="AST or textual code template.")
    macro_template: str | None = Field(default=None, description="Macro substitution template string.")
    args: dict[str, str | int | float | bool] | None = Field(default=None, description="Mapped argument dictionary.")
    import_requirements: list[str] = Field(default_factory=list, description="Required module imports for code generation.")
    min_version: str | None = Field(default=None, description="Minimum supported framework version.")
    max_version: str | None = Field(default=None, description="Maximum supported framework version.")
    custom_code: str | None = Field(default=None, description="Custom emitter code snippet.")
    target_api: str | None = Field(default=None, description="Target backend API endpoint.")
    supported: bool = Field(default=True, description="Whether the operation is supported by the target backend.")
    model_config = ConfigDict(extra="allow")


class MathematicalSemanticsModel(BaseModel):
    """Declarative mathematical semantics and invariant properties for an operation."""

    is_commutative: bool = Field(default=False, description="Whether the operation is commutative.")
    is_associative: bool = Field(default=False, description="Whether the operation is associative.")
    is_idempotent: bool = Field(default=False, description="Whether the operation is idempotent.")
    identity_element: int | float | str | None = Field(default=None, description="Identity element value if applicable.")
    pure_math_equivalent: str | None = Field(default=None, description="Canonical mathematical expression or symbolic representation.")
    differentiable: bool = Field(default=True, description="Whether the operation is mathematically differentiable.")
    model_config = ConfigDict(extra="allow")


class OpArgConfig(BaseModel):
    """Configuration for an argument to an op."""

    name: str = Field(description="Argument name.")
    type: str = Field(description="Argument type annotation string.")
    is_variadic: bool | None = Field(default=False, description="Whether the argument is variadic.")
    kind: str | None = Field(default=None, description="Argument passing convention kind.")
    default: str | int | float | bool | list[int] | list[float] | list[str] | None = Field(default=None, description="Default parameter value.")
    model_config = ConfigDict(extra="allow")


class OpSignatureModel(BaseModel):
    """Declarative function signature model specifying typed arguments and shape formulas."""

    args: list[OpArgConfig] = Field(default_factory=list, description="List of ordered positional/keyword argument specifications.")
    input_tensors: list[str] = Field(default_factory=list, description="Input tensor parameter specifications.")
    output_tensors: list[str] = Field(default_factory=lambda: ["out"], description="Output tensor specifications.")
    keyword_constraints: dict[str, str | int | float | bool | None] = Field(default_factory=dict, description="Constraints on keyword arguments.")
    keyword_defaults: dict[str, str | int | float | bool | None] = Field(default_factory=dict, description="Default parameter values for keyword arguments.")
    has_variadic_args: bool = Field(default=False, description="Whether the operation accepts variadic positional arguments (*args).")
    has_variadic_kwargs: bool = Field(default=False, description="Whether the operation accepts variadic keyword arguments (**kwargs).")
    output_shape_formula: str | None = Field(default=None, description="Symbolic formula computing output shape from inputs.")
    output_dtype_formula: str | None = Field(default=None, description="Symbolic formula computing output data type.")
    model_config = ConfigDict(extra="allow")


class DtypeRulesModel(BaseModel):
    """Declarative data type promotion and validation rules for an operation."""

    promotion_matrix: str = Field(default="standard", description="Promotion table or matrix strategy.")
    allowed_input_dtypes: list[str] = Field(default_factory=lambda: ["all"], description="Permitted input tensor dtypes.")
    output_dtype_inference: str = Field(default="promote(in0, in1)", description="Symbolic rule or logic for inferring output dtype.")
    model_config = ConfigDict(extra="allow")


class ShapeSignatureModel(BaseModel):
    """Declarative shape signature and symbolic propagation specification."""

    pattern: str | None = Field(default=None, description="Shape pattern name or formula (e.g. broadcast(in0, in1)).")
    symbolic_expression: str | None = Field(default=None, description="Symbolic input-to-output shape formula (e.g. [B, M, K], [B, K, N] -> [B, M, N]).")
    category: str | None = Field(default=None, description="Operation category.")
    model_config = ConfigDict(extra="allow")


class InvariantsModel(BaseModel):
    """Declarative mathematical invariants for an operation."""

    is_commutative: bool = Field(default=False, description="Whether the operation is commutative.")
    is_associative: bool = Field(default=False, description="Whether the operation is associative.")
    is_idempotent: bool = Field(default=False, description="Whether the operation is idempotent.")
    identity_value: int | float | str | None = Field(default=None, description="Identity value if applicable.")
    absorbing_value: int | float | str | None = Field(default=None, description="Absorbing value if applicable.")
    model_config = ConfigDict(extra="allow")


class AutodiffConfig(BaseModel):
    """Configuration for autodiff rules."""

    vjp_rule_id: str | None = Field(default=None, description="Identifier of the declarative VJP rule.")
    jvp_rule_id: str | None = Field(default=None, description="Identifier of the declarative JVP rule.")
    jvp: str | None = Field(default=None, description="Symbolic JVP expression or rule name.")
    vjp: list[str] | None = Field(default=None, description="Symbolic VJP expression or rule names.")
    model_config = ConfigDict(extra="allow")


class OperationDefinitionModel(BaseModel):
    """Unified schema-validated declarative definition model for compiler operations."""

    operation: str | None = Field(default=None, description="Canonical operation opcode or name.")
    opcode: str | None = Field(default=None, description="Alternative opcode field.")
    description: str | None = Field(default=None, description="Docstring describing the operation.")
    docstring: str | None = Field(default=None, description="Alternative docstring field.")
    domain: str = Field(default="core", description="Mathematical or functional domain.")
    input_tensors: list[str] = Field(default_factory=list, description="Specifications of required input tensors.")
    output_tensors: list[str] = Field(default_factory=list, description="Specifications of returned output tensors.")
    attribute_specs: dict[str, str] = Field(default_factory=dict, description="Static configuration attributes and type constraints.")
    broadcast_semantics: str = Field(default="numpy", description="Broadcasting rules applied to inputs.")
    type_promotion: str = Field(default="standard", description="Type promotion constraints applied across arguments.")
    std_args: list[str | int | float | bool | OpArgConfig | dict[str, str | int | float | bool | list[int] | list[float] | list[str] | None]] | None = Field(default=None, description="Legacy or standard argument specifications.")
    signature: str | OpSignatureModel | None = Field(default=None, description="Structured signature specification or signature string.")
    dtype_rules: DtypeRulesModel | None = Field(default=None, description="Declarative data type promotion and validation rules.")
    shape_signature: str | ShapeSignatureModel | None = Field(default=None, description="Symbolic shape signature relating input shapes to output shapes.")
    invariants: InvariantsModel | None = Field(default=None, description="Declarative mathematical invariants.")
    math_semantics: MathematicalSemanticsModel | None = Field(default=None, description="Declarative mathematical invariants.")
    autodiff: AutodiffConfig | None = Field(default=None, description="Symbolic autodiff JVP and VJP rules.")
    variants: dict[
        str,
        BackendMappingRuleModel | VariantConfig | dict[str, str | int | float | bool | None | dict[str, str | int | float | bool | None]],
    ] = Field(default_factory=dict, description="Backend mapping rules and emission configurations.")
    model_config = ConfigDict(extra="allow")

    @property
    def canonical_name(self) -> str:
        """Return the canonical opcode name.

        Returns:
            str: The opcode or operation name.
        """
        return self.opcode or self.operation or ""

    @property
    def documentation(self) -> str:
        """Return the documentation string.

        Returns:
            str: The docstring or description.
        """
        return self.docstring or self.description or ""


class OpRegistryConfig(BaseModel):
    """Configuration for a specific op in the registry."""

    description: str | None = None
    operation: str | None = None
    std_args: list[str | int | float | bool | OpArgConfig | dict[str, str | int | float | bool | list[int] | list[float] | list[str] | None]] | None = None
    autodiff: AutodiffConfig | None = None
    variants: dict[str, VariantConfig] = Field(default_factory=dict)
    model_config = ConfigDict(extra="allow")


class OpsRegistry(RootModel[dict[str, OpRegistryConfig]]):
    """Configuration for all ops in the registry."""

    root: dict[str, OpRegistryConfig]

    def items(self) -> ItemsView[str, OpRegistryConfig]:
        """Return items from the underlying dictionary.

        Returns:
            ItemsView[str, OpRegistryConfig]: Key-value pairs of the registry.
        """
        return self.root.items()

    def get(self, key: str, default: OpRegistryConfig | None = None) -> OpRegistryConfig | None:
        """Get op config by key.

        Args:
            key (str): Operation identifier.
            default (Optional[OpRegistryConfig]): Default fallback if not found.

        Returns:
            Optional[OpRegistryConfig]: Retrieved config or default.
        """
        return self.root.get(key, default)


class DomainOperationsModel(RootModel[dict[str, OperationDefinitionModel]]):
    """Collection of operation definitions indexed by op name for domain-specific YAML registries."""

    root: dict[str, OperationDefinitionModel]

    def items(self) -> ItemsView[str, OperationDefinitionModel]:
        """Return items from the underlying dictionary.

        Returns:
            ItemsView[str, OperationDefinitionModel]: Key-value pairs of the registry.
        """
        return self.root.items()

    def get(self, key: str, default: OperationDefinitionModel | None = None) -> OperationDefinitionModel | None:
        """Get op definition by key.

        Args:
            key (str): Operation identifier.
            default (Optional[OperationDefinitionModel]): Default fallback if not found.

        Returns:
            Optional[OperationDefinitionModel]: Retrieved model or default.
        """
        return self.root.get(key, default)
