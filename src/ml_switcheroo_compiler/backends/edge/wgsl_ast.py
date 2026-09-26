# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""WGSL AST definitions for robust code generation conforming to WebGPU specifications."""

import re
from typing import Optional, Union

RESERVED_WGSL_KEYWORDS: set[str] = {
    "array",
    "atomic",
    "bool",
    "f32",
    "f16",
    "i32",
    "u32",
    "vec2",
    "vec3",
    "vec4",
    "mat2x2",
    "mat3x3",
    "mat4x4",
    "true",
    "false",
    "fn",
    "let",
    "var",
    "const",
    "struct",
    "if",
    "else",
    "for",
    "while",
    "loop",
    "break",
    "continue",
    "return",
    "switch",
    "case",
    "default",
    "discard",
    "type",
    "alias",
    "storage",
    "uniform",
    "workgroup",
    "private",
    "function",
    "read",
    "write",
    "read_write",
}

VALID_IDENTIFIER_PATTERN: re.Pattern[str] = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
WORKGROUP_SIZE_PATTERN: re.Pattern[str] = re.compile(r"@workgroup_size\s*\(\s*(\d+)\s*(?:,\s*(\d+))?\s*(?:,\s*(\d+))?\s*\)")


class WGSLValidationError(ValueError):
    """Exception raised when WGSL AST node or emitted code violates WebGPU specifications."""


def validate_identifier(name: str) -> None:
    """Validate that an identifier adheres to WGSL naming specifications.

    Args:
        name (str): Identifier name to validate.

    Raises:
        WGSLValidationError: If the identifier is invalid or a reserved keyword.
    """
    if not name or not isinstance(name, str):
        raise WGSLValidationError(f"Invalid WGSL identifier: '{name}' (must be non-empty string)")
    if not VALID_IDENTIFIER_PATTERN.match(name):
        raise WGSLValidationError(f"Identifier '{name}' does not match WGSL identifier syntax")
    if name in RESERVED_WGSL_KEYWORDS:
        raise WGSLValidationError(f"Identifier '{name}' is a reserved WGSL keyword")


def validate_workgroup_size(x: int, y: int = 1, z: int = 1) -> None:
    """Validate workgroup dimensions against WebGPU guaranteed hardware limits.

    Args:
        x (int): Workgroup size X dimension.
        y (int, optional): Workgroup size Y dimension. Defaults to 1.
        z (int, optional): Workgroup size Z dimension. Defaults to 1.

    Raises:
        WGSLValidationError: If any dimension or total invocations exceed WebGPU limits.
    """
    if x < 1 or y < 1 or z < 1:
        raise WGSLValidationError(f"Workgroup dimensions must be >= 1, got ({x}, {y}, {z})")
    if x > 256 or y > 256 or z > 64:
        raise WGSLValidationError(f"Workgroup dimension exceeds WebGPU limits (max: x<=256, y<=256, z<=64), got ({x}, {y}, {z})")
    if x * y * z > 256:
        raise WGSLValidationError(f"Total workgroup invocations ({x * y * z}) exceed WebGPU minimum guaranteed limit of 256")


def validate_memory_layout(address_space: str, access_mode: Optional[str] = None) -> None:
    """Validate memory layout address space and access mode against WebGPU specifications.

    Args:
        address_space (str): Address space qualifier (e.g., 'storage', 'uniform', 'workgroup').
        access_mode (Optional[str], optional): Access mode qualifier (e.g., 'read', 'read_write').

    Raises:
        WGSLValidationError: If address space or access mode is invalid according to WebGPU.
    """
    valid_address_spaces: set[str] = {"storage", "uniform", "workgroup", "private", "function"}
    if address_space not in valid_address_spaces:
        raise WGSLValidationError(f"Invalid WGSL address space qualifier '{address_space}', expected one of {valid_address_spaces}")
    if access_mode is not None:
        valid_access_modes: set[str] = {"read", "write", "read_write"}
        if access_mode not in valid_access_modes:
            raise WGSLValidationError(f"Invalid WGSL access mode '{access_mode}', expected one of {valid_access_modes}")
        if address_space == "uniform" and access_mode != "read":
            raise WGSLValidationError(f"Uniform address space only supports 'read' access mode, got '{access_mode}'")
        if address_space == "storage" and access_mode not in ("read", "read_write"):
            raise WGSLValidationError(f"Storage buffer address space supports 'read' or 'read_write' access modes, got '{access_mode}'")
        if address_space == "workgroup" and access_mode != "read_write":
            raise WGSLValidationError(f"Workgroup address space only supports 'read_write' access mode, got '{access_mode}'")


def validate_binding_limits(group: int, binding: int) -> None:
    """Validate @group and @binding indices against WebGPU specification limits.

    Args:
        group (int): Bind group index.
        binding (int): Binding index within bind group.

    Raises:
        WGSLValidationError: If group or binding index exceeds WebGPU guaranteed minimum limits.
    """
    if group < 0 or group >= 4:
        raise WGSLValidationError(f"Bind group index {group} exceeds WebGPU minimum guaranteed limit of 4 (allowed: 0..3)")
    if binding < 0 or binding >= 64:
        raise WGSLValidationError(f"Binding index {binding} exceeds WebGPU standard limit of 64 (allowed: 0..63)")


class WGSLNode:
    """Base WGSL AST Node."""

    def validate(self) -> None:
        """Validate AST node against WebGPU specifications.

        Raises:
            WGSLValidationError: If node structure violates specifications.
        """
        for attr_val in self.__dict__.values():
            if isinstance(attr_val, WGSLNode):
                attr_val.validate()
            elif isinstance(attr_val, (list, tuple)):
                for item in attr_val:
                    if isinstance(item, WGSLNode):
                        item.validate()


class WGSLRaw(WGSLNode):
    """Raw string insertion."""

    def __init__(self, code: str) -> None:
        """Initialize WGSLRaw.

        Args:
            code (str): The raw code snippet.
        """
        self.code = code

    def validate(self) -> None:
        """Validate raw string node against WebGPU specifications.

        Raises:
            WGSLValidationError: If delimiters are unbalanced or WebGPU limits are violated.
        """
        pairs: dict[str, str] = {"(": ")", "{": "}", "[": "]"}
        stack: list[str] = []
        for char in self.code:
            if char in pairs:
                stack.append(pairs[char])
            elif char in pairs.values():
                if not stack or stack.pop() != char:
                    raise WGSLValidationError(f"Unbalanced delimiter '{char}' in raw WGSL code")
        if stack:
            raise WGSLValidationError(f"Unclosed delimiter '{stack[-1]}' in raw WGSL code")

        for match in WORKGROUP_SIZE_PATTERN.finditer(self.code):
            x = int(match.group(1))
            y = int(match.group(2)) if match.group(2) else 1
            z = int(match.group(3)) if match.group(3) else 1
            validate_workgroup_size(x, y, z)

        group_pattern: re.Pattern[str] = re.compile(r"@group\s*\(\s*(\d+)\s*\)")
        for g_match in group_pattern.finditer(self.code):
            g_idx = int(g_match.group(1))
            if g_idx < 0 or g_idx >= 4:
                raise WGSLValidationError(f"Bind group index {g_idx} exceeds WebGPU minimum guaranteed limit of 4")

        binding_pattern: re.Pattern[str] = re.compile(r"@binding\s*\(\s*(\d+)\s*\)")
        for b_match in binding_pattern.finditer(self.code):
            b_idx = int(b_match.group(1))
            if b_idx < 0 or b_idx >= 64:
                raise WGSLValidationError(f"Binding index {b_idx} exceeds WebGPU limit of 64")

        var_qual_pattern: re.Pattern[str] = re.compile(r"var\s*<\s*([a-zA-Z_][a-zA-Z0-9_]*)(?:\s*,\s*([a-zA-Z_][a-zA-Z0-9_]*))?\s*>")
        for v_match in var_qual_pattern.finditer(self.code):
            space = v_match.group(1)
            access = v_match.group(2)
            validate_memory_layout(space, access)


class WGSLBinding(WGSLNode):
    """WGSL Resource Binding Declaration (@group(G) @binding(B) var<address_space, access_mode> name: type)."""

    def __init__(
        self,
        group: int,
        binding: int,
        name: str,
        type_str: str,
        address_space: str = "storage",
        access_mode: Optional[str] = "read_write",
    ) -> None:
        """Initialize WGSLBinding.

        Args:
            group (int): Bind group index (0..3).
            binding (int): Binding slot index (0..63).
            name (str): Bound variable identifier.
            type_str (str): WGSL type declaration string.
            address_space (str, optional): Target address space. Defaults to 'storage'.
            access_mode (Optional[str], optional): Access mode. Defaults to 'read_write'.
        """
        self.group: int = group
        self.binding: int = binding
        self.name: str = name
        self.type_str: str = type_str
        self.address_space: str = address_space
        self.access_mode: Optional[str] = access_mode

    def validate(self) -> None:
        """Validate resource binding against WebGPU specification.

        Raises:
            WGSLValidationError: If binding attributes violate WebGPU specifications.
        """
        validate_identifier(self.name)
        validate_binding_limits(self.group, self.binding)
        validate_memory_layout(self.address_space, self.access_mode)


class WGSLVar(WGSLNode):
    """WGSL Variable expression."""

    def __init__(self, name: str) -> None:
        """Initialize WGSLVar.

        Args:
            name (str): The variable name.
        """
        self.name = name

    def validate(self) -> None:
        """Validate variable identifier name against WebGPU naming rules.

        Raises:
            WGSLValidationError: If variable name is invalid.
        """
        validate_identifier(self.name)


class WGSLIndex(WGSLNode):
    """WGSL Array/Buffer Indexing."""

    def __init__(self, buffer: str, index: Union[str, "WGSLNode"]) -> None:
        """Initialize WGSLIndex.

        Args:
            buffer (str): The buffer identifier.
            index (Union[str, WGSLNode]): The index expression or node.
        """
        self.buffer = buffer
        self.index = index

    def validate(self) -> None:
        """Validate array buffer index operation.

        Raises:
            WGSLValidationError: If buffer identifier or index is invalid.
        """
        validate_identifier(self.buffer)
        if isinstance(self.index, WGSLNode):
            self.index.validate()


class WGSLBinaryOp(WGSLNode):
    """WGSL Binary Operation."""

    VALID_OPS: set[str] = {
        "+",
        "-",
        "*",
        "/",
        "%",
        "==",
        "!=",
        "<",
        ">",
        "<=",
        ">=",
        "&&",
        "||",
        "&",
        "|",
        "^",
        "<<",
        ">>",
    }

    def __init__(self, op: str, left: Union[str, "WGSLNode"], right: Union[str, "WGSLNode"]) -> None:
        """Initialize WGSLBinaryOp.

        Args:
            op (str): The binary operator.
            left (Union[str, WGSLNode]): The left operand.
            right (Union[str, WGSLNode]): The right operand.
        """
        self.op = op
        self.left = left
        self.right = right

    def validate(self) -> None:
        """Validate binary operator and operands.

        Raises:
            WGSLValidationError: If operator is unsupported or operands are invalid.
        """
        if self.op not in self.VALID_OPS:
            raise WGSLValidationError(f"Unsupported WGSL binary operator: '{self.op}'")
        if isinstance(self.left, WGSLNode):
            self.left.validate()
        if isinstance(self.right, WGSLNode):
            self.right.validate()


class WGSLUnaryOp(WGSLNode):
    """WGSL Unary Operation."""

    VALID_OPS: set[str] = {"-", "!", "~", "*", "&"}

    def __init__(self, op: str, expr: Union[str, "WGSLNode"]) -> None:
        """Initialize WGSLUnaryOp.

        Args:
            op (str): The unary operator.
            expr (Union[str, WGSLNode]): The operand expression.
        """
        self.op = op
        self.expr = expr

    def validate(self) -> None:
        """Validate unary operator and operand.

        Raises:
            WGSLValidationError: If operator is unsupported or operand is invalid.
        """
        if self.op not in self.VALID_OPS:
            raise WGSLValidationError(f"Unsupported WGSL unary operator: '{self.op}'")
        if isinstance(self.expr, WGSLNode):
            self.expr.validate()


class WGSLAssign(WGSLNode):
    """WGSL Assignment Statement."""

    def __init__(self, target: Union[str, "WGSLNode"], value: Union[str, "WGSLNode"]) -> None:
        """Initialize WGSLAssign.

        Args:
            target (Union[str, WGSLNode]): The assignment target.
            value (Union[str, WGSLNode]): The value expression.
        """
        self.target = target
        self.value = value

    def validate(self) -> None:
        """Validate assignment target and value.

        Raises:
            WGSLValidationError: If target or value node is invalid.
        """
        if isinstance(self.target, WGSLNode):
            self.target.validate()
        if isinstance(self.value, WGSLNode):
            self.value.validate()


class WGSLDecl(WGSLNode):
    """WGSL Variable Declaration (let, var, or const)."""

    VALID_KINDS: set[str] = {"let", "var", "const"}

    def __init__(self, kind: str, name: str, value: Optional[Union[str, "WGSLNode"]] = None, type_annotation: Optional[str] = None) -> None:
        """Initialize WGSLDecl.

        Args:
            kind (str): Declaration kind ('let', 'var', 'const').
            name (str): Identifier name.
            value (Union[str, WGSLNode], optional): Initial value expression. Defaults to None.
            type_annotation (str, optional): WGSL type annotation. Defaults to None.
        """
        self.kind = kind
        self.name = name
        self.value = value
        self.type_annotation = type_annotation

    def validate(self) -> None:
        """Validate variable declaration against WebGPU specifications.

        Raises:
            WGSLValidationError: If kind is invalid, name is invalid, or value is invalid.
        """
        if self.kind not in self.VALID_KINDS:
            raise WGSLValidationError(f"Invalid declaration kind '{self.kind}', expected one of {self.VALID_KINDS}")
        validate_identifier(self.name)
        if isinstance(self.value, WGSLNode):
            self.value.validate()


class WGSLIf(WGSLNode):
    """WGSL If Statement."""

    def __init__(self, condition: Union[str, "WGSLNode"], body: list["WGSLNode"]) -> None:
        """Initialize WGSLIf.

        Args:
            condition (Union[str, WGSLNode]): The condition expression.
            body (list[WGSLNode]): List of body statements.
        """
        self.condition = condition
        self.body = body

    def validate(self) -> None:
        """Validate if statement condition and body statements.

        Raises:
            WGSLValidationError: If condition or any body statement is invalid.
        """
        if isinstance(self.condition, WGSLNode):
            self.condition.validate()
        for stmt in self.body:
            if isinstance(stmt, WGSLNode):
                stmt.validate()


class WGSLFor(WGSLNode):
    """WGSL For Loop."""

    def __init__(self, init: Union[str, "WGSLNode"], cond: Union[str, "WGSLNode"], step: Union[str, "WGSLNode"], body: list["WGSLNode"]) -> None:
        """Initialize WGSLFor.

        Args:
            init (Union[str, WGSLNode]): Loop initialization statement.
            cond (Union[str, WGSLNode]): Loop continuation condition.
            step (Union[str, WGSLNode]): Loop step update statement.
            body (list[WGSLNode]): List of body statements.
        """
        self.init = init
        self.cond = cond
        self.step = step
        self.body = body

    def validate(self) -> None:
        """Validate for loop statements.

        Raises:
            WGSLValidationError: If init, cond, step, or body statements are invalid.
        """
        if isinstance(self.init, WGSLNode):
            self.init.validate()
        if isinstance(self.cond, WGSLNode):
            self.cond.validate()
        if isinstance(self.step, WGSLNode):
            self.step.validate()
        for stmt in self.body:
            if isinstance(stmt, WGSLNode):
                stmt.validate()


class WGSLFunction(WGSLNode):
    """WGSL Function Definition."""

    def __init__(self, name: str, params: list[str], body: list["WGSLNode"], attrs: Optional[list[str]] = None) -> None:
        """Initialize WGSLFunction.

        Args:
            name (str): Function name.
            params (list[str]): Function parameter declarations.
            body (list[WGSLNode]): List of body statements.
            attrs (list[str], optional): Function attributes. Defaults to None.
        """
        self.name = name
        self.params = params
        self.body = body
        self.attrs = attrs or []

    def validate(self) -> None:
        """Validate function attributes, name, and body statements.

        Raises:
            WGSLValidationError: If function name or workgroup size attributes violate WebGPU specifications.
        """
        validate_identifier(self.name)
        for attr in self.attrs:
            match = WORKGROUP_SIZE_PATTERN.search(attr)
            if match:
                x = int(match.group(1))
                y = int(match.group(2)) if match.group(2) else 1
                z = int(match.group(3)) if match.group(3) else 1
                validate_workgroup_size(x, y, z)
        for stmt in self.body:
            if isinstance(stmt, WGSLNode):
                stmt.validate()


class WGSLEmitter:
    """Emits WGSL code from AST and validates specifications."""

    def __init__(self) -> None:
        """Initialize WGSLEmitter."""
        self.indent = 0

    def validate(self, node: Union[str, WGSLNode]) -> None:
        """Validate an AST node against WebGPU specifications.

        Args:
            node (Union[str, WGSLNode]): AST node to validate.

        Raises:
            WGSLValidationError: If node violates WebGPU specifications.
        """
        if isinstance(node, WGSLNode):
            node.validate()

    def emit(self, node: Union[str, WGSLNode], validate: bool = True) -> str:
        """Emit WGSL string for a node with specification validation.

        Args:
            node (Union[str, WGSLNode]): The node to emit.
            validate (bool, optional): Whether to validate the node. Defaults to True.

        Returns:
            str: Generated WGSL code.

        Raises:
            WGSLValidationError: If validate is True and node violates specifications.
        """
        if validate and isinstance(node, WGSLNode):
            node.validate()

        if isinstance(node, str):
            return node
        if isinstance(node, WGSLRaw):
            return node.code
        if isinstance(node, WGSLBinding):
            qual: str = f"<{node.address_space}, {node.access_mode}>" if node.access_mode else f"<{node.address_space}>"
            return f"@group({node.group}) @binding({node.binding}) var{qual} {node.name}: {node.type_str};"
        if isinstance(node, WGSLVar):
            return node.name
        elif isinstance(node, WGSLIndex):
            return f"{node.buffer}[{self.emit(node.index, validate=validate)}]"
        elif isinstance(node, WGSLBinaryOp):
            return f"{self.emit(node.left, validate=validate)} {node.op} {self.emit(node.right, validate=validate)}"
        elif isinstance(node, WGSLUnaryOp):
            return f"{node.op}{self.emit(node.expr, validate=validate)}"
        elif isinstance(node, WGSLAssign):
            return f"{self.emit(node.target, validate=validate)} = {self.emit(node.value, validate=validate)};"
        elif isinstance(node, WGSLDecl):
            type_str: str = f": {node.type_annotation}" if node.type_annotation else ""
            val_str: str = f" = {self.emit(node.value, validate=validate)}" if node.value else ""
            return f"{node.kind} {node.name}{type_str}{val_str};"
        elif isinstance(node, WGSLIf):
            lines: list[str] = [f"if ({self.emit(node.condition, validate=validate)}) {{"]
            self.indent += 2
            for stmt in node.body:
                lines.append(" " * self.indent + self.emit(stmt, validate=validate))
            self.indent -= 2
            lines.append(" " * self.indent + "}")
            return "\n".join(lines)
        elif isinstance(node, WGSLFor):
            lines = [f"for ({self.emit(node.init, validate=validate)} {self.emit(node.cond, validate=validate)}; {self.emit(node.step, validate=validate)}) {{"]
            self.indent += 2
            for stmt in node.body:
                lines.append(" " * self.indent + self.emit(stmt, validate=validate))
            self.indent -= 2
            lines.append(" " * self.indent + "}")
            return "\n".join(lines)
        elif isinstance(node, WGSLFunction):
            lines = []
            for attr in node.attrs:
                lines.append(attr)
            params_str: str = ", ".join(node.params)
            lines.append(f"fn {node.name}({params_str}) {{")
            self.indent += 2
            for stmt in node.body:
                lines.append(" " * self.indent + self.emit(stmt, validate=validate))
            self.indent -= 2
            lines.append("}")
            return "\n".join(lines)
        return ""
