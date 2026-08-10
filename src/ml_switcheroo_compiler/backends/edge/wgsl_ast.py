# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""WGSL AST definitions for robust code generation."""

from typing import Any, Union


class WGSLNode:
    """Base WGSL AST Node."""


class WGSLRaw(WGSLNode):
    """Raw string insertion."""

    def __init__(self, code: str):
        self.code = code


class WGSLVar(WGSLNode):
    """WGSL Variable expression."""

    def __init__(self, name: str):
        self.name = name


class WGSLIndex(WGSLNode):
    """WGSL Array/Buffer Indexing."""

    def __init__(self, buffer: str, index: Union[str, "WGSLNode"]):
        self.buffer = buffer
        self.index = index


class WGSLBinaryOp(WGSLNode):
    """WGSL Binary Operation."""

    def __init__(self, op: str, left: Union[str, "WGSLNode"], right: Union[str, "WGSLNode"]):
        self.op = op
        self.left = left
        self.right = right


class WGSLUnaryOp(WGSLNode):
    """WGSL Unary Operation."""

    def __init__(self, op: str, expr: Union[str, "WGSLNode"]):
        self.op = op
        self.expr = expr


class WGSLAssign(WGSLNode):
    """WGSL Assignment Statement."""

    def __init__(self, target: Union[str, "WGSLNode"], value: Union[str, "WGSLNode"]):
        self.target = target
        self.value = value


class WGSLDecl(WGSLNode):
    """WGSL Variable Declaration (let or var)."""

    def __init__(self, kind: str, name: str, value: Any = None, type_annotation: Any = None):
        self.kind = kind
        self.name = name
        self.value = value
        self.type_annotation = type_annotation


class WGSLIf(WGSLNode):
    """WGSL If Statement."""

    def __init__(self, condition: Union[str, "WGSLNode"], body: list["WGSLNode"]):
        self.condition = condition
        self.body = body


class WGSLFor(WGSLNode):
    """WGSL For Loop."""

    def __init__(self, init: Union[str, "WGSLNode"], cond: Union[str, "WGSLNode"], step: Union[str, "WGSLNode"], body: list["WGSLNode"]):
        self.init = init
        self.cond = cond
        self.step = step
        self.body = body


class WGSLFunction(WGSLNode):
    """WGSL Function Definition."""

    def __init__(self, name: str, params: list[str], body: list["WGSLNode"], attrs: Any = None):
        self.name = name
        self.params = params
        self.body = body
        self.attrs = attrs or []


class WGSLEmitter:
    """Emits WGSL code from AST."""

    def __init__(self):  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
        self.indent = 0

    def emit(self, node: Union[str, WGSLNode]) -> str:
        """Emit WGSL string for a node."""
        if isinstance(node, str):
            return node
        if isinstance(node, WGSLRaw):
            return node.code
        if isinstance(node, WGSLVar):
            return node.name
        elif isinstance(node, WGSLIndex):
            return f"{node.buffer}[{self.emit(node.index)}]"
        elif isinstance(node, WGSLBinaryOp):
            return f"{self.emit(node.left)} {node.op} {self.emit(node.right)}"
        elif isinstance(node, WGSLUnaryOp):
            return f"{node.op}{self.emit(node.expr)}"
        elif isinstance(node, WGSLAssign):
            return f"{self.emit(node.target)} = {self.emit(node.value)};"
        elif isinstance(node, WGSLDecl):
            type_str = f": {node.type_annotation}" if node.type_annotation else ""
            val_str = f" = {self.emit(node.value)}" if node.value else ""
            return f"{node.kind} {node.name}{type_str}{val_str};"
        elif isinstance(node, WGSLIf):
            lines = [f"if ({self.emit(node.condition)}) {{"]
            self.indent += 2
            for stmt in node.body:
                lines.append(" " * self.indent + self.emit(stmt))
            self.indent -= 2
            lines.append(" " * self.indent + "}")
            return "\n".join(lines)
        elif isinstance(node, WGSLFor):
            lines = [f"for ({self.emit(node.init)} {self.emit(node.cond)}; {self.emit(node.step)}) {{"]
            self.indent += 2
            for stmt in node.body:
                lines.append(" " * self.indent + self.emit(stmt))
            self.indent -= 2
            lines.append(" " * self.indent + "}")
            return "\n".join(lines)
        elif isinstance(node, WGSLFunction):
            lines = []
            for attr in node.attrs:
                lines.append(attr)
            params_str = ", ".join(node.params)
            lines.append(f"fn {node.name}({params_str}) {{")
            self.indent += 2
            for stmt in node.body:
                lines.append(" " * self.indent + self.emit(stmt))
            self.indent -= 2
            lines.append("}")
            return "\n".join(lines)
        return ""
