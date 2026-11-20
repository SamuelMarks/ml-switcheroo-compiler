"""AST slice parsing and indexed array mutation helpers for the IR interpreter."""

from __future__ import annotations

import ast
import builtins
import sys
import typing
from types import ModuleType

from ml_switcheroo_compiler.core.shape import OpValue, Scalar, SliceProtocol
from ml_switcheroo_compiler.interpreter.environment import InterpreterValue, as_eval_value

if typing.TYPE_CHECKING:
    if sys.version_info >= (3, 10):  # noqa: UP036
        from types import EllipsisType as _EllipsisType
    else:
        from builtins import ellipsis as _EllipsisType

SliceKey = typing.Union[  # noqa: UP007
    Scalar,
    str,
    None,
    SliceProtocol,
    "_EllipsisType",
    list["SliceKey"],
    tuple["SliceKey", ...],
]


@typing.runtime_checkable
class _EvalSliceableProtocol(typing.Protocol):
    """Protocol for evaluated arrays supporting index/slice lookup."""

    @property
    def ndim(self) -> int:
        """Return array rank.

        Returns:
            int: Number of dimensions.
        """
        ...

    def __getitem__(self, key: SliceKey) -> OpValue:
        """Index array by slice or key.

        Args:
            key (SliceKey): Slice or index specification.

        Returns:
            OpValue: Indexed array view or element.
        """
        ...


@typing.runtime_checkable
class _EvalSettableProtocol(typing.Protocol):
    """Protocol for evaluated arrays supporting in-place slice/index mutation."""

    @property
    def ndim(self) -> int:
        """Return array rank.

        Returns:
            int: Number of dimensions.
        """
        ...

    def __setitem__(self, key: SliceKey, val: InterpreterValue) -> None:
        """Assign ``val`` into ``key`` slice/index.

        Args:
            key (SliceKey): Slice or index specification.
            val (InterpreterValue): Update value.
        """
        ...


def _as_eval_sliceable(arr: OpValue | InterpreterValue | _EvalSliceableProtocol) -> _EvalSliceableProtocol:
    """Verify and narrow ``arr`` to ``_EvalSliceableProtocol``.

    Args:
        arr (OpValue | InterpreterValue | _EvalSliceableProtocol): Candidate array.

    Returns:
        _EvalSliceableProtocol: Verified sliceable array.
    """
    assert isinstance(arr, _EvalSliceableProtocol) and not isinstance(arr, ModuleType)
    return arr


def _as_eval_settable(arr: OpValue | InterpreterValue | _EvalSettableProtocol) -> _EvalSettableProtocol:
    """Verify and narrow ``arr`` to ``_EvalSettableProtocol``.

    Args:
        arr (OpValue | InterpreterValue | _EvalSettableProtocol): Candidate mutable array.

    Returns:
        _EvalSettableProtocol: Verified mutable array.
    """
    assert isinstance(arr, _EvalSettableProtocol) and not isinstance(arr, ModuleType)
    return arr


def _eval_getitem(arr: OpValue | InterpreterValue | _EvalSliceableProtocol, key: SliceKey) -> InterpreterValue:
    """Index an evaluated backend array with ``key``.

    Args:
        arr (OpValue | InterpreterValue | _EvalSliceableProtocol): Target array.
        key (SliceKey): Parsed slice or index key.

    Returns:
        InterpreterValue: Indexed result.
    """
    return as_eval_value(_as_eval_sliceable(arr)[key])


def _eval_setitem(
    arr: OpValue | InterpreterValue | _EvalSettableProtocol, key: SliceKey, val: InterpreterValue
) -> None:
    """Mutate an evaluated backend array at ``key`` with ``val``.

    Args:
        arr (OpValue | InterpreterValue | _EvalSettableProtocol): Target cloned array.
        key (SliceKey): Parsed slice or index key.
        val (InterpreterValue): Replacement value.
    """
    _as_eval_settable(arr)[key] = val


def _eval_to_int(val: OpValue | InterpreterValue, default: int = 0) -> int:
    """Coerce an evaluation value or dimension to int.

    Args:
        val (OpValue | InterpreterValue): Raw integer or dimension value.
        default (int): Default fallback integer.

    Returns:
        int: Coerced integer value.
    """
    return int(val) if isinstance(val, (int, float)) else default


def _parse_slice_call(node: ast.Call) -> SliceKey:
    """Parse a slice call AST node into a Python slice or sequence key.

    Args:
        node (ast.Call): The Call AST node.

    Returns:
        SliceKey: The parsed slice specification.
    """
    if not isinstance(node.func, ast.Name):
        return None
    if node.func.id == "slice":
        slice_args: list[int | float | str | None] = [
            v if isinstance((v := _parse_slice_node(arg)), (int, float, str)) else None for arg in node.args
        ]
        return builtins.slice(*slice_args)
    if node.func.id == "array":
        return _parse_slice_node(node.args[0])
    return None


def _parse_tuple(node: ast.Tuple) -> tuple[SliceKey, ...]:
    """Parse a tuple AST node into a tuple of slice keys.

    Args:
        node (ast.Tuple): The Tuple AST node.

    Returns:
        tuple[SliceKey, ...]: The parsed tuple object.
    """
    return tuple(_parse_slice_node(elt) for elt in node.elts)


def _parse_list(node: ast.List) -> list[SliceKey]:
    """Parse a list AST node into a list of slice keys.

    Args:
        node (ast.List): The List AST node.

    Returns:
        list[SliceKey]: The parsed list object.
    """
    return [_parse_slice_node(elt) for elt in node.elts]


def _parse_constant(node: ast.Constant) -> SliceKey:
    """Parse a constant AST node into a primitive slice key.

    Args:
        node (ast.Constant): The Constant AST node.

    Returns:
        SliceKey: The parsed constant value.
    """
    val = node.value
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace")
    assert val is Ellipsis or val is None or isinstance(val, (int, float, str, bool))
    return val


def _parse_name(node: ast.Name) -> SliceKey:
    """Parse a name AST node into standard slice singletons.

    Args:
        node (ast.Name): The Name AST node.

    Returns:
        SliceKey: The parsed singleton (`None`, `Ellipsis`, `False`, `True`).
    """
    name_map: dict[str, SliceKey] = {"None": None, "Ellipsis": Ellipsis, "False": False, "True": True}
    return name_map.get(node.id)


def _parse_unary(node: ast.UnaryOp) -> SliceKey:
    """Parse a unary operation AST node for negative slice bounds.

    Args:
        node (ast.UnaryOp): The UnaryOp AST node.

    Returns:
        SliceKey: Negated numeric literal or `None`.
    """
    if (
        isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, (int, float))
    ):
        return -node.operand.value
    return None


def _parse_slice_node(node: ast.AST) -> SliceKey:
    """Dispatch and parse an AST node for a slice representation into native Python types.

    Args:
        node (ast.AST): The parsed AST node representing a portion of a slice index.

    Returns:
        SliceKey: The interpreted Python slice, tuple, list, or constant value.

    Raises:
        ValueError: If the AST node type is not supported in slice expressions.
    """
    if isinstance(node, (ast.Tuple, ast.List)):
        return _parse_tuple(node) if isinstance(node, ast.Tuple) else _parse_list(node)
    if isinstance(node, ast.Constant):
        return _parse_constant(node)
    if isinstance(node, ast.Name):
        return _parse_name(node)
    if isinstance(node, ast.UnaryOp):
        return _parse_unary(node)
    if isinstance(node, ast.Call):
        return _parse_slice_call(node)
    msg = f"Unsupported slice expression node: {type(node)}"
    raise ValueError(msg)


def _parse_slice_string(s: str) -> SliceKey:
    """Parse a string representation of a slice index.

    Args:
        s (str): The slice string to evaluate.

    Returns:
        SliceKey: The parsed Python slice object.
    """
    return _parse_slice_node(ast.parse(str(s), mode="eval").body)
