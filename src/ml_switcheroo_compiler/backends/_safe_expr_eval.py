"""Safe AST-walking expression evaluator for declarative YAML custom_code mappings."""

from __future__ import annotations

import ast
import operator
from collections.abc import Sequence

from ml_switcheroo_compiler.backends.eager_registry import (
    EagerBoundCallable,
    coerce_float,
    coerce_int,
    coerce_operand_list,
    coerce_opt_int,
    require_callable,
)
from ml_switcheroo_compiler.backends.protocols import adapt_backend_module
from ml_switcheroo_compiler.core.shape import OpValue as AttributeValue
from ml_switcheroo_compiler.core.shape import SliceProtocol

_SAFE_BUILTIN_NAMES: frozenset[str] = frozenset(
    {"abs", "bool", "float", "hasattr", "int", "len", "list", "max", "min", "range", "slice", "str", "sum", "tuple"}
)

_BIN_OP_CALLABLES: dict[type[ast.operator], EagerBoundCallable] = {
    ast.Add: require_callable(operator.add),
    ast.Sub: require_callable(operator.sub),
    ast.Mult: require_callable(operator.mul),
    ast.Div: require_callable(operator.truediv),
    ast.FloorDiv: require_callable(operator.floordiv),
    ast.Mod: require_callable(operator.mod),
    ast.Pow: require_callable(operator.pow),
}

_CMP_OP_CALLABLES: dict[type[ast.cmpop], EagerBoundCallable] = {
    ast.Eq: require_callable(operator.eq),
    ast.NotEq: require_callable(operator.ne),
    ast.Lt: require_callable(operator.lt),
    ast.LtE: require_callable(operator.le),
    ast.Gt: require_callable(operator.gt),
    ast.GtE: require_callable(operator.ge),
}

_UADD_FN: EagerBoundCallable = require_callable(operator.pos)
_USUB_FN: EagerBoundCallable = require_callable(operator.neg)
_ABS_FN: EagerBoundCallable = require_callable(operator.abs)
_GETITEM_FN: EagerBoundCallable = require_callable(operator.getitem)
_LEN_FN: EagerBoundCallable = require_callable(len)


def _apply_bin_op(op: ast.operator, lhs: AttributeValue, rhs: AttributeValue) -> AttributeValue:
    """Apply an allowlisted AST binary operator to two evaluated operands.

    Args:
        op (ast.operator): Binary operator AST node.
        lhs (AttributeValue): Evaluated left operand.
        rhs (AttributeValue): Evaluated right operand.

    Returns:
        AttributeValue: Computed binary operation result.

    Raises:
        ValueError: If `op` is not in the allowlisted binary operators.
    """
    fn = _BIN_OP_CALLABLES.get(type(op))
    if fn is not None:
        return fn(lhs, rhs)
    raise ValueError(f"Unsupported binary operator: {type(op).__name__}")


def _apply_unary_op(op: ast.unaryop, val: AttributeValue) -> AttributeValue:
    """Apply an allowlisted AST unary operator to an evaluated operand.

    Args:
        op (ast.unaryop): Unary operator AST node.
        val (AttributeValue): Evaluated operand.

    Returns:
        AttributeValue: Computed unary operation result.

    Raises:
        ValueError: If `op` is not in the allowlisted unary operators.
    """
    if isinstance(op, ast.UAdd):
        return _UADD_FN(val)
    if isinstance(op, ast.USub):
        return _USUB_FN(val)
    if isinstance(op, ast.Not):
        return not bool(val)
    raise ValueError(f"Unsupported unary operator: {type(op).__name__}")


def _apply_cmp_op(op: ast.cmpop, lhs: AttributeValue, rhs: AttributeValue) -> AttributeValue:
    """Apply an allowlisted AST comparison operator to two evaluated operands.

    Args:
        op (ast.cmpop): Comparison operator AST node.
        lhs (AttributeValue): Evaluated left operand.
        rhs (AttributeValue): Evaluated right operand.

    Returns:
        AttributeValue: Boolean or elementwise tensor comparison result.

    Raises:
        ValueError: If `op` is not in the allowlisted comparison operators.
    """
    fn = _CMP_OP_CALLABLES.get(type(op))
    if fn is not None:
        return fn(lhs, rhs)
    raise ValueError(f"Unsupported comparison operator: {type(op).__name__}")


def _call_seq_builtin(name: str, a0: AttributeValue) -> AttributeValue:
    """Execute single-operand sequence conversion or summation builtins.

    Args:
        name (str): Builtin name (`len`, `tuple`, `list`, `sum`).
        a0 (AttributeValue): Sequence operand.

    Returns:
        AttributeValue: Computed sequence result.
    """
    if name == "len":
        return coerce_int(_LEN_FN(a0))
    if name == "tuple":
        return tuple(coerce_operand_list(a0))
    if name == "list":
        return coerce_operand_list(a0)
    acc: AttributeValue = 0
    add_fn = _BIN_OP_CALLABLES[ast.Add]
    for elem in coerce_operand_list(a0):
        acc = add_fn(acc, elem)
    return acc


def _call_numeric_or_seq_builtin(name: str, a0: AttributeValue) -> AttributeValue:
    """Execute single-operand scalar or container conversion builtins.

    Args:
        name (str): Builtin name (`len`, `bool`, `int`, `float`, `str`, `abs`, `tuple`, `list`, `sum`).
        a0 (AttributeValue): First operand.

    Returns:
        AttributeValue: Converted result.
    """
    if name == "bool":
        return bool(a0)
    if name == "int":
        return coerce_int(a0)
    if name == "float":
        return coerce_float(a0)
    if name == "str":
        return str(a0)
    if name == "abs":
        return _ABS_FN(a0)
    return _call_seq_builtin(name, a0)


def _call_safe_builtin(name: str, pos_args: list[AttributeValue]) -> AttributeValue:
    """Execute an allowlisted Python builtin by name without exposing `__builtins__`.

    Args:
        name (str): Allowlisted builtin function name.
        pos_args (list[AttributeValue]): Evaluated positional arguments.

    Returns:
        AttributeValue: Result of the builtin operation.
    """
    if name in ("len", "bool", "int", "float", "str", "abs", "tuple", "list", "sum"):
        return _call_numeric_or_seq_builtin(name, pos_args[0])
    if name == "range":
        int_args = [coerce_int(x) for x in pos_args]
        r_obj: Sequence[int] = (
            range(int_args[0])
            if len(int_args) == 1
            else (
                range(int_args[0], int_args[1]) if len(int_args) == 2 else range(int_args[0], int_args[1], int_args[2])
            )
        )
        return r_obj
    if name == "slice":
        opt_args = [coerce_opt_int(x) for x in pos_args]
        sl_obj: SliceProtocol = (
            slice(opt_args[0])
            if len(opt_args) == 1
            else (
                slice(opt_args[0], opt_args[1]) if len(opt_args) == 2 else slice(opt_args[0], opt_args[1], opt_args[2])
            )
        )
        return sl_obj
    if name == "hasattr":
        attr_name = str(pos_args[1])
        return not attr_name.startswith("_") and adapt_backend_module(pos_args[0]).has_symbol(attr_name)
    items = coerce_operand_list(pos_args[0]) if len(pos_args) == 1 else pos_args
    best = items[0]
    gt_fn = _CMP_OP_CALLABLES[ast.Gt]
    lt_fn = _CMP_OP_CALLABLES[ast.Lt]
    for item in items[1:]:
        if (name == "max" and bool(gt_fn(item, best))) or (name == "min" and bool(lt_fn(item, best))):
            best = item
    return best


def _bind_lambda_scope(
    node: ast.Lambda,
    closure_env: dict[str, AttributeValue],
    default_vals: list[AttributeValue],
    call_args: tuple[AttributeValue, ...],
    call_kwargs: dict[str, AttributeValue],
) -> dict[str, AttributeValue]:
    """Bind positional, default, vararg, and keyword arguments for an AST Lambda invocation.

    Args:
        node (ast.Lambda): Parsed AST Lambda node.
        closure_env (dict[str, AttributeValue]): Captured lexical scope.
        default_vals (list[AttributeValue]): Pre-evaluated positional default values.
        call_args (tuple[AttributeValue, ...]): Runtime positional arguments.
        call_kwargs (dict[str, AttributeValue]): Runtime keyword arguments.

    Returns:
        dict[str, AttributeValue]: Populated local execution scope.

    Raises:
        TypeError: If required positional arguments are missing or unexpected kwargs are passed.
    """
    local_scope = dict(closure_env)
    pos_names = [a.arg for a in node.args.args]
    num_non_default = len(pos_names) - len(default_vals)
    remaining_kw: dict[str, AttributeValue] = dict(call_kwargs)

    for idx, p_name in enumerate(pos_names):
        if idx < len(call_args):
            local_scope[p_name] = call_args[idx]
        elif p_name in remaining_kw:
            local_scope[p_name] = remaining_kw.pop(p_name)
        elif idx >= num_non_default:
            local_scope[p_name] = default_vals[idx - num_non_default]
        else:
            raise TypeError(f"Missing required argument: {p_name}")

    if node.args.vararg is not None:
        local_scope[node.args.vararg.arg] = tuple(call_args[len(pos_names) :])
    elif len(call_args) > len(pos_names):
        raise TypeError("Too many positional arguments for lambda")

    if node.args.kwarg is not None:
        local_scope[node.args.kwarg.arg] = remaining_kw
    elif remaining_kw:
        raise TypeError(f"Unexpected keyword arguments: {sorted(remaining_kw.keys())}")
    return local_scope


def _build_lambda(node: ast.Lambda, env: dict[str, AttributeValue]) -> AttributeValue:
    """Construct a safe Python callable from an `ast.Lambda` node.

    Args:
        node (ast.Lambda): Lambda AST node.
        env (dict[str, AttributeValue]): Enclosing scope mapping.

    Returns:
        AttributeValue: Callable evaluating `node.body` on invocation.
    """
    default_vals = [_eval_node(d, env) for d in node.args.defaults]
    captured = dict(env)

    def _invoker(*args: AttributeValue, **kwargs: AttributeValue) -> AttributeValue:
        """Execute the AST-interpreted lambda body.

        Args:
            *args (AttributeValue): Positional call arguments.
            **kwargs (AttributeValue): Keyword call arguments.

        Returns:
            AttributeValue: Evaluated expression result.
        """
        scope = _bind_lambda_scope(node, captured, default_vals, args, kwargs)
        return _eval_node(node.body, scope)

    return _invoker


def _eval_call(node: ast.Call, env: dict[str, AttributeValue]) -> AttributeValue:
    """Evaluate a safe `ast.Call` node with positional, starred, and keyword arguments.

    Args:
        node (ast.Call): Call AST node.
        env (dict[str, AttributeValue]): Current variable scope.

    Returns:
        AttributeValue: Result of invoking the resolved callable.
    """
    pos_args: list[AttributeValue] = []
    for arg in node.args:
        if isinstance(arg, ast.Starred):
            for elem in coerce_operand_list(_eval_node(arg.value, env)):
                pos_args.append(elem)
        else:
            pos_args.append(_eval_node(arg, env))
    if (
        isinstance(node.func, ast.Name)
        and node.func.id not in env
        and node.func.id in _SAFE_BUILTIN_NAMES
        and not node.keywords
    ):
        return _call_safe_builtin(node.func.id, pos_args)
    fn = require_callable(_eval_node(node.func, env))
    kw_args: dict[str, AttributeValue] = {}
    for kw in node.keywords:
        val = _eval_node(kw.value, env)
        if kw.arg is None:
            for mk_k in coerce_operand_list(val):
                kw_args[str(mk_k)] = _GETITEM_FN(val, mk_k)
        else:
            kw_args[kw.arg] = val
    return fn(*pos_args, **kw_args)


def _eval_comprehension(
    elt: ast.expr,
    generators: list[ast.comprehension],
    env: dict[str, AttributeValue],
) -> list[AttributeValue]:
    """Evaluate a single-clause list or generator comprehension safely.

    Args:
        elt (ast.expr): Element expression AST node.
        generators (list[ast.comprehension]): Comprehension generator clauses.
        env (dict[str, AttributeValue]): Enclosing variable scope.

    Returns:
        list[AttributeValue]: Evaluated comprehension elements.

    Raises:
        ValueError: If unsupported multi-target or multi-clause comprehensions are encountered.
    """
    if len(generators) != 1 or not isinstance(generators[0].target, ast.Name) or generators[0].ifs:
        raise ValueError("Unsupported comprehension structure in custom_code")
    gen = generators[0]
    target_node = gen.target
    assert isinstance(target_node, ast.Name)
    target_name = target_node.id
    iterable = coerce_operand_list(_eval_node(gen.iter, env))
    results: list[AttributeValue] = []
    for item in iterable:
        inner_env = dict(env)
        inner_env[target_name] = item
        results.append(_eval_node(elt, inner_env))
    return results


def _eval_container_or_access(node: ast.AST, env: dict[str, AttributeValue]) -> AttributeValue:
    """Evaluate attribute access, subscripting, slices, tuples, lists, or comprehensions.

    Args:
        node (ast.AST): Expression AST node.
        env (dict[str, AttributeValue]): Current variable scope.

    Returns:
        AttributeValue: Evaluated container or member value.

    Raises:
        ValueError: If private attributes or unsupported AST nodes are encountered.
    """
    if isinstance(node, ast.Attribute):
        if node.attr.startswith("_"):
            raise ValueError(f"Disallowed private attribute access: {node.attr}")
        owner = _eval_node(node.value, env)
        return adapt_backend_module(owner).require_symbol(node.attr)
    if isinstance(node, ast.Subscript):
        target = _eval_node(node.value, env)
        idx = _eval_node(node.slice, env)
        return _GETITEM_FN(target, idx)
    if isinstance(node, ast.Slice):
        low = coerce_opt_int(_eval_node(node.lower, env)) if node.lower is not None else None
        up = coerce_opt_int(_eval_node(node.upper, env)) if node.upper is not None else None
        step = coerce_opt_int(_eval_node(node.step, env)) if node.step is not None else None
        sl_res: SliceProtocol = slice(low, up, step)
        return sl_res
    if isinstance(node, ast.Tuple):
        return tuple(_eval_node(e, env) for e in node.elts)
    if isinstance(node, ast.List):
        return [_eval_node(e, env) for e in node.elts]
    if isinstance(node, (ast.ListComp, ast.GeneratorExp)):
        items = _eval_comprehension(node.elt, node.generators, env)
        return items if isinstance(node, ast.ListComp) else tuple(items)
    raise ValueError(f"Unsupported AST expression node: {type(node).__name__}")


def _eval_operator_expr(node: ast.AST, env: dict[str, AttributeValue]) -> AttributeValue:
    """Evaluate binary, unary, boolean, conditional, or comparison AST expressions.

    Args:
        node (ast.AST): Operator expression AST node.
        env (dict[str, AttributeValue]): Current variable scope.

    Returns:
        AttributeValue: Computed result.
    """
    if isinstance(node, ast.BinOp):
        return _apply_bin_op(node.op, _eval_node(node.left, env), _eval_node(node.right, env))
    if isinstance(node, ast.UnaryOp):
        return _apply_unary_op(node.op, _eval_node(node.operand, env))
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        return _apply_cmp_op(node.ops[0], _eval_node(node.left, env), _eval_node(node.comparators[0], env))
    if isinstance(node, ast.BoolOp):
        vals = [_eval_node(v, env) for v in node.values]
        res = all(bool(v) for v in vals) if isinstance(node.op, ast.And) else any(bool(v) for v in vals)
        return res
    if isinstance(node, ast.IfExp):
        branch = node.body if bool(_eval_node(node.test, env)) else node.orelse
        return _eval_node(branch, env)
    return _eval_container_or_access(node, env)


def _eval_node(node: ast.AST, env: dict[str, AttributeValue]) -> AttributeValue:
    """Recursively evaluate an allowlisted Python expression AST node without `eval()`.

    Args:
        node (ast.AST): Parsed expression AST node.
        env (dict[str, AttributeValue]): Bound identifier environment.

    Returns:
        AttributeValue: Evaluated expression result.

    Raises:
        NameError: If an unbound or private identifier is referenced.
    """
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id.startswith("_"):
            raise NameError(f"Disallowed private identifier: {node.id}")
        if node.id in env:
            return env[node.id]
        raise NameError(f"Unbound identifier in custom_code: {node.id}")
    if isinstance(node, ast.Lambda):
        return _build_lambda(node, env)
    if isinstance(node, ast.Call):
        return _eval_call(node, env)
    return _eval_operator_expr(node, env)


def evaluate_safe_expression(expr_src: str, env: dict[str, AttributeValue]) -> AttributeValue:
    """Parse and evaluate a declarative `custom_code` expression without Python `eval()`.

    Args:
        expr_src (str): Expression source string (typically `lambda ...: ...`).
        env (dict[str, AttributeValue]): Scoped bindings (e.g., `backend_module`).

    Returns:
        AttributeValue: Evaluated callable or value.
    """
    parsed = ast.parse(expr_src, mode="eval")
    return _eval_node(parsed.body, dict(env))
