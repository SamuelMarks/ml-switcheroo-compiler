"""Safe AST-walking expression evaluator for declarative YAML custom_code mappings."""

from __future__ import annotations

import ast

from ml_switcheroo_compiler.backends.eager_registry import (
    EagerTensorProtocol,
    ensure_callable,
    ensure_eager_tensor,
)
from ml_switcheroo_compiler.backends.protocols import get_backend_member, has_backend_member
from ml_switcheroo_compiler.core.shape import AttrValue as AttributeValue

_SAFE_BUILTIN_NAMES: frozenset[str] = frozenset({"abs", "bool", "float", "hasattr", "int", "len", "list", "max", "min", "range", "slice", "str", "sum", "tuple"})


def _apply_multiplicative_op(op: ast.operator, lhs: EagerTensorProtocol, rhs: EagerTensorProtocol) -> EagerTensorProtocol:
    """Apply multiplicative, modulo, or power AST binary operators.

    Args:
        op (ast.operator): Binary operator AST node.
        lhs (EagerTensorProtocol): Evaluated left operand.
        rhs (EagerTensorProtocol): Evaluated right operand.

    Returns:
        EagerTensorProtocol: Computed binary operation result.

    Raises:
        ValueError: If `op` is not in the allowlisted arithmetic operators.
    """
    if isinstance(op, ast.Mult):
        return lhs * rhs
    if isinstance(op, ast.Div):
        return lhs / rhs
    if isinstance(op, ast.FloorDiv):
        return lhs // rhs
    if isinstance(op, ast.Mod):
        return lhs % rhs
    if isinstance(op, ast.Pow):
        return lhs**rhs
    raise ValueError(f"Unsupported binary operator: {type(op).__name__}")


def _apply_bin_op(op: ast.operator, lhs: EagerTensorProtocol, rhs: EagerTensorProtocol) -> EagerTensorProtocol:
    """Apply an allowlisted AST binary operator to two evaluated operands.

    Args:
        op (ast.operator): Binary operator AST node.
        lhs (EagerTensorProtocol): Evaluated left operand.
        rhs (EagerTensorProtocol): Evaluated right operand.

    Returns:
        EagerTensorProtocol: Computed binary operation result.
    """
    if isinstance(op, ast.Add):
        return lhs + rhs
    if isinstance(op, ast.Sub):
        return lhs - rhs
    return _apply_multiplicative_op(op, lhs, rhs)


def _apply_unary_op(op: ast.unaryop, val: EagerTensorProtocol) -> EagerTensorProtocol:
    """Apply an allowlisted AST unary operator to an evaluated operand.

    Args:
        op (ast.unaryop): Unary operator AST node.
        val (EagerTensorProtocol): Evaluated operand.

    Returns:
        EagerTensorProtocol: Computed unary operation result.

    Raises:
        ValueError: If `op` is not in the allowlisted unary operators.
    """
    if isinstance(op, ast.UAdd):
        return +val
    if isinstance(op, ast.USub):
        return -val
    if isinstance(op, ast.Not):
        return ensure_eager_tensor(not bool(val))
    raise ValueError(f"Unsupported unary operator: {type(op).__name__}")


def _apply_cmp_op(op: ast.cmpop, lhs: EagerTensorProtocol, rhs: EagerTensorProtocol) -> EagerTensorProtocol:
    """Apply an allowlisted AST comparison operator to two evaluated operands.

    Args:
        op (ast.cmpop): Comparison operator AST node.
        lhs (EagerTensorProtocol): Evaluated left operand.
        rhs (EagerTensorProtocol): Evaluated right operand.

    Returns:
        EagerTensorProtocol: Boolean or elementwise tensor comparison result.

    Raises:
        ValueError: If `op` is not in the allowlisted comparison operators.
    """
    if isinstance(op, ast.Eq):
        return ensure_eager_tensor(lhs == rhs)
    if isinstance(op, ast.NotEq):
        return ensure_eager_tensor(lhs != rhs)
    if isinstance(op, ast.Lt):
        return lhs < rhs
    if isinstance(op, ast.LtE):
        return lhs <= rhs
    if isinstance(op, ast.Gt):
        return lhs > rhs
    if isinstance(op, ast.GtE):
        return lhs >= rhs
    raise ValueError(f"Unsupported comparison operator: {type(op).__name__}")


def _call_seq_builtin(name: str, a0: EagerTensorProtocol) -> EagerTensorProtocol:
    """Execute single-operand sequence conversion or summation builtins.

    Args:
        name (str): Builtin name (`len`, `tuple`, `list`, `sum`).
        a0 (EagerTensorProtocol): Sequence operand.

    Returns:
        EagerTensorProtocol: Computed sequence result.
    """
    if name == "len":
        return ensure_eager_tensor(len(a0))
    if name == "tuple":
        return ensure_eager_tensor(tuple(x for x in a0))
    if name == "list":
        return ensure_eager_tensor([x for x in a0])
    acc = ensure_eager_tensor(0)
    for elem in a0:
        acc = acc + elem
    return acc


def _call_numeric_or_seq_builtin(name: str, a0: EagerTensorProtocol) -> EagerTensorProtocol:
    """Execute single-operand scalar or container conversion builtins.

    Args:
        name (str): Builtin name (`len`, `bool`, `int`, `float`, `str`, `abs`, `tuple`, `list`, `sum`).
        a0 (EagerTensorProtocol): First operand.

    Returns:
        EagerTensorProtocol: Converted result.
    """
    if name == "bool":
        return ensure_eager_tensor(bool(a0))
    if name == "int":
        return ensure_eager_tensor(int(a0))
    if name == "float":
        return ensure_eager_tensor(float(a0))
    if name == "str":
        return ensure_eager_tensor(str(a0))
    if name == "abs":
        return abs(a0)
    return _call_seq_builtin(name, a0)


def _call_safe_builtin(name: str, pos_args: list[EagerTensorProtocol]) -> EagerTensorProtocol:
    """Execute an allowlisted Python builtin by name without exposing `__builtins__`.

    Args:
        name (str): Allowlisted builtin function name.
        pos_args (list[EagerTensorProtocol]): Evaluated positional arguments.

    Returns:
        EagerTensorProtocol: Result of the builtin operation.
    """
    if name in ("len", "bool", "int", "float", "str", "abs", "tuple", "list", "sum"):
        return _call_numeric_or_seq_builtin(name, pos_args[0])
    if name == "range":
        int_args = [int(x) for x in pos_args]
        r_obj = range(int_args[0]) if len(int_args) == 1 else (range(int_args[0], int_args[1]) if len(int_args) == 2 else range(int_args[0], int_args[1], int_args[2]))
        return ensure_eager_tensor(r_obj)
    if name == "slice":
        sl_obj = slice(pos_args[0]) if len(pos_args) == 1 else (slice(pos_args[0], pos_args[1]) if len(pos_args) == 2 else slice(pos_args[0], pos_args[1], pos_args[2]))
        return ensure_eager_tensor(sl_obj)
    if name == "hasattr":
        attr_name = str(pos_args[1])
        return ensure_eager_tensor(not attr_name.startswith("_") and has_backend_member(pos_args[0], attr_name))
    items = [x for x in pos_args[0]] if len(pos_args) == 1 else pos_args
    best = items[0]
    for item in items[1:]:
        if (name == "max" and bool(item > best)) or (name == "min" and bool(item < best)):
            best = item
    return best


def _bind_lambda_scope(
    node: ast.Lambda,
    closure_env: dict[str, EagerTensorProtocol],
    default_vals: list[EagerTensorProtocol],
    call_args: tuple[AttributeValue, ...],
    call_kwargs: dict[str, AttributeValue],
) -> dict[str, EagerTensorProtocol]:
    """Bind positional, default, vararg, and keyword arguments for an AST Lambda invocation.

    Args:
        node (ast.Lambda): Parsed AST Lambda node.
        closure_env (dict[str, EagerTensorProtocol]): Captured lexical scope.
        default_vals (list[EagerTensorProtocol]): Pre-evaluated positional default values.
        call_args (tuple[AttributeValue, ...]): Runtime positional arguments.
        call_kwargs (dict[str, AttributeValue]): Runtime keyword arguments.

    Returns:
        dict[str, EagerTensorProtocol]: Populated local execution scope.

    Raises:
        TypeError: If required positional arguments are missing or unexpected kwargs are passed.
    """
    local_scope = dict(closure_env)
    pos_names = [a.arg for a in node.args.args]
    num_non_default = len(pos_names) - len(default_vals)
    remaining_kw: dict[str, EagerTensorProtocol] = {k: ensure_eager_tensor(v) for k, v in call_kwargs.items()}

    for idx, p_name in enumerate(pos_names):
        if idx < len(call_args):
            local_scope[p_name] = ensure_eager_tensor(call_args[idx])
        elif p_name in remaining_kw:
            local_scope[p_name] = remaining_kw.pop(p_name)
        elif idx >= num_non_default:
            local_scope[p_name] = default_vals[idx - num_non_default]
        else:
            raise TypeError(f"Missing required argument: {p_name}")

    if node.args.vararg is not None:
        local_scope[node.args.vararg.arg] = ensure_eager_tensor(tuple(call_args[len(pos_names) :]))
    elif len(call_args) > len(pos_names):
        raise TypeError("Too many positional arguments for lambda")

    if node.args.kwarg is not None:
        local_scope[node.args.kwarg.arg] = ensure_eager_tensor(remaining_kw)
    elif remaining_kw:
        raise TypeError(f"Unexpected keyword arguments: {sorted(remaining_kw.keys())}")
    return local_scope


def _build_lambda(node: ast.Lambda, env: dict[str, EagerTensorProtocol]) -> EagerTensorProtocol:
    """Construct a safe Python callable from an `ast.Lambda` node.

    Args:
        node (ast.Lambda): Lambda AST node.
        env (dict[str, EagerTensorProtocol]): Enclosing scope mapping.

    Returns:
        EagerTensorProtocol: Callable evaluating `node.body` on invocation.
    """
    default_vals = [_eval_node(d, env) for d in node.args.defaults]
    captured = dict(env)

    def _invoker(*args: AttributeValue, **kwargs: AttributeValue) -> EagerTensorProtocol:
        """Execute the AST-interpreted lambda body.

        Args:
            *args (AttributeValue): Positional call arguments.
            **kwargs (AttributeValue): Keyword call arguments.

        Returns:
            EagerTensorProtocol: Evaluated expression result.
        """
        scope = _bind_lambda_scope(node, captured, default_vals, args, kwargs)
        return _eval_node(node.body, scope)

    return ensure_eager_tensor(_invoker)


def _eval_call(node: ast.Call, env: dict[str, EagerTensorProtocol]) -> EagerTensorProtocol:
    """Evaluate a safe `ast.Call` node with positional, starred, and keyword arguments.

    Args:
        node (ast.Call): Call AST node.
        env (dict[str, EagerTensorProtocol]): Current variable scope.

    Returns:
        EagerTensorProtocol: Result of invoking the resolved callable.
    """
    pos_args: list[EagerTensorProtocol] = []
    for arg in node.args:
        if isinstance(arg, ast.Starred):
            for elem in _eval_node(arg.value, env):
                pos_args.append(elem)
        else:
            pos_args.append(_eval_node(arg, env))
    if isinstance(node.func, ast.Name) and node.func.id not in env and node.func.id in _SAFE_BUILTIN_NAMES and not node.keywords:
        return _call_safe_builtin(node.func.id, pos_args)
    fn = ensure_callable(_eval_node(node.func, env))
    kw_args: dict[str, EagerTensorProtocol] = {}
    for kw in node.keywords:
        val = _eval_node(kw.value, env)
        if kw.arg is None:
            for mk_k in val:
                kw_args[str(mk_k)] = val[mk_k]
        else:
            kw_args[kw.arg] = val
    return fn(*pos_args, **kw_args)


def _eval_comprehension(
    elt: ast.expr,
    generators: list[ast.comprehension],
    env: dict[str, EagerTensorProtocol],
) -> list[EagerTensorProtocol]:
    """Evaluate a single-clause list or generator comprehension safely.

    Args:
        elt (ast.expr): Element expression AST node.
        generators (list[ast.comprehension]): Comprehension generator clauses.
        env (dict[str, EagerTensorProtocol]): Enclosing variable scope.

    Returns:
        list[EagerTensorProtocol]: Evaluated comprehension elements.

    Raises:
        ValueError: If unsupported multi-target or multi-clause comprehensions are encountered.
    """
    if len(generators) != 1 or not isinstance(generators[0].target, ast.Name) or generators[0].ifs:
        raise ValueError("Unsupported comprehension structure in custom_code")
    gen = generators[0]
    target_node = gen.target
    assert isinstance(target_node, ast.Name)
    target_name = target_node.id
    iterable = _eval_node(gen.iter, env)
    results: list[EagerTensorProtocol] = []
    for item in iterable:
        inner_env = dict(env)
        inner_env[target_name] = item
        results.append(_eval_node(elt, inner_env))
    return results


def _eval_container_or_access(node: ast.AST, env: dict[str, EagerTensorProtocol]) -> EagerTensorProtocol:
    """Evaluate attribute access, subscripting, slices, tuples, lists, or comprehensions.

    Args:
        node (ast.AST): Expression AST node.
        env (dict[str, EagerTensorProtocol]): Current variable scope.

    Returns:
        EagerTensorProtocol: Evaluated container or member value.

    Raises:
        ValueError: If private attributes or unsupported AST nodes are encountered.
    """
    if isinstance(node, ast.Attribute):
        if node.attr.startswith("_"):
            raise ValueError(f"Disallowed private attribute access: {node.attr}")
        owner = _eval_node(node.value, env)
        return ensure_eager_tensor(get_backend_member(owner, node.attr))
    if isinstance(node, ast.Subscript):
        target = _eval_node(node.value, env)
        idx = _eval_node(node.slice, env)
        return target[idx]
    if isinstance(node, ast.Slice):
        low = _eval_node(node.lower, env) if node.lower is not None else None
        up = _eval_node(node.upper, env) if node.upper is not None else None
        step = _eval_node(node.step, env) if node.step is not None else None
        return ensure_eager_tensor(slice(low, up, step))
    if isinstance(node, ast.Tuple):
        return ensure_eager_tensor(tuple(_eval_node(e, env) for e in node.elts))
    if isinstance(node, ast.List):
        return ensure_eager_tensor([_eval_node(e, env) for e in node.elts])
    if isinstance(node, (ast.ListComp, ast.GeneratorExp)):
        items = _eval_comprehension(node.elt, node.generators, env)
        return ensure_eager_tensor(items) if isinstance(node, ast.ListComp) else ensure_eager_tensor(iter(items))
    raise ValueError(f"Unsupported AST expression node: {type(node).__name__}")


def _eval_operator_expr(node: ast.AST, env: dict[str, EagerTensorProtocol]) -> EagerTensorProtocol:
    """Evaluate binary, unary, boolean, conditional, or comparison AST expressions.

    Args:
        node (ast.AST): Operator expression AST node.
        env (dict[str, EagerTensorProtocol]): Current variable scope.

    Returns:
        EagerTensorProtocol: Computed result.
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
        return ensure_eager_tensor(res)
    if isinstance(node, ast.IfExp):
        branch = node.body if bool(_eval_node(node.test, env)) else node.orelse
        return _eval_node(branch, env)
    return _eval_container_or_access(node, env)


def _eval_node(node: ast.AST, env: dict[str, EagerTensorProtocol]) -> EagerTensorProtocol:
    """Recursively evaluate an allowlisted Python expression AST node without `eval()`.

    Args:
        node (ast.AST): Parsed expression AST node.
        env (dict[str, EagerTensorProtocol]): Bound identifier environment.

    Returns:
        EagerTensorProtocol: Evaluated expression result.

    Raises:
        NameError: If an unbound or private identifier is referenced.
    """
    if isinstance(node, ast.Constant):
        return ensure_eager_tensor(node.value)
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
    narrowed_env: dict[str, EagerTensorProtocol] = {k: ensure_eager_tensor(v) for k, v in env.items()}
    return _eval_node(parsed.body, narrowed_env)
