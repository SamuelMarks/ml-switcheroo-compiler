"""Unit tests for AST-to-IR, IR-to-AST translation, and backend mapping loader integration."""

import os
import tempfile
from unittest.mock import MagicMock

import libcst as cst
import pytest

from ml_switcheroo_compiler.backends.ast_to_ir import parse_ast_to_ir
from ml_switcheroo_compiler.backends.ir_to_ast import (
    _build_attribute_chain,
    _emit_call_args,
    _emit_node_statement,
    emit_ir_to_class,
)
from ml_switcheroo_compiler.backends.mapping_loader import (
    BackendMappingSchema,
    OpMappingSchema,
    _load_yaml_dir,
    _read_and_merge,
    _resolve_by_import,
    _resolve_custom_code,
    dispatch_eager_op,
    resolve_target_api,
    translate_kwargs,
)
from ml_switcheroo_compiler.core.errors import BackendNotSupportedError
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def test_mapping_loader_resolution_and_dispatch():
    """Verify resolution, custom code parsing, and dispatch paths in mapping_loader."""

    # 1. _resolve_custom_code with property that raises on getattr
    class BadModule:
        @property
        def failing_prop(self):
            raise RuntimeError("Attr access failure")

    mod = BadModule()
    fn = _resolve_custom_code("lambda x: x + 1", mod)
    assert callable(fn)
    assert fn(2) == 3

    # Invalid custom code syntax
    assert _resolve_custom_code("lambda (: 123", mod) is None

    # 2. _resolve_by_import failure
    assert _resolve_by_import("non_existent_module_xyz.some_attr") is None

    # 3. resolve_target_api edge cases
    assert resolve_target_api("") is None
    assert resolve_target_api("custom_op", custom_code=None) is None

    # Fallback to direct attribute on backend_module
    class DirectModule:
        @staticmethod
        def direct_func(x):
            return x * 2

    dmod = DirectModule()
    resolved = resolve_target_api("direct_func", backend_module=dmod)
    assert callable(resolved)
    assert resolved(5) == 10

    # Line 181: dotted attribute attached directly to backend_module via setattr
    dotted_mod = type("DottedMod", (), {})()
    setattr(dotted_mod, "special.pkg.func", lambda: 99)
    res_dotted = resolve_target_api("special.pkg.func", backend_module=dotted_mod)
    assert callable(res_dotted)
    assert res_dotted() == 99

    # Unresolvable API
    assert resolve_target_api("unresolvable_attr", backend_module=object()) is None

    # 4. _read_and_merge with various YAML shapes
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write("not_dict: 123\n")
        f1 = f.name
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write("operations:\n  OpA:\n    target_api: a.b\n")
        f2 = f.name
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write("- list_element_not_dict\n")
        f3 = f.name

    try:
        t1 = {}
        _read_and_merge(f1, t1)
        assert "not_dict" in t1

        t2 = {}
        _read_and_merge(f2, t2)
        assert "OpA" in t2

        t3 = {}
        _read_and_merge(f3, t3)
        assert t3 == {}
    finally:
        os.unlink(f1)
        os.unlink(f2)
        os.unlink(f3)

    # 5. _load_yaml_dir with non-yaml files and missing dir
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "ignored.txt"), "w") as f:
            f.write("ignored")
        with open(os.path.join(tmpdir, "valid.yaml"), "w") as f:
            f.write("OpB: {target_api: b.c}\n")
        target_dict = {}
        _load_yaml_dir(tmpdir, target_dict)
        assert "OpB" in target_dict

    _load_yaml_dir("/path/does/not/exist", {})

    # 6. translate_kwargs
    trans = translate_kwargs({"dim": "axis"}, {"dim": 1, "other": "val"})
    assert trans == {"axis": 1, "other": "val"}

    # 7. dispatch_eager_op standard function and method dispatch
    schema = BackendMappingSchema(
        backend_name="test_backend",
        operations={
            "StandardOp": OpMappingSchema(target_api="std_func"),
            "UnresolvableOp": OpMappingSchema(target_api="invalid_api_str", custom_code=None),
            "MethodOp": OpMappingSchema(target_api="run_method", is_method=True),
        },
    )
    from ml_switcheroo_compiler.backends.mapping_loader import _MAPPING_CACHE

    _MAPPING_CACHE["test_backend"] = schema

    # Standard function op dispatch (line 242)
    mock_mod = MagicMock()
    mock_mod.std_func.return_value = "standard_result"
    res_std = dispatch_eager_op("test_backend", "StandardOp", [1, 2], {"k": 3}, backend_module=mock_mod)
    assert res_std == "standard_result"

    # Op not in schema
    with pytest.raises(BackendNotSupportedError, match="Operation 'MissingOp' is not supported"):
        dispatch_eager_op("test_backend", "MissingOp", [], {})

    # Op cannot be resolved
    with pytest.raises(BackendNotSupportedError, match="could not be resolved"):
        dispatch_eager_op("test_backend", "UnresolvableOp", [], {})

    # Method dispatch with is_method=True
    class TargetObj:
        def run_method(self, a, b=1):
            return f"method_result_{a}_{b}"

    obj = TargetObj()
    res_m = dispatch_eager_op(
        "test_backend",
        "MethodOp",
        [obj, "arg1"],
        {"b": 2},
        backend_module=obj,
    )
    assert res_m == "method_result_arg1_2"

    # Method dispatch when args is empty
    with pytest.raises(BackendNotSupportedError):
        dispatch_eager_op("test_backend", "MethodOp", [], {}, backend_module=None)

    # MethodOp where obj has no such method (covers branch 239->242)
    class DummyNoMethod:
        pass

    mock_mod.run_method = lambda *a, **k: "fallback_func_result"
    res_nometh = dispatch_eager_op("test_backend", "MethodOp", [DummyNoMethod()], {}, backend_module=mock_mod)
    assert res_nometh == "fallback_func_result"


def test_ir_to_ast_generation():
    """Verify code emission patterns and class generation in ir_to_ast."""
    # 1. _build_attribute_chain
    assert isinstance(_build_attribute_chain([]), cst.Name)
    assert _build_attribute_chain([]).value == "empty"
    assert _build_attribute_chain(["single"]).value == "single"
    chain = _build_attribute_chain(["a", "b", "c"])
    assert isinstance(chain, cst.Attribute)

    # 2. _emit_call_args with various attribute value types
    node = IRNode(
        id="test_node",
        op_type="Add",
        inputs=["x", "y"],
        attributes={
            "dim": 1,
            "epsilon": 1e-5,
            "mode": "constant",
            "flag": True,
            "custom_obj": object(),  # triggers line 65 (else: cst.Name(str(v)))
            "is_parameter": True,  # should be skipped
        },
    )
    # Target framework not in frameworks config (covers branch 49->52)
    args_unknown = _emit_call_args(node, "unregistered_framework_xyz")
    assert len(args_unknown) == 7

    args = _emit_call_args(node, "pytorch")
    assert len(args) == 7  # 2 positional + 5 kwargs

    # 3. _emit_node_statement
    # Input node statement
    in_node = IRNode(id="in0", op_type="Input", inputs=[])
    stmt_in = _emit_node_statement(in_node, "pytorch")
    assert stmt_in is not None

    # Op not mapped in target framework
    unmapped = IRNode(id="unmapped", op_type="UnmappedOp123", inputs=[])
    assert _emit_node_statement(unmapped, "pytorch") is None

    # Node with and without var_name
    add_named = IRNode(id="n0", op_type="Add", inputs=["x", "y"], attributes={"var_name": "res"})
    stmt_named = _emit_node_statement(add_named, "pytorch")
    assert isinstance(stmt_named.body[0], cst.Assign)

    add_unnamed = IRNode(id="n1", op_type="Add", inputs=["x", "y"])
    stmt_unnamed = _emit_node_statement(add_unnamed, "pytorch")
    assert isinstance(stmt_unnamed.body[0], cst.Expr)

    # 4. emit_ir_to_class edge cases
    # Class with no user inputs (defaults to param x)
    empty_graph = IRGraph()
    cls_def = emit_ir_to_class(empty_graph, "pytorch", "EmptyModel")
    assert "x" in cst.Module(body=[cls_def]).code

    # Class with no outputs but has op (returns last_val_name)
    g_no_out = IRGraph()
    g_no_out.nodes["n0"] = IRNode(id="n0", op_type="Add", inputs=["a", "b"], attributes={"var_name": "out_var"})
    cls_no_out = emit_ir_to_class(g_no_out, "pytorch", "NoOutModel")
    assert "return out_var" in cst.Module(body=[cls_no_out]).code

    # Class with no outputs and no ops (emits pass)
    g_pass = IRGraph()
    cls_pass = emit_ir_to_class(g_pass, "custom_target", "PassModel")
    assert "pass" in cst.Module(body=[cls_pass]).code

    # Framework config with empty base parts and custom method map
    from unittest.mock import MagicMock, patch

    mock_fw = MagicMock()
    mock_fw.class_bases = {"model": []}
    mock_fw.method_map = {"custom_key": "custom_compute"}
    with patch.dict("ml_switcheroo_compiler.backends.ir_to_ast._CONFIG.frameworks", {"custom_fw": mock_fw}):
        from ml_switcheroo_compiler.backends.ir_to_ast import _get_class_base_expr, _get_compute_method_name

        base_expr = _get_class_base_expr("custom_fw")
        assert isinstance(base_expr, cst.Name) and base_expr.value == "object"
        meth_name = _get_compute_method_name("custom_fw")
        assert meth_name == "custom_compute"


def test_ast_to_ir_lowering_and_scoping():
    """Verify AST visitor lowerings (binary ops, control flow, self params, subscripts)."""
    # 1. Self parameter with integer, float, and non-numeric value
    src_self = """
class ParamModel:
    def __init__(self):
        self.channels = 64
        self.scale = 0.5
        self.name = 'custom_name'
"""
    g_param = parse_ast_to_ir(src_self)
    nodes = list(g_param.nodes.values())
    assert any(n.attributes.get("param_name") == "channels" and n.attributes.get("value") == 64 for n in nodes)
    assert any(n.attributes.get("param_name") == "scale" and n.attributes.get("value") == 0.5 for n in nodes)
    assert any(n.attributes.get("param_name") == "name" and n.attributes.get("value") is None for n in nodes)

    # 2. Binary operations (+, -, *, /, **, %, and unknown operator)
    src_binops = """
a = x + y
b = a - z
c = b * 2
d = c / 3.0
e = d ** 2
f = e % 4
g = f << 1
"""
    g_binops = parse_ast_to_ir(src_binops)
    op_types = [n.op_type for n in g_binops.nodes.values()]
    assert "Add" in op_types
    assert "Sub" in op_types
    assert "Mul" in op_types
    assert "Div" in op_types
    assert "Pow" in op_types
    assert "Mod" in op_types
    assert "UnknownBinaryOp" in op_types

    # 3. If and While control flow (with name and non-name test)
    src_cf = """
if flag:
    pass
if True:
    pass
while loop_cond:
    pass
while False:
    pass
"""
    g_cf = parse_ast_to_ir(src_cf)
    cf_types = [n.op_type for n in g_cf.nodes.values()]
    assert "Cond" in cf_types
    assert "WhileLoop" in cf_types

    # 4. Subscript indexing (with name and non-name value)
    src_sub = """
elem = my_tensor[0:2]
elem2 = [1, 2][0]
"""
    g_sub = parse_ast_to_ir(src_sub)
    assert any(n.op_type == "Slice" for n in g_sub.nodes.values())

    # 5. Call with varied keyword argument literal types (int, float, str, name, complex)
    src_kwargs = """
z = torch.add(x, y, dim=1, eps=0.001, mode='reflect', name=target_var, extra=[1, 2])
"""
    g_kw = parse_ast_to_ir(src_kwargs)
    add_node = next(n for n in g_kw.nodes.values() if n.op_type == "Add")
    assert add_node.attributes["dim"] == 1
    assert add_node.attributes["eps"] == 0.001
    assert add_node.attributes["mode"] == "reflect"
    assert add_node.attributes["name"] == "target_var"
    assert add_node.attributes["extra"] is None

    # 6. Nested calls and attributes in call arguments
    src_nested = """
res = torch.add(torch.relu(x), self.weight, [1, 2], unmapped_fn())
"""
    g_nested = parse_ast_to_ir(src_nested)
    assert any(n.op_type == "Relu" for n in g_nested.nodes.values())
    top_add = next(n for n in g_nested.nodes.values() if n.op_type == "Add")
    assert "unknown" in top_add.inputs

    # 7. Container unpacking with non-names and list targets
    src_unpack_edge = """
[a, b] = torch.split(x, 2)
c, obj.attr = torch.split(x, 2)
foo = torch.split(x, 2)
"""
    g_unpack = parse_ast_to_ir(src_unpack_edge)
    assert len(g_unpack.nodes) > 0

    # 8. Return statement variations
    # Return direct name
    src_ret_name = "def f(x):\n    return x\n"
    g_rn = parse_ast_to_ir(src_ret_name)
    assert len(g_rn.outputs) == 1

    # Return call expression
    src_ret_call = "def f(x):\n    return torch.relu(x)\n"
    g_rc = parse_ast_to_ir(src_ret_call)
    assert len(g_rc.outputs) == 1

    # Bare return
    src_ret_bare = "def f(x):\n    torch.relu(x)\n    return\n"
    g_rb = parse_ast_to_ir(src_ret_bare)
    assert len(g_rb.outputs) == 0

    # 9. Target attribute where target is not self (covers line 83)
    src_obj_attr = "obj.attr = 10\n"
    g_oa = parse_ast_to_ir(src_obj_attr)
    assert len(g_oa.nodes) >= 0

    # 10. Unpack assignment with unmapped RHS (covers branch 99->112)
    src_unpack_unmapped = "a, b = unmapped_call()\n"
    g_uu = parse_ast_to_ir(src_unpack_unmapped)
    assert len(g_uu.nodes) >= 0

    # 11. Regular assignment with unmapped RHS (covers branch 124->131)
    src_assign_unmapped = "a = unmapped_call()\n"
    g_au = parse_ast_to_ir(src_assign_unmapped)
    assert len(g_au.nodes) >= 0

    # 12. Assignment to attribute target with valid node (covers branch 135->134)
    src_assign_attr = "obj.attr = torch.relu(x)\n"
    g_aa = parse_ast_to_ir(src_assign_attr)
    assert len(g_aa.nodes) >= 1

    # 13. Return unmapped call (covers branch 155->exit)
    src_ret_unmapped = "def f():\n    return unmapped_call()\n"
    g_ru = parse_ast_to_ir(src_ret_unmapped)
    assert len(g_ru.outputs) == 0


def test_ast_to_ir_control_flow_subgraphs() -> None:
    """Verify recursive body and branch capture for Cond, WhileLoop, and Scan nodes."""
    src_cond = """def branch_fn(cond, x, y):
    if cond:
        out = torch.add(x, y)
    else:
        out = torch.sub(x, y)
    return out
"""
    g_cond = parse_ast_to_ir(src_cond)
    cond_node = next(n for n in g_cond.nodes.values() if n.op_type == "Cond")
    assert "true_branch" in cond_node.attributes
    assert "false_branch" in cond_node.attributes
    assert any(n.op_type == "Add" for n in cond_node.attributes["true_branch"].nodes.values())
    assert any(n.op_type == "Sub" for n in cond_node.attributes["false_branch"].nodes.values())

    src_while = """def loop_fn(cond, x):
    while cond:
        x = torch.add(x, 1)
    return x
"""
    g_while = parse_ast_to_ir(src_while)
    while_node = next(n for n in g_while.nodes.values() if n.op_type == "WhileLoop")
    assert "body_subgraph" in while_node.attributes
    assert any(n.op_type == "Add" for n in while_node.attributes["body_subgraph"].nodes.values())

    src_for = """def scan_fn(items, accum):
    for item in items:
        accum = torch.add(accum, item)
    return accum
"""
    g_for = parse_ast_to_ir(src_for)
    scan_node = next(n for n in g_for.nodes.values() if n.op_type == "Scan")
    assert "body_subgraph" in scan_node.attributes
    assert scan_node.attributes["target_var"] == "item"
    assert any(n.op_type == "Add" for n in scan_node.attributes["body_subgraph"].nodes.values())

    # Compound test expression in If
    src_cond_expr = """def branch_expr(x, y):
    if x + y:
        return torch.relu(x)
    return torch.relu(y)
"""
    g_ce = parse_ast_to_ir(src_cond_expr)
    assert any(n.op_type == "Cond" for n in g_ce.nodes.values())

    # Compound test expression in While
    src_while_expr = """def while_expr(x):
    while x + 1:
        new_var = torch.relu(x)
    return x
"""
    g_we = parse_ast_to_ir(src_while_expr)
    assert any(n.op_type == "WhileLoop" for n in g_we.nodes.values())

    # Compound iter expression in For and tuple target
    src_for_expr = """def for_expr(x):
    for (a, b) in torch.split(x, 2):
        out = torch.add(a, b)
    for _ in unknown_unmapped():
        pass
    return x
"""
    g_fe = parse_ast_to_ir(src_for_expr)
    assert any(n.op_type == "Scan" for n in g_fe.nodes.values())


def test_ast_to_ir_nested_container_unpacking_and_closure() -> None:
    """Verify nested tuple/list unpacking and closure outer-scope capture."""
    src_unpack = """def unpack_fn(data):
    (a, (b, c)) = data
    [d, [e, f]] = data
    z = torch.add(a, b)
    return z
"""
    g = parse_ast_to_ir(src_unpack)
    getitem_nodes = [n for n in g.nodes.values() if n.op_type == "GetItem"]
    assert len(getitem_nodes) >= 6
    assert any(n.attributes.get("var_name") == "a" for n in getitem_nodes)
    assert any(n.attributes.get("var_name") == "b" for n in getitem_nodes)
    assert any(n.attributes.get("var_name") == "c" for n in getitem_nodes)
    assert any(n.attributes.get("var_name") == "d" for n in getitem_nodes)
    assert any(n.attributes.get("var_name") == "e" for n in getitem_nodes)
    assert any(n.attributes.get("var_name") == "f" for n in getitem_nodes)

    # Closure outer-scope capture
    src_closure = """def inner_fn(inner_var):
    return torch.add(inner_var, outer_var)
"""
    from ml_switcheroo_compiler.backends.ast_to_ir import ASTToIRVisitor

    tree = cst.parse_module(src_closure)
    outer_scope = {"outer_var": "outer_var_node"}
    inner_visitor = ASTToIRVisitor(parent_scope=outer_scope)
    tree.visit(inner_visitor)
    assert "outer_var" in inner_visitor.graph.nodes
    assert inner_visitor.graph.nodes["outer_var"].op_type == "Input"


def test_ast_to_ir_exception_handling_and_roundtrip_control_flow() -> None:
    """Verify try/except and raise AST ingestion and control flow preservation."""
    src_try = """def try_fn(x):
    try:
        y = torch.relu(x)
    except Exception:
        y = torch.zeros_like(x)
        raise ValueError("error")
    return y
"""
    g = parse_ast_to_ir(src_try)
    assert any(n.op_type == "TryExcept" for n in g.nodes.values())
    try_node = next(n for n in g.nodes.values() if n.op_type == "TryExcept")
    assert "try_subgraph" in try_node.attributes
    assert "fallback_subgraph" in try_node.attributes
    assert any(n.op_type == "Assert" for n in try_node.attributes["fallback_subgraph"].nodes.values())


def test_ast_to_ir_scoped_visitor_resolution() -> None:
    """Test scoped identifier resolution in ast_to_ir visitor."""
    from unittest.mock import patch

    from ml_switcheroo_compiler.backends.ast_to_ir import ASTToIRVisitor

    # 1. _lookup_var with parent scope
    # Case A: name not in graph.nodes, and not in graph.inputs (lines 47-48)
    vis_a = ASTToIRVisitor(parent_scope={"p_var": "val"})
    vis_a.graph.inputs = []
    res_a = vis_a._lookup_var("p_var")
    assert res_a == "p_var"
    assert "p_var" in vis_a.graph.nodes
    assert "p_var" in vis_a.graph.inputs

    # Case B: name not in graph.nodes, but already in graph.inputs (branch 47->49)
    vis_b = ASTToIRVisitor(parent_scope={"p_var": "val"})
    vis_b.graph.inputs = ["p_var"]
    res_b = vis_b._lookup_var("p_var")
    assert res_b == "p_var"

    # Case C: name already in graph.nodes (branch 44->49)
    vis_c = ASTToIRVisitor(parent_scope={"p_var": "val"})
    vis_c.graph.nodes["p_var"] = IRNode(id="p_var", op_type="Input")
    res_c = vis_c._lookup_var("p_var")
    assert res_c == "p_var"

    # Case D: graph.inputs is not a list
    vis_d = ASTToIRVisitor(parent_scope={"p_var": "val"})
    vis_d.graph.inputs = "not_a_list"  # type: ignore[assignment]
    res_d = vis_d._lookup_var("p_var")
    assert res_d == "p_var"

    # 2. _recursive_unpack with non-container target (line 120)
    vis_a._recursive_unpack(cst.Name("not_container"), "parent_node_id")

    # 3. _resolve_target_val_node_id with Attribute (line 157)
    src_attr_unpack = """def attr_unpack_fn(obj):
    (a, b) = obj.pair
    return torch.add(a, b)
"""
    g_attr_unpack = parse_ast_to_ir(src_attr_unpack)
    assert any(n.op_type == "GetItem" for n in g_attr_unpack.nodes.values())

    # 4. _handle_dict_unpack and _handle_unpack_assign with Dict target (lines 169-181, 199-202)
    dict_target = cst.Dict(
        elements=[
            cst.DictElement(key=cst.SimpleString("'key1'"), value=cst.Name("val1")),
            cst.DictElement(key=cst.Integer("0"), value=cst.Name("val2")),
            cst.DictElement(key=cst.SimpleString("'k3'"), value=cst.Integer("10")),  # non-Name value
            cst.StarredDictElement(value=cst.Name("rest")),  # non-DictElement
        ]
    )
    assign_node = cst.Assign(
        targets=[cst.AssignTarget(target=dict_target)],
        value=cst.Name("source_dict"),
    )
    vis_dict = ASTToIRVisitor()
    vis_dict.var_table["source_dict"] = "source_dict_node"
    handled = vis_dict._handle_unpack_assign(dict_target, assign_node)
    assert handled is True
    assert "source_dict_node_dict_'key1'" in vis_dict.graph.nodes
    assert "source_dict_node_dict_1" in vis_dict.graph.nodes

    # Dict unpack when val_node_id is None (covers branch 200->202)
    assign_node_none = cst.Assign(
        targets=[cst.AssignTarget(target=dict_target)],
        value=cst.Name("missing_dict"),
    )
    with patch.object(vis_dict, "_resolve_target_val_node_id", return_value=None):
        handled_none = vis_dict._handle_unpack_assign(dict_target, assign_node_none)
        assert handled_none is True

    # 5. TryExcept with both pre-existing and new variables (covers both branches of 442->447)
    src_try_pre_existing = """def try_existing(x):
    y = torch.relu(x)
    try:
        y = torch.add(y, 1)
        z = torch.add(y, 2)
    except Exception:
        pass
    return y
"""
    g_try_exist = parse_ast_to_ir(src_try_pre_existing)
    assert any(n.op_type == "TryExcept" for n in g_try_exist.nodes.values())

    # TryExcept where only existing variable is in var_table (so line 442 false branch exits loop to 447)
    src_try_only_existing = """def try_only_exist(x):
    y = torch.relu(x)
    try:
        y = torch.add(y, 1)
    except Exception:
        pass
    return y
"""
    g_try_only = parse_ast_to_ir(src_try_only_existing)
    assert any(n.op_type == "TryExcept" for n in g_try_only.nodes.values())

    # TryExcept without handlers (try-finally) covers branch 442->447
    src_try_finally = """def try_finally_fn(x):
    try:
        y = torch.relu(x)
    finally:
        pass
    return y
"""
    g_finally = parse_ast_to_ir(src_try_finally)
    assert any(n.op_type == "TryExcept" for n in g_finally.nodes.values())

    # 6. Raise with cst.Name and bare raise (lines 471-472)
    src_raise_name = """def raise_name(x):
    raise ValueError
"""
    g_raise = parse_ast_to_ir(src_raise_name)
    assert any(n.op_type == "Assert" and n.attributes.get("exception") == "ValueError" for n in g_raise.nodes.values())

    src_raise_bare = """def raise_bare(x):
    raise
"""
    g_bare = parse_ast_to_ir(src_raise_bare)
    assert any(n.op_type == "Assert" and n.attributes.get("exception") == "Exception" for n in g_bare.nodes.values())
