import pytest

from ml_switcheroo_compiler.backends.edge.wgsl_ast import (
    WGSLAssign,
    WGSLBinaryOp,
    WGSLDecl,
    WGSLEmitter,
    WGSLFor,
    WGSLFunction,
    WGSLIf,
    WGSLIndex,
    WGSLNode,
    WGSLRaw,
    WGSLUnaryOp,
    WGSLValidationError,
    WGSLVar,
    validate_identifier,
    validate_workgroup_size,
)


def test_wgsl_emitter_raw_and_var():
    emitter = WGSLEmitter()
    assert emitter.emit("string_node") == "string_node"

    raw = WGSLRaw("var x = 1;")
    assert emitter.emit(raw) == "var x = 1;"

    var = WGSLVar("my_var")
    assert emitter.emit(var) == "my_var"

    base_node = WGSLNode()
    assert emitter.emit(base_node) == ""


def test_wgsl_emitter_index():
    emitter = WGSLEmitter()
    idx = WGSLIndex("my_buf", "0")
    assert emitter.emit(idx) == "my_buf[0]"


def test_wgsl_emitter_ops():
    emitter = WGSLEmitter()
    binop = WGSLBinaryOp("+", "a", "b")
    assert emitter.emit(binop) == "a + b"

    unop = WGSLUnaryOp("-", "a")
    assert emitter.emit(unop) == "-a"


def test_wgsl_emitter_assign():
    emitter = WGSLEmitter()
    assign = WGSLAssign("a", "b")
    assert emitter.emit(assign) == "a = b;"


def test_wgsl_emitter_decl():
    emitter = WGSLEmitter()
    decl1 = WGSLDecl("let", "a")
    assert emitter.emit(decl1) == "let a;"

    decl2 = WGSLDecl("var", "b", "2")
    assert emitter.emit(decl2) == "var b = 2;"

    decl3 = WGSLDecl("var", "c", "3", "u32")
    assert emitter.emit(decl3) == "var c: u32 = 3;"


def test_wgsl_emitter_if():
    emitter = WGSLEmitter()
    if_node = WGSLIf("a == b", [WGSLRaw("return;")])
    expected = "if (a == b) {\n  return;\n}"
    assert emitter.emit(if_node) == expected


def test_wgsl_emitter_for():
    emitter = WGSLEmitter()
    for_node = WGSLFor(WGSLDecl("var", "i", "0"), "i < 10", "i++", [WGSLRaw("break;")])
    expected = "for (var i = 0; i < 10; i++) {\n  break;\n}"
    assert emitter.emit(for_node) == expected


def test_wgsl_emitter_function():
    emitter = WGSLEmitter()
    func = WGSLFunction(name="my_func", params=["a: u32"], body=[WGSLRaw("return;")], attrs=["@compute"])
    expected = "@compute\nfn my_func(a: u32) {\n  return;\n}"
    assert emitter.emit(func) == expected


def test_validate_identifier_specifications():
    validate_identifier("valid_name_1")
    validate_identifier("_private_var")

    with pytest.raises(WGSLValidationError, match="Invalid WGSL identifier"):
        validate_identifier("")
    with pytest.raises(WGSLValidationError, match="does not match WGSL identifier syntax"):
        validate_identifier("123invalid")
    with pytest.raises(WGSLValidationError, match="reserved WGSL keyword"):
        validate_identifier("var")


def test_validate_workgroup_size_limits():
    validate_workgroup_size(64, 1, 1)
    validate_workgroup_size(16, 16, 1)

    with pytest.raises(WGSLValidationError, match="dimensions must be >= 1"):
        validate_workgroup_size(0, 1, 1)
    with pytest.raises(WGSLValidationError, match="dimension exceeds WebGPU limits"):
        validate_workgroup_size(512, 1, 1)
    with pytest.raises(WGSLValidationError, match="dimension exceeds WebGPU limits"):
        validate_workgroup_size(1, 1, 128)
    with pytest.raises(WGSLValidationError, match="Total workgroup invocations"):
        validate_workgroup_size(16, 17, 1)


def test_wgsl_ast_node_validation():
    emitter = WGSLEmitter()

    # Valid nodes
    var = WGSLVar("tensor_x")
    emitter.validate(var)

    idx = WGSLIndex("tensor_x", WGSLVar("idx"))
    idx.validate()

    binop = WGSLBinaryOp("+", WGSLVar("a"), WGSLVar("b"))
    binop.validate()

    unop = WGSLUnaryOp("-", WGSLVar("a"))
    unop.validate()

    assign = WGSLAssign(WGSLVar("a"), WGSLVar("b"))
    assign.validate()

    decl = WGSLDecl("let", "out_val", WGSLVar("a"))
    decl.validate()

    if_stmt = WGSLIf(WGSLVar("cond"), [assign])
    if_stmt.validate()

    for_stmt = WGSLFor(WGSLDecl("var", "i", "0"), WGSLVar("cond"), WGSLVar("step"), [assign])
    for_stmt.validate()

    fn_stmt = WGSLFunction("kernel_main", ["in0: f32"], [assign], ["@compute @workgroup_size(64, 1, 1)"])
    fn_stmt.validate()

    # Invalid cases
    with pytest.raises(WGSLValidationError):
        WGSLBinaryOp("invalid_op", "a", "b").validate()

    with pytest.raises(WGSLValidationError):
        WGSLUnaryOp("invalid_op", "a").validate()

    with pytest.raises(WGSLValidationError, match="Invalid declaration kind"):
        WGSLDecl("invalid_kind", "x").validate()

    with pytest.raises(WGSLValidationError):
        WGSLFunction("kernel_main", [], [], ["@compute @workgroup_size(512, 1, 1)"]).validate()


def test_wgsl_ast_string_statements_validation():
    """Verify validation when AST nodes contain plain string statements."""
    emitter = WGSLEmitter()
    emitter.validate("plain_string_node")

    if_node = WGSLIf("cond", ["return;"])
    if_node.validate()

    for_node = WGSLFor("var i = 0u;", "i < 10u", "i = i + 1u", ["break;"])
    for_node.validate()

    func = WGSLFunction("my_func", [], ["return;"])
    func.validate()
