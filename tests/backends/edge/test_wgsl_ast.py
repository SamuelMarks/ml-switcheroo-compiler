from ml_switcheroo_compiler.backends.edge.wgsl_ast import WGSLAssign, WGSLBinaryOp, WGSLDecl, WGSLEmitter, WGSLFor, WGSLFunction, WGSLIf, WGSLIndex, WGSLNode, WGSLRaw, WGSLUnaryOp, WGSLVar


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
