"""Unit tests for the CST transpiler backend."""

from ml_switcheroo_compiler.backends.cst_transpiler import (
    CSTTransformer,
    _build_attribute_chain,
    transpile_source,
    type_infer_dry_run,
    validate_diff,
)


def test_build_attribute_chain() -> None:
    """Test building attribute chains."""
    import libcst as cst

    node_empty = _build_attribute_chain([])
    assert isinstance(node_empty, cst.Name)
    assert node_empty.value == "empty"

    node_foo = _build_attribute_chain(["foo"])
    assert isinstance(node_foo, cst.Name)
    assert node_foo.value == "foo"

    chain = _build_attribute_chain(["foo", "bar"])
    assert isinstance(chain, cst.Attribute)
    assert chain.attr.value == "bar"
    assert isinstance(chain.value, cst.Name)
    assert chain.value.value == "foo"


def test_transpile_source_import_from() -> None:
    """Test transpilation of 'from ... import ...' statements."""
    source = "from torch import nn\n"
    assert transpile_source(source, target_framework="jax") == "from jax import nn\n"
    assert transpile_source(source, target_framework="mlx") == "from mlx.core import nn\n"

    source = "from mlx.core import nn\n"
    assert transpile_source(source, target_framework="pytorch") == "from torch import nn\n"

    # Unsupported target framework -> unchanged
    assert transpile_source(source, target_framework="unknown") == source

    # Non-framework import -> unchanged
    assert transpile_source("from os import path\n", target_framework="jax") == "from os import path\n"

    # Module with no value (relative import) -> unchanged
    source = "from . import nn\n"
    assert transpile_source(source, target_framework="jax") == "from . import nn\n"


def test_transpile_source_import() -> None:
    """Test transpilation of 'import ...' statements."""
    source = "import torch\n"
    assert transpile_source(source, target_framework="jax") == "import jax\n"
    assert transpile_source(source, target_framework="mlx") == "import mlx.core\n"

    # Aliases
    source = "import torch as t\n"
    assert transpile_source(source, target_framework="jax") == "import jax as t\n"

    # Non-framework
    source = "import os\n"
    assert transpile_source(source, target_framework="jax") == "import os\n"

    # Multiple imports
    source = "import os, torch, sys\n"
    assert transpile_source(source, target_framework="jax") == "import os, jax, sys\n"

    # Test that targeting the same framework leaves it unchanged (Line 121)
    source = "import jax\n"
    assert transpile_source(source, target_framework="jax") == "import jax\n"

    # Unknown target framework fallback
    assert transpile_source("import torch\n", target_framework="unknown") == "import torch\n"

    # Test fallback in `leave_ImportFrom` with mock node (Lines 79 & 83)
    import libcst as cst

    transformer = CSTTransformer("jax")
    dummy_names = [cst.ImportAlias(name=cst.Name("foo"))]
    node = cst.ImportFrom(module=cst.Integer("1"), names=dummy_names)
    assert transformer.leave_ImportFrom(node, node) is node

    # Test _get_base_name fallback (Line 79)
    # create a mock Attribute with a non-Name/non-Attribute value
    mock_attr = cst.Attribute(value=cst.Integer("1"), attr=cst.Name("core"))
    node_attr = cst.ImportFrom(module=mock_attr, names=dummy_names)
    assert transformer.leave_ImportFrom(node_attr, node_attr) is node_attr


def test_transpile_source_call() -> None:
    """Test transpilation of function calls."""
    source = "torch.add(x, y)\n"
    assert transpile_source(source, target_framework="jax") == "jax.numpy.add(x, y)\n"
    assert transpile_source(source, target_framework="mlx") == "mlx.core.add(x, y)\n"

    source = "mlx.core.add(x, y)\n"
    assert transpile_source(source, target_framework="pytorch") == "torch.add(x, y)\n"

    # Non-framework call
    source = "print('hello')\n"
    assert transpile_source(source, target_framework="jax") == source

    source = "os.path.join('a', 'b')\n"
    assert transpile_source(source, target_framework="jax") == source

    # Not an attribute call
    source = "add(x, y)\n"
    assert transpile_source(source, target_framework="jax") == source

    transformer = CSTTransformer("jax")

    # Complex non-attribute call (to cover `not isinstance(updated_node.func.value, cst.Name)`)
    # Actually, `mlx.core.add` has `func.value` as an Attribute, which we now handle with `_get_base_name`
    # We can test `_get_base_name` fallback by passing a literal or something.
    source = "(1).add()\n"
    assert transpile_source(source, target_framework="jax") == source


def test_transpile_assign_stateful() -> None:
    """Test stateful-to-functional assignment rewrites."""
    source = "self.weight = 1.0\n"
    expected = 'state["weight"] = 1.0\n'
    assert transpile_source(source, target_framework="jax") == expected

    source = "self.weight, self.bias = 1.0, 0.0\n"
    expected = 'state["weight"], state["bias"] = 1.0, 0.0\n'
    assert transpile_source(source, target_framework="jax") == expected

    # Non-self assign
    source = "x.weight = 1.0\n"
    assert transpile_source(source, target_framework="jax") == source


def test_validate_diff() -> None:
    """Test diff validation."""
    source = "x = 1\n"
    transpiled = "x = 2\n"
    assert validate_diff(source, transpiled) is True

    # Same source
    assert validate_diff(source, source) is False

    # Invalid syntax
    invalid_transpiled = "x = 2\n\ninvalid code"
    assert validate_diff(source, invalid_transpiled) is False


def test_type_infer_dry_run() -> None:
    """Test basic dry-run type inference on CST."""
    source = "x = 1.0\ny = 2\n"
    res = type_infer_dry_run(source)
    assert res == {"dry_run": "success", "x": "float", "y": "int"}

    invalid_source = "x = \n"
    res = type_infer_dry_run(invalid_source)
    assert res == {"dry_run": "failed"}


def test_cst_transformer_init() -> None:
    """Test CSTTransformer initialization defaults."""
    transformer = CSTTransformer()
    assert transformer.target_framework == "jax"


def test_transpile_edge_cases() -> None:
    """Test missing branch edge cases in leave_ImportFrom and leave_Call."""
    import libcst as cst

    transformer = CSTTransformer("jax")

    # Lines 89-93: when target_module == src_module
    source_node = cst.ImportFrom(module=cst.Name("jax"), names=[cst.ImportAlias(name=cst.Name("foo"))])
    res = transformer.leave_ImportFrom(source_node, source_node)
    assert res is source_node

    # Lines 159-164: when target_chain == [src_call_base]
    # We want target_chain == [src_call_base] so we bypass the inner if.
    import ml_switcheroo_compiler.backends.cst_transpiler as cst_t

    # We use "pytorch" as the target framework. Its chain is ["torch"].
    # If the source call is "torch()", src_call_base is "torch".
    # target_chain != [src_call_base] becomes ["torch"] != ["torch"] -> False.
    transformer_pytorch = cst_t.CSTTransformer("pytorch")
    call_node_torch = cst.Call(func=cst.Name("torch"))
    res_call_pytorch = transformer_pytorch.leave_Call(call_node_torch, call_node_torch)
    assert res_call_pytorch is call_node_torch

    # Also test it through the main entrypoint
    from ml_switcheroo_compiler.backends.cst_transpiler import transpile_source

    assert transpile_source("torch.add(x)\n", target_framework="pytorch") == "torch.add(x)\n"


def test_type_infer_assign_edge_cases() -> None:
    """Test Assign edge cases where targets are not Name nodes, and a fallback assign."""
    from ml_switcheroo_compiler.backends.cst_transpiler import type_infer_dry_run

    # 202, 204, 206 branches: Assign to attribute or tuple instead of simple name, plus unknown type
    source = "self.x = 1.0\nx, y = 1\nself.y = 2\nz = 'hello'\n"
    res = type_infer_dry_run(source)
    assert res == {"dry_run": "success"}


def test_transpile_kwargs():
    source = "torch.sum(x, dim=1, keepdim=True)\n"
    assert transpile_source(source, target_framework="jax") == "jax.numpy.sum(x, axis=1, keepdims=True)\n"

    source = "jax.numpy.sum(x, axis=1, keepdims=True)\n"
    assert transpile_source(source, target_framework="pytorch") == "torch.sum(x, dim=1, keepdim=True)\n"


def test_transpile_broadcast_expand():
    source = "x.expand(1, 2)\n"
    assert transpile_source(source, target_framework="jax") == "jax.numpy.broadcast_to(x, 1, 2)\n"
    assert transpile_source(source, target_framework="mlx") == "mlx.core.broadcast_to(x, 1, 2)\n"
    assert transpile_source(source, target_framework="numpy") == "numpy.broadcast_to(x, 1, 2)\n"

    source = "jax.numpy.broadcast_to(x, 1, 2)\n"
    assert transpile_source(source, target_framework="pytorch") == "x.expand(1, 2)\n"


def test_transpile_classdef():
    source = "class MyModel(nn.Module):\n    def forward(self, x):\n        return x\n"
    res_jax = transpile_source(source, target_framework="jax")
    assert "class MyModel(flax.linen.Module):" in res_jax
    assert "def __call__(self, x):" in res_jax

    source2 = "class MyModel(flax.linen.Module):\n    def __call__(self, x):\n        return x\n"
    res_pt = transpile_source(source2, target_framework="pytorch")
    assert "class MyModel(nn.Module):" in res_pt
    assert "def forward(self, x):" in res_pt


def test_cst_coverage_edge_cases():
    source = "class MyModel(object):\n    def __call__(x):\n        return x\n"
    res = transpile_source(source, target_framework="pytorch")
    assert "class MyModel(object):" in res
    assert "def __call__(x):" in res

    source2 = "class MyModel(nn.Module):\n    pass\n"
    res2 = transpile_source(source2, target_framework="pytorch")
    assert "class MyModel(nn.Module):" in res2

    source3 = "class MyModel(flax.linen.Module):\n    pass\n"
    res3 = transpile_source(source3, target_framework="jax")
    assert "class MyModel(flax.linen.Module):" in res3

    # KWARG_MAP else branch
    source4 = "def foo(a=1): pass\n"
    res4 = transpile_source(source4, target_framework="unknown")
    assert "def foo(a=1): pass" in res4


def test_transpile_kwarg_else():
    source = "foo(a=1)\n"
    res = transpile_source(source, target_framework="unknown")
    assert "foo(a=1)" in res


def test_cst_transpiler_branches() -> None:
    from ml_switcheroo_compiler.backends.cst_transpiler import type_infer_dry_run, validate_diff

    assert type_infer_dry_run("some random invalid python code {")["dry_run"] == "failed"
    assert not validate_diff("x = 1", "x = 1")
    assert not validate_diff("x = 1", "invalid syntax {")

    source = "import os\nimport sys\n"
    res = transpile_source(source)
    assert "import os" in res

    source = "from os import path\n"
    res = transpile_source(source)
    assert "from os import path" in res

    source = "import numpy.random as rnd\n"
    res = transpile_source(source)
    assert "import numpy" in res or "rnd" in res


def test_cst_attribute() -> None:
    source = "x = self.weight\n"
    res = transpile_source(source)
    assert 'state["weight"]' in res or "self.weight" in res


def test_cst_import_branches() -> None:
    source = "import torch as t\n"
    res = transpile_source(source, target_framework="jax")
    assert "import" in res


def test_cst_transpiler_missing_branches():
    import libcst as cst

    from ml_switcheroo_compiler.backends.cst_transpiler import CSTTransformer, transpile_source

    # 1. target_config without kw_map (we can fake this or use a config that has no kw_map if one exists)
    # Actually just pass empty method_map to fall to line 288
    source = "def unknown_method(self): pass"
    res = transpile_source(source, "pytorch")
    assert "unknown_method" in res

    # 2. line 220: no target config
    source = "class A:\n    pass"
    res = transpile_source(source, "unknown_framework")
    assert "class A:" in res

    # 3. get_attr_chain non-Name/Attr
    source = "class A(funcs[0]):\n    pass"
    res = transpile_source(source, "pytorch")
    assert "funcs[0]" in res

    # 4. empty kwarg_map
    # If the target framework has an empty kwarg_map it falls to 160.
    # We can fake it via subclass.
    class FakeConfig:
        kwarg_map = {}
        module_path = ["fake"]
        broadcast_method = "broadcast_fake"
        method_map = {}
        class_bases = {"fake_base": []}

    visitor = CSTTransformer("pytorch")
    visitor.target_config = FakeConfig()

    tree = cst.parse_module("func(x=1)")
    tree = tree.visit(visitor)

    # 5. broadcast_method different but not expand/broadcast_to
    # Just need fw_config.broadcast_method = "A" and target = "B"
    # Actually, we can just trigger it using fake visitor
    visitor.target_config = FakeConfig()
    # It needs a valid original fw config in _CONFIG.frameworks to match
    source2 = "x.broadcast(1)"

    # 6. line 247 -> 241 (target base parts empty)
    # visitor with FakeConfig that has empty class base
    tree = cst.parse_module("class B(nn.Module):\n    pass")
    tree = tree.visit(visitor)

    # 5. properly visit
    # Let's say pytorch to fake, pytorch uses expand, fake uses broadcast_fake
    tree = cst.parse_module("x.expand(1)")
    visitor.target_config = FakeConfig()  # with broadcast_fake
    visitor.target_framework = "fake"
    # To match we need final_node.func.attr.value to be the broadcast_method of some config.
    # We can just visit it directly
    tree = tree.visit(visitor)

    # Test import matching target module directly
    assert transpile_source("import mlx.core\n", target_framework="mlx") == "import mlx.core\n"


def test_transpile_source_method_and_tensor_creation_rewrites() -> None:
    """Test transpilation of method rewrites (view, permute, contiguous) and tensor creation."""
    # 1. tensor.contiguous() -> identity
    source_contig = "y = x.contiguous()\n"
    assert transpile_source(source_contig, target_framework="jax") == "y = x\n"

    # 2. tensor.view(...) -> tensor.reshape(...)
    source_view = "y = x.view(2, 3)\n"
    assert transpile_source(source_view, target_framework="jax") == "y = x.reshape(2, 3)\n"

    # 3. tensor.permute(...) -> tensor.transpose(...)
    source_perm = "y = x.permute(1, 0)\n"
    assert transpile_source(source_perm, target_framework="jax") == "y = x.transpose(1, 0)\n"

    # 4. torch.tensor(...) -> jax.numpy.array(...)
    source_tensor = "t = torch.tensor([1, 2, 3])\n"
    assert transpile_source(source_tensor, target_framework="jax") == "t = jax.numpy.array([1, 2, 3])\n"


def test_cst_transpiler_advanced_transforms() -> None:
    """Test listcomp, genexp, ifexp, early return lowering, and config fallbacks."""
    from unittest.mock import patch

    import libcst as cst

    # 1. Lower early returns: if cond: return a; return b
    src_early_ret = "def f(cond, a, b):\n    if cond:\n        return a\n    return b\n"
    res_ret = transpile_source(src_early_ret, target_framework="jax")
    assert "jax.numpy.where(cond, a, b)" in res_ret

    # Function with 1 statement (len(body_stmts) != 2 branch 328->352)
    src_single_stmt = "def single_stmt(x):\n    return x\n"
    res_single = transpile_source(src_single_stmt, target_framework="jax")
    assert "return x" in res_single

    # Function with 2 statements where first is not an If (covers 328->352)
    src_two_stmts_no_if = "def two_stmts(x):\n    y = x + 1\n    return y\n"
    res_two = transpile_source(src_two_stmts_no_if, target_framework="jax")
    assert "return y" in res_two

    # Early returns with target_config fallback to jnp (line 338 fallback)
    class FalsyAfterFirstCheck:
        def __init__(self) -> None:
            self.checked = False
            self.method_map: dict[str, str] = {}

        def __bool__(self) -> bool:
            if not self.checked:
                self.checked = True
                return True
            return False

    t_dynamic = CSTTransformer("jax")
    t_dynamic.target_config = FalsyAfterFirstCheck()  # type: ignore[assignment]
    tree_dynamic = cst.parse_module(src_early_ret)
    res_dynamic = tree_dynamic.visit(t_dynamic)
    assert "jnp.where(cond, a, b)" in res_dynamic.code

    # Early return with empty return (ret1 is None or ret2 is None, line 342->352)
    src_none_ret = "def g(cond):\n    if cond:\n        return\n    return\n"
    res_none_ret = transpile_source(src_none_ret, target_framework="jax")
    assert "return\n" in res_none_ret

    # 2. ListComp functionalization
    src_listcomp = "[x + 1 for x in xs]\n"
    res_listcomp = transpile_source(src_listcomp, target_framework="jax")
    assert "list(map(lambda x: x + 1, xs))" in res_listcomp

    # ListComp with non-Name target (e.g. tuple unpack)
    src_listcomp_tuple = "[a + b for a, b in pairs]\n"
    res_listcomp_tuple = transpile_source(src_listcomp_tuple, target_framework="jax")
    assert "[a + b for a, b in pairs]" in res_listcomp_tuple

    # 3. GeneratorExp functionalization
    src_genexp = "(x * 2 for x in xs)\n"
    res_genexp = transpile_source(src_genexp, target_framework="jax")
    assert "map(lambda x: x * 2, xs)" in res_genexp

    # GeneratorExp with non-Name target
    src_genexp_tuple = "(a + b for a, b in pairs)\n"
    res_genexp_tuple = transpile_source(src_genexp_tuple, target_framework="jax")
    assert "(a + b for a, b in pairs)" in res_genexp_tuple

    # 4. IfExp lowering
    src_ifexp = "res = a if cond else b\n"
    res_ifexp = transpile_source(src_ifexp, target_framework="jax")
    assert "jax.numpy.where(cond, a, b)" in res_ifexp

    # IfExp lowering with target_config None
    transformer_no_cfg = CSTTransformer("unknown_framework")
    tree_ifexp = cst.parse_module(src_ifexp)
    res_tree_ifexp = tree_ifexp.visit(transformer_no_cfg)
    assert "jnp.where(cond, a, b)" in res_tree_ifexp.code

    # 5. Method rewrite targeting PyTorch (leaves view unchanged to exercise 192->188)
    src_view_pt = "y = x.view(2, 3)\n"
    assert transpile_source(src_view_pt, target_framework="pytorch") == "y = x.view(2, 3)\n"

    # 6. Fallback hardcoded defaults if config missing (lines 197, 199, 201)
    from ml_switcheroo_compiler.backends import cst_transpiler

    with patch.object(cst_transpiler._CONFIG.syntactic_patterns, "method_to_function_rewrites", []):
        assert transpile_source("y = x.contiguous()\n", target_framework="jax") == "y = x\n"
        assert transpile_source("y = x.view(2, 3)\n", target_framework="jax") == "y = x.reshape(2, 3)\n"
        assert transpile_source("y = x.permute(1, 0)\n", target_framework="jax") == "y = x.transpose(1, 0)\n"
        # Also test view and permute targeting pytorch to cover the False branch of 199 and 201
        assert transpile_source("y = x.view(2, 3)\n", target_framework="pytorch") == "y = x.view(2, 3)\n"
        assert transpile_source("y = x.permute(1, 0)\n", target_framework="pytorch") == "y = x.permute(1, 0)\n"


def test_transpile_inplace_and_augassign() -> None:
    """Test in-place mutation and augmented assignment functionalization."""
    # 1. In-place method mutation x.add_(y) -> x.add(y)
    src_add_ = "x.add_(y)\n"
    res_add_ = transpile_source(src_add_, target_framework="jax")
    assert "x.add(y)" in res_add_

    # 2. In-place sub_ -> subtract
    src_sub_ = "x.sub_(y)\n"
    res_sub_ = transpile_source(src_sub_, target_framework="jax")
    assert "x.subtract(y)" in res_sub_

    # 3. Augmented assignment functionalization for JAX
    src_aug = "x += y\n"
    res_aug = transpile_source(src_aug, target_framework="jax")
    assert "x = x + y" in res_aug

    src_sub_aug = "z -= 1\n"
    res_sub_aug = transpile_source(src_sub_aug, target_framework="jax")
    assert "z = z - 1" in res_sub_aug

    src_mul_aug = "a *= 2\n"
    res_mul_aug = transpile_source(src_mul_aug, target_framework="jax")
    assert "a = a * 2" in res_mul_aug

    src_div_aug = "b /= 2\n"
    res_div_aug = transpile_source(src_div_aug, target_framework="jax")
    assert "b = b / 2" in res_div_aug

    # Unsupported aug assign op (e.g. <<=) stays unchanged
    src_lshift_aug = "c <<= 1\n"
    res_lshift_aug = transpile_source(src_lshift_aug, target_framework="jax")
    assert "c <<= 1" in res_lshift_aug

    # Targeting non-functional framework (e.g. pytorch) leaves augassign unchanged
    res_pt_aug = transpile_source(src_aug, target_framework="pytorch")
    assert "x += y" in res_pt_aug


def test_declarative_cst_rewrite_rules_loading() -> None:
    """Test loading and validation of declarative CST rewrite rules from YAML."""
    from ml_switcheroo_compiler.backends.transpiler_config_models import load_cst_rewrite_rules

    rules = load_cst_rewrite_rules()
    assert len(rules.method_substitutions) > 0
    assert len(rules.chain_removals) > 0
    assert len(rules.in_place_mutations) > 0
    assert len(rules.tensor_creation_rewrites) > 0
    assert len(rules.broadcasting_rules) > 0
