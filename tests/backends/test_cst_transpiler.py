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
    node = cst.ImportFrom(module=cst.Integer("1"), names=dummy_names)  # type: ignore
    assert transformer.leave_ImportFrom(node, node) is node

    # Test _get_base_name fallback (Line 79)
    # create a mock Attribute with a non-Name/non-Attribute value
    mock_attr = cst.Attribute(value=cst.Integer("1"), attr=cst.Name("core"))  # type: ignore
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
