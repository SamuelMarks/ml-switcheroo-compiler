import pytest

from ml_switcheroo_compiler.backends.pytorch.generator import PyTorchCodeGenerator
from ml_switcheroo_compiler.ir.core import IRGraph


def test_pytorch_generator_no_numpy_crashes():
    """Dynamically execute the PyTorchCodeGenerator AST to ensure no NameErrors."""
    graph = IRGraph()
    # Mocking some basic nodes
    # Normally we would build a full graph, but just initializing the generator
    # and testing if the prefix string executes without numpy is sufficient for the hotfix test
    gen = PyTorchCodeGenerator(graph)
    prefix_lines = gen._get_prefix_code()
    prefix_str = "\n".join(prefix_lines)

    # We will execute the prefix in a dictionary
    exec_globals = {}
    try:
        exec(prefix_str, exec_globals)
    except NameError as e:
        pytest.fail(f"NameError during PyTorch AST prefix execution: {e}")
    except ImportError as e:
        pytest.fail(f"ImportError during PyTorch AST prefix execution: {e}")
    except Exception as e:
        pytest.fail(f"Unexpected error during PyTorch AST prefix execution: {e}")

    assert "np" not in exec_globals and "numpy" not in exec_globals, "Numpy import should not be in prefix to prevent leaks."
