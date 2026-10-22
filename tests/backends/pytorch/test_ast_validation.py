"""Tests for test_ast_validation."""

from __future__ import annotations

import typing

import pytest

from ml_switcheroo_compiler.backends.pytorch.generator import PyTorchCodeGenerator
from ml_switcheroo_compiler.ir.core import IRGraph
from tests.testing_utils import as_tensor_dict

if typing.TYPE_CHECKING:
    pass


def test_pytorch_generator_no_numpy_crashes() -> None:
    """Dynamically execute the PyTorchCodeGenerator AST to ensure no NameErrors."""
    graph = IRGraph()
    # Mocking some basic nodes
    # Normally we would build a full graph, but just initializing the generator
    # and testing if the prefix string executes without numpy is sufficient for the hotfix test
    gen = PyTorchCodeGenerator(graph)
    prefix_lines = gen._get_prefix_code()
    prefix_str = "\n".join(prefix_lines)

    # We will execute the prefix in a dictionary
    import importlib.util
    import sys
    from unittest.mock import MagicMock, patch

    mock_torch = MagicMock()
    ctx = (
        patch.dict(
            sys.modules,
            {
                "torch": mock_torch,
                "torch.nn": mock_torch.nn,
                "torch.nn.functional": mock_torch.nn.functional,
                "torch.fft": mock_torch.fft,
                "torch.linalg": mock_torch.linalg,
                "torch.special": mock_torch.special,
            },
        )
        if importlib.util.find_spec("torch") is None
        else MagicMock()
    )
    exec_globals = as_tensor_dict({})
    with ctx:
        try:
            exec(prefix_str, exec_globals)
        except NameError as e:
            pytest.fail(f"NameError during PyTorch AST prefix execution: {e}")
        except ImportError as e:
            pytest.fail(f"ImportError during PyTorch AST prefix execution: {e}")
        except Exception as e:
            pytest.fail(f"Unexpected error during PyTorch AST prefix execution: {e}")

    assert "np" not in exec_globals and "numpy" not in exec_globals, "Numpy import should not be in prefix to prevent leaks."
