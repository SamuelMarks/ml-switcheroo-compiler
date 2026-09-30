"""Unit tests verifying ecosystem export utilities for TypeScript, C++, Pydantic, and Protobuf."""

from __future__ import annotations

import ml_ecosystem_snapshots.export as exp
from ml_ecosystem_snapshots.models import ExtendedGhostParam, ExtendedGhostRef, ParameterKind


def _create_sample_ghost_ref(op_name: str) -> ExtendedGhostRef:
    """Create a sample ExtendedGhostRef fixture for export testing.

    Args:
        op_name (str): Operation name to embed in reference.

    Returns:
        ExtendedGhostRef: Instantiated ghost reference model.
    """
    return ExtendedGhostRef(
        name=op_name,
        api_path=f"torch.{op_name}",
        kind="function",
        params=[
            ExtendedGhostParam(
                name="input",
                kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                annotation="Tensor",
            ),
            ExtendedGhostParam(
                name="other",
                kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                annotation="Tensor",
            ),
        ],
    )


def test_export_to_typescript_interface() -> None:
    """Verify TypeScript interface export generation from ghost references."""
    ref = _create_sample_ghost_ref("matmul")
    ts_code = exp.to_typescript_interface(ref)
    assert "export interface matmul" in ts_code
    assert "input:" in ts_code
    assert "other:" in ts_code


def test_export_to_cpp_header() -> None:
    """Verify C++ struct and header export generation from ghost references."""
    ref = _create_sample_ghost_ref("matmul")
    cpp_code = exp.to_cpp_header(ref)
    assert "#pragma once" in cpp_code
    assert "namespace ml_ecosystem" in cpp_code
    assert "struct matmul" in cpp_code


def test_export_to_pydantic() -> None:
    """Verify Pydantic model export generation from ghost references."""
    ref = _create_sample_ghost_ref("matmul")
    pydantic_code = exp.to_pydantic(ref)
    assert "class matmul(BaseModel):" in pydantic_code
    assert "input:" in pydantic_code


def test_export_to_protobuf() -> None:
    """Verify Protobuf message export generation from ghost references."""
    ref = _create_sample_ghost_ref("matmul")
    pb_code = exp.to_protobuf(ref)
    assert 'syntax = "proto3";' in pb_code
    assert "message matmul {" in pb_code
    assert "TensorProto input = 1;" in pb_code
