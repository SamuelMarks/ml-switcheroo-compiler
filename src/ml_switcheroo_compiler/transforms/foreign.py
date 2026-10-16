"""Re-exports of foreign graph ingestion utilities from the top-level adapter layer."""

from __future__ import annotations

from ml_switcheroo_compiler.foreign import (
    _FX_MODULE_TYPE_MAP,
    _extract_jaxpr_constants,
    _handle_fx_call_function,
    _handle_fx_call_method,
    _handle_fx_call_module,
    _handle_fx_get_attr,
    _handle_fx_output,
    _handle_fx_placeholder,
    _resolve_fx_module_op_type,
    _translate_jax_equation,
    ingest_jaxpr,
    ingest_torch_fx,
)

__all__ = [
    "_FX_MODULE_TYPE_MAP",
    "_extract_jaxpr_constants",
    "_handle_fx_call_function",
    "_handle_fx_call_method",
    "_handle_fx_call_module",
    "_handle_fx_get_attr",
    "_handle_fx_output",
    "_handle_fx_placeholder",
    "_resolve_fx_module_op_type",
    "_translate_jax_equation",
    "ingest_jaxpr",
    "ingest_torch_fx",
]
