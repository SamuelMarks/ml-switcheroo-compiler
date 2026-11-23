"""JAX Jaxpr to LogicalGraph ingestion adapter."""

from __future__ import annotations

import typing
from collections.abc import Mapping, Sequence

from ml_switcheroo_ir import AttributeValue, LogicalGraph, LogicalNode

from ml_switcheroo_compiler.core._mock_protocol import is_mock_with_attrs
from ml_switcheroo_compiler.core.shape import OpValue, ShapeProtocol


@typing.runtime_checkable
class _HasNameProtocol(typing.Protocol):
    """Protocol for primitives exposing a ``name`` attribute."""

    @property
    def name(self) -> str:
        """Return primitive name.

        Returns:
            str: Primitive name string.
        """
        ...


@typing.runtime_checkable
class _JaxAvalProtocol(typing.Protocol):
    """Protocol for JAX ShapedArray abstract values exposing ``shape``."""

    @property
    def shape(self) -> Sequence[int]:
        """Return abstract value shape.

        Returns:
            Sequence[int]: Shape tuple.
        """
        ...


@typing.runtime_checkable
class _JaxVarProtocol(typing.Protocol):
    """Protocol for JAX variables exposing ``aval``."""

    @property
    def aval(self) -> _JaxAvalProtocol | OpValue:
        """Return variable abstract value.

        Returns:
            _JaxAvalProtocol | OpValue: Abstract value.
        """
        ...


@typing.runtime_checkable
class JaxprEqnProtocol(typing.Protocol):
    """Protocol for JAXpr equation objects."""

    @property
    def primitive(self) -> _HasNameProtocol | str:
        """Return JAX primitive descriptor.

        Returns:
            _HasNameProtocol | str: Primitive object or name.
        """
        ...

    @property
    def invars(self) -> Sequence[OpValue]:
        """Return equation input variables.

        Returns:
            Sequence[OpValue]: Input variables.
        """
        ...

    @property
    def outvars(self) -> Sequence[OpValue]:
        """Return equation output variables.

        Returns:
            Sequence[OpValue]: Output variables.
        """
        ...


@typing.runtime_checkable
class _JaxprEqnParamsProtocol(typing.Protocol):
    """Protocol for JAXpr equations exposing ``params``."""

    @property
    def params(self) -> Mapping[str, OpValue]:
        """Return equation parameter mapping.

        Returns:
            Mapping[str, OpValue]: Equation parameters.
        """
        ...


@typing.runtime_checkable
class JaxprProtocol(typing.Protocol):
    """Protocol for JAXpr graph objects exposing ``eqns``."""

    @property
    def eqns(self) -> Sequence[JaxprEqnProtocol | OpValue]:
        """Return sequence of JAXpr equations.

        Returns:
            Sequence[JaxprEqnProtocol | OpValue]: Equations sequence.
        """
        ...


@typing.runtime_checkable
class _JaxprConstsProtocol(typing.Protocol):
    """Protocol for JAXpr objects exposing ``consts``."""

    @property
    def consts(self) -> Sequence[OpValue]:
        """Return constant values.

        Returns:
            Sequence[OpValue]: Constant values.
        """
        ...


@typing.runtime_checkable
class _JaxprConstvarsProtocol(typing.Protocol):
    """Protocol for JAXpr objects exposing ``constvars``."""

    @property
    def constvars(self) -> Sequence[OpValue] | None:
        """Return constant variables.

        Returns:
            Sequence[OpValue] | None: Constant variables.
        """
        ...


def _has_jaxpr_name(val: _HasNameProtocol | str) -> typing.TypeGuard[_HasNameProtocol]:
    """Check whether `val` exposes `.name`.

    Args:
        val (_HasNameProtocol | str): Candidate primitive.

    Returns:
        typing.TypeGuard[_HasNameProtocol]: `True` when `.name` exists.
    """
    found: bool = isinstance(val, _HasNameProtocol)
    return found or is_mock_with_attrs(val, "name")


def _has_jaxpr_aval(val: OpValue | _JaxVarProtocol | None) -> typing.TypeGuard[_JaxVarProtocol]:
    """Check whether `val` exposes `.aval`.

    Args:
        val (OpValue | _JaxVarProtocol | None): Candidate JAX variable.

    Returns:
        typing.TypeGuard[_JaxVarProtocol]: `True` when `.aval` exists.
    """
    found: bool = isinstance(val, _JaxVarProtocol)
    return found or is_mock_with_attrs(val, "aval")


def _has_aval_shape(val: OpValue | _JaxAvalProtocol) -> typing.TypeGuard[_JaxAvalProtocol]:
    """Check whether `val` exposes `.shape`.

    Args:
        val (OpValue | _JaxAvalProtocol): Candidate abstract value.

    Returns:
        typing.TypeGuard[_JaxAvalProtocol]: `True` when `.shape` exists.
    """
    found: bool = isinstance(val, _JaxAvalProtocol)
    return found or is_mock_with_attrs(val, "shape")


def _has_const_shape(val: OpValue) -> typing.TypeGuard[ShapeProtocol]:
    """Check whether `val` exposes `.shape`.

    Args:
        val (OpValue): Candidate constant value.

    Returns:
        typing.TypeGuard[ShapeProtocol]: `True` when `.shape` exists.
    """
    found: bool = isinstance(val, ShapeProtocol)
    return found or is_mock_with_attrs(val, "shape")


def _is_jaxpr_eqn(val: JaxprEqnProtocol | OpValue) -> typing.TypeGuard[JaxprEqnProtocol]:
    """Check whether `val` exposes `primitive`, `invars`, and `outvars`.

    Args:
        val (JaxprEqnProtocol | OpValue): Candidate JAXpr equation.

    Returns:
        typing.TypeGuard[JaxprEqnProtocol]: `True` when `val` is a JAXpr equation.
    """
    found: bool = isinstance(val, JaxprEqnProtocol)
    return found or is_mock_with_attrs(val, "primitive", "invars", "outvars")


def _has_eqn_params(
    val: JaxprEqnProtocol | OpValue | _JaxprEqnParamsProtocol,
) -> typing.TypeGuard[_JaxprEqnParamsProtocol]:
    """Check whether `val` exposes `.params`.

    Args:
        val (JaxprEqnProtocol | OpValue | _JaxprEqnParamsProtocol): Candidate equation.

    Returns:
        typing.TypeGuard[_JaxprEqnParamsProtocol]: `True` when `.params` exists.
    """
    found: bool = isinstance(val, _JaxprEqnParamsProtocol)
    return found or is_mock_with_attrs(val, "params")


def _is_jaxpr_graph(val: OpValue) -> typing.TypeGuard[JaxprProtocol]:
    """Check whether `val` exposes `.eqns`.

    Args:
        val (OpValue): Candidate JAXpr graph.

    Returns:
        typing.TypeGuard[JaxprProtocol]: `True` when `.eqns` exists.
    """
    found: bool = isinstance(val, JaxprProtocol)
    return found or is_mock_with_attrs(val, "eqns")


def _has_jaxpr_consts(
    val: OpValue | _JaxprConstvarsProtocol | _JaxprConstsProtocol,
) -> typing.TypeGuard[_JaxprConstsProtocol]:
    """Check whether `val` exposes `.consts`.

    Args:
        val (OpValue | _JaxprConstvarsProtocol | _JaxprConstsProtocol): Candidate JAXpr object.

    Returns:
        typing.TypeGuard[_JaxprConstsProtocol]: `True` when `.consts` exists.
    """
    found: bool = isinstance(val, _JaxprConstsProtocol)
    return found or is_mock_with_attrs(val, "consts")


def _has_jaxpr_constvars(
    val: OpValue | _JaxprConstvarsProtocol | _JaxprConstsProtocol,
) -> typing.TypeGuard[_JaxprConstvarsProtocol]:
    """Check whether `val` exposes `.constvars`.

    Args:
        val (OpValue | _JaxprConstvarsProtocol | _JaxprConstsProtocol): Candidate JAXpr object.

    Returns:
        typing.TypeGuard[_JaxprConstvarsProtocol]: `True` when `.constvars` exists.
    """
    found: bool = isinstance(val, _JaxprConstvarsProtocol)
    return found or is_mock_with_attrs(val, "constvars")


def _extract_const_shape(val: OpValue) -> tuple[int, ...]:
    """Extract integer shape tuple from a constant operand if available.

    Args:
        val (OpValue): Candidate constant operand.

    Returns:
        tuple[int, ...]: Extracted integer shape tuple.
    """
    if _has_const_shape(val):
        return tuple(int(d) if isinstance(d, (int, float)) else 0 for d in val.shape)
    return ()


def _extract_jaxpr_constants(jaxpr: OpValue, graph: LogicalGraph) -> None:
    """Extract constant values from a JAX ``jaxpr`` into ``Constant`` ``LogicalNode`` entries.

    Args:
        jaxpr (OpValue): JAX ``jaxpr`` object containing constants.
        graph (LogicalGraph): Target ``LogicalGraph`` being populated.
    """
    cand_jaxpr: OpValue | _JaxprConstvarsProtocol | _JaxprConstsProtocol = jaxpr
    constvars = cand_jaxpr.constvars if _has_jaxpr_constvars(cand_jaxpr) else None
    if _has_jaxpr_consts(cand_jaxpr):
        for idx, const_val in enumerate(cand_jaxpr.consts):
            const_id = str(id(constvars[idx])) if constvars else f"const_{idx}"
            c_shape = _extract_const_shape(const_val)
            c_attr: AttributeValue = (
                const_val
                if isinstance(const_val, (int, float, str, bool, bytes)) or const_val is None
                else str(const_val)
            )
            graph.nodes[const_id] = LogicalNode(
                id=const_id,
                op_type="Constant",
                domain="ai.onnx",
                version=1,
                attributes={"value": c_attr},
                inputs=[],
                shape_metadata=c_shape,
            )


def _translate_jax_equation(eqn: JaxprEqnProtocol | OpValue, graph: LogicalGraph) -> None:
    """Translate a single JAXpr equation into a ``LogicalNode``.

    Args:
        eqn (JaxprEqnProtocol | OpValue): JAXpr equation object.
        graph (LogicalGraph): Target ``LogicalGraph`` being populated.
    """
    if _is_jaxpr_eqn(eqn):
        prim = eqn.primitive
        primitive_name = str(prim.name) if _has_jaxpr_name(prim) else str(prim)
        invars: Sequence[OpValue] = eqn.invars
        outvars: list[OpValue] = list(eqn.outvars)
    else:
        primitive_name = ""
        invars = ()
        outvars = []

    op_type = "Unknown"
    if primitive_name == "add":
        op_type = "Add"
    elif primitive_name == "mul":
        op_type = "Mul"

    inputs = [str(id(invar)) for invar in invars]
    out_id = str(id(outvars[0])) if outvars else "out"

    attributes: dict[str, AttributeValue] = {}
    cand_eqn: JaxprEqnProtocol | OpValue | _JaxprEqnParamsProtocol = eqn
    if _has_eqn_params(cand_eqn) and isinstance(cand_eqn.params, Mapping):
        for k, v in cand_eqn.params.items():
            if isinstance(v, (list, tuple)):
                attributes[str(k)] = [x for x in v if isinstance(x, (int, float, str, bool))]
            elif isinstance(v, (int, float, str, bool)):
                attributes[str(k)] = v

    shape_metadata: tuple[int, ...] = ()
    first_outvar: OpValue | _JaxVarProtocol | None = outvars[0] if outvars else None
    if _has_jaxpr_aval(first_outvar):
        raw_aval: OpValue | _JaxAvalProtocol = first_outvar.aval
        if _has_aval_shape(raw_aval):
            shape_metadata = tuple(int(d) for d in raw_aval.shape)

    graph.nodes[out_id] = LogicalNode(
        id=out_id,
        op_type=op_type,
        attributes=attributes,
        inputs=inputs,
        shape_metadata=shape_metadata,
    )


def ingest_jaxpr(jaxpr: OpValue) -> LogicalGraph:
    """Ingest a JAX ``jaxpr`` into a canonical ``LogicalGraph``.

    Args:
        jaxpr (OpValue): JAX ``jaxpr`` object to convert.

    Returns:
        LogicalGraph: Converted compiler intermediate representation graph.

    Raises:
        ValueError: If ``jaxpr`` is ``None``.
    """
    if jaxpr is None:
        raise ValueError("JAX jaxpr cannot be None")

    graph = LogicalGraph(name="jaxpr_ingested")
    _extract_jaxpr_constants(jaxpr, graph)

    if _is_jaxpr_graph(jaxpr):
        for eqn in jaxpr.eqns:
            _translate_jax_equation(eqn, graph)

    return graph


__all__ = [
    "JaxprEqnProtocol",
    "JaxprProtocol",
    "_extract_jaxpr_constants",
    "_translate_jax_equation",
    "ingest_jaxpr",
]
