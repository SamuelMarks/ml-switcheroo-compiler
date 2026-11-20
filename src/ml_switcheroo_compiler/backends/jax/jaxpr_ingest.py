"""JAX Jaxpr to LogicalGraph ingestion adapter."""

from __future__ import annotations

import typing
from collections.abc import Mapping, Sequence

from ml_switcheroo_ir import AttributeValue, LogicalGraph, LogicalNode

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


def _extract_const_shape(val: OpValue) -> tuple[int, ...]:
    """Extract integer shape tuple from a constant operand if available.

    Args:
        val (OpValue): Candidate constant operand.

    Returns:
        tuple[int, ...]: Extracted integer shape tuple.
    """
    if isinstance(val, ShapeProtocol):
        return tuple(int(d) if isinstance(d, (int, float)) else 0 for d in val.shape)
    return ()


def _extract_jaxpr_constants(jaxpr: OpValue, graph: LogicalGraph) -> None:
    """Extract constant values from a JAX ``jaxpr`` into ``Constant`` ``LogicalNode`` entries.

    Args:
        jaxpr (OpValue): JAX ``jaxpr`` object containing constants.
        graph (LogicalGraph): Target ``LogicalGraph`` being populated.
    """
    cand_jaxpr: OpValue | _JaxprConstvarsProtocol | _JaxprConstsProtocol = jaxpr
    constvars = cand_jaxpr.constvars if isinstance(cand_jaxpr, _JaxprConstvarsProtocol) else None
    if isinstance(cand_jaxpr, _JaxprConstsProtocol):
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
    if isinstance(eqn, JaxprEqnProtocol):
        prim = eqn.primitive
        primitive_name = str(prim.name) if isinstance(prim, _HasNameProtocol) else str(prim)
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
    if isinstance(cand_eqn, _JaxprEqnParamsProtocol) and isinstance(cand_eqn.params, Mapping):
        for k, v in cand_eqn.params.items():
            if isinstance(v, (list, tuple)):
                attributes[str(k)] = [x for x in v if isinstance(x, (int, float, str, bool))]
            elif isinstance(v, (int, float, str, bool)):
                attributes[str(k)] = v

    shape_metadata: tuple[int, ...] = ()
    first_outvar: OpValue | _JaxVarProtocol | None = outvars[0] if outvars else None
    if isinstance(first_outvar, _JaxVarProtocol):
        raw_aval: OpValue | _JaxAvalProtocol = first_outvar.aval
        if isinstance(raw_aval, _JaxAvalProtocol):
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

    if isinstance(jaxpr, JaxprProtocol):
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
