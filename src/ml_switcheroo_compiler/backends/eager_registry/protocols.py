"""Cohesive structural protocols for arrays, RNG streams, context managers, and eager callables."""

from __future__ import annotations

import types
import typing
from collections.abc import Callable, Generator

from .domain_ops_a_f import _DomainOpsAFProto
from .domain_ops_g_n import _DomainOpsGNProto
from .domain_ops_o_s import _DomainOpsOSProto
from .domain_ops_private import _DomainOpsPrivateProto
from .domain_ops_t_z import _DomainOpsTZProto
from .domain_symbols_upper import _UpperCaseSymbolProto
from .dunders_bitwise import _ArrayBitwiseAndReverseDundersProto, _ExtendedTensorDundersProto
from .dunders_numeric import _ArrayArithmeticAndIndexingProto
from .types_base import BackendArg, BackendOperand, _passthrough_tensor_val

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.numpy.eager import _np_module


@typing.runtime_checkable
class EagerBoundCallable(typing.Protocol):
    """Runtime-checkable protocol for bound backend operation callables."""

    def __call__(self, *args: types.ModuleType | BackendOperand, **kwargs: BackendOperand) -> EagerTensorProtocol:
        """Execute bound backend operation.

        Args:
            *args (types.ModuleType | BackendOperand): Positional arguments.
            **kwargs (BackendOperand): Keyword arguments.

        Returns:
            EagerTensorProtocol: Resulting backend array or scalar.
        """
        ...


@typing.runtime_checkable
class EagerRegisteredCallable(typing.Protocol):
    """Runtime-checkable protocol for registered eager operation callables."""

    __name__: str

    def __call__(self, *args: types.ModuleType | BackendOperand, **kwargs: BackendOperand) -> EagerTensorProtocol:
        """Execute registered eager operation.

        Args:
            *args (types.ModuleType | BackendOperand): Positional backend module and tensor/scalar arguments.
            **kwargs (BackendOperand): Keyword arguments.

        Returns:
            EagerTensorProtocol: Resulting backend array or scalar.
        """
        ...


class EagerOpCallable(typing.Protocol):
    """Protocol for eager operation callables and indexable helper factories."""

    __name__: str
    __module__: str
    __wrapped__: EagerCallableProtocol
    __all__: list[str]
    __abstractmethods__: frozenset[str]
    _autograph_do_not_convert: EagerCallableProtocol

    def __getitem__(self, key: BackendArg) -> EagerTensorProtocol:
        """Index into an indexable helper object (e.g., `.at[...]`, `r_[...]`, `s_[...]`).

        Args:
            key (BackendArg): Index key or slice specification.

        Returns:
            EagerTensorProtocol: Indexed tensor or helper object.
        """
        ...

    def __iter__(self) -> Generator[EagerTensorProtocol, None, None]:
        """Iterate over elements when the attribute represents an iterable sequence.

        Yields:
            EagerTensorProtocol: Element tensor.
        """
        ...

    def __index__(self) -> int:
        """Convert scalar helper attribute to slice index `int`.

        Returns:
            int: Integer index.
        """
        ...

    def __int__(self) -> int:
        """Convert scalar helper attribute to Python `int`.

        Returns:
            int: Integer value.
        """
        ...

    def __float__(self) -> float:
        """Convert scalar helper attribute to Python `float`.

        Returns:
            float: Floating-point value.
        """
        ...

    def __call__(self, *args: BackendOperand, **kwargs: BackendOperand) -> EagerTensorProtocol:
        """Invoke the eager backend callable with positional and keyword operands.

        Args:
            *args (BackendOperand): Positional operands.
            **kwargs (BackendOperand): Keyword operands.

        Returns:
            EagerTensorProtocol: Computed backend tensor or value.
        """
        ...


class EagerCallableProtocol(EagerOpCallable, typing.Protocol):
    """Explicitly typed eager callable protocol supporting RNG factory access."""

    default_rng: EagerCallableProtocol


class BackendContextManagerProtocol(typing.Protocol):
    """Structural protocol for backend device, stream, and gradient context managers."""

    def __enter__(self) -> EagerTensorProtocol:
        """Enter runtime backend context scope.

        Returns:
            EagerTensorProtocol: Active context object.
        """
        ...

    def __exit__(self, exc_type: BackendArg, exc_val: BackendArg, exc_tb: BackendArg) -> BackendArg:
        """Exit runtime backend context scope.

        Args:
            exc_type (BackendArg): Raised exception class or `None`.
            exc_val (BackendArg): Raised exception instance or `None`.
            exc_tb (BackendArg): Traceback object or `None`.

        Returns:
            BackendArg: Boolean exception-suppression indicator.
        """
        ...


class BackendRNGProtocol(typing.Protocol):
    """Structural protocol for backend pseudo-random number generator objects and namespaces."""

    default_rng: EagerCallableProtocol
    seed: EagerCallableProtocol
    normal: EagerCallableProtocol
    uniform: EagerCallableProtocol
    randint: EagerCallableProtocol
    binomial: EagerCallableProtocol
    poisson: EagerCallableProtocol
    gamma: EagerCallableProtocol
    beta: EagerCallableProtocol
    permutation: EagerCallableProtocol
    choice: EagerCallableProtocol
    shuffle: EagerCallableProtocol
    random: EagerTensorProtocol
    rand: EagerCallableProtocol
    randn: EagerCallableProtocol


@typing.runtime_checkable
class BackendArrayProtocol(
    _ArrayArithmeticAndIndexingProto,
    _ArrayBitwiseAndReverseDundersProto,
    typing.Protocol,
):
    """Cohesive runtime-checkable structural protocol for backend multidimensional array/tensor values."""

    ndim: int

    @property
    def shape(self) -> tuple[int, ...]:
        """Return array dimension sizes.

        Returns:
            tuple[int, ...]: Tuple of axis lengths.
        """
        ...

    @property
    def dtype(self) -> _np_module.dtype[_np_module.float64]:
        """Return array element data type descriptor.

        Returns:
            _np_module.dtype[_np_module.float64]: Element dtype object.
        """
        ...


class _BaseTensorProto(
    BackendArrayProtocol,
    _ExtendedTensorDundersProto,
    BackendContextManagerProtocol,
    typing.Protocol,
):
    """Base protocol combining `BackendArrayProtocol`, `BackendContextManagerProtocol`, and callable metadata."""

    T: EagerTensorProtocol
    real: EagerTensorProtocol
    imag: EagerTensorProtocol
    flat: EagerTensorProtocol
    tolist: Callable[[], list[int]]
    tobytes: Callable[[], bytes]
    __name__: str
    __module__: str
    __wrapped__: EagerCallableProtocol
    __all__: list[str]
    __abstractmethods__: frozenset[str]

    def item(self, *args: int) -> float | int | bool | complex | str | bytes:
        """Extract Python scalar value from a 0-D or indexed array element.

        Args:
            *args (int): Optional flat or multi-dimensional coordinates.

        Returns:
            float | int | bool | complex | str | bytes: Extracted Python scalar.
        """
        ...

    def __array__(self) -> _np_module.ndarray[tuple[int, ...], _np_module.dtype[_np_module.float64]]:
        """Return host array representation for `_SupportsArray` interoperability.

        Returns:
            _np_module.ndarray[tuple[int, ...], _np_module.dtype[_np_module.float64]]: Host array buffer.
        """
        ...

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: BackendArg,
        handler: EagerTensorProtocol,
    ) -> BackendArg:
        """Build Pydantic plain-validator schema preserving runtime backend tensor instances.

        Args:
            source_type (BackendArg): Annotated field source type.
            handler (EagerTensorProtocol): Pydantic core schema generation callback.

        Returns:
            BackendArg: Plain-validator core schema dictionary.
        """
        del source_type, handler
        fn_dict: dict[str, BackendArg] = {"type": "no-info", "function": _passthrough_tensor_val}
        schema_dict: dict[str, BackendArg] = {"type": "function-plain", "function": fn_dict}
        return schema_dict

    def __call__(self, *args: BackendOperand, **kwargs: BackendOperand) -> EagerTensorProtocol:
        """Invoke when the protocol reference represents a callable backend function or module.

        Args:
            *args (BackendOperand): Positional operands.
            **kwargs (BackendOperand): Keyword operands.

        Returns:
            EagerTensorProtocol: Invoked result tensor or object.
        """
        ...


class EagerTensorProtocol(
    _BaseTensorProto,
    _UpperCaseSymbolProto,
    _DomainOpsAFProto,
    _DomainOpsGNProto,
    _DomainOpsOSProto,
    _DomainOpsTZProto,
    _DomainOpsPrivateProto,
    typing.Protocol,
):
    """Composite eager backend tensor and namespace protocol."""

    ndim: int


BackendArray = EagerTensorProtocol
