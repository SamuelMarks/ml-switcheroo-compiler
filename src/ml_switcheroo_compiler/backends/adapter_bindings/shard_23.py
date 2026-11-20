"""Static structural protocol bindings shard 23 (free_vars..gelu)."""

from __future__ import annotations

import types
import typing
from collections.abc import Callable

from ml_switcheroo_compiler.backends.eager_registry.protocols import EagerTensorProtocol
from ml_switcheroo_compiler.backends.eager_registry.types_base import BackendOperand

MemberReader = Callable[
    [typing.Union[types.ModuleType, BackendOperand]],  # noqa: UP007
    tuple[bool, typing.Optional[EagerTensorProtocol]],  # noqa: UP045
]


@typing.runtime_checkable
class _HasMember23_00(typing.Protocol):
    """Structural protocol for objects exposing `free_vars`."""

    free_vars: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember23_01(typing.Protocol):
    """Structural protocol for objects exposing `frexp`."""

    frexp: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember23_02(typing.Protocol):
    """Structural protocol for objects exposing `from_array`."""

    from_array: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember23_03(typing.Protocol):
    """Structural protocol for objects exposing `from_arrays`."""

    from_arrays: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember23_04(typing.Protocol):
    """Structural protocol for objects exposing `from_batches`."""

    from_batches: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember23_05(typing.Protocol):
    """Structural protocol for objects exposing `from_buffer`."""

    from_buffer: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember23_06(typing.Protocol):
    """Structural protocol for objects exposing `from_config`."""

    from_config: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember23_07(typing.Protocol):
    """Structural protocol for objects exposing `from_cupy_dtype`."""

    from_cupy_dtype: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember23_08(typing.Protocol):
    """Structural protocol for objects exposing `from_dask_dtype`."""

    from_dask_dtype: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember23_09(typing.Protocol):
    """Structural protocol for objects exposing `from_dlpack`."""

    from_dlpack: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember23_10(typing.Protocol):
    """Structural protocol for objects exposing `from_iter`."""

    from_iter: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember23_11(typing.Protocol):
    """Structural protocol for objects exposing `from_jax_dtype`."""

    from_jax_dtype: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember23_12(typing.Protocol):
    """Structural protocol for objects exposing `from_keras_dtype`."""

    from_keras_dtype: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember23_13(typing.Protocol):
    """Structural protocol for objects exposing `from_mlx_dtype`."""

    from_mlx_dtype: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember23_14(typing.Protocol):
    """Structural protocol for objects exposing `from_numpy`."""

    from_numpy: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember23_15(typing.Protocol):
    """Structural protocol for objects exposing `from_numpy_dtype`."""

    from_numpy_dtype: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember23_16(typing.Protocol):
    """Structural protocol for objects exposing `from_pydict`."""

    from_pydict: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember23_17(typing.Protocol):
    """Structural protocol for objects exposing `from_regular`."""

    from_regular: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember23_18(typing.Protocol):
    """Structural protocol for objects exposing `from_tf_dtype`."""

    from_tf_dtype: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember23_19(typing.Protocol):
    """Structural protocol for objects exposing `from_torch_dtype`."""

    from_torch_dtype: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember23_20(typing.Protocol):
    """Structural protocol for objects exposing `frombuffer`."""

    frombuffer: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember23_21(typing.Protocol):
    """Structural protocol for objects exposing `fromfile`."""

    fromfile: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember23_22(typing.Protocol):
    """Structural protocol for objects exposing `fromfunction`."""

    fromfunction: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember23_23(typing.Protocol):
    """Structural protocol for objects exposing `fromiter`."""

    fromiter: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember23_24(typing.Protocol):
    """Structural protocol for objects exposing `frompyfunc`."""

    frompyfunc: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember23_25(typing.Protocol):
    """Structural protocol for objects exposing `fromstring`."""

    fromstring: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember23_26(typing.Protocol):
    """Structural protocol for objects exposing `full`."""

    full: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember23_27(typing.Protocol):
    """Structural protocol for objects exposing `full_like`."""

    full_like: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember23_28(typing.Protocol):
    """Structural protocol for objects exposing `fun`."""

    fun: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember23_29(typing.Protocol):
    """Structural protocol for objects exposing `func`."""

    func: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember23_30(typing.Protocol):
    """Structural protocol for objects exposing `func2`."""

    func2: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember23_31(typing.Protocol):
    """Structural protocol for objects exposing `func_add`."""

    func_add: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember23_32(typing.Protocol):
    """Structural protocol for objects exposing `func_mul`."""

    func_mul: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember23_33(typing.Protocol):
    """Structural protocol for objects exposing `func_sub`."""

    func_sub: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember23_34(typing.Protocol):
    """Structural protocol for objects exposing `functional`."""

    functional: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember23_35(typing.Protocol):
    """Structural protocol for objects exposing `fwd`."""

    fwd: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember23_36(typing.Protocol):
    """Structural protocol for objects exposing `gamma`."""

    gamma: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember23_37(typing.Protocol):
    """Structural protocol for objects exposing `gather`."""

    gather: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember23_38(typing.Protocol):
    """Structural protocol for objects exposing `gather_mm`."""

    gather_mm: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember23_39(typing.Protocol):
    """Structural protocol for objects exposing `gather_nd`."""

    gather_nd: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember23_40(typing.Protocol):
    """Structural protocol for objects exposing `gaussian_blur`."""

    gaussian_blur: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember23_41(typing.Protocol):
    """Structural protocol for objects exposing `gaussian_nll_loss`."""

    gaussian_nll_loss: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember23_42(typing.Protocol):
    """Structural protocol for objects exposing `gcd`."""

    gcd: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember23_43(typing.Protocol):
    """Structural protocol for objects exposing `ge`."""

    ge: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember23_44(typing.Protocol):
    """Structural protocol for objects exposing `gelu`."""

    gelu: EagerTensorProtocol


SHARD_23_READERS: dict[str, MemberReader] = {
    "free_vars": lambda t: (True, t.free_vars) if isinstance(t, _HasMember23_00) else (False, None),
    "frexp": lambda t: (True, t.frexp) if isinstance(t, _HasMember23_01) else (False, None),
    "from_array": lambda t: (True, t.from_array) if isinstance(t, _HasMember23_02) else (False, None),
    "from_arrays": lambda t: (True, t.from_arrays) if isinstance(t, _HasMember23_03) else (False, None),
    "from_batches": lambda t: (True, t.from_batches) if isinstance(t, _HasMember23_04) else (False, None),
    "from_buffer": lambda t: (True, t.from_buffer) if isinstance(t, _HasMember23_05) else (False, None),
    "from_config": lambda t: (True, t.from_config) if isinstance(t, _HasMember23_06) else (False, None),
    "from_cupy_dtype": lambda t: (True, t.from_cupy_dtype) if isinstance(t, _HasMember23_07) else (False, None),
    "from_dask_dtype": lambda t: (True, t.from_dask_dtype) if isinstance(t, _HasMember23_08) else (False, None),
    "from_dlpack": lambda t: (True, t.from_dlpack) if isinstance(t, _HasMember23_09) else (False, None),
    "from_iter": lambda t: (True, t.from_iter) if isinstance(t, _HasMember23_10) else (False, None),
    "from_jax_dtype": lambda t: (True, t.from_jax_dtype) if isinstance(t, _HasMember23_11) else (False, None),
    "from_keras_dtype": lambda t: (True, t.from_keras_dtype) if isinstance(t, _HasMember23_12) else (False, None),
    "from_mlx_dtype": lambda t: (True, t.from_mlx_dtype) if isinstance(t, _HasMember23_13) else (False, None),
    "from_numpy": lambda t: (True, t.from_numpy) if isinstance(t, _HasMember23_14) else (False, None),
    "from_numpy_dtype": lambda t: (True, t.from_numpy_dtype) if isinstance(t, _HasMember23_15) else (False, None),
    "from_pydict": lambda t: (True, t.from_pydict) if isinstance(t, _HasMember23_16) else (False, None),
    "from_regular": lambda t: (True, t.from_regular) if isinstance(t, _HasMember23_17) else (False, None),
    "from_tf_dtype": lambda t: (True, t.from_tf_dtype) if isinstance(t, _HasMember23_18) else (False, None),
    "from_torch_dtype": lambda t: (True, t.from_torch_dtype) if isinstance(t, _HasMember23_19) else (False, None),
    "frombuffer": lambda t: (True, t.frombuffer) if isinstance(t, _HasMember23_20) else (False, None),
    "fromfile": lambda t: (True, t.fromfile) if isinstance(t, _HasMember23_21) else (False, None),
    "fromfunction": lambda t: (True, t.fromfunction) if isinstance(t, _HasMember23_22) else (False, None),
    "fromiter": lambda t: (True, t.fromiter) if isinstance(t, _HasMember23_23) else (False, None),
    "frompyfunc": lambda t: (True, t.frompyfunc) if isinstance(t, _HasMember23_24) else (False, None),
    "fromstring": lambda t: (True, t.fromstring) if isinstance(t, _HasMember23_25) else (False, None),
    "full": lambda t: (True, t.full) if isinstance(t, _HasMember23_26) else (False, None),
    "full_like": lambda t: (True, t.full_like) if isinstance(t, _HasMember23_27) else (False, None),
    "fun": lambda t: (True, t.fun) if isinstance(t, _HasMember23_28) else (False, None),
    "func": lambda t: (True, t.func) if isinstance(t, _HasMember23_29) else (False, None),
    "func2": lambda t: (True, t.func2) if isinstance(t, _HasMember23_30) else (False, None),
    "func_add": lambda t: (True, t.func_add) if isinstance(t, _HasMember23_31) else (False, None),
    "func_mul": lambda t: (True, t.func_mul) if isinstance(t, _HasMember23_32) else (False, None),
    "func_sub": lambda t: (True, t.func_sub) if isinstance(t, _HasMember23_33) else (False, None),
    "functional": lambda t: (True, t.functional) if isinstance(t, _HasMember23_34) else (False, None),
    "fwd": lambda t: (True, t.fwd) if isinstance(t, _HasMember23_35) else (False, None),
    "gamma": lambda t: (True, t.gamma) if isinstance(t, _HasMember23_36) else (False, None),
    "gather": lambda t: (True, t.gather) if isinstance(t, _HasMember23_37) else (False, None),
    "gather_mm": lambda t: (True, t.gather_mm) if isinstance(t, _HasMember23_38) else (False, None),
    "gather_nd": lambda t: (True, t.gather_nd) if isinstance(t, _HasMember23_39) else (False, None),
    "gaussian_blur": lambda t: (True, t.gaussian_blur) if isinstance(t, _HasMember23_40) else (False, None),
    "gaussian_nll_loss": lambda t: (True, t.gaussian_nll_loss) if isinstance(t, _HasMember23_41) else (False, None),
    "gcd": lambda t: (True, t.gcd) if isinstance(t, _HasMember23_42) else (False, None),
    "ge": lambda t: (True, t.ge) if isinstance(t, _HasMember23_43) else (False, None),
    "gelu": lambda t: (True, t.gelu) if isinstance(t, _HasMember23_44) else (False, None),
}
