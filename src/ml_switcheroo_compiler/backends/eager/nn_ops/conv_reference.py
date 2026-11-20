"""Generic backend-parametric eager reference operations for `nn_ops.conv_reference`."""

from __future__ import annotations

import itertools
import logging
import types
from collections.abc import Sequence

import pydantic

from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    EagerTensorProtocol,
    coerce_int,
    coerce_int_list,
    coerce_operand_list,
    coerce_shape,
    coerce_str,
    global_eager_registry,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, adapt_backend_module
from ml_switcheroo_compiler.core.constants import MATRIX_RANK, MIN_CONV_RANK
from ml_switcheroo_compiler.core.shape import OpValue as OpValue
from ml_switcheroo_compiler.core.shape import SliceType
from ml_switcheroo_compiler.ops.configs import ConvConfig


def _ensure_arr(val: BackendArg) -> EagerTensorProtocol:
    """Coerce `val` to an eager array via the active backend module.

    Args:
        val (BackendArg): Input array or scalar value.

    Returns:
        EagerTensorProtocol: Coerced backend array.
    """
    return _resolve_eager_mod(None).asarray(val)


class ConvDimSpecs(pydantic.BaseModel):
    """Configuration class for conv dim specs."""

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    spatial_dims: int
    lhs_spec: Sequence[int]
    rhs_spec: Sequence[int]
    out_spec: tuple[int, ...]


class ConvExecutionState(pydantic.BaseModel):
    """Configuration class for conv execution state."""

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    lhs_pad: EagerTensorProtocol
    rhs_c: EagerTensorProtocol
    out: EagerTensorProtocol
    config: ConvConfig
    spatial_dims: int


def _get_transpose(spec: BackendArg, default: str) -> tuple[int, ...]:
    """Get transpose.

    Args:
        spec (BackendArg): Input `spec` argument (spec).
        default (str): Input `default` argument (default).

    Returns:
        tuple: Computed `tuple` output.

    Raises:
        TypeError: If invalid inputs or unsupported runtime conditions trigger `TypeError`.
    """
    if isinstance(spec, str):
        try:
            return tuple(spec.index(c) for c in default)
        except (ValueError, TypeError) as e:
            logging.error(f"CRASH: spec={spec}, default={default}")
            raise e
    if isinstance(spec, (int, float, bool)):
        raise TypeError(f"Expected dimension specification string or integer sequence, got {type(spec).__name__}")
    return tuple(coerce_shape(spec))


def _get_conv_defaults(spatial_dims: int) -> tuple[str, str]:
    """Execute the eager get conv defaults operation on the active backend array.

    Args:
        spatial_dims (int): Input `spatial_dims` argument (spatial dims).

    Returns:
        tuple[str, str]: Computed `tuple[str, str]` output.
    """
    if spatial_dims == 1:
        return ("NCW", "OIW")
    if spatial_dims == MATRIX_RANK:
        return ("NCHW", "OIHW")
    return ("NCDHW", "OIDHW")


def _parse_conv_dimension_numbers(
    lhs_ndim: int, rhs_ndim: int, spatial_dims: int, dimension_numbers: BackendArg
) -> ConvDimSpecs:
    """Parse dimension numbers for convolution.

    Args:
        lhs_ndim (int): Input `lhs_ndim` argument (lhs ndim).
        rhs_ndim (int): Input `rhs_ndim` argument (rhs ndim).
        spatial_dims (int): Input `spatial_dims` argument (spatial dims).
        dimension_numbers (BackendArg): Input `dimension_numbers` argument (dimension numbers).

    Returns:
        ConvDimSpecs: Computed `ConvDimSpecs` output.
    """
    if dimension_numbers is None:
        lhs_spec: tuple[BackendArg, ...] = (0, 1) + tuple(range(2, lhs_ndim))
        rhs_spec: tuple[BackendArg, ...] = (0, 1) + tuple(range(2, rhs_ndim))
        out_spec: tuple[BackendArg, ...] = (0, 1) + tuple(range(2, lhs_ndim))
        return ConvDimSpecs(
            spatial_dims=spatial_dims,
            lhs_spec=coerce_shape(lhs_spec),
            rhs_spec=coerce_shape(rhs_spec),
            out_spec=coerce_shape(out_spec),
        )
    if isinstance(dimension_numbers, tuple) and len(dimension_numbers) == MIN_CONV_RANK:
        (lhs_spec, rhs_spec, out_spec) = dimension_numbers
        (lhs_default, rhs_default) = _get_conv_defaults(spatial_dims)
        return ConvDimSpecs(
            spatial_dims=spatial_dims,
            lhs_spec=_get_transpose(lhs_spec, lhs_default),
            rhs_spec=_get_transpose(rhs_spec, rhs_default),
            out_spec=_get_transpose(out_spec, lhs_default),
        )
    return ConvDimSpecs(
        spatial_dims=spatial_dims,
        lhs_spec=coerce_shape(adapt_backend_module(dimension_numbers).require_symbol("lhs_spec")),
        rhs_spec=coerce_shape(adapt_backend_module(dimension_numbers).require_symbol("rhs_spec")),
        out_spec=coerce_shape(adapt_backend_module(dimension_numbers).require_symbol("out_spec")),
    )


def _calculate_same_padding(
    lhs_shape: tuple[int, ...],
    rhs_shape: tuple[int, ...],
    rhs_dilation: Sequence[int],
    window_strides: Sequence[int],
) -> list[tuple[int, int]]:
    """Execute the eager calculate same padding operation on the active backend array.

    Args:
        lhs_shape (tuple[int, ...]): Input `lhs_shape` argument (lhs shape).
        rhs_shape (tuple[int, ...]): Input `rhs_shape` argument (rhs shape).
        rhs_dilation (Sequence[int]): Input `rhs_dilation` argument (rhs dilation).
        window_strides (Sequence[int]): Input `window_strides` argument (window strides).

    Returns:
        list[tuple[int, int]]: Computed `list[tuple[int, int]]` output.
    """
    arr_mod = _resolve_eager_mod(None)
    spatial_dims: int = len(lhs_shape) - 2
    pad_list: list[tuple[int, int]] = []
    for i in range(spatial_dims):
        in_size: int = lhs_shape[2 + i]
        filter_size: int = (rhs_shape[2 + i] - 1) * rhs_dilation[i] + 1
        out_size: int = int(arr_mod.ceil(float(in_size) / window_strides[i]))
        pad_total: int = max((out_size - 1) * window_strides[i] + filter_size - in_size, 0)
        pad_front: int = pad_total // 2
        pad_back: int = pad_total - pad_front
        pad_list.append((pad_front, pad_back))
    return pad_list


def _calculate_conv_padding(
    config: ConvConfig, lhs_shape: tuple[int, ...], rhs_shape: tuple[int, ...]
) -> list[tuple[int, int]]:
    """Calculate convolution padding.

    Args:
        config (ConvConfig): Input `config` argument (config).
        lhs_shape (tuple[int, ...]): Input `lhs_shape` argument (lhs shape).
        rhs_shape (tuple[int, ...]): Input `rhs_shape` argument (rhs shape).

    Returns:
        list: Computed `list` output.
    """
    spatial_dims: int = len(lhs_shape) - 2
    padding: Sequence[tuple[int, int]] | str | None = config.padding
    if not isinstance(padding, str):
        if padding is None:
            return [(0, 0)] * spatial_dims
        return list(padding)
    if padding == "VALID":
        return [(0, 0)] * spatial_dims
    if padding == "SAME":
        rhs_dilation: Sequence[int] = config.rhs_dilation if config.rhs_dilation is not None else [1] * spatial_dims
        return _calculate_same_padding(lhs_shape, rhs_shape, rhs_dilation, config.window_strides)
    return [(0, 0)] * spatial_dims


def _apply_conv_dilation(tensor: BackendArg, dilation: Sequence[int], spatial_dims: int) -> BackendArg:
    """Apply dilation to a convolution tensor.

    Args:
        tensor (BackendArg): Input `tensor` argument (tensor).
        dilation (Sequence[int]): Input `dilation` argument (dilation).
        spatial_dims (int): Input `spatial_dims` argument (spatial dims).

    Returns:
        BackendArg: Computed `BackendArg` output.
    """
    arr_mod = _resolve_eager_mod(None)
    if not any(d > 1 for d in dilation):
        return tensor
    new_shape: list[BackendArg] = list(adapt_backend_module(tensor).require_symbol("shape"))
    for i, d in enumerate(dilation):
        new_shape[2 + i] = (adapt_backend_module(tensor).require_symbol("shape")[2 + i] - 1) * d + 1
    dilated: EagerTensorProtocol = _ensure_arr(
        adapt_backend_module(arr_mod).require_symbol("zeros")(
            new_shape, dtype=adapt_backend_module(tensor).require_symbol("dtype")
        )
    )
    slices = [slice(None), slice(None)] + [slice(None, None, d) for d in dilation]
    dilated[tuple(slices)] = _ensure_arr(tensor)
    return dilated


class PatchConfig(pydantic.BaseModel):
    """Configuration for convolution patches."""

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    axes_lhs: list[int]
    axes_rhs: list[int]
    group_in_c: int
    group_out_c: int
    feature_group_count: int


def _compute_conv_patch_group(lhs_patch: BackendArg, rhs_c: BackendArg, config: PatchConfig, g: int) -> BackendArg:
    """Execute the eager compute conv patch group operation on the active backend array.

    Args:
        lhs_patch (BackendArg): Input `lhs_patch` argument (lhs patch).
        rhs_c (BackendArg): Input `rhs_c` argument (rhs c).
        config (PatchConfig): Input `config` argument (config).
        g (int): Input `g` argument (g).

    Returns:
        BackendArg: Computed `BackendArg` output.
    """
    arr_mod = _resolve_eager_mod(None)
    from ml_switcheroo_compiler.backends.eager_registry import coerce_array

    lhs_arr = coerce_array(arr_mod, lhs_patch)
    rhs_arr = coerce_array(arr_mod, rhs_c)
    group_in_c: int = config.group_in_c
    group_out_c: int = config.group_out_c
    lp_g = lhs_arr[:, g * group_in_c : (g + 1) * group_in_c, ...]
    rc_g = rhs_arr[g * group_out_c : (g + 1) * group_out_c, :, ...]
    return adapt_backend_module(arr_mod).require_symbol("tensordot")(
        lp_g, rc_g, axes=(config.axes_lhs, config.axes_rhs)
    )


def _compute_single_patch_grouped(
    lhs_patch: BackendArg, rhs_c: BackendArg, out: BackendArg, spatial_indices: tuple[int, ...], config: PatchConfig
) -> None:
    """Execute the eager compute single patch grouped operation on the active backend array.

    Args:
        lhs_patch (BackendArg): Input `lhs_patch` argument (lhs patch).
        rhs_c (BackendArg): Input `rhs_c` argument (rhs c).
        out (BackendArg): Input `out` argument (out).
        spatial_indices (tuple[int, ...]): Input `spatial_indices` argument (spatial indices).
        config (PatchConfig): Input `config` argument (config).
    """
    from ml_switcheroo_compiler.backends.eager_registry import coerce_array

    out_arr = coerce_array(_resolve_eager_mod(None), out)
    for g in range(config.feature_group_count):
        res: EagerTensorProtocol = _ensure_arr(_compute_conv_patch_group(lhs_patch, rhs_c, config, g))
        s_c: SliceType = slice(g * config.group_out_c, (g + 1) * config.group_out_c)
        out_arr[tuple([slice(None), s_c] + list(spatial_indices))] = res


def _get_patch_slices(
    spatial_indices: tuple[int, ...], window_strides: Sequence[int], rhs_shape: tuple[int, ...]
) -> tuple[SliceType, ...]:
    """Execute the eager get patch slices operation on the active backend array.

    Args:
        spatial_indices (tuple[int, ...]): Input `spatial_indices` argument (spatial indices).
        window_strides (Sequence[int]): Input `window_strides` argument (window strides).
        rhs_shape (tuple[int, ...]): Input `rhs_shape` argument (rhs shape).

    Returns:
        tuple[SliceType, ...]: Computed `tuple[SliceType, ...]` output.
    """
    slices: list[SliceType] = [slice(None), slice(None)]
    for i, idx in enumerate(spatial_indices):
        start: int = idx * window_strides[i]
        end: int = start + rhs_shape[2 + i]
        slices.append(slice(start, end))
    return tuple(slices)


def _compute_single_patch(state: ConvExecutionState, spatial_indices: tuple[int, ...]) -> None:
    """Execute the eager compute single patch operation on the active backend array.

    Args:
        state (ConvExecutionState): Input `state` argument (state).
        spatial_indices (tuple[int, ...]): Input `spatial_indices` argument (spatial indices).
    """
    arr_mod = _resolve_eager_mod(None)
    slices: tuple[BackendArg, ...] = _get_patch_slices(spatial_indices, state.config.window_strides, state.rhs_c.shape)
    lhs_patch: EagerTensorProtocol = _ensure_arr(state.lhs_pad[slices])
    axes_lhs: list[int] = [1] + list(range(2, 2 + state.spatial_dims))
    axes_rhs: list[int] = [1] + list(range(2, 2 + state.spatial_dims))
    if state.config.feature_group_count > 1:
        in_channels: int = lhs_patch.shape[1]
        out_channels: int = state.rhs_c.shape[0]
        patch_config: PatchConfig = PatchConfig(
            axes_lhs=axes_lhs,
            axes_rhs=axes_rhs,
            group_in_c=in_channels // state.config.feature_group_count,
            group_out_c=out_channels // state.config.feature_group_count,
            feature_group_count=state.config.feature_group_count,
        )
        _compute_single_patch_grouped(lhs_patch, state.rhs_c, state.out, spatial_indices, patch_config)
    else:
        res: EagerTensorProtocol = _ensure_arr(
            adapt_backend_module(arr_mod).require_symbol("tensordot")(lhs_patch, state.rhs_c, axes=(axes_lhs, axes_rhs))
        )
        state.out[tuple([slice(None), slice(None)] + list(spatial_indices))] = res


def _compute_conv_patches(lhs_pad: BackendArg, rhs_c: BackendArg, out: BackendArg, config: ConvConfig) -> None:
    """Execute the eager compute conv patches operation on the active backend array.

    Args:
        lhs_pad (BackendArg): Input `lhs_pad` argument (lhs pad).
        rhs_c (BackendArg): Input `rhs_c` argument (rhs c).
        out (BackendArg): Input `out` argument (out).
        config (ConvConfig): Input `config` argument (config).
    """
    spatial_dims: int = len(adapt_backend_module(lhs_pad).require_symbol("shape")) - 2
    out_spatial = coerce_shape(adapt_backend_module(out).require_symbol("shape"))[2:]
    for spatial_indices in itertools.product(*[range(int(d)) for d in out_spatial]):
        state: ConvExecutionState = ConvExecutionState(
            lhs_pad=_ensure_arr(lhs_pad),
            rhs_c=_ensure_arr(rhs_c),
            out=_ensure_arr(out),
            config=config,
            spatial_dims=spatial_dims,
        )
        _compute_single_patch(state, spatial_indices)


def _apply_conv_padding_helper(lhs_c: BackendArg, rhs_c: BackendArg, config: ConvConfig) -> BackendArg:
    """Execute the eager apply conv padding helper operation on the active backend array.

    Args:
        lhs_c (BackendArg): Input `lhs_c` argument (lhs c).
        rhs_c (BackendArg): Input `rhs_c` argument (rhs c).
        config (ConvConfig): Input `config` argument (config).

    Returns:
        BackendArg: Computed `BackendArg` output.
    """
    arr_mod = _resolve_eager_mod(None)
    lhs_sh = coerce_shape(adapt_backend_module(lhs_c).require_symbol("shape"))
    rhs_sh = coerce_shape(adapt_backend_module(rhs_c).require_symbol("shape"))
    pad_list: list[tuple[int, int]] = _calculate_conv_padding(config, lhs_sh, rhs_sh)
    pad_width: tuple[tuple[int, int], ...] = tuple((int(x[0]), int(x[1])) for x in [(0, 0), (0, 0)] + pad_list)
    new_shape = tuple(int(s) + int(p[0]) + int(p[1]) for s, p in zip(lhs_sh, pad_width))
    padded = arr_mod.zeros(new_shape, dtype=adapt_backend_module(lhs_c).require_symbol("dtype"))
    slices = tuple(slice(int(p[0]), int(p[0]) + int(s)) for s, p in zip(lhs_sh, pad_width))
    padded[slices] = _ensure_arr(lhs_c)
    return padded


def _preprocess_conv_tensors(
    lhs: BackendArg,
    rhs: BackendArg,
    config: ConvConfig,
    specs: ConvDimSpecs,
) -> tuple[EagerTensorProtocol, EagerTensorProtocol]:
    """Execute the eager preprocess conv tensors operation on the active backend array.

    Args:
        lhs (BackendArg): Input `lhs` argument (lhs).
        rhs (BackendArg): Input `rhs` argument (rhs).
        config (ConvConfig): Input `config` argument (config).
        specs (ConvDimSpecs): Input `specs` argument (specs).

    Returns:
        tuple[EagerTensorProtocol, EagerTensorProtocol]:
            Computed `tuple[EagerTensorProtocol, EagerTensorProtocol]` output.
    """
    arr_mod = _resolve_eager_mod(None)
    lhs_c: EagerTensorProtocol = _ensure_arr(
        adapt_backend_module(arr_mod).require_symbol("transpose")(lhs, specs.lhs_spec)
    )
    rhs_c: EagerTensorProtocol = _ensure_arr(
        adapt_backend_module(arr_mod).require_symbol("transpose")(rhs, specs.rhs_spec)
    )
    if config.feature_group_count > 1:
        in_channels: int = lhs_c.shape[1]
        expected_rhs_in: int = in_channels // config.feature_group_count
        if rhs_c.shape[1] != expected_rhs_in:
            if rhs_c.shape[1] == in_channels:
                permutation: tuple[int, ...] = (1, 0) + tuple(range(2, rhs_c.ndim))
                rhs_c = arr_mod.transpose(rhs_c, permutation)
                new_shape: tuple[int, ...] = (
                    rhs_c.shape[0] * rhs_c.shape[1] // expected_rhs_in,
                    expected_rhs_in,
                ) + rhs_c.shape[2:]
                rhs_c = arr_mod.reshape(rhs_c, new_shape)
    lhs_dilation: Sequence[int] = config.lhs_dilation if config.lhs_dilation is not None else [1] * specs.spatial_dims
    rhs_dilation: Sequence[int] = config.rhs_dilation if config.rhs_dilation is not None else [1] * specs.spatial_dims
    lhs_dilated: EagerTensorProtocol = _ensure_arr(_apply_conv_dilation(lhs_c, lhs_dilation, specs.spatial_dims))
    rhs_c = _ensure_arr(_apply_conv_dilation(rhs_c, rhs_dilation, specs.spatial_dims))
    lhs_pad: EagerTensorProtocol = _ensure_arr(_apply_conv_padding_helper(lhs_dilated, rhs_c, config))
    return (lhs_pad, rhs_c)


def _compute_out_shape(
    lhs_pad_shape: tuple[int, ...],
    rhs_c_shape: tuple[int, ...],
    spatial_dims: int,
    window_strides: Sequence[int],
) -> list[int]:
    """Execute the eager compute out shape operation on the active backend array.

    Args:
        lhs_pad_shape (tuple[int, ...]): Input `lhs_pad_shape` argument (lhs pad shape).
        rhs_c_shape (tuple[int, ...]): Input `rhs_c_shape` argument (rhs c shape).
        spatial_dims (int): Input `spatial_dims` argument (spatial dims).
        window_strides (Sequence[int]): Input `window_strides` argument (window strides).

    Returns:
        list[int]: Computed `list[int]` output.
    """
    out_spatial: list[int] = [
        (lhs_pad_shape[2 + i] - rhs_c_shape[2 + i]) // window_strides[i] + 1 for i in range(spatial_dims)
    ]
    return [lhs_pad_shape[0], rhs_c_shape[0]] + out_spatial


def _get_inv_out_spec(out_spec: tuple[int, ...]) -> list[int]:
    """Execute the eager get inv out spec operation on the active backend array.

    Args:
        out_spec (tuple[int, ...]): Input `out_spec` argument (out spec).

    Returns:
        list[int]: Computed `list[int]` output.
    """
    inv_out_spec: list[int] = [0] * len(out_spec)
    for i, p in enumerate(out_spec):
        inv_out_spec[p] = i
    return inv_out_spec


def _conv_general_dilated(lhs: BackendArg, rhs: BackendArg, config: ConvConfig, **kwargs: BackendArg) -> BackendArg:
    """Evaluate.

    Args:
        lhs (BackendArg): Input `lhs` argument (lhs).
        rhs (BackendArg): Input `rhs` argument (rhs).
        config (ConvConfig): Input `config` argument (config).
        **kwargs (BackendArg): Keyword args.

    Returns:
        BackendArg: Computed `BackendArg` output.
    """
    arr_mod = _resolve_eager_mod(None)
    lhs_arr: EagerTensorProtocol = arr_mod.asarray(lhs)
    rhs_arr: EagerTensorProtocol = arr_mod.asarray(rhs)
    spatial_dims: int = lhs_arr.ndim - 2
    specs: ConvDimSpecs = _parse_conv_dimension_numbers(
        lhs_arr.ndim, rhs_arr.ndim, spatial_dims, config.dimension_numbers
    )
    lhs_pad, rhs_c = _preprocess_conv_tensors(lhs_arr, rhs_arr, config, specs)
    out_shape: list[int] = _compute_out_shape(lhs_pad.shape, rhs_c.shape, spatial_dims, config.window_strides)
    out: EagerTensorProtocol = arr_mod.zeros(out_shape, dtype=lhs_arr.dtype)
    _compute_conv_patches(lhs_pad, rhs_c, out, config)
    inv_out_spec: list[int] = _get_inv_out_spec(specs.out_spec)
    return arr_mod.transpose(out, inv_out_spec)


@global_eager_registry.register("ConvGeneralDilated")
def _np_conv_general_dilated(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np conv general dilated operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
        BackendArg: Computed `BackendArg` output.
    """
    from ml_switcheroo_compiler.ops.configs import ConvConfig

    if len(args) == 4:
        from ml_switcheroo_compiler.backends.eager_registry import unpack_pair

        lhs, rhs, window_strides, padding = args
        pad_val = (
            padding
            if isinstance(padding, str)
            else [
                (int(coerce_int(lo)), int(coerce_int(hi)))
                for lo, hi in (unpack_pair(p) for p in coerce_operand_list(padding))
            ]
        )
        config = ConvConfig(window_strides=coerce_shape(window_strides), padding=pad_val)
        return _conv_general_dilated(lhs, rhs, config, **kwargs)
    return _conv_general_dilated(
        args[0],
        args[1],
        args[2] if len(args) > 2 and isinstance(args[2], ConvConfig) else ConvConfig(),
        **kwargs,
    )


@global_eager_registry.register("Conv2D")
def _np_conv2d(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Evaluate Conv2D operation in NumPy eager mode.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): The backend module (np).
        *args (BackendArg): Positional args (lhs, rhs).
        **kwargs (BackendArg): Keyword args (stride, padding, groups).

    Returns:
        BackendArg: Result of 2D convolution.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    from ml_switcheroo_compiler.ops.configs import ConvConfig

    lhs, rhs = arr_mod.asarray(args[0]), arr_mod.asarray(args[1])
    stride = kwargs.get("stride", kwargs.get("strides", [1, 1]))
    if isinstance(stride, int):
        stride = [stride, stride]
    raw_padding = kwargs.get("padding", "SAME")
    padding: str = raw_padding if isinstance(raw_padding, str) else "SAME"
    groups = int(coerce_int(kwargs.get("groups", 1)))

    if groups > 1:
        c_in_g = lhs.shape[1] // groups
        c_out_g = rhs.shape[0] // groups
        outs = []
        for g in range(groups):
            lhs_g = lhs[:, g * c_in_g : (g + 1) * c_in_g, :, :]
            rhs_g = rhs[g * c_out_g : (g + 1) * c_out_g, :, :, :]
            cfg = ConvConfig(window_strides=coerce_shape(stride), padding=padding)
            outs.append(_conv_general_dilated(lhs_g, rhs_g, cfg))
        return adapt_backend_module(arr_mod).require_symbol("concatenate")(outs, axis=1)

    config = ConvConfig(window_strides=coerce_shape(stride), padding=padding)
    return _conv_general_dilated(lhs, rhs, config)


def _calculate_conv_transpose_padding(
    spatial_in: Sequence[int],
    spatial_k: Sequence[int],
    strides: Sequence[int],
    padding: str,
) -> tuple[list[int], list[tuple[int, int]]]:
    """Calculate output spatial shapes and paddings for transposed convolution.

    Args:
        spatial_in (Sequence[int]): Input spatial dimensions.
        spatial_k (Sequence[int]): Kernel spatial dimensions.
        strides (Sequence[int]): Stride values along spatial dimensions.
        padding (str): Padding mode ('VALID' or 'SAME').

    Returns:
        tuple[list[int], list[tuple[int, int]]]: Output spatial dimensions and padding pairs.
    """
    out_spatial: list[int] = []
    pads: list[tuple[int, int]] = []
    for s_in, k, st in zip(spatial_in, spatial_k, strides):
        s_out: int = (s_in - 1) * st + k if padding == "VALID" else s_in * st
        out_spatial.append(s_out)
        total_pad: int = s_out - s_in * st + st + k - 2
        pad_r: int = total_pad // 2
        pad_l: int = total_pad - pad_r
        pads.append((pad_l, pad_r))
    return out_spatial, pads


def _build_conv_transpose_config(
    spatial_in: tuple[int, ...] | list[int],
    spatial_k: tuple[int, ...] | list[int],
    strides: tuple[int, ...] | list[int],
    pads: list[tuple[int, int]],
) -> tuple[tuple[slice[int | None, int | None, int | None], ...], ConvConfig]:
    """Build the configuration and reversed slices for ConvTranspose.

    Args:
        spatial_in: Input spatial shape.
        spatial_k: Kernel spatial shape.
        strides: Stride tuple.
        pads: Padding list.

    Returns:
        tuple: (slices, config_obj)
    """
    from ml_switcheroo_compiler.ops.configs import ConvConfig

    slices: tuple[slice[int | None, int | None, int | None], ...] = (slice(None), slice(None)) + tuple(
        slice(None, None, -1) for _ in spatial_k
    )
    spatial_rank: int = len(spatial_in)
    n_dims: int = spatial_rank + 2
    lhs_spec: tuple[int, ...] = (0, 1) + tuple(range(2, n_dims))
    rhs_spec: tuple[int, ...] = (0, 1) + tuple(range(2, n_dims))
    out_spec: tuple[int, ...] = lhs_spec
    config_obj: ConvConfig = ConvConfig(
        window_strides=(1,) * spatial_rank,
        padding=pads,
        lhs_dilation=strides,
        rhs_dilation=(1,) * spatial_rank,
        dimension_numbers=(lhs_spec, rhs_spec, out_spec),
    )
    return slices, config_obj


@global_eager_registry.register("ConvTranspose")
@global_eager_registry.register("ConvTranspose2D")
@global_eager_registry.register("Conv2DTranspose")
def _np_conv_transpose(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np conv transpose operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
        BackendArg: Computed `BackendArg` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    lhs: EagerTensorProtocol = arr_mod.asarray(args[0])
    rhs: EagerTensorProtocol = arr_mod.asarray(args[1])
    raw_strides = args[2] if len(args) > 2 else kwargs.get("strides", kwargs.get("stride", 1))
    padding: str = coerce_str(args[3] if len(args) > 3 else kwargs.get("padding", "VALID"))
    spatial_in: tuple[int, ...] = lhs.shape[2:]
    spatial_k: tuple[int, ...] = rhs.shape[2:]
    strides: Sequence[int] = (
        (raw_strides,) * len(spatial_in) if isinstance(raw_strides, int) else coerce_shape(raw_strides)
    )
    out_spatial, pads = _calculate_conv_transpose_padding(spatial_in, spatial_k, strides, padding)
    slices, config_obj = _build_conv_transpose_config(spatial_in, spatial_k, list(strides), pads)
    rhs_rev: EagerTensorProtocol = rhs[slices]
    return _np_conv_general_dilated(backend_module, lhs, rhs_rev, config_obj)


def _compute_conv2d_padding_values(
    in_hw: tuple[int, int],
    k_hw: tuple[int, int],
    s_hw: tuple[int, int],
    d_hw: tuple[int, int],
    padding: BackendArg,
    out_hw: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Calculate 4-sided padding values (top, bottom, left, right) for 2D convolution.

    Args:
        in_hw (tuple[int, int]): Input spatial shape (in_h, in_w).
        k_hw (tuple[int, int]): Kernel spatial shape (k_h, k_w).
        s_hw (tuple[int, int]): Stride spatial values (s_h, s_w).
        d_hw (tuple[int, int]): Dilation spatial values (d_h, d_w).
        padding (BackendArg): Padding specification ("SAME", "VALID", int, or tuple).
        out_hw (tuple[int, int]): Output spatial shape (out_h, out_w).

    Returns:
        tuple[int, int, int, int]: Padding tuple (p_top, p_bottom, p_left, p_right).
    """
    in_h, in_w = in_hw
    k_h, k_w = k_hw
    s_h, s_w = s_hw
    d_h, d_w = d_hw
    out_h, out_w = out_hw
    if isinstance(padding, (list, tuple)):
        if len(padding) == 2:
            return (
                int(coerce_int(padding[0])),
                int(coerce_int(padding[0])),
                int(coerce_int(padding[1])),
                int(coerce_int(padding[1])),
            )
        if len(padding) == 4:
            return (
                int(coerce_int(padding[0])),
                int(coerce_int(padding[1])),
                int(coerce_int(padding[2])),
                int(coerce_int(padding[3])),
            )
    if isinstance(padding, int):
        return padding, padding, padding, padding
    if isinstance(padding, str) and padding.upper() == "SAME":
        p_h = max(0, (out_h - 1) * s_h + (k_h - 1) * d_h + 1 - in_h)
        p_w = max(0, (out_w - 1) * s_w + (k_w - 1) * d_w + 1 - in_w)
        p_top = p_h // 2
        p_bottom = p_h - p_top
        p_left = p_w // 2
        p_right = p_w - p_left
        return p_top, p_bottom, p_left, p_right
    return 0, 0, 0, 0


@global_eager_registry.register("Conv2DInputGrad")
def _np_conv2d_input_grad(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> EagerTensorProtocol:
    """Compute exact gradient of Conv2D with respect to the input tensor.

    Supports arbitrary strides, padding, dilation, and group convolutions.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Backend provider module.
        *args (BackendArg): Positional args [d_out, weight].
        **kwargs (BackendArg): Conv attributes (stride, padding, dilation, groups, target_shape).

    Returns:
        EagerTensorProtocol: Computed gradient tensor with shape matching the original input.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    d_out: EagerTensorProtocol = arr_mod.asarray(args[0])
    weight: EagerTensorProtocol = arr_mod.asarray(args[1])

    stride = kwargs.get("stride", kwargs.get("strides", (1, 1)))
    s_dims = (stride, stride) if isinstance(stride, int) else coerce_shape(stride)
    s_h, s_w = s_dims[0], s_dims[1]

    dilation = kwargs.get("dilation", kwargs.get("dilations", (1, 1)))
    d_dims = (dilation, dilation) if isinstance(dilation, int) else coerce_shape(dilation)
    d_h, d_w = d_dims[0], d_dims[1]

    groups = int(coerce_int(kwargs.get("groups", 1)))
    padding = kwargs.get("padding", "SAME")

    target_shape = kwargs.get("target_shape")
    if target_shape is not None:
        n_batch, c_in, in_h, in_w = (int(s) for s in coerce_int_list(target_shape))
    else:
        n_batch = d_out.shape[0]
        c_in = weight.shape[1] * groups
        in_h = (d_out.shape[2] - 1) * s_h + 1
        in_w = (d_out.shape[3] - 1) * s_w + 1

    c_out = weight.shape[0]
    k_h, k_w = weight.shape[2], weight.shape[3]
    out_h, out_w = d_out.shape[2], d_out.shape[3]

    p_top, p_bottom, p_left, p_right = _compute_conv2d_padding_values(
        (in_h, in_w), (k_h, k_w), (s_h, s_w), (d_h, d_w), padding, (out_h, out_w)
    )

    pad_h = in_h + p_top + p_bottom
    pad_w = in_w + p_left + p_right
    dx_pad: EagerTensorProtocol = arr_mod.zeros((n_batch, c_in, pad_h, pad_w), dtype=d_out.dtype)

    c_in_g = c_in // groups
    c_out_g = c_out // groups

    for g in range(groups):
        dy_g = d_out[:, g * c_out_g : (g + 1) * c_out_g, :, :]
        w_g = weight[g * c_out_g : (g + 1) * c_out_g, :, :, :]

        for kh in range(k_h):
            h_start = kh * d_h
            h_end = h_start + out_h * s_h
            for kw in range(k_w):
                w_start = kw * d_w
                w_end = w_start + out_w * s_w
                w_sub = w_g[:, :, kh, kw]
                term = _ensure_arr(
                    adapt_backend_module(arr_mod).require_symbol("einsum")("n o h w, o i -> n i h w", dy_g, w_sub)
                )
                dx_pad[:, g * c_in_g : (g + 1) * c_in_g, h_start:h_end:s_h, w_start:w_end:s_w] += term

    return dx_pad[:, :, p_top : p_top + in_h, p_left : p_left + in_w]


@global_eager_registry.register("Conv2DWeightGrad")
def _np_conv2d_weight_grad(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> EagerTensorProtocol:
    """Compute exact gradient of Conv2D with respect to the weight tensor.

    Supports arbitrary strides, padding, dilation, and group convolutions.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Backend provider module.
        *args (BackendArg): Positional args [input, d_out].
        **kwargs (BackendArg): Conv attributes (stride, padding, dilation, groups, target_shape).

    Returns:
        EagerTensorProtocol: Computed gradient tensor with shape matching original weight.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    x: EagerTensorProtocol = arr_mod.asarray(args[0])
    d_out: EagerTensorProtocol = arr_mod.asarray(args[1])

    stride = kwargs.get("stride", kwargs.get("strides", (1, 1)))
    s_dims = (stride, stride) if isinstance(stride, int) else coerce_shape(stride)
    s_h, s_w = s_dims[0], s_dims[1]

    dilation = kwargs.get("dilation", kwargs.get("dilations", (1, 1)))
    d_dims = (dilation, dilation) if isinstance(dilation, int) else coerce_shape(dilation)
    d_h, d_w = d_dims[0], d_dims[1]

    groups = int(coerce_int(kwargs.get("groups", 1)))
    padding = kwargs.get("padding", "SAME")

    target_shape = kwargs.get("target_shape")
    if target_shape is not None:
        c_out, c_in_g, k_h, k_w = (int(s) for s in coerce_int_list(target_shape))
    else:
        c_out = d_out.shape[1]
        c_in_g = x.shape[1] // groups
        k_h, k_w = 3, 3

    in_h, in_w = x.shape[2], x.shape[3]
    out_h, out_w = d_out.shape[2], d_out.shape[3]

    p_top, p_bottom, p_left, p_right = _compute_conv2d_padding_values(
        (in_h, in_w), (k_h, k_w), (s_h, s_w), (d_h, d_w), padding, (out_h, out_w)
    )

    if p_top > 0 or p_bottom > 0 or p_left > 0 or p_right > 0:
        pad_h = in_h + p_top + p_bottom
        pad_w = in_w + p_left + p_right
        x_pad: EagerTensorProtocol = arr_mod.zeros((x.shape[0], x.shape[1], pad_h, pad_w), dtype=x.dtype)
        x_pad[:, :, p_top : p_top + in_h, p_left : p_left + in_w] = x
    else:
        x_pad = x

    dw = arr_mod.zeros((c_out, c_in_g, k_h, k_w), dtype=x.dtype)

    c_out_g = c_out // groups

    for g in range(groups):
        x_g = x_pad[:, g * c_in_g : (g + 1) * c_in_g, :, :]
        dy_g = d_out[:, g * c_out_g : (g + 1) * c_out_g, :, :]

        for kh in range(k_h):
            h_start = kh * d_h
            h_end = h_start + out_h * s_h
            for kw in range(k_w):
                w_start = kw * d_w
                w_end = w_start + out_w * s_w
                x_slice = x_g[:, :, h_start:h_end:s_h, w_start:w_end:s_w]
                dw[g * c_out_g : (g + 1) * c_out_g, :, kh, kw] = _ensure_arr(
                    adapt_backend_module(arr_mod).require_symbol("einsum")("n o h w, n i h w -> o i", dy_g, x_slice)
                )

    return dw


@global_eager_registry.register("Conv2DBiasGrad")
def _np_conv2d_bias_grad(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> EagerTensorProtocol:
    """Compute exact gradient of Conv2D with respect to bias vector.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Backend provider module.
        *args (BackendArg): Positional args [d_out].
        **kwargs (BackendArg): Optional keyword args.

    Returns:
        EagerTensorProtocol: Bias gradient vector.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    d_out: EagerTensorProtocol = arr_mod.asarray(args[0])
    return _ensure_arr(arr_mod.sum(d_out, axis=(0, 2, 3)))
