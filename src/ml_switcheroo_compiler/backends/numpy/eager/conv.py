# ruff: noqa: E501
"""Convolution Ops."""

import itertools
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Union

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.core.constants import MAGIC_VAL_2, MAGIC_VAL_3
from ml_switcheroo_compiler.ops.configs import ConvConfig


@dataclass
class ConvDimSpecs:
    """Configuration class for conv dim specs."""

    spatial_dims: int
    lhs_spec: list[int]
    rhs_spec: list[int]
    out_spec: tuple[int, ...]


@dataclass
class ConvExecutionState:
    """Configuration class for conv execution state."""

    lhs_pad: np.ndarray
    rhs_c: np.ndarray
    out: np.ndarray
    config: ConvConfig
    spatial_dims: int


def _get_transpose(spec: Union[str, Iterable[int]], default: str) -> tuple[int, ...]:
    """Get transpose.

    Args:
        spec (Union): The spec parameter.
        default (str): The default parameter.

    Returns:
        tuple: Result.

    Raises:
        TypeError: An exception.
    """
    if isinstance(spec, str):
        try:
            return tuple(spec.index(c) for c in default)
        except (ValueError, TypeError) as e:
            logging.error(f"CRASH: spec={spec}, default={default}")
            raise e
    return tuple(spec)


def _get_conv_defaults(spatial_dims: int) -> tuple[str, str]:
    """Evaluate _get_conv_defaults operation.

    Args:
        spatial_dims (int): The spatial_dims parameter.

    Returns:
        object: Result.
    """
    if spatial_dims == 1:
        return ("NCW", "OIW")
    if spatial_dims == MAGIC_VAL_2:
        return ("NCHW", "OIHW")
    return ("NCDHW", "OIDHW")


def _parse_conv_dimension_numbers(lhs_ndim: int, rhs_ndim: int, spatial_dims: int, dimension_numbers: object) -> ConvDimSpecs:
    """Parse dimension numbers for convolution.

    Args:
        lhs_ndim (int): The lhs_ndim parameter.
        rhs_ndim (int): The rhs_ndim parameter.
        spatial_dims (int): The spatial_dims parameter.
        dimension_numbers (object): The dimension_numbers parameter.

    Returns:
        ConvDimSpecs: Result.
    """
    if dimension_numbers is None:
        lhs_spec = (0, 1) + tuple(range(2, lhs_ndim))
        rhs_spec = (0, 1) + tuple(range(2, rhs_ndim))
        out_spec = (0, 1) + tuple(range(2, lhs_ndim))
        return ConvDimSpecs(spatial_dims, lhs_spec, rhs_spec, out_spec)
    if isinstance(dimension_numbers, tuple) and len(dimension_numbers) == MAGIC_VAL_3:
        (lhs_spec, rhs_spec, out_spec) = dimension_numbers
        (lhs_default, rhs_default) = _get_conv_defaults(spatial_dims)
        return ConvDimSpecs(
            spatial_dims,
            _get_transpose(lhs_spec, lhs_default),
            _get_transpose(rhs_spec, rhs_default),
            _get_transpose(out_spec, lhs_default),
        )
    return ConvDimSpecs(
        spatial_dims,
        tuple(dimension_numbers.lhs_spec),
        tuple(dimension_numbers.rhs_spec),
        tuple(dimension_numbers.out_spec),
    )


def _calculate_same_padding(
    lhs_shape: tuple[int, ...],
    rhs_shape: tuple[int, ...],
    rhs_dilation: list[int],
    window_strides: Union[tuple[int, ...], list[int]],
) -> list[tuple[int, int]]:
    """Evaluate _calculate_same_padding operation.

    Args:
        lhs_shape (object): The lhs_shape parameter.
        rhs_shape (object): The rhs_shape parameter.
        rhs_dilation (object): The rhs_dilation parameter.
        window_strides (object): The window_strides parameter.

    Returns:
        object: Result.
    """
    spatial_dims = len(lhs_shape) - 2
    pad_list = []
    for i in range(spatial_dims):
        in_size = lhs_shape[2 + i]
        filter_size = (rhs_shape[2 + i] - 1) * rhs_dilation[i] + 1
        out_size = int(np.ceil(float(in_size) / window_strides[i]))
        pad_total = max((out_size - 1) * window_strides[i] + filter_size - in_size, 0)
        pad_front = pad_total // 2
        pad_back = pad_total - pad_front
        pad_list.append((pad_front, pad_back))
    return pad_list


def _calculate_conv_padding(config: ConvConfig, lhs_shape: tuple[int, ...], rhs_shape: tuple[int, ...]) -> list[tuple[int, int]]:
    """Calculate convolution padding.

    Args:
        config (ConvConfig): The config parameter.
        lhs_shape (tuple): The lhs_shape parameter.
        rhs_shape (tuple): The rhs_shape parameter.

    Returns:
        list: Result.
    """
    spatial_dims = len(lhs_shape) - 2
    padding = config.padding
    if not isinstance(padding, str):
        if padding is None:
            return [(0, 0)] * spatial_dims
        return list(padding)
    if padding == "VALID":
        return [(0, 0)] * spatial_dims
    if padding == "SAME":
        rhs_dilation = config.rhs_dilation if config.rhs_dilation is not None else [1] * spatial_dims
        return _calculate_same_padding(lhs_shape, rhs_shape, rhs_dilation, config.window_strides)
    return [(0, 0)] * spatial_dims


def _apply_conv_dilation(tensor: np.ndarray, dilation: list[int], spatial_dims: int) -> np.ndarray:
    """Apply dilation to a convolution tensor.

    Args:
        tensor (object): The tensor parameter.
        dilation (object): The dilation parameter.
        spatial_dims (int): The spatial_dims parameter.

    Returns:
        object: Result.
    """
    if not any(d > 1 for d in dilation):
        return tensor
    new_shape = list(tensor.shape)
    for i, d in enumerate(dilation):
        new_shape[2 + i] = (tensor.shape[2 + i] - 1) * d + 1
    dilated = np.zeros(new_shape, dtype=tensor.dtype)
    slices = [slice(None), slice(None)] + [slice(None, None, d) for d in dilation]
    dilated[tuple(slices)] = tensor
    return dilated


@dataclass
class PatchConfig:
    """Configuration for convolution patches."""

    axes_lhs: list[int]
    axes_rhs: list[int]
    group_in_c: int
    group_out_c: int
    feature_group_count: int


def _compute_conv_patch_group(lhs_patch: np.ndarray, rhs_c: np.ndarray, config: PatchConfig, g: int) -> np.ndarray:
    """Evaluate _compute_conv_patch_group operation.

    Args:
        lhs_patch (object): The lhs_patch parameter.
        rhs_c (object): The rhs_c parameter.
        config (PatchConfig): The config parameter.
        g (int): The g parameter.

    Returns:
        object: Result.
    """
    group_in_c = config.group_in_c
    group_out_c = config.group_out_c
    lp_g = lhs_patch[:, g * group_in_c : (g + 1) * group_in_c, ...]
    rc_g = rhs_c[g * group_out_c : (g + 1) * group_out_c, :, ...]
    return np.tensordot(lp_g, rc_g, axes=(config.axes_lhs, config.axes_rhs))


def _compute_single_patch_grouped(lhs_patch: np.ndarray, rhs_c: np.ndarray, out: np.ndarray, spatial_indices: tuple[int, ...], config: PatchConfig) -> None:
    """Evaluate _compute_single_patch_grouped operation.

    Args:
        lhs_patch (object): The lhs_patch parameter.
        rhs_c (object): The rhs_c parameter.
        out (object): The out parameter.
        spatial_indices (tuple): The spatial_indices parameter.
        config (PatchConfig): The config parameter.
    """
    for g in range(config.feature_group_count):
        res = _compute_conv_patch_group(lhs_patch, rhs_c, config, g)
        s_c = slice(g * config.group_out_c, (g + 1) * config.group_out_c)
        out[tuple([slice(None), s_c] + list(spatial_indices))] = res


def _get_patch_slices(spatial_indices: tuple[int, ...], window_strides: Union[tuple[int, ...], list[int]], rhs_shape: tuple[int, ...]) -> tuple[slice, ...]:
    """Evaluate _get_patch_slices operation.

    Args:
        spatial_indices (object): The spatial_indices parameter.
        window_strides (object): The window_strides parameter.
        rhs_shape (object): The rhs_shape parameter.

    Returns:
        object: Result.
    """
    slices = [slice(None), slice(None)]
    for i, idx in enumerate(spatial_indices):
        start = idx * window_strides[i]
        end = start + rhs_shape[2 + i]
        slices.append(slice(start, end))
    return tuple(slices)


def _compute_single_patch(state: ConvExecutionState, spatial_indices: tuple[int, ...]) -> None:
    """Evaluate _compute_single_patch operation.

    Args:
        state (ConvExecutionState): The state parameter.
        spatial_indices (tuple): The spatial_indices parameter.
    """
    slices = _get_patch_slices(spatial_indices, state.config.window_strides, state.rhs_c.shape)
    lhs_patch = state.lhs_pad[slices]
    axes_lhs = [1] + list(range(2, 2 + state.spatial_dims))
    axes_rhs = [1] + list(range(2, 2 + state.spatial_dims))
    if state.config.feature_group_count > 1:
        in_channels = lhs_patch.shape[1]
        out_channels = state.rhs_c.shape[0]
        patch_config = PatchConfig(
            axes_lhs=axes_lhs,
            axes_rhs=axes_rhs,
            group_in_c=in_channels // state.config.feature_group_count,
            group_out_c=out_channels // state.config.feature_group_count,
            feature_group_count=state.config.feature_group_count,
        )
        _compute_single_patch_grouped(lhs_patch, state.rhs_c, state.out, spatial_indices, patch_config)
    else:
        res = np.tensordot(lhs_patch, state.rhs_c, axes=(axes_lhs, axes_rhs))
        state.out[tuple([slice(None), slice(None)] + list(spatial_indices))] = res


def _compute_conv_patches(lhs_pad: np.ndarray, rhs_c: np.ndarray, out: np.ndarray, config: ConvConfig) -> None:
    """Evaluate _compute_conv_patches operation.

    Args:
        lhs_pad (object): The lhs_pad parameter.
        rhs_c (object): The rhs_c parameter.
        out (object): The out parameter.
        config (ConvConfig): The config parameter.
    """
    spatial_dims = len(lhs_pad.shape) - 2
    out_spatial = out.shape[2:]
    for spatial_indices in itertools.product(*[range(d) for d in out_spatial]):
        state = ConvExecutionState(lhs_pad=lhs_pad, rhs_c=rhs_c, out=out, config=config, spatial_dims=spatial_dims)
        _compute_single_patch(state, spatial_indices)


def _apply_conv_padding_helper(lhs_c: np.ndarray, rhs_c: np.ndarray, config: ConvConfig) -> np.ndarray:
    """Evaluate _apply_conv_padding_helper operation.

    Args:
        lhs_c (object): The lhs_c parameter.
        rhs_c (object): The rhs_c parameter.
        config (ConvConfig): The config parameter.

    Returns:
        object: Result.
    """
    pad_list = _calculate_conv_padding(config, lhs_c.shape, rhs_c.shape)
    pad_width = tuple((int(x[0]), int(x[1])) for x in [(0, 0), (0, 0)] + pad_list)
    return np.pad(lhs_c, pad_width, mode="constant", constant_values=0)


def _preprocess_conv_tensors(lhs: np.ndarray, rhs: np.ndarray, config: ConvConfig, specs: ConvDimSpecs) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate _preprocess_conv_tensors operation.

    Args:
        lhs (object): The lhs parameter.
        rhs (object): The rhs parameter.
        config (ConvConfig): The config parameter.
        specs (ConvDimSpecs): The specs parameter.

    Returns:
        tuple: Result.
    """
    lhs_c = np.transpose(lhs, specs.lhs_spec)
    rhs_c = np.transpose(rhs, specs.rhs_spec)
    if config.feature_group_count > 1:
        in_channels = lhs_c.shape[1]
        expected_rhs_in = in_channels // config.feature_group_count
        if rhs_c.shape[1] != expected_rhs_in:
            if rhs_c.shape[1] == in_channels:
                permutation = (1, 0) + tuple(range(2, rhs_c.ndim))
                rhs_c = np.transpose(rhs_c, permutation)
                new_shape = (rhs_c.shape[0] * rhs_c.shape[1] // expected_rhs_in, expected_rhs_in) + rhs_c.shape[2:]
                rhs_c = np.reshape(rhs_c, new_shape)
    lhs_dilation = config.lhs_dilation if config.lhs_dilation is not None else [1] * specs.spatial_dims
    rhs_dilation = config.rhs_dilation if config.rhs_dilation is not None else [1] * specs.spatial_dims
    lhs_dilated = _apply_conv_dilation(lhs_c, lhs_dilation, specs.spatial_dims)
    rhs_c = _apply_conv_dilation(rhs_c, rhs_dilation, specs.spatial_dims)
    lhs_pad = _apply_conv_padding_helper(lhs_dilated, rhs_c, config)
    return (lhs_pad, rhs_c)


def _compute_out_shape(
    lhs_pad_shape: tuple[int, ...],
    rhs_c_shape: tuple[int, ...],
    spatial_dims: int,
    window_strides: Union[tuple[int, ...], list[int]],
) -> list[int]:
    """Evaluate _compute_out_shape operation.

    Args:
        lhs_pad_shape (object): The lhs_pad_shape parameter.
        rhs_c_shape (object): The rhs_c_shape parameter.
        spatial_dims (int): The spatial_dims parameter.
        window_strides (object): The window_strides parameter.

    Returns:
        object: Result.
    """
    out_spatial = [(lhs_pad_shape[2 + i] - rhs_c_shape[2 + i]) // window_strides[i] + 1 for i in range(spatial_dims)]
    return [lhs_pad_shape[0], rhs_c_shape[0]] + out_spatial


def _get_inv_out_spec(out_spec: tuple[int, ...]) -> list[int]:
    """Evaluate _get_inv_out_spec operation.

    Args:
        out_spec (object): The out_spec parameter.

    Returns:
        object: Result.
    """
    inv_out_spec = [0] * len(out_spec)
    for i, p in enumerate(out_spec):
        inv_out_spec[p] = i
    return inv_out_spec


def _conv_general_dilated(lhs: object, rhs: object, config: ConvConfig, **kwargs: object) -> object:
    """Evaluate.

    Args:
        lhs (object): The lhs parameter.
        rhs (object): The rhs parameter.
        config (ConvConfig): The config parameter.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    lhs = np.asarray(lhs)
    rhs = np.asarray(rhs)
    spatial_dims = lhs.ndim - 2
    specs = _parse_conv_dimension_numbers(lhs.ndim, rhs.ndim, spatial_dims, config.dimension_numbers)
    (lhs_pad, rhs_c) = _preprocess_conv_tensors(lhs, rhs, config, specs)
    out_shape = _compute_out_shape(lhs_pad.shape, rhs_c.shape, spatial_dims, config.window_strides)
    out = np.zeros(out_shape, dtype=lhs.dtype)
    _compute_conv_patches(lhs_pad, rhs_c, out, config)
    inv_out_spec = _get_inv_out_spec(specs.out_spec)
    return np.transpose(out, inv_out_spec)


@numpy_eager_registry.register("ConvGeneralDilated")
def _np_conv_general_dilated(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_conv_general_dilated operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    from ml_switcheroo_compiler.ops.configs import ConvConfig

    if len(args) == 4:
        lhs, rhs, window_strides, padding = args
        config = ConvConfig(window_strides=window_strides, padding=padding)
        return _conv_general_dilated(lhs, rhs, config, **kwargs)
    return _conv_general_dilated(*args, **kwargs)


def _calculate_conv_transpose_padding(spatial_in: tuple, spatial_k: tuple, strides: tuple, padding: str) -> tuple:
    """Calculate output spatial shapes and paddings for transposed convolution.

    Args:
        spatial_in (tuple): The spatial_in parameter.
        spatial_k (tuple): The spatial_k parameter.
        strides (tuple): The strides parameter.
        padding (str): The padding parameter.

    Returns:
        tuple: Result.
    """
    out_spatial = []
    pads = []
    for s_in, k, st in zip(spatial_in, spatial_k, strides):
        if padding == "VALID":
            s_out = (s_in - 1) * st + k
        else:
            s_out = s_in * st
        out_spatial.append(s_out)
        total_pad = s_out - s_in * st + st + k - 2
        pad_r = total_pad // 2
        pad_l = total_pad - pad_r
        pads.append((pad_l, pad_r))
    return out_spatial, pads


def _build_conv_transpose_config(spatial_in: tuple, spatial_k: tuple, strides: tuple, pads: list) -> tuple:
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

    slices = (slice(None), slice(None)) + tuple(slice(None, None, -1) for _ in spatial_k)
    spatial_rank = len(spatial_in)
    n_dims = spatial_rank + 2
    lhs_spec = (0, 1) + tuple(range(2, n_dims))
    rhs_spec = (0, 1) + tuple(range(2, n_dims))
    out_spec = lhs_spec
    config_obj = ConvConfig(
        window_strides=(1,) * spatial_rank,
        padding=pads,
        lhs_dilation=strides,
        rhs_dilation=(1,) * spatial_rank,
        dimension_numbers=(lhs_spec, rhs_spec, out_spec),
    )
    return slices, config_obj


@numpy_eager_registry.register("ConvTranspose")
def _np_conv_transpose(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_conv_transpose operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    lhs = np.asarray(args[0])
    rhs = np.asarray(args[1])
    strides = args[2] if len(args) > 2 else kwargs.get("strides", 1)
    padding = args[3] if len(args) > 3 else kwargs.get("padding", "VALID")
    spatial_in = lhs.shape[2:]
    spatial_k = rhs.shape[2:]
    if isinstance(strides, int):
        strides = (strides,) * len(spatial_in)
    out_spatial, pads = _calculate_conv_transpose_padding(spatial_in, spatial_k, strides, padding)
    slices, config_obj = _build_conv_transpose_config(spatial_in, spatial_k, strides, pads)
    rhs_rev = rhs[slices]
    return _np_conv_general_dilated(backend_module, lhs, rhs_rev, config_obj)
