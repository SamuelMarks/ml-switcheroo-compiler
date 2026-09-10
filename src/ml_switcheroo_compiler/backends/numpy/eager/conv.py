# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Module conv.py."""

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

    lhs_pad: object
    rhs_c: object
    out: object
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


def _parse_conv_dimension_numbers(lhs_ndim: int, rhs_ndim: int, spatial_dims: int, dimension_numbers) -> ConvDimSpecs:
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
        lhs_spec: tuple = (0, 1) + tuple(range(2, lhs_ndim))
        rhs_spec: tuple = (0, 1) + tuple(range(2, rhs_ndim))
        out_spec: tuple = (0, 1) + tuple(range(2, lhs_ndim))
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
    spatial_dims: int = len(lhs_shape) - 2
    pad_list: list = []
    for i in range(spatial_dims):
        in_size: int = lhs_shape[2 + i]
        filter_size: int = (rhs_shape[2 + i] - 1) * rhs_dilation[i] + 1
        out_size: int = int(np.ceil(float(in_size) / window_strides[i]))
        pad_total: int = max((out_size - 1) * window_strides[i] + filter_size - in_size, 0)
        pad_front: int = pad_total // 2
        pad_back: np.ndarray = pad_total - pad_front
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
    spatial_dims: int = len(lhs_shape) - 2
    padding: str = config.padding
    if not isinstance(padding, str):
        if padding is None:
            return [(0, 0)] * spatial_dims
        return list(padding)
    if padding == "VALID":
        return [(0, 0)] * spatial_dims
    if padding == "SAME":
        rhs_dilation: np.ndarray = config.rhs_dilation if config.rhs_dilation is not None else [1] * spatial_dims
        return _calculate_same_padding(lhs_shape, rhs_shape, rhs_dilation, config.window_strides)
    return [(0, 0)] * spatial_dims


def _apply_conv_dilation(tensor, dilation: list[int], spatial_dims: int):
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
    new_shape: list = list(tensor.shape)
    for i, d in enumerate(dilation):
        new_shape[2 + i] = (tensor.shape[2 + i] - 1) * d + 1
    dilated: np.ndarray = np.zeros(new_shape, dtype=tensor.dtype)
    slices: tuple = [slice(None), slice(None)] + [slice(None, None, d) for d in dilation]
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


def _compute_conv_patch_group(lhs_patch, rhs_c, config: PatchConfig, g: int):
    """Evaluate _compute_conv_patch_group operation.

    Args:
        lhs_patch (object): The lhs_patch parameter.
        rhs_c (object): The rhs_c parameter.
        config (PatchConfig): The config parameter.
        g (int): The g parameter.

    Returns:
        object: Result.
    """
    group_in_c: int = config.group_in_c
    group_out_c: int = config.group_out_c
    lp_g: np.ndarray = lhs_patch[:, g * group_in_c : (g + 1) * group_in_c, ...]
    rc_g: np.ndarray = rhs_c[g * group_out_c : (g + 1) * group_out_c, :, ...]
    return np.tensordot(lp_g, rc_g, axes=(config.axes_lhs, config.axes_rhs))


def _compute_single_patch_grouped(lhs_patch, rhs_c, out, spatial_indices: tuple[int, ...], config: PatchConfig) -> None:
    """Evaluate _compute_single_patch_grouped operation.

    Args:
        lhs_patch (object): The lhs_patch parameter.
        rhs_c (object): The rhs_c parameter.
        out (object): The out parameter.
        spatial_indices (tuple): The spatial_indices parameter.
        config (PatchConfig): The config parameter.
    """
    for g in range(config.feature_group_count):
        res: np.ndarray = _compute_conv_patch_group(lhs_patch, rhs_c, config, g)
        s_c: slice = slice(g * config.group_out_c, (g + 1) * config.group_out_c)
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
    slices: tuple = [slice(None), slice(None)]
    for i, idx in enumerate(spatial_indices):
        start: int = idx * window_strides[i]
        end: int = start + rhs_shape[2 + i]
        slices.append(slice(start, end))
    return tuple(slices)


def _compute_single_patch(state: ConvExecutionState, spatial_indices: tuple[int, ...]) -> None:
    """Evaluate _compute_single_patch operation.

    Args:
        state (ConvExecutionState): The state parameter.
        spatial_indices (tuple): The spatial_indices parameter.
    """
    slices: tuple = _get_patch_slices(spatial_indices, state.config.window_strides, state.rhs_c.shape)
    lhs_patch: np.ndarray = state.lhs_pad[slices]
    axes_lhs: list = [1] + list(range(2, 2 + state.spatial_dims))
    axes_rhs: list = [1] + list(range(2, 2 + state.spatial_dims))
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
        res: np.ndarray = np.tensordot(lhs_patch, state.rhs_c, axes=(axes_lhs, axes_rhs))
        state.out[tuple([slice(None), slice(None)] + list(spatial_indices))] = res


def _compute_conv_patches(lhs_pad, rhs_c, out, config: ConvConfig) -> None:
    """Evaluate _compute_conv_patches operation.

    Args:
        lhs_pad (object): The lhs_pad parameter.
        rhs_c (object): The rhs_c parameter.
        out (object): The out parameter.
        config (ConvConfig): The config parameter.
    """
    spatial_dims: int = len(lhs_pad.shape) - 2
    out_spatial: tuple = out.shape[2:]
    for spatial_indices in itertools.product(*[range(d) for d in out_spatial]):
        state: ConvExecutionState = ConvExecutionState(lhs_pad=lhs_pad, rhs_c=rhs_c, out=out, config=config, spatial_dims=spatial_dims)
        _compute_single_patch(state, spatial_indices)


def _apply_conv_padding_helper(lhs_c, rhs_c, config: ConvConfig):
    """Evaluate _apply_conv_padding_helper operation.

    Args:
        lhs_c (object): The lhs_c parameter.
        rhs_c (object): The rhs_c parameter.
        config (ConvConfig): The config parameter.

    Returns:
        object: Result.
    """
    pad_list: list = _calculate_conv_padding(config, lhs_c.shape, rhs_c.shape)
    pad_width: tuple = tuple((int(x[0]), int(x[1])) for x in [(0, 0), (0, 0)] + pad_list)
    new_shape = tuple(int(s) + int(p[0]) + int(p[1]) for s, p in zip(lhs_c.shape, pad_width))
    padded = np.zeros(new_shape, dtype=lhs_c.dtype)
    slices = tuple(slice(int(p[0]), int(p[0]) + int(s)) for s, p in zip(lhs_c.shape, pad_width))
    padded[slices] = lhs_c
    return padded


def _preprocess_conv_tensors(lhs, rhs, config: ConvConfig, specs: ConvDimSpecs):
    """Evaluate _preprocess_conv_tensors operation.

    Args:
        lhs (object): The lhs parameter.
        rhs (object): The rhs parameter.
        config (ConvConfig): The config parameter.
        specs (ConvDimSpecs): The specs parameter.

    Returns:
        tuple: Result.
    """
    lhs_c: np.ndarray = np.transpose(lhs, specs.lhs_spec)
    rhs_c: np.ndarray = np.transpose(rhs, specs.rhs_spec)
    if config.feature_group_count > 1:
        in_channels: int = lhs_c.shape[1]
        expected_rhs_in: np.ndarray = in_channels // config.feature_group_count
        if rhs_c.shape[1] != expected_rhs_in:
            if rhs_c.shape[1] == in_channels:
                permutation: np.ndarray = (1, 0) + tuple(range(2, rhs_c.ndim))
                rhs_c: np.ndarray = np.transpose(rhs_c, permutation)
                new_shape: list = (rhs_c.shape[0] * rhs_c.shape[1] // expected_rhs_in, expected_rhs_in) + rhs_c.shape[2:]
                rhs_c: np.ndarray = np.reshape(rhs_c, new_shape)
    lhs_dilation: np.ndarray = config.lhs_dilation if config.lhs_dilation is not None else [1] * specs.spatial_dims
    rhs_dilation: np.ndarray = config.rhs_dilation if config.rhs_dilation is not None else [1] * specs.spatial_dims
    lhs_dilated: np.ndarray = _apply_conv_dilation(lhs_c, lhs_dilation, specs.spatial_dims)
    rhs_c: slice = _apply_conv_dilation(rhs_c, rhs_dilation, specs.spatial_dims)
    lhs_pad: np.ndarray = _apply_conv_padding_helper(lhs_dilated, rhs_c, config)
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
    out_spatial: tuple = [(lhs_pad_shape[2 + i] - rhs_c_shape[2 + i]) // window_strides[i] + 1 for i in range(spatial_dims)]
    return [lhs_pad_shape[0], rhs_c_shape[0]] + out_spatial


def _get_inv_out_spec(out_spec: tuple[int, ...]) -> list[int]:
    """Evaluate _get_inv_out_spec operation.

    Args:
        out_spec (object): The out_spec parameter.

    Returns:
        object: Result.
    """
    inv_out_spec: list[int] = [0] * len(out_spec)
    for i, p in enumerate(out_spec):
        inv_out_spec[p] = i
    return inv_out_spec


def _conv_general_dilated(lhs, rhs, config: ConvConfig, **kwargs):
    """Evaluate.

    Args:
        lhs (object): The lhs parameter.
        rhs (object): The rhs parameter.
        config (ConvConfig): The config parameter.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    lhs_arr: np.ndarray = np.asarray(lhs)
    rhs_arr: np.ndarray = np.asarray(rhs)
    spatial_dims: int = lhs_arr.ndim - 2
    specs: list = _parse_conv_dimension_numbers(lhs_arr.ndim, rhs_arr.ndim, spatial_dims, config.dimension_numbers)
    (lhs_pad, rhs_c) = _preprocess_conv_tensors(lhs_arr, rhs_arr, config, specs)
    out_shape: list = _compute_out_shape(lhs_pad.shape, rhs_c.shape, spatial_dims, config.window_strides)
    out: np.ndarray = np.zeros(out_shape, dtype=lhs_arr.dtype)
    _compute_conv_patches(lhs_pad, rhs_c, out, config)
    inv_out_spec: tuple = _get_inv_out_spec(specs.out_spec)
    return np.transpose(out, inv_out_spec)


@numpy_eager_registry.register("ConvGeneralDilated")
def _np_conv_general_dilated(backend_module, *args, **kwargs):
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
        config: ConvConfig = ConvConfig(window_strides=window_strides, padding=padding)
        return _conv_general_dilated(lhs, rhs, config, **kwargs)
    return _conv_general_dilated(*args, **kwargs)


@numpy_eager_registry.register("Conv2D")
def _np_conv2d(backend_module, *args, **kwargs):
    """Evaluate Conv2D operation in NumPy eager mode.

    Args:
        backend_module (object): The backend module (np).
        *args (object): Positional args (lhs, rhs).
        **kwargs (object): Keyword args (stride, padding, groups).

    Returns:
        object: Result of 2D convolution.
    """
    from ml_switcheroo_compiler.ops.configs import ConvConfig

    lhs, rhs = np.asarray(args[0]), np.asarray(args[1])
    stride = kwargs.get("stride", kwargs.get("strides", [1, 1]))
    if isinstance(stride, int):
        stride = [stride, stride]
    padding = kwargs.get("padding", "SAME")
    if isinstance(padding, (list, tuple)):
        padding = "SAME"
    groups = int(kwargs.get("groups", 1))

    if groups > 1:
        c_in_g = lhs.shape[1] // groups
        c_out_g = rhs.shape[0] // groups
        outs = []
        for g in range(groups):
            lhs_g = lhs[:, g * c_in_g : (g + 1) * c_in_g, :, :]
            rhs_g = rhs[g * c_out_g : (g + 1) * c_out_g, :, :, :]
            cfg = ConvConfig(window_strides=stride, padding=padding)
            outs.append(_conv_general_dilated(lhs_g, rhs_g, cfg))
        return np.concatenate(outs, axis=1)

    config: ConvConfig = ConvConfig(window_strides=stride, padding=padding)
    return _conv_general_dilated(lhs, rhs, config)


def _calculate_conv_transpose_padding(spatial_in, spatial_k, strides, padding: str):
    """Calculate output spatial shapes and paddings for transposed convolution.

    Args:
        spatial_in (tuple): The spatial_in parameter.
        spatial_k (tuple): The spatial_k parameter.
        strides (tuple): The strides parameter.
        padding (str): The padding parameter.

    Returns:
        tuple: Result.
    """
    out_spatial: tuple = []
    pads: list = []
    for s_in, k, st in zip(spatial_in, spatial_k, strides):
        if padding == "VALID":
            s_out: np.ndarray = (s_in - 1) * st + k
        else:
            s_out: np.ndarray = s_in * st
        out_spatial.append(s_out)
        total_pad: int = s_out - s_in * st + st + k - 2
        pad_r: int = total_pad // 2
        pad_l: int = total_pad - pad_r
        pads.append((pad_l, pad_r))
    return out_spatial, pads


def _build_conv_transpose_config(spatial_in, spatial_k, strides, pads):
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

    slices: tuple = (slice(None), slice(None)) + tuple(slice(None, None, -1) for _ in spatial_k)
    spatial_rank: int = len(spatial_in)
    n_dims: int = spatial_rank + 2
    lhs_spec: tuple = (0, 1) + tuple(range(2, n_dims))
    rhs_spec: tuple = (0, 1) + tuple(range(2, n_dims))
    out_spec: tuple = lhs_spec
    config_obj: ConvConfig = ConvConfig(
        window_strides=(1,) * spatial_rank,
        padding=pads,
        lhs_dilation=strides,
        rhs_dilation=(1,) * spatial_rank,
        dimension_numbers=(lhs_spec, rhs_spec, out_spec),
    )
    return slices, config_obj


@numpy_eager_registry.register("ConvTranspose")
@numpy_eager_registry.register("ConvTranspose2D")
@numpy_eager_registry.register("Conv2DTranspose")
def _np_conv_transpose(backend_module, *args, **kwargs):
    """Evaluate _np_conv_transpose operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    lhs: np.ndarray = np.asarray(args[0])
    rhs: np.ndarray = np.asarray(args[1])
    strides: list = args[2] if len(args) > 2 else kwargs.get("strides", kwargs.get("stride", 1))
    padding: str = args[3] if len(args) > 3 else kwargs.get("padding", "VALID")
    spatial_in: np.ndarray = lhs.shape[2:]
    spatial_k: np.ndarray = rhs.shape[2:]
    if isinstance(strides, int):
        strides: list = (strides,) * len(spatial_in)
    out_spatial, pads = _calculate_conv_transpose_padding(spatial_in, spatial_k, strides, padding)
    slices, config_obj = _build_conv_transpose_config(spatial_in, spatial_k, strides, pads)
    rhs_rev: np.ndarray = rhs[slices]
    return _np_conv_general_dilated(backend_module, lhs, rhs_rev, config_obj)


def _compute_conv2d_padding_values(
    in_h: int,
    in_w: int,
    k_h: int,
    k_w: int,
    s_h: int,
    s_w: int,
    d_h: int,
    d_w: int,
    padding: object,
    out_h: int,
    out_w: int,
) -> tuple[int, int, int, int]:
    """Calculate 4-sided padding values (top, bottom, left, right) for 2D convolution.

    Args:
        in_h (int): Input height.
        in_w (int): Input width.
        k_h (int): Kernel height.
        k_w (int): Kernel width.
        s_h (int): Stride height.
        s_w (int): Stride width.
        d_h (int): Dilation height.
        d_w (int): Dilation width.
        padding (object): Padding specification ("SAME", "VALID", int, or tuple).
        out_h (int): Output height.
        out_w (int): Output width.

    Returns:
        tuple[int, int, int, int]: Padding tuple (p_top, p_bottom, p_left, p_right).
    """
    if isinstance(padding, (list, tuple)):
        if len(padding) == 2:
            return int(padding[0]), int(padding[0]), int(padding[1]), int(padding[1])
        if len(padding) == 4:
            return int(padding[0]), int(padding[1]), int(padding[2]), int(padding[3])
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


@numpy_eager_registry.register("Conv2DInputGrad")
def _np_conv2d_input_grad(backend_module, *args, **kwargs) -> np.ndarray:
    """Compute exact gradient of Conv2D with respect to the input tensor.

    Supports arbitrary strides, padding, dilation, and group convolutions.

    Args:
        backend_module (object): Backend provider module.
        *args (object): Positional args [d_out, weight].
        **kwargs (object): Conv attributes (stride, padding, dilation, groups, target_shape).

    Returns:
        np.ndarray: Computed gradient tensor with shape matching the original input.
    """
    d_out: np.ndarray = np.asarray(args[0])
    weight: np.ndarray = np.asarray(args[1])

    stride = kwargs.get("stride", kwargs.get("strides", (1, 1)))
    s_h, s_w = (stride, stride) if isinstance(stride, int) else (stride[0], stride[1])

    dilation = kwargs.get("dilation", kwargs.get("dilations", (1, 1)))
    d_h, d_w = (dilation, dilation) if isinstance(dilation, int) else (dilation[0], dilation[1])

    groups = int(kwargs.get("groups", 1))
    padding = kwargs.get("padding", "SAME")

    target_shape = kwargs.get("target_shape")
    if target_shape is not None:
        n_batch, c_in, in_h, in_w = (int(s) for s in target_shape)
    else:
        n_batch = d_out.shape[0]
        c_in = weight.shape[1] * groups
        in_h = (d_out.shape[2] - 1) * s_h + 1
        in_w = (d_out.shape[3] - 1) * s_w + 1

    c_out = weight.shape[0]
    k_h, k_w = weight.shape[2], weight.shape[3]
    out_h, out_w = d_out.shape[2], d_out.shape[3]

    p_top, p_bottom, p_left, p_right = _compute_conv2d_padding_values(in_h, in_w, k_h, k_w, s_h, s_w, d_h, d_w, padding, out_h, out_w)

    pad_h = in_h + p_top + p_bottom
    pad_w = in_w + p_left + p_right
    dx_pad = np.zeros((n_batch, c_in, pad_h, pad_w), dtype=d_out.dtype)

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
                term = np.einsum("n o h w, o i -> n i h w", dy_g, w_sub)
                dx_pad[:, g * c_in_g : (g + 1) * c_in_g, h_start:h_end:s_h, w_start:w_end:s_w] += term

    return dx_pad[:, :, p_top : p_top + in_h, p_left : p_left + in_w]


@numpy_eager_registry.register("Conv2DWeightGrad")
def _np_conv2d_weight_grad(backend_module, *args, **kwargs) -> np.ndarray:
    """Compute exact gradient of Conv2D with respect to the weight tensor.

    Supports arbitrary strides, padding, dilation, and group convolutions.

    Args:
        backend_module (object): Backend provider module.
        *args (object): Positional args [input, d_out].
        **kwargs (object): Conv attributes (stride, padding, dilation, groups, target_shape).

    Returns:
        np.ndarray: Computed gradient tensor with shape matching original weight.
    """
    x: np.ndarray = np.asarray(args[0])
    d_out: np.ndarray = np.asarray(args[1])

    stride = kwargs.get("stride", kwargs.get("strides", (1, 1)))
    s_h, s_w = (stride, stride) if isinstance(stride, int) else (stride[0], stride[1])

    dilation = kwargs.get("dilation", kwargs.get("dilations", (1, 1)))
    d_h, d_w = (dilation, dilation) if isinstance(dilation, int) else (dilation[0], dilation[1])

    groups = int(kwargs.get("groups", 1))
    padding = kwargs.get("padding", "SAME")

    target_shape = kwargs.get("target_shape")
    if target_shape is not None:
        c_out, c_in_g, k_h, k_w = (int(s) for s in target_shape)
    else:
        c_out = d_out.shape[1]
        c_in_g = x.shape[1] // groups
        k_h, k_w = 3, 3

    in_h, in_w = x.shape[2], x.shape[3]
    out_h, out_w = d_out.shape[2], d_out.shape[3]

    p_top, p_bottom, p_left, p_right = _compute_conv2d_padding_values(in_h, in_w, k_h, k_w, s_h, s_w, d_h, d_w, padding, out_h, out_w)

    if p_top > 0 or p_bottom > 0 or p_left > 0 or p_right > 0:
        pad_h = in_h + p_top + p_bottom
        pad_w = in_w + p_left + p_right
        x_pad = np.zeros((x.shape[0], x.shape[1], pad_h, pad_w), dtype=x.dtype)
        x_pad[:, :, p_top : p_top + in_h, p_left : p_left + in_w] = x
    else:
        x_pad = x

    dw = np.zeros((c_out, c_in_g, k_h, k_w), dtype=x.dtype)

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
                dw[g * c_out_g : (g + 1) * c_out_g, :, kh, kw] = np.einsum("n o h w, n i h w -> o i", dy_g, x_slice)

    return dw


@numpy_eager_registry.register("Conv2DBiasGrad")
def _np_conv2d_bias_grad(backend_module, *args, **kwargs) -> np.ndarray:
    """Compute exact gradient of Conv2D with respect to bias vector.

    Args:
        backend_module (object): Backend provider module.
        *args (object): Positional args [d_out].
        **kwargs (object): Optional keyword args.

    Returns:
        np.ndarray: Bias gradient vector.
    """
    d_out: np.ndarray = np.asarray(args[0])
    return np.sum(d_out, axis=(0, 2, 3))
