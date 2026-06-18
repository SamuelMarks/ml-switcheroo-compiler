"""Convolution Ops."""

import numpy as np
from dataclasses import dataclass
import itertools
from typing import Union
from ml_switcheroo_compiler.ops.configs import ConvConfig
from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


def _parse_conv_dimension_numbers(
    lhs_ndim: int,
    rhs_ndim: int,
    spatial_dims: int,
    dimension_numbers: object,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Parse dimension numbers for convolution."""
    if dimension_numbers is None:
        lhs_spec = (0, 1) + tuple(range(2, lhs_ndim))
        rhs_spec = (0, 1) + tuple(range(2, rhs_ndim))
        out_spec = (0, 1) + tuple(range(2, lhs_ndim))
        return lhs_spec, rhs_spec, out_spec

    if isinstance(dimension_numbers, tuple) and len(dimension_numbers) == 3:
        lhs_spec, rhs_spec, out_spec = dimension_numbers

        def _get_transpose(spec: object, default: str) -> tuple[int, ...]:
            if isinstance(spec, str):
                try:
                    return tuple(spec.index(c) for c in default)
                except Exception as e:
                    print(f"CRASH: spec={spec}, default={default}")
                    raise e
            return tuple(spec)  # type: ignore

        lhs_default = "NCW" if spatial_dims == 1 else "NCHW" if spatial_dims == 2 else "NCDHW"
        rhs_default = "OIW" if spatial_dims == 1 else "OIHW" if spatial_dims == 2 else "OIDHW"

        return (
            _get_transpose(lhs_spec, lhs_default),
            _get_transpose(rhs_spec, rhs_default),
            _get_transpose(out_spec, lhs_default),
        )

    return (
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


def _calculate_conv_padding(
    config: ConvConfig,
    lhs_shape: tuple[int, ...],
    rhs_shape: tuple[int, ...],
) -> list[tuple[int, int]]:
    """Calculate convolution padding."""
    spatial_dims = len(lhs_shape) - 2
    padding = config.padding

    if not isinstance(padding, str):
        return list(padding)  # type: ignore

    if padding == "VALID":
        return [(0, 0)] * spatial_dims

    if padding == "SAME":
        rhs_dilation = (
            config.rhs_dilation if config.rhs_dilation is not None else [1] * spatial_dims
        )
        return _calculate_same_padding(lhs_shape, rhs_shape, rhs_dilation, config.window_strides)

    return [(0, 0)] * spatial_dims


def _apply_conv_dilation(tensor: np.ndarray, dilation: list[int], spatial_dims: int) -> np.ndarray:
    """Apply dilation to a convolution tensor."""
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


def _compute_conv_patch_group(
    lhs_patch: np.ndarray,
    rhs_c: np.ndarray,
    config: PatchConfig,
    g: int,
) -> np.ndarray:
    group_in_c = config.group_in_c
    group_out_c = config.group_out_c
    lp_g = lhs_patch[:, g * group_in_c : (g + 1) * group_in_c, ...]
    rc_g = rhs_c[g * group_out_c : (g + 1) * group_out_c, :, ...]
    return np.tensordot(lp_g, rc_g, axes=(config.axes_lhs, config.axes_rhs))


def _compute_single_patch_grouped(
    lhs_patch: np.ndarray,
    rhs_c: np.ndarray,
    out: np.ndarray,
    spatial_indices: tuple[int, ...],
    config: PatchConfig,
) -> None:
    for g in range(config.feature_group_count):
        res = _compute_conv_patch_group(lhs_patch, rhs_c, config, g)
        s_c = slice(g * config.group_out_c, (g + 1) * config.group_out_c)
        out[tuple([slice(None), s_c] + list(spatial_indices))] = res


def _get_patch_slices(
    spatial_indices: tuple[int, ...],
    window_strides: Union[tuple[int, ...], list[int]],
    rhs_shape: tuple[int, ...],
) -> tuple[slice, ...]:
    slices = [slice(None), slice(None)]
    for i, idx in enumerate(spatial_indices):
        start = idx * window_strides[i]
        end = start + rhs_shape[2 + i]
        slices.append(slice(start, end))
    return tuple(slices)


def _compute_single_patch(
    lhs_pad: np.ndarray,
    rhs_c: np.ndarray,
    out: np.ndarray,
    spatial_indices: tuple[int, ...],
    config: ConvConfig,
    spatial_dims: int,
) -> None:
    slices = _get_patch_slices(spatial_indices, config.window_strides, rhs_c.shape)
    lhs_patch = lhs_pad[slices]
    axes_lhs = [1] + list(range(2, 2 + spatial_dims))
    axes_rhs = [1] + list(range(2, 2 + spatial_dims))

    if config.feature_group_count > 1:
        in_channels = lhs_patch.shape[1]
        out_channels = rhs_c.shape[0]
        patch_config = PatchConfig(
            axes_lhs=axes_lhs,
            axes_rhs=axes_rhs,
            group_in_c=in_channels // config.feature_group_count,
            group_out_c=out_channels // config.feature_group_count,
            feature_group_count=config.feature_group_count,
        )
        _compute_single_patch_grouped(lhs_patch, rhs_c, out, spatial_indices, patch_config)
    else:
        res = np.tensordot(lhs_patch, rhs_c, axes=(axes_lhs, axes_rhs))
        out[tuple([slice(None), slice(None)] + list(spatial_indices))] = res


def _compute_conv_patches(
    lhs_pad: np.ndarray,
    rhs_c: np.ndarray,
    out: np.ndarray,
    config: ConvConfig,
) -> None:
    """Compute convolution patches."""
    spatial_dims = len(lhs_pad.shape) - 2
    out_spatial = out.shape[2:]

    for spatial_indices in itertools.product(*[range(d) for d in out_spatial]):
        _compute_single_patch(lhs_pad, rhs_c, out, spatial_indices, config, spatial_dims)


def _apply_conv_padding_helper(
    lhs_c: np.ndarray, rhs_c: np.ndarray, config: ConvConfig
) -> np.ndarray:
    pad_list = _calculate_conv_padding(config, lhs_c.shape, rhs_c.shape)
    pad_width = [(0, 0), (0, 0)] + pad_list
    return np.pad(lhs_c, pad_width, mode="constant", constant_values=0)


def _preprocess_conv_tensors(
    lhs: np.ndarray,
    rhs: np.ndarray,
    config: ConvConfig,
    spatial_dims: int,
    lhs_spec: list[int],
    rhs_spec: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    lhs_c = np.transpose(lhs, lhs_spec)
    rhs_c = np.transpose(rhs, rhs_spec)

    lhs_dilation = config.lhs_dilation if config.lhs_dilation is not None else [1] * spatial_dims
    rhs_dilation = config.rhs_dilation if config.rhs_dilation is not None else [1] * spatial_dims

    # Dilate before padding!
    lhs_dilated = _apply_conv_dilation(lhs_c, lhs_dilation, spatial_dims)
    rhs_c = _apply_conv_dilation(rhs_c, rhs_dilation, spatial_dims)

    # Pass dilated lhs to calculate padding, because SAME padding needs the dilated shape
    lhs_pad = _apply_conv_padding_helper(lhs_dilated, rhs_c, config)

    return lhs_pad, rhs_c


def _compute_out_shape(
    lhs_pad_shape: tuple[int, ...],
    rhs_c_shape: tuple[int, ...],
    spatial_dims: int,
    window_strides: Union[tuple[int, ...], list[int]],
) -> list[int]:
    out_spatial = [
        (lhs_pad_shape[2 + i] - rhs_c_shape[2 + i]) // window_strides[i] + 1
        for i in range(spatial_dims)
    ]
    return [lhs_pad_shape[0], rhs_c_shape[0]] + out_spatial


def _get_inv_out_spec(out_spec: tuple[int, ...]) -> list[int]:
    inv_out_spec = [0] * len(out_spec)
    for i, p in enumerate(out_spec):
        inv_out_spec[p] = i
    return inv_out_spec


def _conv_general_dilated(
    lhs: object,
    rhs: object,
    config: ConvConfig,
    **kwargs: object,
) -> object:
    """Evaluate."""
    lhs = np.asarray(lhs)
    print("conv eager lhs shape:", lhs.shape, "config:", config.dimension_numbers)
    print("conv eager lhs shape:", lhs.shape, "config:", config.dimension_numbers)
    rhs = np.asarray(rhs)
    spatial_dims = lhs.ndim - 2

    lhs_spec, rhs_spec, out_spec = _parse_conv_dimension_numbers(
        lhs.ndim, rhs.ndim, spatial_dims, config.dimension_numbers
    )

    lhs_pad, rhs_c = _preprocess_conv_tensors(lhs, rhs, config, spatial_dims, lhs_spec, rhs_spec)

    out_shape = _compute_out_shape(lhs_pad.shape, rhs_c.shape, spatial_dims, config.window_strides)
    out = np.zeros(out_shape, dtype=lhs.dtype)

    _compute_conv_patches(lhs_pad, rhs_c, out, config)

    inv_out_spec = _get_inv_out_spec(out_spec)
    return np.transpose(out, inv_out_spec)


@numpy_eager_registry.register("ConvGeneralDilated")
def _np_conv_general_dilated(backend_module: object, *args: object, **kwargs: object) -> object:
    return _conv_general_dilated(*args, **kwargs)
