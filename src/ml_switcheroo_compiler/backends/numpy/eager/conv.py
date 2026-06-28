# ruff: noqa: E402
"""Convolution Ops."""

from ml_switcheroo_compiler.core.constants import MAGIC_VAL_2
from ml_switcheroo_compiler.core.constants import MAGIC_VAL_3

import itertools
from dataclasses import dataclass
from typing import Union
from collections.abc import Iterable

import numpy as np

from ml_switcheroo_compiler.ops.configs import ConvConfig


@dataclass
class ConvDimSpecs:
    """Class docstring."""

    spatial_dims: int
    lhs_spec: list[int]
    rhs_spec: list[int]
    out_spec: tuple[int, ...]


@dataclass
class ConvExecutionState:
    """Class docstring."""

    lhs_pad: np.ndarray
    rhs_c: np.ndarray
    out: np.ndarray
    config: ConvConfig
    spatial_dims: int


from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


def _get_transpose(spec: Union[str, Iterable[int]], default: str) -> tuple[int, ...]:
    """Get transpose.

    Args:
        spec (Union[str, Iterable[int]]): Spec.
        default (str): Default.

    Returns:
        tuple[int, ...]: Transpose.
    """
    if isinstance(spec, str):
        try:
            return tuple(spec.index(c) for c in default)
        except (ValueError, TypeError) as e:  # pragma: no cover
            import logging  # pragma: no cover

            logging.error(f"CRASH: spec={spec}, default={default}")  # pragma: no cover
            raise e  # pragma: no cover
    return tuple(spec)


def _get_conv_defaults(spatial_dims: int) -> tuple[str, str]:
    """Function docstring.

    Args:
        spatial_dims: Arg.
    """
    if spatial_dims == 1:
        return "NCW", "OIW"
    if spatial_dims == MAGIC_VAL_2:
        return "NCHW", "OIHW"
    return "NCDHW", "OIDHW"


def _parse_conv_dimension_numbers(
    lhs_ndim: int,
    rhs_ndim: int,
    spatial_dims: int,
    dimension_numbers: object,
) -> ConvDimSpecs:
    """Parse dimension numbers for convolution."""
    if dimension_numbers is None:
        lhs_spec = (0, 1) + tuple(range(2, lhs_ndim))
        rhs_spec = (0, 1) + tuple(range(2, rhs_ndim))
        out_spec = (0, 1) + tuple(range(2, lhs_ndim))
        return ConvDimSpecs(spatial_dims, lhs_spec, rhs_spec, out_spec)

    if isinstance(dimension_numbers, tuple) and len(dimension_numbers) == MAGIC_VAL_3:
        lhs_spec, rhs_spec, out_spec = dimension_numbers
        lhs_default, rhs_default = _get_conv_defaults(spatial_dims)

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
    """Function docstring.

    Args:
        lhs_shape: Arg.
        rhs_shape: Arg.
        rhs_dilation: Arg.
        window_strides: Arg.
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


def _calculate_conv_padding(
    config: ConvConfig,
    lhs_shape: tuple[int, ...],
    rhs_shape: tuple[int, ...],
) -> list[tuple[int, int]]:
    """Calculate convolution padding."""
    spatial_dims = len(lhs_shape) - 2
    padding = config.padding

    if not isinstance(padding, str):
        if padding is None:  # pragma: no branch
            return [(0, 0)] * spatial_dims  # pragma: no cover
        return list(padding)  # pragma: no cover

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
    """Function docstring.

    Args:
        lhs_patch: Arg.
        rhs_c: Arg.
        config: Arg.
        g: Arg.
    """
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
    """Function docstring.

    Args:
        lhs_patch: Arg.
        rhs_c: Arg.
        out: Arg.
        spatial_indices: Arg.
        config: Arg.
    """
    for g in range(config.feature_group_count):
        res = _compute_conv_patch_group(lhs_patch, rhs_c, config, g)
        s_c = slice(g * config.group_out_c, (g + 1) * config.group_out_c)
        out[tuple([slice(None), s_c] + list(spatial_indices))] = res


def _get_patch_slices(
    spatial_indices: tuple[int, ...],
    window_strides: Union[tuple[int, ...], list[int]],
    rhs_shape: tuple[int, ...],
) -> tuple[slice, ...]:
    """Function docstring.

    Args:
        spatial_indices: Arg.
        window_strides: Arg.
        rhs_shape: Arg.
    """
    slices = [slice(None), slice(None)]
    for i, idx in enumerate(spatial_indices):
        start = idx * window_strides[i]
        end = start + rhs_shape[2 + i]
        slices.append(slice(start, end))
    return tuple(slices)


def _compute_single_patch(
    state: ConvExecutionState,
    spatial_indices: tuple[int, ...],
) -> None:
    """Function docstring.

    Args:
        state: Arg.
        spatial_indices: Arg.
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
        _compute_single_patch_grouped(
            lhs_patch, state.rhs_c, state.out, spatial_indices, patch_config
        )
    else:
        res = np.tensordot(lhs_patch, state.rhs_c, axes=(axes_lhs, axes_rhs))
        state.out[tuple([slice(None), slice(None)] + list(spatial_indices))] = res


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
        state = ConvExecutionState(
            lhs_pad=lhs_pad, rhs_c=rhs_c, out=out, config=config, spatial_dims=spatial_dims
        )
        _compute_single_patch(state, spatial_indices)


def _apply_conv_padding_helper(
    lhs_c: np.ndarray, rhs_c: np.ndarray, config: ConvConfig
) -> np.ndarray:
    """Function docstring.

    Args:
        lhs_c: Arg.
        rhs_c: Arg.
        config: Arg.
    """
    pad_list = _calculate_conv_padding(config, lhs_c.shape, rhs_c.shape)
    pad_width = tuple((int(x[0]), int(x[1])) for x in [(0, 0), (0, 0)] + pad_list)
    return np.pad(lhs_c, pad_width, mode="constant", constant_values=0)


def _preprocess_conv_tensors(
    lhs: np.ndarray,
    rhs: np.ndarray,
    config: ConvConfig,
    specs: ConvDimSpecs,
) -> tuple[np.ndarray, np.ndarray]:
    """Function docstring.

    Args:
        lhs: Arg.
        rhs: Arg.
        config: Arg.
        specs: Arg.
    """
    lhs_c = np.transpose(lhs, specs.lhs_spec)
    rhs_c = np.transpose(rhs, specs.rhs_spec)

    if config.feature_group_count > 1:
        in_channels = lhs_c.shape[1]
        expected_rhs_in = in_channels // config.feature_group_count
        if rhs_c.shape[1] != expected_rhs_in:  # pragma: no branch
            if rhs_c.shape[1] == in_channels:  # pragma: no cover
                permutation = (1, 0) + tuple(range(2, rhs_c.ndim))  # pragma: no cover
                rhs_c = np.transpose(rhs_c, permutation)  # pragma: no cover
                new_shape = (rhs_c.shape[0] * rhs_c.shape[1], expected_rhs_in) + rhs_c.shape[
                    2:
                ]  # pragma: no cover
                rhs_c = np.reshape(rhs_c, new_shape)  # pragma: no cover

    lhs_dilation = (
        config.lhs_dilation if config.lhs_dilation is not None else [1] * specs.spatial_dims
    )
    rhs_dilation = (
        config.rhs_dilation if config.rhs_dilation is not None else [1] * specs.spatial_dims
    )

    # Dilate before padding!
    lhs_dilated = _apply_conv_dilation(lhs_c, lhs_dilation, specs.spatial_dims)
    rhs_c = _apply_conv_dilation(rhs_c, rhs_dilation, specs.spatial_dims)

    # Pass dilated lhs to calculate padding, because SAME padding needs the dilated shape
    lhs_pad = _apply_conv_padding_helper(lhs_dilated, rhs_c, config)

    return lhs_pad, rhs_c


def _compute_out_shape(
    lhs_pad_shape: tuple[int, ...],
    rhs_c_shape: tuple[int, ...],
    spatial_dims: int,
    window_strides: Union[tuple[int, ...], list[int]],
) -> list[int]:
    """Function docstring.

    Args:
        lhs_pad_shape: Arg.
        rhs_c_shape: Arg.
        spatial_dims: Arg.
        window_strides: Arg.
    """
    out_spatial = [
        (lhs_pad_shape[2 + i] - rhs_c_shape[2 + i]) // window_strides[i] + 1
        for i in range(spatial_dims)
    ]
    return [lhs_pad_shape[0], rhs_c_shape[0]] + out_spatial


def _get_inv_out_spec(out_spec: tuple[int, ...]) -> list[int]:
    """Function docstring.

    Args:
        out_spec: Arg.
    """
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
    rhs = np.asarray(rhs)
    spatial_dims = lhs.ndim - 2

    specs = _parse_conv_dimension_numbers(
        lhs.ndim, rhs.ndim, spatial_dims, config.dimension_numbers
    )

    lhs_pad, rhs_c = _preprocess_conv_tensors(lhs, rhs, config, specs)

    out_shape = _compute_out_shape(lhs_pad.shape, rhs_c.shape, spatial_dims, config.window_strides)
    out = np.zeros(out_shape, dtype=lhs.dtype)

    _compute_conv_patches(lhs_pad, rhs_c, out, config)

    inv_out_spec = _get_inv_out_spec(specs.out_spec)
    return np.transpose(out, inv_out_spec)


@numpy_eager_registry.register("ConvGeneralDilated")
def _np_conv_general_dilated(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return _conv_general_dilated(*args, **kwargs)
