"""Lax primitives."""

from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.dispatch import dispatch


def argmax_p(*args: object, **kwargs: object) -> object:
    """Execute argmax_p."""
    return dispatch("lax", "argmax_p", *args, **kwargs)


def argmin_p(*args: object, **kwargs: object) -> object:
    """Execute argmin_p."""
    return dispatch("lax", "argmin_p", *args, **kwargs)


def axis_index(*args: object, **kwargs: object) -> object:
    """Execute axis_index."""
    return dispatch("lax", "axis_index", *args, **kwargs)


def axis_index_p(*args: object, **kwargs: object) -> object:
    """Execute axis_index_p."""
    return dispatch("lax", "axis_index_p", *args, **kwargs)


def broadcast_in_dim_p(*args: object, **kwargs: object) -> object:
    """Execute broadcast_in_dim_p."""
    return dispatch("lax", "broadcast_in_dim_p", *args, **kwargs)


def broadcast_to_rank(*args: object, **kwargs: object) -> object:
    """Execute broadcast_to_rank."""
    return dispatch("lax", "broadcast_to_rank", *args, **kwargs)


def broadcasted_iota(*args: object, **kwargs: object) -> object:
    """Execute broadcasted_iota."""
    return dispatch("lax", "broadcasted_iota", *args, **kwargs)


def collapse(*args: object, **kwargs: object) -> object:
    """Execute collapse."""
    return dispatch("lax", "collapse", *args, **kwargs)


def concatenate_p(*args: object, **kwargs: object) -> object:
    """Execute concatenate_p."""
    return dispatch("lax", "concatenate_p", *args, **kwargs)


def copy_p(*args: object, **kwargs: object) -> object:
    """Execute copy_p."""
    return dispatch("lax", "copy_p", *args, **kwargs)


def create_token(*args: object, **kwargs: object) -> object:
    """Execute create_token."""
    return dispatch("lax", "create_token", *args, **kwargs)


def create_token_p(*args: object, **kwargs: object) -> object:
    """Execute create_token_p."""
    return dispatch("lax", "create_token_p", *args, **kwargs)


def custom_root(*args: object, **kwargs: object) -> object:
    """Execute custom_root."""
    return dispatch("lax", "custom_root", *args, **kwargs)


def device_put_p(*args: object, **kwargs: object) -> object:
    """Execute device_put_p."""
    return dispatch("lax", "device_put_p", *args, **kwargs)


def dynamic_index_in_dim(*args: object, **kwargs: object) -> object:
    """Execute dynamic_index_in_dim."""
    return dispatch("lax", "dynamic_index_in_dim", *args, **kwargs)


def dynamic_slice_in_dim(*args: object, **kwargs: object) -> object:
    """Execute dynamic_slice_in_dim."""
    return dispatch("lax", "dynamic_slice_in_dim", *args, **kwargs)


def dynamic_slice_p(*args: object, **kwargs: object) -> object:
    """Execute dynamic_slice_p."""
    return dispatch("lax", "dynamic_slice_p", *args, **kwargs)


def dynamic_update_index_in_dim(*args: object, **kwargs: object) -> object:
    """Execute dynamic_update_index_in_dim."""
    return dispatch("lax", "dynamic_update_index_in_dim", *args, **kwargs)


def dynamic_update_slice_in_dim(*args: object, **kwargs: object) -> object:
    """Execute dynamic_update_slice_in_dim."""
    return dispatch("lax", "dynamic_update_slice_in_dim", *args, **kwargs)


def dynamic_update_slice_p(*args: object, **kwargs: object) -> object:
    """Execute dynamic_update_slice_p."""
    return dispatch("lax", "dynamic_update_slice_p", *args, **kwargs)


def fft_p(*args: object, **kwargs: object) -> object:
    """Execute fft_p."""
    return dispatch("lax", "fft_p", *args, **kwargs)


def index_in_dim(*args: object, **kwargs: object) -> object:
    """Execute index_in_dim."""
    return dispatch("lax", "index_in_dim", *args, **kwargs)


def index_take(*args: object, **kwargs: object) -> object:
    """Execute index_take."""
    return dispatch("lax", "index_take", *args, **kwargs)


def infeed(*args: object, **kwargs: object) -> object:
    """Execute infeed."""
    return dispatch("lax", "infeed", *args, **kwargs)


def infeed_p(*args: object, **kwargs: object) -> object:
    """Execute infeed_p."""
    return dispatch("lax", "infeed_p", *args, **kwargs)


def iota(*args: object, **kwargs: object) -> object:
    """Execute iota."""
    return dispatch("lax", "iota", *args, **kwargs)


def iota_p(*args: object, **kwargs: object) -> object:
    """Execute iota_p."""
    return dispatch("lax", "iota_p", *args, **kwargs)


def outfeed(*args: object, **kwargs: object) -> object:
    """Execute outfeed."""
    return dispatch("lax", "outfeed", *args, **kwargs)


def outfeed_p(*args: object, **kwargs: object) -> object:
    """Execute outfeed_p."""
    return dispatch("lax", "outfeed_p", *args, **kwargs)


def pad_p(*args: object, **kwargs: object) -> object:
    """Execute pad_p."""
    return dispatch("lax", "pad_p", *args, **kwargs)


def platform_dependent(*args: object, **kwargs: object) -> object:
    """Execute platform_dependent."""
    return dispatch("lax", "platform_dependent", *args, **kwargs)


def random_gamma_grad(*args: object, **kwargs: object) -> object:
    """Execute random_gamma_grad."""
    return dispatch("lax", "random_gamma_grad", *args, **kwargs)


def random_gamma_grad_p(*args: object, **kwargs: object) -> object:
    """Execute random_gamma_grad_p."""
    return dispatch("lax", "random_gamma_grad_p", *args, **kwargs)


def regularized_incomplete_beta_p(*args: object, **kwargs: object) -> object:
    """Execute regularized_incomplete_beta_p."""
    return dispatch("lax", "regularized_incomplete_beta_p", *args, **kwargs)


def reshape_p(*args: object, **kwargs: object) -> object:
    """Execute reshape_p."""
    return dispatch("lax", "reshape_p", *args, **kwargs)


def rev(*args: object, **kwargs: object) -> object:
    """Execute rev."""
    return dispatch("lax", "rev", *args, **kwargs)


def rev_p(*args: object, **kwargs: object) -> object:
    """Execute rev_p."""
    return dispatch("lax", "rev_p", *args, **kwargs)


def scatter_apply(*args: object, **kwargs: object) -> object:
    """Execute scatter_apply."""
    return dispatch("lax", "scatter_apply", *args, **kwargs)


def scatter_p(*args: object, **kwargs: object) -> object:
    """Execute scatter_p."""
    return dispatch("lax", "scatter_p", *args, **kwargs)


def select_n(*args: object, **kwargs: object) -> object:
    """Execute select_n."""
    return dispatch("lax", "select_n", *args, **kwargs)


def select_n_p(*args: object, **kwargs: object) -> object:
    """Execute select_n_p."""
    return dispatch("lax", "select_n_p", *args, **kwargs)


def slice_in_dim(*args: object, **kwargs: object) -> object:
    """Execute slice_in_dim."""
    return dispatch("lax", "slice_in_dim", *args, **kwargs)


def slice_p(*args: object, **kwargs: object) -> object:
    """Execute slice_p."""
    return dispatch("lax", "slice_p", *args, **kwargs)


def sort_key_val(*args: object, **kwargs: object) -> object:
    """Execute sort_key_val."""
    return dispatch("lax", "sort_key_val", *args, **kwargs)


def sort_p(*args: object, **kwargs: object) -> object:
    """Execute sort_p."""
    return dispatch("lax", "sort_p", *args, **kwargs)


def squeeze_p(*args: object, **kwargs: object) -> object:
    """Execute squeeze_p."""
    return dispatch("lax", "squeeze_p", *args, **kwargs)


def stop_gradient_p(*args: object, **kwargs: object) -> object:
    """Execute stop_gradient_p."""
    return dispatch("lax", "stop_gradient_p", *args, **kwargs)


def top_k_p(*args: object, **kwargs: object) -> object:
    """Execute top_k_p."""
    return dispatch("lax", "top_k_p", *args, **kwargs)


def transpose_p(*args: object, **kwargs: object) -> object:
    """Execute transpose_p."""
    return dispatch("lax", "transpose_p", *args, **kwargs)


def zeros_like_array(*args: object, **kwargs: object) -> object:
    """Execute zeros_like_array."""
    return dispatch("lax", "zeros_like_array", *args, **kwargs)


dtype = DType
