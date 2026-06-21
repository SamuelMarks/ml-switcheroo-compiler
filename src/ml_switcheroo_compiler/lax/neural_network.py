"""Lax primitives."""

from ml_switcheroo_compiler.core.dispatch import dispatch


def bitcast_convert_type(*args: object, **kwargs: object) -> object:
    """Execute bitcast_convert_type."""
    return dispatch("lax", "bitcast_convert_type", *args, **kwargs)


def bitcast_convert_type_p(*args: object, **kwargs: object) -> object:
    """Execute bitcast_convert_type_p."""
    return dispatch("lax", "bitcast_convert_type_p", *args, **kwargs)


def conv(*args: object, **kwargs: object) -> object:
    """Execute conv."""
    return dispatch("lax", "conv", *args, **kwargs)


def conv_dimension_numbers(*args: object, **kwargs: object) -> object:
    """Execute conv_dimension_numbers."""
    return dispatch("lax", "conv_dimension_numbers", *args, **kwargs)


def conv_general_dilated_local(*args: object, **kwargs: object) -> object:
    """Execute conv_general_dilated_local."""
    return dispatch("lax", "conv_general_dilated_local", *args, **kwargs)


def conv_general_dilated_p(*args: object, **kwargs: object) -> object:
    """Execute conv_general_dilated_p."""
    return dispatch("lax", "conv_general_dilated_p", *args, **kwargs)


def conv_general_dilated_patches(*args: object, **kwargs: object) -> object:
    """Execute conv_general_dilated_patches."""
    return dispatch("lax", "conv_general_dilated_patches", *args, **kwargs)


def conv_general_permutations(*args: object, **kwargs: object) -> object:
    """Execute conv_general_permutations."""
    return dispatch("lax", "conv_general_permutations", *args, **kwargs)


def conv_general_shape_tuple(*args: object, **kwargs: object) -> object:
    """Execute conv_general_shape_tuple."""
    return dispatch("lax", "conv_general_shape_tuple", *args, **kwargs)


def conv_shape_tuple(*args: object, **kwargs: object) -> object:
    """Execute conv_shape_tuple."""
    return dispatch("lax", "conv_shape_tuple", *args, **kwargs)


def conv_transpose(*args: object, **kwargs: object) -> object:
    """Execute conv_transpose."""
    return dispatch("lax", "conv_transpose", *args, **kwargs)


def conv_transpose_shape_tuple(*args: object, **kwargs: object) -> object:
    """Execute conv_transpose_shape_tuple."""
    return dispatch("lax", "conv_transpose_shape_tuple", *args, **kwargs)


def conv_with_general_padding(*args: object, **kwargs: object) -> object:
    """Execute conv_with_general_padding."""
    return dispatch("lax", "conv_with_general_padding", *args, **kwargs)


def convert_element_type(*args: object, **kwargs: object) -> object:
    """Execute convert_element_type."""
    return dispatch("lax", "convert_element_type", *args, **kwargs)


def convert_element_type_p(*args: object, **kwargs: object) -> object:
    """Execute convert_element_type_p."""
    return dispatch("lax", "convert_element_type_p", *args, **kwargs)


def reduce_window_max_p(*args: object, **kwargs: object) -> object:
    """Execute reduce_window_max_p."""
    return dispatch("lax", "reduce_window_max_p", *args, **kwargs)


def reduce_window_min_p(*args: object, **kwargs: object) -> object:
    """Execute reduce_window_min_p."""
    return dispatch("lax", "reduce_window_min_p", *args, **kwargs)


def reduce_window_p(*args: object, **kwargs: object) -> object:
    """Execute reduce_window_p."""
    return dispatch("lax", "reduce_window_p", *args, **kwargs)


def reduce_window_shape_tuple(*args: object, **kwargs: object) -> object:
    """Execute reduce_window_shape_tuple."""
    return dispatch("lax", "reduce_window_shape_tuple", *args, **kwargs)


def reduce_window_sum_p(*args: object, **kwargs: object) -> object:
    """Execute reduce_window_sum_p."""
    return dispatch("lax", "reduce_window_sum_p", *args, **kwargs)
