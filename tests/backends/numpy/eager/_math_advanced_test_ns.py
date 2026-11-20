"""Composite test namespace re-exporting unified reference math and SciPy kernel helpers."""

from __future__ import annotations

from ml_switcheroo_compiler.backends.eager.nn_ops.layers import _np_convolve as _np_convolve
from ml_switcheroo_compiler.backends.eager.reference_math import (
    math_bit as math_bit,
)
from ml_switcheroo_compiler.backends.eager.reference_math import (
    math_creation as math_creation,
)
from ml_switcheroo_compiler.backends.eager.reference_math import (
    math_gamma as math_gamma,
)
from ml_switcheroo_compiler.backends.eager.reference_math import (
    math_general as math_general,
)
from ml_switcheroo_compiler.backends.eager.reference_math import (
    math_internal as math_internal,
)
from ml_switcheroo_compiler.backends.eager.reference_math import (
    math_linalg as math_linalg,
)
from ml_switcheroo_compiler.backends.eager.reference_math import (
    math_log_exp as math_log_exp,
)
from ml_switcheroo_compiler.backends.eager.reference_math import (
    math_manipulation as math_manipulation,
)
from ml_switcheroo_compiler.backends.eager.reference_math import (
    math_matrix_utils as math_matrix_utils,
)
from ml_switcheroo_compiler.backends.eager.reference_math import (
    math_random_sampling as math_random_sampling,
)
from ml_switcheroo_compiler.backends.eager.reference_math import (
    math_segment as math_segment,
)
from ml_switcheroo_compiler.backends.eager.reference_math import (
    math_set as math_set,
)
from ml_switcheroo_compiler.backends.eager.reference_math import (
    math_signal as math_signal,
)
from ml_switcheroo_compiler.backends.eager.reference_math import (
    math_sorting as math_sorting,
)
from ml_switcheroo_compiler.backends.eager.reference_math import (
    math_string_io as math_string_io,
)
from ml_switcheroo_compiler.backends.eager.reference_math import (
    math_testing as math_testing,
)
from ml_switcheroo_compiler.backends.eager.reference_math import (
    math_tri as math_tri,
)
from ml_switcheroo_compiler.backends.eager.reference_math.math_bit import (
    _np_bitcast_convert_type as _np_bitcast_convert_type,
)
from ml_switcheroo_compiler.backends.eager.reference_math.math_bit import _np_bitwise_count as _np_bitwise_count
from ml_switcheroo_compiler.backends.eager.reference_math.math_bit import _np_clz as _np_clz
from ml_switcheroo_compiler.backends.eager.reference_math.math_bit import _np_packbits as _np_packbits
from ml_switcheroo_compiler.backends.eager.reference_math.math_bit import _np_unpackbits as _np_unpackbits
from ml_switcheroo_compiler.backends.eager.reference_math.math_creation import _np_arange_ as _np_arange_
from ml_switcheroo_compiler.backends.eager.reference_math.math_creation import _np_empty_ as _np_empty_
from ml_switcheroo_compiler.backends.eager.reference_math.math_creation import _np_empty_like_ as _np_empty_like_
from ml_switcheroo_compiler.backends.eager.reference_math.math_creation import _np_frombuffer as _np_frombuffer
from ml_switcheroo_compiler.backends.eager.reference_math.math_creation import _np_fromdlpack as _np_fromdlpack
from ml_switcheroo_compiler.backends.eager.reference_math.math_creation import _np_fromfunction_ as _np_fromfunction_
from ml_switcheroo_compiler.backends.eager.reference_math.math_creation import _np_fromiter_ as _np_fromiter_
from ml_switcheroo_compiler.backends.eager.reference_math.math_creation import _np_frompyfunc_ as _np_frompyfunc_
from ml_switcheroo_compiler.backends.eager.reference_math.math_creation import _np_full_ as _np_full_
from ml_switcheroo_compiler.backends.eager.reference_math.math_creation import _np_full_like_ as _np_full_like_
from ml_switcheroo_compiler.backends.eager.reference_math.math_creation import _np_geomspace_ as _np_geomspace_
from ml_switcheroo_compiler.backends.eager.reference_math.math_creation import (
    _np_linearoperatoridentity as _np_linearoperatoridentity,
)
from ml_switcheroo_compiler.backends.eager.reference_math.math_creation import (
    _np_linearoperatorscaledidentity as _np_linearoperatorscaledidentity,
)
from ml_switcheroo_compiler.backends.eager.reference_math.math_creation import (
    _np_linearoperatorzeros as _np_linearoperatorzeros,
)
from ml_switcheroo_compiler.backends.eager.reference_math.math_creation import _np_ones_ as _np_ones_
from ml_switcheroo_compiler.backends.eager.reference_math.math_creation import _np_ones_like_ as _np_ones_like_
from ml_switcheroo_compiler.backends.eager.reference_math.math_creation import _np_zeros_ as _np_zeros_
from ml_switcheroo_compiler.backends.eager.reference_math.math_creation import _np_zeros_like_ as _np_zeros_like_
from ml_switcheroo_compiler.backends.eager.reference_math.math_gamma import _np_mvlgamma as _np_mvlgamma
from ml_switcheroo_compiler.backends.eager.reference_math.math_general import (
    _build_dot_general_einsum_str as _build_dot_general_einsum_str,
)
from ml_switcheroo_compiler.backends.eager.reference_math.math_general import _get_np_arg as _get_np_arg
from ml_switcheroo_compiler.backends.eager.reference_math.math_general import _np_apply_over_axes as _np_apply_over_axes
from ml_switcheroo_compiler.backends.eager.reference_math.math_general import _np_array_repr_ as _np_array_repr_
from ml_switcheroo_compiler.backends.eager.reference_math.math_general import _np_array_str_ as _np_array_str_
from ml_switcheroo_compiler.backends.eager.reference_math.math_general import _np_atleast_1d as _np_atleast_1d
from ml_switcheroo_compiler.backends.eager.reference_math.math_general import _np_atleast_2d as _np_atleast_2d
from ml_switcheroo_compiler.backends.eager.reference_math.math_general import _np_atleast_3d as _np_atleast_3d
from ml_switcheroo_compiler.backends.eager.reference_math.math_general import _np_average as _np_average
from ml_switcheroo_compiler.backends.eager.reference_math.math_general import _np_block as _np_block
from ml_switcheroo_compiler.backends.eager.reference_math.math_general import _np_callable as _np_callable
from ml_switcheroo_compiler.backends.eager.reference_math.math_general import _np_clip as _np_clip
from ml_switcheroo_compiler.backends.eager.reference_math.math_general import _np_corrcoef as _np_corrcoef
from ml_switcheroo_compiler.backends.eager.reference_math.math_general import _np_cov as _np_cov
from ml_switcheroo_compiler.backends.eager.reference_math.math_general import _np_debuginfs as _np_debuginfs
from ml_switcheroo_compiler.backends.eager.reference_math.math_general import _np_debugnans as _np_debugnans
from ml_switcheroo_compiler.backends.eager.reference_math.math_general import _np_descriptive_2 as _np_descriptive
from ml_switcheroo_compiler.backends.eager.reference_math.math_general import _np_diff_ as _np_diff_
from ml_switcheroo_compiler.backends.eager.reference_math.math_general import _np_digitize_ as _np_digitize_
from ml_switcheroo_compiler.backends.eager.reference_math.math_general import _np_dotgeneral as _np_dotgeneral
from ml_switcheroo_compiler.backends.eager.reference_math.math_general import _np_ediff1d_ as _np_ediff1d_
from ml_switcheroo_compiler.backends.eager.reference_math.math_general import _np_fabs_ as _np_fabs_
from ml_switcheroo_compiler.backends.eager.reference_math.math_general import _np_interp_ as _np_interp_
from ml_switcheroo_compiler.backends.eager.reference_math.math_general import _np_iterable_ as _np_iterable_
from ml_switcheroo_compiler.backends.eager.reference_math.math_general import _np_ix__ as _np_ix__
from ml_switcheroo_compiler.backends.eager.reference_math.math_general import _np_key as _np_key
from ml_switcheroo_compiler.backends.eager.reference_math.math_general import _np_kron_ as _np_kron_
from ml_switcheroo_compiler.backends.eager.reference_math.math_general import _np_load_ as _np_load_
from ml_switcheroo_compiler.backends.eager.reference_math.math_general import _np_modf_ as _np_modf_
from ml_switcheroo_compiler.backends.eager.reference_math.math_general import _np_one_hot as _np_one_hot
from ml_switcheroo_compiler.backends.eager.reference_math.math_general import _np_raggeddot as _np_raggeddot
from ml_switcheroo_compiler.backends.eager.reference_math.math_general import (
    _np_ravel_multi_index_ as _np_ravel_multi_index_,
)
from ml_switcheroo_compiler.backends.eager.reference_math.math_general import _np_rem_2 as _np_rem
from ml_switcheroo_compiler.backends.eager.reference_math.math_general import (
    _np_serialize_tensor as _np_serialize_tensor,
)
from ml_switcheroo_compiler.backends.eager.reference_math.math_general import (
    _np_serialize_tensor_camel as _np_serialize_tensor_camel,
)
from ml_switcheroo_compiler.backends.eager.reference_math.math_general import _np_tensor as _np_tensor
from ml_switcheroo_compiler.backends.eager.reference_math.math_general import _np_trapz as _np_trapz
from ml_switcheroo_compiler.backends.eager.reference_math.math_general import _np_trapz_ as _np_trapz_
from ml_switcheroo_compiler.backends.eager.reference_math.math_general import _np_truncate_div as _np_truncate_div
from ml_switcheroo_compiler.backends.eager.reference_math.math_general import _np_truncate_mod as _np_truncate_mod
from ml_switcheroo_compiler.backends.eager.reference_math.math_general import _np_uint as _np_uint
from ml_switcheroo_compiler.backends.eager.reference_math.math_general import _np_uint8 as _np_uint8
from ml_switcheroo_compiler.backends.eager.reference_math.math_general import _np_unravelindex as _np_unravelindex
from ml_switcheroo_compiler.backends.eager.reference_math.math_general import _np_unwrap as _np_unwrap
from ml_switcheroo_compiler.backends.eager.reference_math.math_general import _np_vectorize as _np_vectorize
from ml_switcheroo_compiler.backends.eager.reference_math.math_internal import _np_affineconfig as _np_affineconfig
from ml_switcheroo_compiler.backends.eager.reference_math.math_internal import _np_blurconfig as _np_blurconfig
from ml_switcheroo_compiler.backends.eager.reference_math.math_internal import _np_customroot as _np_customroot
from ml_switcheroo_compiler.backends.eager.reference_math.math_internal import _np_elasticconfig as _np_elasticconfig
from ml_switcheroo_compiler.backends.eager.reference_math.math_internal import (
    _np_get_printoptions_ as _np_get_printoptions_,
)
from ml_switcheroo_compiler.backends.eager.reference_math.math_internal import _np_linearoperator as _np_linearoperator
from ml_switcheroo_compiler.backends.eager.reference_math.math_internal import (
    _np_linearoperatoradjoint as _np_linearoperatoradjoint,
)
from ml_switcheroo_compiler.backends.eager.reference_math.math_internal import (
    _np_linearoperatorcirculant as _np_linearoperatorcirculant,
)
from ml_switcheroo_compiler.backends.eager.reference_math.math_internal import (
    _np_linearoperatorcirculant2d as _np_linearoperatorcirculant2d,
)
from ml_switcheroo_compiler.backends.eager.reference_math.math_internal import (
    _np_linearoperatorcirculant3d as _np_linearoperatorcirculant3d,
)
from ml_switcheroo_compiler.backends.eager.reference_math.math_internal import (
    _np_linearoperatorcomposition as _np_linearoperatorcomposition,
)
from ml_switcheroo_compiler.backends.eager.reference_math.math_internal import (
    _np_linearoperatorhouseholder as _np_linearoperatorhouseholder,
)
from ml_switcheroo_compiler.backends.eager.reference_math.math_internal import (
    _np_linearoperatorkronecker as _np_linearoperatorkronecker,
)
from ml_switcheroo_compiler.backends.eager.reference_math.math_internal import (
    _np_linearoperatorlowrankupdate as _np_linearoperatorlowrankupdate,
)
from ml_switcheroo_compiler.backends.eager.reference_math.math_internal import (
    _np_linearoperatortoeplitz as _np_linearoperatortoeplitz,
)
from ml_switcheroo_compiler.backends.eager.reference_math.math_internal import (
    _np_perspectiveconfig as _np_perspectiveconfig,
)
from ml_switcheroo_compiler.backends.eager.reference_math.math_internal import (
    _np_population_count as _np_population_count,
)
from ml_switcheroo_compiler.backends.eager.reference_math.math_internal import _np_rawmatmul as _np_rawmatmul
from ml_switcheroo_compiler.backends.eager.reference_math.math_internal import _np_rawmerge as _np_rawmerge
from ml_switcheroo_compiler.backends.eager.reference_math.math_internal import _np_rawop as _np_rawop
from ml_switcheroo_compiler.backends.eager.reference_math.math_internal import _np_rawswitch as _np_rawswitch
from ml_switcheroo_compiler.backends.eager.reference_math.math_internal import _np_scanop as _np_scanop
from ml_switcheroo_compiler.backends.eager.reference_math.math_internal import _np_switchop as _np_switchop
from ml_switcheroo_compiler.backends.eager.reference_math.math_internal import (
    _np_tensorarrayread as _np_tensorarrayread,
)
from ml_switcheroo_compiler.backends.eager.reference_math.math_internal import (
    _np_tensorarraywrite as _np_tensorarraywrite,
)
from ml_switcheroo_compiler.backends.eager.reference_math.math_internal import _np_tensorconfig as _np_tensorconfig
from ml_switcheroo_compiler.backends.eager.reference_math.math_linalg import (
    _np_customlinearsolve as _np_customlinearsolve,
)
from ml_switcheroo_compiler.backends.eager.reference_math.math_linalg import (
    _np_linalg_cholesky_ as _np_linalg_cholesky_,
)
from ml_switcheroo_compiler.backends.eager.reference_math.math_linalg import _np_linalg_det_ as _np_linalg_det_
from ml_switcheroo_compiler.backends.eager.reference_math.math_linalg import _np_linalg_inv_ as _np_linalg_inv_
from ml_switcheroo_compiler.backends.eager.reference_math.math_linalg import _np_linalg_pinv_ as _np_linalg_pinv_
from ml_switcheroo_compiler.backends.eager.reference_math.math_linalg import _np_linalg_svd_ as _np_linalg_svd_
from ml_switcheroo_compiler.backends.eager.reference_math.math_linalg import (
    _np_linearoperatorinversion as _np_linearoperatorinversion,
)
from ml_switcheroo_compiler.backends.eager.reference_math.math_linalg import _np_rsqrt as _np_rsqrt
from ml_switcheroo_compiler.backends.eager.reference_math.math_linalg import _np_trace as _np_trace
from ml_switcheroo_compiler.backends.eager.reference_math.math_linalg import _np_vander as _np_vander
from ml_switcheroo_compiler.backends.eager.reference_math.math_linalg import _np_vecdot as _np_vecdot
from ml_switcheroo_compiler.backends.eager.reference_math.math_log_exp import _np_expand_dims_ as _np_expand_dims_
from ml_switcheroo_compiler.backends.eager.reference_math.math_log_exp import _np_log1p2 as _np_log1p2
from ml_switcheroo_compiler.backends.eager.reference_math.math_log_exp import _np_log_softmax as _np_log_softmax
from ml_switcheroo_compiler.backends.eager.reference_math.math_log_exp import _np_logsumexp as _np_logsumexp
from ml_switcheroo_compiler.backends.eager.reference_math.math_log_exp import _np_xlogy as _np_xlogy
from ml_switcheroo_compiler.backends.eager.reference_math.math_manipulation import _np_append as _np_append
from ml_switcheroo_compiler.backends.eager.reference_math.math_manipulation import _np_argwhere as _np_argwhere
from ml_switcheroo_compiler.backends.eager.reference_math.math_manipulation import _np_choose as _np_choose
from ml_switcheroo_compiler.backends.eager.reference_math.math_manipulation import _np_column_stack as _np_column_stack
from ml_switcheroo_compiler.backends.eager.reference_math.math_manipulation import _np_compress as _np_compress
from ml_switcheroo_compiler.backends.eager.reference_math.math_manipulation import _np_delete_ as _np_delete_
from ml_switcheroo_compiler.backends.eager.reference_math.math_manipulation import _np_extract_ as _np_extract_
from ml_switcheroo_compiler.backends.eager.reference_math.math_manipulation import (
    _np_extractpatchesoptions as _np_extractpatchesoptions,
)
from ml_switcheroo_compiler.backends.eager.reference_math.math_manipulation import _np_flatnonzero_ as _np_flatnonzero_
from ml_switcheroo_compiler.backends.eager.reference_math.math_manipulation import _np_flip_op_ as _np_flip_op_
from ml_switcheroo_compiler.backends.eager.reference_math.math_manipulation import (
    _np_flip_reverse_ as _np_flip_reverse_,
)
from ml_switcheroo_compiler.backends.eager.reference_math.math_manipulation import _np_fliplr_ as _np_fliplr_
from ml_switcheroo_compiler.backends.eager.reference_math.math_manipulation import _np_flipud_ as _np_flipud_
from ml_switcheroo_compiler.backends.eager.reference_math.math_manipulation import _np_insert_ as _np_insert_
from ml_switcheroo_compiler.backends.eager.reference_math.math_manipulation import _np_nonzero_ as _np_nonzero_
from ml_switcheroo_compiler.backends.eager.reference_math.math_manipulation import _np_resize_ as _np_resize_
from ml_switcheroo_compiler.backends.eager.reference_math.math_manipulation import (
    _np_tensorarraystack as _np_tensorarraystack,
)
from ml_switcheroo_compiler.backends.eager.reference_math.math_matrix_utils import (
    _np_diag_indices_ as _np_diag_indices_,
)
from ml_switcheroo_compiler.backends.eager.reference_math.math_matrix_utils import (
    _np_diag_indices_from_ as _np_diag_indices_from_,
)
from ml_switcheroo_compiler.backends.eager.reference_math.math_matrix_utils import _np_diagflat_ as _np_diagflat_
from ml_switcheroo_compiler.backends.eager.reference_math.math_matrix_utils import _np_diagonal_ as _np_diagonal_
from ml_switcheroo_compiler.backends.eager.reference_math.math_matrix_utils import _np_indices_ as _np_indices_
from ml_switcheroo_compiler.backends.eager.reference_math.math_matrix_utils import (
    _np_linearoperatorblockdiag as _np_linearoperatorblockdiag,
)
from ml_switcheroo_compiler.backends.eager.reference_math.math_matrix_utils import (
    _np_linearoperatordiag as _np_linearoperatordiag,
)
from ml_switcheroo_compiler.backends.eager.reference_math.math_matrix_utils import (
    _np_mask_indices_ as _np_mask_indices_,
)
from ml_switcheroo_compiler.backends.eager.reference_math.math_random_sampling import (
    _np_linearoperatorpermutation as _np_linearoperatorpermutation,
)
from ml_switcheroo_compiler.backends.eager.reference_math.math_random_sampling import (
    _np_randombernoulli as _np_randombernoulli,
)
from ml_switcheroo_compiler.backends.eager.reference_math.math_random_sampling import (
    _np_randomcategorical as _np_randomcategorical,
)
from ml_switcheroo_compiler.backends.eager.reference_math.math_random_sampling import (
    _np_randompermutation as _np_randompermutation,
)
from ml_switcheroo_compiler.backends.eager.reference_math.math_random_sampling import (
    _np_randomtruncatednormal as _np_randomtruncatednormal,
)
from ml_switcheroo_compiler.backends.eager.reference_math.math_random_sampling import _np_sobolsample as _np_sobolsample
from ml_switcheroo_compiler.backends.eager.reference_math.math_segment import _np_segment_sum as _np_segment_sum
from ml_switcheroo_compiler.backends.eager.reference_math.math_segment import (
    _np_sparsesegmentsum as _np_sparsesegmentsum,
)
from ml_switcheroo_compiler.backends.eager.reference_math.math_set import _np_intersect1d_ as _np_intersect1d_
from ml_switcheroo_compiler.backends.eager.reference_math.math_set import _np_isin_ as _np_isin_
from ml_switcheroo_compiler.backends.eager.reference_math.math_set import _np_union1d as _np_union1d
from ml_switcheroo_compiler.backends.eager.reference_math.math_signal import _np_blackman_ as _np_blackman_
from ml_switcheroo_compiler.backends.eager.reference_math.math_signal import _np_correlate as _np_correlate
from ml_switcheroo_compiler.backends.eager.reference_math.math_signal import _np_fftn as _np_fftn
from ml_switcheroo_compiler.backends.eager.reference_math.math_signal import _np_fftnd as _np_fftnd
from ml_switcheroo_compiler.backends.eager.reference_math.math_signal import _np_fftshift as _np_fftshift
from ml_switcheroo_compiler.backends.eager.reference_math.math_signal import _np_hamming_ as _np_hamming_
from ml_switcheroo_compiler.backends.eager.reference_math.math_signal import _np_hanning_ as _np_hanning_
from ml_switcheroo_compiler.backends.eager.reference_math.math_signal import _np_hfft as _np_hfft
from ml_switcheroo_compiler.backends.eager.reference_math.math_signal import _np_ifft as _np_ifft
from ml_switcheroo_compiler.backends.eager.reference_math.math_signal import _np_ifft2 as _np_ifft2
from ml_switcheroo_compiler.backends.eager.reference_math.math_signal import _np_ifftn as _np_ifftn
from ml_switcheroo_compiler.backends.eager.reference_math.math_signal import _np_ifftnd as _np_ifftnd
from ml_switcheroo_compiler.backends.eager.reference_math.math_signal import _np_ifftshift as _np_ifftshift
from ml_switcheroo_compiler.backends.eager.reference_math.math_signal import _np_irfft2 as _np_irfft2
from ml_switcheroo_compiler.backends.eager.reference_math.math_signal import _np_irfftn as _np_irfftn
from ml_switcheroo_compiler.backends.eager.reference_math.math_signal import _np_irfftnd as _np_irfftnd
from ml_switcheroo_compiler.backends.eager.reference_math.math_signal import _np_kaiser_ as _np_kaiser_
from ml_switcheroo_compiler.backends.eager.reference_math.math_signal import _np_rfft as _np_rfft
from ml_switcheroo_compiler.backends.eager.reference_math.math_signal import _np_rfft2 as _np_rfft2
from ml_switcheroo_compiler.backends.eager.reference_math.math_signal import _np_rfftfreq as _np_rfftfreq
from ml_switcheroo_compiler.backends.eager.reference_math.math_signal import _np_rfftn as _np_rfftn
from ml_switcheroo_compiler.backends.eager.reference_math.math_signal import _np_rfftnd as _np_rfftnd
from ml_switcheroo_compiler.backends.eager.reference_math.math_signal import _np_rrelu as _np_rrelu
from ml_switcheroo_compiler.backends.eager.reference_math.math_signal import _np_sigmoid as _np_sigmoid
from ml_switcheroo_compiler.backends.eager.reference_math.math_signal import _np_softmax as _np_softmax
from ml_switcheroo_compiler.backends.eager.reference_math.math_sorting import _np_argpartition as _np_argpartition
from ml_switcheroo_compiler.backends.eager.reference_math.math_sorting import _np_lexsort_ as _np_lexsort_
from ml_switcheroo_compiler.backends.eager.reference_math.math_sorting import _np_median_ as _np_median_
from ml_switcheroo_compiler.backends.eager.reference_math.math_string_io import _get_csv_data as _get_csv_data
from ml_switcheroo_compiler.backends.eager.reference_math.math_string_io import _get_csv_defaults as _get_csv_defaults
from ml_switcheroo_compiler.backends.eager.reference_math.math_string_io import _np_decode_csv as _np_decode_csv
from ml_switcheroo_compiler.backends.eager.reference_math.math_string_io import (
    _np_decode_csv_camel as _np_decode_csv_camel,
)
from ml_switcheroo_compiler.backends.eager.reference_math.math_string_io import _np_decode_image as _np_decode_image
from ml_switcheroo_compiler.backends.eager.reference_math.math_string_io import (
    _np_decode_image_camel as _np_decode_image_camel,
)
from ml_switcheroo_compiler.backends.eager.reference_math.math_string_io import _np_parse_tensor as _np_parse_tensor
from ml_switcheroo_compiler.backends.eager.reference_math.math_string_io import (
    _np_parse_tensor_camel as _np_parse_tensor_camel,
)
from ml_switcheroo_compiler.backends.eager.reference_math.math_string_io import _np_read_file as _np_read_file
from ml_switcheroo_compiler.backends.eager.reference_math.math_string_io import (
    _np_read_file_camel as _np_read_file_camel,
)
from ml_switcheroo_compiler.backends.eager.reference_math.math_string_io import (
    _np_sparsedensematmul as _np_sparsedensematmul,
)
from ml_switcheroo_compiler.backends.eager.reference_math.math_string_io import (
    _np_sparsemapvalues as _np_sparsemapvalues,
)
from ml_switcheroo_compiler.backends.eager.reference_math.math_string_io import _np_sparsereshape as _np_sparsereshape
from ml_switcheroo_compiler.backends.eager.reference_math.math_string_io import (
    _np_sparsesampledadd as _np_sparsesampledadd,
)
from ml_switcheroo_compiler.backends.eager.reference_math.math_string_io import (
    _np_sparsetranspose as _np_sparsetranspose,
)
from ml_switcheroo_compiler.backends.eager.reference_math.math_string_io import _np_write_file as _np_write_file
from ml_switcheroo_compiler.backends.eager.reference_math.math_string_io import (
    _np_write_file_camel as _np_write_file_camel,
)
from ml_switcheroo_compiler.backends.eager.reference_math.math_string_io import _parse_csv_row as _parse_csv_row
from ml_switcheroo_compiler.backends.eager.reference_math.math_string_io import _parse_scanop_args as _parse_scanop_args
from ml_switcheroo_compiler.backends.eager.reference_math.math_testing import (
    _np_apply_along_axis as _np_apply_along_axis,
)
from ml_switcheroo_compiler.backends.eager.reference_math.math_testing import _np_array_equiv_ as _np_array_equiv_
from ml_switcheroo_compiler.backends.eager.reference_math.math_testing import _np_assertop as _np_assertop
from ml_switcheroo_compiler.backends.eager.reference_math.math_testing import (
    _np_broadcast_arrays_ as _np_broadcast_arrays_,
)
from ml_switcheroo_compiler.backends.eager.reference_math.math_testing import _np_can_cast_ as _np_can_cast_
from ml_switcheroo_compiler.backends.eager.reference_math.math_testing import _np_histogram2d_ as _np_histogram2d_
from ml_switcheroo_compiler.backends.eager.reference_math.math_testing import _np_histogram_ as _np_histogram_
from ml_switcheroo_compiler.backends.eager.reference_math.math_testing import (
    _np_histogram_bin_edges_ as _np_histogram_bin_edges_,
)
from ml_switcheroo_compiler.backends.eager.reference_math.math_testing import _np_histogramdd_ as _np_histogramdd_
from ml_switcheroo_compiler.backends.eager.reference_math.math_testing import _np_iscomplex_ as _np_iscomplex_
from ml_switcheroo_compiler.backends.eager.reference_math.math_testing import _np_iscomplexobj_ as _np_iscomplexobj_
from ml_switcheroo_compiler.backends.eager.reference_math.math_testing import _np_isreal_ as _np_isreal_
from ml_switcheroo_compiler.backends.eager.reference_math.math_testing import _np_isrealobj_ as _np_isrealobj_
from ml_switcheroo_compiler.backends.eager.reference_math.math_testing import _np_isscalar_ as _np_isscalar_
from ml_switcheroo_compiler.backends.eager.reference_math.math_testing import (
    _np_issubdtype_issubdtype_ as _np_issubdtype_issubdtype_,
)
from ml_switcheroo_compiler.backends.eager.reference_math.math_testing import _np_issubdtype_op_ as _np_issubdtype_op_
from ml_switcheroo_compiler.backends.eager.reference_math.math_testing import _np_piecewise as _np_piecewise
from ml_switcheroo_compiler.backends.eager.reference_math.math_testing import _np_promotetypes as _np_promotetypes
from ml_switcheroo_compiler.backends.eager.reference_math.math_testing import _np_result_type_ as _np_result_type_
from ml_switcheroo_compiler.backends.eager.reference_math.math_tri import _np_asstringconfig as _np_asstringconfig
from ml_switcheroo_compiler.backends.eager.reference_math.math_tri import _np_confusion_matrix as _np_confusion_matrix
from ml_switcheroo_compiler.backends.eager.reference_math.math_tri import _np_distributions as _np_distributions
from ml_switcheroo_compiler.backends.eager.reference_math.math_tri import _np_fromstring_ as _np_fromstring_
from ml_switcheroo_compiler.backends.eager.reference_math.math_tri import (
    _np_linalg_matrix_power_ as _np_linalg_matrix_power_,
)
from ml_switcheroo_compiler.backends.eager.reference_math.math_tri import (
    _np_linearoperatorblocklowertriangular as _np_linearoperatorblocklowertriangular,
)
from ml_switcheroo_compiler.backends.eager.reference_math.math_tri import (
    _np_linearoperatorfullmatrix as _np_linearoperatorfullmatrix,
)
from ml_switcheroo_compiler.backends.eager.reference_math.math_tri import (
    _np_linearoperatorlowertriangular as _np_linearoperatorlowertriangular,
)
from ml_switcheroo_compiler.backends.eager.reference_math.math_tri import (
    _np_linearoperatortridiag as _np_linearoperatortridiag,
)
from ml_switcheroo_compiler.backends.eager.reference_math.math_tri import _np_stridedslice as _np_stridedslice
from ml_switcheroo_compiler.backends.eager.reference_math.math_tri import _np_tri as _np_tri
from ml_switcheroo_compiler.backends.eager.reference_math.math_tri import _np_trilindices as _np_trilindices
from ml_switcheroo_compiler.backends.eager.reference_math.math_tri import _np_trilindicesfrom as _np_trilindicesfrom
from ml_switcheroo_compiler.backends.eager.reference_math.math_tri import _np_trimzeros as _np_trimzeros
from ml_switcheroo_compiler.backends.eager.reference_math.math_tri import _np_triuindices as _np_triuindices
from ml_switcheroo_compiler.backends.eager.reference_math.math_tri import _np_triuindicesfrom as _np_triuindicesfrom
from ml_switcheroo_compiler.backends.eager.reference_math.reductions import (
    _np_reduce_precision as _np_reduce_precision,
)
from ml_switcheroo_compiler.backends.eager.reference_math.reductions import (
    _np_sparsereducemax as _np_sparsereducemax,
)
from ml_switcheroo_compiler.backends.eager_registry import (
    global_eager_registry as global_eager_registry,
)
from ml_switcheroo_compiler.backends.eager_registry import (
    numpy_eager_registry as numpy_eager_registry,
)
from ml_switcheroo_compiler.backends.numpy.eager import (
    scipy_bessel_poly_kernels as math_bessel,
)
from ml_switcheroo_compiler.backends.numpy.eager import (
    scipy_bessel_poly_kernels as math_poly,
)
from ml_switcheroo_compiler.backends.numpy.eager.distributed import (
    _np_pmean as _np_pmean,
)
from ml_switcheroo_compiler.backends.numpy.eager.distributed import (
    _np_psum as _np_psum,
)
from ml_switcheroo_compiler.backends.numpy.eager.scipy_bessel_poly_kernels import _np_bessel_i0e as _np_bessel_i0e
from ml_switcheroo_compiler.backends.numpy.eager.scipy_bessel_poly_kernels import _np_bessel_i1e as _np_bessel_i1e
from ml_switcheroo_compiler.backends.numpy.eager.scipy_bessel_poly_kernels import (
    _np_chebyshev_polynomial_t as _np_chebyshev_polynomial_t,
)
from ml_switcheroo_compiler.backends.numpy.eager.scipy_bessel_poly_kernels import (
    _np_chebyshev_polynomial_u as _np_chebyshev_polynomial_u,
)
from ml_switcheroo_compiler.backends.numpy.eager.scipy_bessel_poly_kernels import (
    _np_hermite_polynomial_h as _np_hermite_polynomial_h,
)
from ml_switcheroo_compiler.backends.numpy.eager.scipy_bessel_poly_kernels import (
    _np_hermite_polynomial_he as _np_hermite_polynomial_he,
)
from ml_switcheroo_compiler.backends.numpy.eager.scipy_bessel_poly_kernels import (
    _np_laguerre_polynomial_l as _np_laguerre_polynomial_l,
)
from ml_switcheroo_compiler.backends.numpy.eager.scipy_bessel_poly_kernels import (
    _np_legendre_polynomial_p as _np_legendre_polynomial_p,
)
from ml_switcheroo_compiler.backends.numpy.eager.scipy_bessel_poly_kernels import (
    _np_modified_bessel_i0 as _np_modified_bessel_i0,
)
from ml_switcheroo_compiler.backends.numpy.eager.scipy_bessel_poly_kernels import (
    _np_modified_bessel_i1 as _np_modified_bessel_i1,
)
from ml_switcheroo_compiler.backends.numpy.eager.scipy_bessel_poly_kernels import (
    _np_modified_bessel_k0 as _np_modified_bessel_k0,
)
from ml_switcheroo_compiler.backends.numpy.eager.scipy_bessel_poly_kernels import (
    _np_modified_bessel_k1 as _np_modified_bessel_k1,
)
from ml_switcheroo_compiler.backends.numpy.eager.scipy_bessel_poly_kernels import (
    _np_shifted_chebyshev_polynomial_t as _np_shifted_chebyshev_polynomial_t,
)
from ml_switcheroo_compiler.backends.numpy.eager.scipy_bessel_poly_kernels import (
    _np_shifted_chebyshev_polynomial_u as _np_shifted_chebyshev_polynomial_u,
)
from ml_switcheroo_compiler.backends.numpy.eager.scipy_bessel_poly_kernels import (
    _np_shifted_chebyshev_polynomial_v as _np_shifted_chebyshev_polynomial_v,
)
from ml_switcheroo_compiler.backends.numpy.eager.scipy_bessel_poly_kernels import (
    _np_shifted_chebyshev_polynomial_w as _np_shifted_chebyshev_polynomial_w,
)
from ml_switcheroo_compiler.backends.numpy.eager.scipy_bessel_poly_kernels import _poly_recurrence as _poly_recurrence
from ml_switcheroo_compiler.backends.numpy.eager.scipy_ndimage_kernels import (
    _np_convgeneraldilatedlocal as _np_convgeneraldilatedlocal,
)
from ml_switcheroo_compiler.backends.numpy.eager.scipy_ndimage_kernels import (
    _np_convgeneraldilatedpatches as _np_convgeneraldilatedpatches,
)
from ml_switcheroo_compiler.backends.numpy.eager.scipy_ndimage_kernels import (
    _np_convwithgeneralpadding as _np_convwithgeneralpadding,
)
from ml_switcheroo_compiler.backends.numpy.eager.scipy_ndimage_kernels import _np_rawconv2d as _np_rawconv2d
from ml_switcheroo_compiler.backends.numpy.eager.scipy_special_kernels import _get_sc as _get_sc
from ml_switcheroo_compiler.backends.numpy.eager.scipy_special_kernels import _np_betainc as _np_betainc
from ml_switcheroo_compiler.backends.numpy.eager.scipy_special_kernels import _np_einsum_path_ as _np_einsum_path_

__all__ = [
    "_build_dot_general_einsum_str",
    "_get_csv_data",
    "_get_csv_defaults",
    "_get_np_arg",
    "_get_sc",
    "_np_affineconfig",
    "_np_append",
    "_np_apply_along_axis",
    "_np_apply_over_axes",
    "_np_arange_",
    "_np_argpartition",
    "_np_argwhere",
    "_np_array_equiv_",
    "_np_array_repr_",
    "_np_array_str_",
    "_np_assertop",
    "_np_asstringconfig",
    "_np_atleast_1d",
    "_np_atleast_2d",
    "_np_atleast_3d",
    "_np_average",
    "_np_bessel_i0e",
    "_np_bessel_i1e",
    "_np_betainc",
    "_np_bitcast_convert_type",
    "_np_bitwise_count",
    "_np_blackman_",
    "_np_block",
    "_np_blurconfig",
    "_np_broadcast_arrays_",
    "_np_callable",
    "_np_can_cast_",
    "_np_chebyshev_polynomial_t",
    "_np_chebyshev_polynomial_u",
    "_np_choose",
    "_np_clip",
    "_np_clz",
    "_np_column_stack",
    "_np_compress",
    "_np_confusion_matrix",
    "_np_convgeneraldilatedlocal",
    "_np_convgeneraldilatedpatches",
    "_np_convolve",
    "_np_convwithgeneralpadding",
    "_np_corrcoef",
    "_np_correlate",
    "_np_cov",
    "_np_customlinearsolve",
    "_np_customroot",
    "_np_debuginfs",
    "_np_debugnans",
    "_np_decode_csv",
    "_np_decode_csv_camel",
    "_np_decode_image",
    "_np_decode_image_camel",
    "_np_delete_",
    "_np_descriptive",
    "_np_diag_indices_",
    "_np_diag_indices_from_",
    "_np_diagflat_",
    "_np_diagonal_",
    "_np_diff_",
    "_np_digitize_",
    "_np_distributions",
    "_np_dotgeneral",
    "_np_ediff1d_",
    "_np_einsum_path_",
    "_np_elasticconfig",
    "_np_empty_",
    "_np_empty_like_",
    "_np_expand_dims_",
    "_np_extract_",
    "_np_extractpatchesoptions",
    "_np_fabs_",
    "_np_fftn",
    "_np_fftnd",
    "_np_fftshift",
    "_np_flatnonzero_",
    "_np_flip_op_",
    "_np_flip_reverse_",
    "_np_fliplr_",
    "_np_flipud_",
    "_np_frombuffer",
    "_np_fromdlpack",
    "_np_fromfunction_",
    "_np_fromiter_",
    "_np_frompyfunc_",
    "_np_fromstring_",
    "_np_full_",
    "_np_full_like_",
    "_np_geomspace_",
    "_np_get_printoptions_",
    "_np_hamming_",
    "_np_hanning_",
    "_np_hermite_polynomial_h",
    "_np_hermite_polynomial_he",
    "_np_hfft",
    "_np_histogram2d_",
    "_np_histogram_",
    "_np_histogram_bin_edges_",
    "_np_histogramdd_",
    "_np_ifft",
    "_np_ifft2",
    "_np_ifftn",
    "_np_ifftnd",
    "_np_ifftshift",
    "_np_indices_",
    "_np_insert_",
    "_np_interp_",
    "_np_intersect1d_",
    "_np_irfft2",
    "_np_irfftn",
    "_np_irfftnd",
    "_np_iscomplex_",
    "_np_iscomplexobj_",
    "_np_isin_",
    "_np_isreal_",
    "_np_isrealobj_",
    "_np_isscalar_",
    "_np_issubdtype_issubdtype_",
    "_np_issubdtype_op_",
    "_np_iterable_",
    "_np_ix__",
    "_np_kaiser_",
    "_np_key",
    "_np_kron_",
    "_np_laguerre_polynomial_l",
    "_np_legendre_polynomial_p",
    "_np_lexsort_",
    "_np_linalg_cholesky_",
    "_np_linalg_det_",
    "_np_linalg_inv_",
    "_np_linalg_matrix_power_",
    "_np_linalg_pinv_",
    "_np_linalg_svd_",
    "_np_linearoperator",
    "_np_linearoperatoradjoint",
    "_np_linearoperatorblockdiag",
    "_np_linearoperatorblocklowertriangular",
    "_np_linearoperatorcirculant",
    "_np_linearoperatorcirculant2d",
    "_np_linearoperatorcirculant3d",
    "_np_linearoperatorcomposition",
    "_np_linearoperatordiag",
    "_np_linearoperatorfullmatrix",
    "_np_linearoperatorhouseholder",
    "_np_linearoperatoridentity",
    "_np_linearoperatorinversion",
    "_np_linearoperatorkronecker",
    "_np_linearoperatorlowertriangular",
    "_np_linearoperatorlowrankupdate",
    "_np_linearoperatorpermutation",
    "_np_linearoperatorscaledidentity",
    "_np_linearoperatortoeplitz",
    "_np_linearoperatortridiag",
    "_np_linearoperatorzeros",
    "_np_load_",
    "_np_log1p2",
    "_np_log_softmax",
    "_np_logsumexp",
    "_np_mask_indices_",
    "_np_median_",
    "_np_modf_",
    "_np_modified_bessel_i0",
    "_np_modified_bessel_i1",
    "_np_modified_bessel_k0",
    "_np_modified_bessel_k1",
    "_np_mvlgamma",
    "_np_nonzero_",
    "_np_one_hot",
    "_np_ones_",
    "_np_ones_like_",
    "_np_packbits",
    "_np_parse_tensor",
    "_np_parse_tensor_camel",
    "_np_perspectiveconfig",
    "_np_piecewise",
    "_np_population_count",
    "_np_promotetypes",
    "_np_raggeddot",
    "_np_randombernoulli",
    "_np_randomcategorical",
    "_np_randompermutation",
    "_np_randomtruncatednormal",
    "_np_ravel_multi_index_",
    "_np_rawconv2d",
    "_np_rawmatmul",
    "_np_rawmerge",
    "_np_rawop",
    "_np_rawswitch",
    "_np_read_file",
    "_np_read_file_camel",
    "_np_rem",
    "_np_resize_",
    "_np_result_type_",
    "_np_rfft",
    "_np_rfft2",
    "_np_rfftfreq",
    "_np_rfftn",
    "_np_rfftnd",
    "_np_rrelu",
    "_np_rsqrt",
    "_np_scanop",
    "_np_segment_sum",
    "_np_serialize_tensor",
    "_np_serialize_tensor_camel",
    "_np_shifted_chebyshev_polynomial_t",
    "_np_shifted_chebyshev_polynomial_u",
    "_np_shifted_chebyshev_polynomial_v",
    "_np_shifted_chebyshev_polynomial_w",
    "_np_sigmoid",
    "_np_sobolsample",
    "_np_softmax",
    "_np_sparsedensematmul",
    "_np_sparsemapvalues",
    "_np_sparsereshape",
    "_np_sparsesampledadd",
    "_np_sparsesegmentsum",
    "_np_sparsetranspose",
    "_np_stridedslice",
    "_np_switchop",
    "_np_tensor",
    "_np_tensorarrayread",
    "_np_tensorarraystack",
    "_np_tensorarraywrite",
    "_np_tensorconfig",
    "_np_trace",
    "_np_trapz",
    "_np_trapz_",
    "_np_tri",
    "_np_trilindices",
    "_np_trilindicesfrom",
    "_np_trimzeros",
    "_np_triuindices",
    "_np_triuindicesfrom",
    "_np_truncate_div",
    "_np_truncate_mod",
    "_np_uint",
    "_np_uint8",
    "_np_union1d",
    "_np_unpackbits",
    "_np_unravelindex",
    "_np_unwrap",
    "_np_vander",
    "_np_vecdot",
    "_np_vectorize",
    "_np_write_file",
    "_np_write_file_camel",
    "_np_xlogy",
    "_np_zeros_",
    "_np_zeros_like_",
    "_parse_csv_row",
    "_parse_scanop_args",
    "_poly_recurrence",
    "global_eager_registry",
    "math_bessel",
    "math_bit",
    "math_creation",
    "math_gamma",
    "math_general",
    "math_internal",
    "math_linalg",
    "math_log_exp",
    "math_manipulation",
    "math_matrix_utils",
    "math_poly",
    "math_random_sampling",
    "math_segment",
    "math_set",
    "math_signal",
    "math_sorting",
    "math_string_io",
    "math_testing",
    "math_tri",
    "numpy_eager_registry",
]
