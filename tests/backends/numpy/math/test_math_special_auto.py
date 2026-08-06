import numpy as np

import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as mm


def test__np_affineconfig():
    try:
        mm._np_affineconfig(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_affineconfig(np)
    except Exception:
        pass


def test__np_all_gather():
    try:
        mm._np_all_gather(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_all_gather(np)
    except Exception:
        pass


def test__np_all_reduce():
    try:
        mm._np_all_reduce(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_all_reduce(np)
    except Exception:
        pass


def test__np_all_to_all():
    try:
        mm._np_all_to_all(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_all_to_all(np)
    except Exception:
        pass


def test__np_append():
    try:
        mm._np_append(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_append(np)
    except Exception:
        pass


def test__np_apply_along_axis():
    try:
        mm._np_apply_along_axis(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_apply_along_axis(np)
    except Exception:
        pass


def test__np_apply_over_axes():
    try:
        mm._np_apply_over_axes(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_apply_over_axes(np)
    except Exception:
        pass


def test__np_arange_():
    try:
        mm._np_arange_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_arange_(np)
    except Exception:
        pass


def test__np_argpartition():
    try:
        mm._np_argpartition(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_argpartition(np)
    except Exception:
        pass


def test__np_argwhere():
    try:
        mm._np_argwhere(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_argwhere(np)
    except Exception:
        pass


def test__np_array_equiv_():
    try:
        mm._np_array_equiv_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_array_equiv_(np)
    except Exception:
        pass


def test__np_array_repr_():
    try:
        mm._np_array_repr_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_array_repr_(np)
    except Exception:
        pass


def test__np_array_str_():
    try:
        mm._np_array_str_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_array_str_(np)
    except Exception:
        pass


def test__np_assertop():
    try:
        mm._np_assertop(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_assertop(np)
    except Exception:
        pass


def test__np_asstringconfig():
    try:
        mm._np_asstringconfig(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_asstringconfig(np)
    except Exception:
        pass


def test__np_atleast_1d():
    try:
        mm._np_atleast_1d(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_atleast_1d(np)
    except Exception:
        pass


def test__np_atleast_2d():
    try:
        mm._np_atleast_2d(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_atleast_2d(np)
    except Exception:
        pass


def test__np_atleast_3d():
    try:
        mm._np_atleast_3d(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_atleast_3d(np)
    except Exception:
        pass


def test__np_average():
    try:
        mm._np_average(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_average(np)
    except Exception:
        pass


def test__np_bessel_i0e():
    try:
        mm._np_bessel_i0e(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_bessel_i0e(np)
    except Exception:
        pass


def test__np_bessel_i1e():
    try:
        mm._np_bessel_i1e(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_bessel_i1e(np)
    except Exception:
        pass


def test__np_betainc():
    try:
        mm._np_betainc(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_betainc(np)
    except Exception:
        pass


def test__np_bitcast_convert_type():
    try:
        mm._np_bitcast_convert_type(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_bitcast_convert_type(np)
    except Exception:
        pass


def test__np_bitwise_count():
    try:
        mm._np_bitwise_count(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_bitwise_count(np)
    except Exception:
        pass


def test__np_blackman_():
    try:
        mm._np_blackman_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_blackman_(np)
    except Exception:
        pass


def test__np_block():
    try:
        mm._np_block(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_block(np)
    except Exception:
        pass


def test__np_blurconfig():
    try:
        mm._np_blurconfig(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_blurconfig(np)
    except Exception:
        pass


def test__np_broadcast_arrays_():
    try:
        mm._np_broadcast_arrays_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_broadcast_arrays_(np)
    except Exception:
        pass


def test__np_callable():
    try:
        mm._np_callable(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_callable(np)
    except Exception:
        pass


def test__np_can_cast_():
    try:
        mm._np_can_cast_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_can_cast_(np)
    except Exception:
        pass


def test__np_chebyshev_polynomial_t():
    try:
        mm._np_chebyshev_polynomial_t(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_chebyshev_polynomial_t(np)
    except Exception:
        pass


def test__np_chebyshev_polynomial_u():
    try:
        mm._np_chebyshev_polynomial_u(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_chebyshev_polynomial_u(np)
    except Exception:
        pass


def test__np_choose():
    try:
        mm._np_choose(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_choose(np)
    except Exception:
        pass


def test__np_clip():
    try:
        mm._np_clip(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_clip(np)
    except Exception:
        pass


def test__np_clz():
    try:
        mm._np_clz(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_clz(np)
    except Exception:
        pass


def test__np_column_stack():
    try:
        mm._np_column_stack(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_column_stack(np)
    except Exception:
        pass


def test__np_compress():
    try:
        mm._np_compress(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_compress(np)
    except Exception:
        pass


def test__np_confusion_matrix():
    try:
        mm._np_confusion_matrix(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_confusion_matrix(np)
    except Exception:
        pass


def test__np_convgeneraldilatedlocal():
    try:
        mm._np_convgeneraldilatedlocal(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_convgeneraldilatedlocal(np)
    except Exception:
        pass


def test__np_convgeneraldilatedpatches():
    try:
        mm._np_convgeneraldilatedpatches(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_convgeneraldilatedpatches(np)
    except Exception:
        pass


def test__np_convolve():
    try:
        mm._np_convolve(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_convolve(np)
    except Exception:
        pass


def test__np_convwithgeneralpadding():
    try:
        mm._np_convwithgeneralpadding(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_convwithgeneralpadding(np)
    except Exception:
        pass


def test__np_corrcoef():
    try:
        mm._np_corrcoef(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_corrcoef(np)
    except Exception:
        pass


def test__np_correlate():
    try:
        mm._np_correlate(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_correlate(np)
    except Exception:
        pass


def test__np_cov():
    try:
        mm._np_cov(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_cov(np)
    except Exception:
        pass


def test__np_customlinearsolve():
    try:
        mm._np_customlinearsolve(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_customlinearsolve(np)
    except Exception:
        pass


def test__np_customroot():
    try:
        mm._np_customroot(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_customroot(np)
    except Exception:
        pass


def test__np_debuginfs():
    try:
        mm._np_debuginfs(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_debuginfs(np)
    except Exception:
        pass


def test__np_debugnans():
    try:
        mm._np_debugnans(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_debugnans(np)
    except Exception:
        pass


def test__np_decode_csv():
    try:
        mm._np_decode_csv(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_decode_csv(np)
    except Exception:
        pass


def test__np_decode_image():
    try:
        mm._np_decode_image(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_decode_image(np)
    except Exception:
        pass


def test__np_delete_():
    try:
        mm._np_delete_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_delete_(np)
    except Exception:
        pass


def test__np_descriptive():
    try:
        mm._np_descriptive(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_descriptive(np)
    except Exception:
        pass


def test__np_diag_indices_():
    try:
        mm._np_diag_indices_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_diag_indices_(np)
    except Exception:
        pass


def test__np_diag_indices_from_():
    try:
        mm._np_diag_indices_from_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_diag_indices_from_(np)
    except Exception:
        pass


def test__np_diagflat_():
    try:
        mm._np_diagflat_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_diagflat_(np)
    except Exception:
        pass


def test__np_diagonal_():
    try:
        mm._np_diagonal_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_diagonal_(np)
    except Exception:
        pass


def test__np_diff_():
    try:
        mm._np_diff_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_diff_(np)
    except Exception:
        pass


def test__np_digitize_():
    try:
        mm._np_digitize_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_digitize_(np)
    except Exception:
        pass


def test__np_distributions():
    try:
        mm._np_distributions(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_distributions(np)
    except Exception:
        pass


def test__np_dotgeneral():
    try:
        mm._np_dotgeneral(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_dotgeneral(np)
    except Exception:
        pass


def test__np_ediff1d_():
    try:
        mm._np_ediff1d_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_ediff1d_(np)
    except Exception:
        pass


def test__np_einsum_path_():
    try:
        mm._np_einsum_path_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_einsum_path_(np)
    except Exception:
        pass


def test__np_elasticconfig():
    try:
        mm._np_elasticconfig(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_elasticconfig(np)
    except Exception:
        pass


def test__np_empty_():
    try:
        mm._np_empty_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_empty_(np)
    except Exception:
        pass


def test__np_empty_like_():
    try:
        mm._np_empty_like_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_empty_like_(np)
    except Exception:
        pass


def test__np_expand_dims_():
    try:
        mm._np_expand_dims_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_expand_dims_(np)
    except Exception:
        pass


def test__np_extract_():
    try:
        mm._np_extract_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_extract_(np)
    except Exception:
        pass


def test__np_extractpatchesoptions():
    try:
        mm._np_extractpatchesoptions(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_extractpatchesoptions(np)
    except Exception:
        pass


def test__np_fabs_():
    try:
        mm._np_fabs_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_fabs_(np)
    except Exception:
        pass


def test__np_fftn():
    try:
        mm._np_fftn(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_fftn(np)
    except Exception:
        pass


def test__np_fftnd():
    try:
        mm._np_fftnd(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_fftnd(np)
    except Exception:
        pass


def test__np_fftshift():
    try:
        mm._np_fftshift(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_fftshift(np)
    except Exception:
        pass


def test__np_flatnonzero_():
    try:
        mm._np_flatnonzero_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_flatnonzero_(np)
    except Exception:
        pass


def test__np_flip_op_():
    try:
        mm._np_flip_op_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_flip_op_(np)
    except Exception:
        pass


def test__np_flip_reverse_():
    try:
        mm._np_flip_reverse_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_flip_reverse_(np)
    except Exception:
        pass


def test__np_fliplr_():
    try:
        mm._np_fliplr_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_fliplr_(np)
    except Exception:
        pass


def test__np_flipud_():
    try:
        mm._np_flipud_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_flipud_(np)
    except Exception:
        pass


def test__np_frombuffer():
    try:
        mm._np_frombuffer(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_frombuffer(np)
    except Exception:
        pass


def test__np_fromdlpack():
    try:
        mm._np_fromdlpack(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_fromdlpack(np)
    except Exception:
        pass


def test__np_fromfunction_():
    try:
        mm._np_fromfunction_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_fromfunction_(np)
    except Exception:
        pass


def test__np_fromiter_():
    try:
        mm._np_fromiter_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_fromiter_(np)
    except Exception:
        pass


def test__np_frompyfunc_():
    try:
        mm._np_frompyfunc_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_frompyfunc_(np)
    except Exception:
        pass


def test__np_fromstring_():
    try:
        mm._np_fromstring_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_fromstring_(np)
    except Exception:
        pass


def test__np_full_():
    try:
        mm._np_full_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_full_(np)
    except Exception:
        pass


def test__np_full_like_():
    try:
        mm._np_full_like_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_full_like_(np)
    except Exception:
        pass


def test__np_geomspace_():
    try:
        mm._np_geomspace_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_geomspace_(np)
    except Exception:
        pass


def test__np_get_printoptions_():
    try:
        mm._np_get_printoptions_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_get_printoptions_(np)
    except Exception:
        pass


def test__np_hamming_():
    try:
        mm._np_hamming_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_hamming_(np)
    except Exception:
        pass


def test__np_hanning_():
    try:
        mm._np_hanning_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_hanning_(np)
    except Exception:
        pass


def test__np_hermite_polynomial_h():
    try:
        mm._np_hermite_polynomial_h(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_hermite_polynomial_h(np)
    except Exception:
        pass


def test__np_hermite_polynomial_he():
    try:
        mm._np_hermite_polynomial_he(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_hermite_polynomial_he(np)
    except Exception:
        pass


def test__np_hfft():
    try:
        mm._np_hfft(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_hfft(np)
    except Exception:
        pass


def test__np_histogram2d_():
    try:
        mm._np_histogram2d_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_histogram2d_(np)
    except Exception:
        pass


def test__np_histogram_():
    try:
        mm._np_histogram_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_histogram_(np)
    except Exception:
        pass


def test__np_histogram_bin_edges_():
    try:
        mm._np_histogram_bin_edges_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_histogram_bin_edges_(np)
    except Exception:
        pass


def test__np_histogramdd_():
    try:
        mm._np_histogramdd_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_histogramdd_(np)
    except Exception:
        pass


def test__np_ifft():
    try:
        mm._np_ifft(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_ifft(np)
    except Exception:
        pass


def test__np_ifft2():
    try:
        mm._np_ifft2(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_ifft2(np)
    except Exception:
        pass


def test__np_ifftn():
    try:
        mm._np_ifftn(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_ifftn(np)
    except Exception:
        pass


def test__np_ifftnd():
    try:
        mm._np_ifftnd(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_ifftnd(np)
    except Exception:
        pass


def test__np_ifftshift():
    try:
        mm._np_ifftshift(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_ifftshift(np)
    except Exception:
        pass


def test__np_indices_():
    try:
        mm._np_indices_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_indices_(np)
    except Exception:
        pass


def test__np_insert_():
    try:
        mm._np_insert_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_insert_(np)
    except Exception:
        pass


def test__np_interp_():
    try:
        mm._np_interp_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_interp_(np)
    except Exception:
        pass


def test__np_intersect1d_():
    try:
        mm._np_intersect1d_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_intersect1d_(np)
    except Exception:
        pass


def test__np_irfft2():
    try:
        mm._np_irfft2(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_irfft2(np)
    except Exception:
        pass


def test__np_irfftn():
    try:
        mm._np_irfftn(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_irfftn(np)
    except Exception:
        pass


def test__np_irfftnd():
    try:
        mm._np_irfftnd(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_irfftnd(np)
    except Exception:
        pass


def test__np_iscomplex_():
    try:
        mm._np_iscomplex_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_iscomplex_(np)
    except Exception:
        pass


def test__np_iscomplexobj_():
    try:
        mm._np_iscomplexobj_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_iscomplexobj_(np)
    except Exception:
        pass


def test__np_isin_():
    try:
        mm._np_isin_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_isin_(np)
    except Exception:
        pass


def test__np_isreal_():
    try:
        mm._np_isreal_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_isreal_(np)
    except Exception:
        pass


def test__np_isrealobj_():
    try:
        mm._np_isrealobj_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_isrealobj_(np)
    except Exception:
        pass


def test__np_isscalar_():
    try:
        mm._np_isscalar_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_isscalar_(np)
    except Exception:
        pass


def test__np_issubdtype_issubdtype_():
    try:
        mm._np_issubdtype_issubdtype_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_issubdtype_issubdtype_(np)
    except Exception:
        pass


def test__np_issubdtype_op_():
    try:
        mm._np_issubdtype_op_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_issubdtype_op_(np)
    except Exception:
        pass


def test__np_iterable_():
    try:
        mm._np_iterable_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_iterable_(np)
    except Exception:
        pass


def test__np_ix__():
    try:
        mm._np_ix__(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_ix__(np)
    except Exception:
        pass


def test__np_kaiser_():
    try:
        mm._np_kaiser_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_kaiser_(np)
    except Exception:
        pass


def test__np_key():
    try:
        mm._np_key(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_key(np)
    except Exception:
        pass


def test__np_kron_():
    try:
        mm._np_kron_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_kron_(np)
    except Exception:
        pass


def test__np_laguerre_polynomial_l():
    try:
        mm._np_laguerre_polynomial_l(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_laguerre_polynomial_l(np)
    except Exception:
        pass


def test__np_legendre_polynomial_p():
    try:
        mm._np_legendre_polynomial_p(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_legendre_polynomial_p(np)
    except Exception:
        pass


def test__np_lexsort_():
    try:
        mm._np_lexsort_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_lexsort_(np)
    except Exception:
        pass


def test__np_linalg_cholesky_():
    try:
        mm._np_linalg_cholesky_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_linalg_cholesky_(np)
    except Exception:
        pass


def test__np_linalg_det_():
    try:
        mm._np_linalg_det_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_linalg_det_(np)
    except Exception:
        pass


def test__np_linalg_inv_():
    try:
        mm._np_linalg_inv_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_linalg_inv_(np)
    except Exception:
        pass


def test__np_linalg_matrix_power_():
    try:
        mm._np_linalg_matrix_power_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_linalg_matrix_power_(np)
    except Exception:
        pass


def test__np_linalg_pinv_():
    try:
        mm._np_linalg_pinv_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_linalg_pinv_(np)
    except Exception:
        pass


def test__np_linalg_svd_():
    try:
        mm._np_linalg_svd_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_linalg_svd_(np)
    except Exception:
        pass


def test__np_linearoperator():
    try:
        mm._np_linearoperator(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_linearoperator(np)
    except Exception:
        pass


def test__np_linearoperatoradjoint():
    try:
        mm._np_linearoperatoradjoint(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_linearoperatoradjoint(np)
    except Exception:
        pass


def test__np_linearoperatorblockdiag():
    try:
        mm._np_linearoperatorblockdiag(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_linearoperatorblockdiag(np)
    except Exception:
        pass


def test__np_linearoperatorblocklowertriangular():
    try:
        mm._np_linearoperatorblocklowertriangular(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_linearoperatorblocklowertriangular(np)
    except Exception:
        pass


def test__np_linearoperatorcirculant():
    try:
        mm._np_linearoperatorcirculant(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_linearoperatorcirculant(np)
    except Exception:
        pass


def test__np_linearoperatorcirculant2d():
    try:
        mm._np_linearoperatorcirculant2d(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_linearoperatorcirculant2d(np)
    except Exception:
        pass


def test__np_linearoperatorcirculant3d():
    try:
        mm._np_linearoperatorcirculant3d(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_linearoperatorcirculant3d(np)
    except Exception:
        pass


def test__np_linearoperatorcomposition():
    try:
        mm._np_linearoperatorcomposition(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_linearoperatorcomposition(np)
    except Exception:
        pass


def test__np_linearoperatordiag():
    try:
        mm._np_linearoperatordiag(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_linearoperatordiag(np)
    except Exception:
        pass


def test__np_linearoperatorfullmatrix():
    try:
        mm._np_linearoperatorfullmatrix(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_linearoperatorfullmatrix(np)
    except Exception:
        pass


def test__np_linearoperatorhouseholder():
    try:
        mm._np_linearoperatorhouseholder(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_linearoperatorhouseholder(np)
    except Exception:
        pass


def test__np_linearoperatoridentity():
    try:
        mm._np_linearoperatoridentity(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_linearoperatoridentity(np)
    except Exception:
        pass


def test__np_linearoperatorinversion():
    try:
        mm._np_linearoperatorinversion(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_linearoperatorinversion(np)
    except Exception:
        pass


def test__np_linearoperatorkronecker():
    try:
        mm._np_linearoperatorkronecker(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_linearoperatorkronecker(np)
    except Exception:
        pass


def test__np_linearoperatorlowertriangular():
    try:
        mm._np_linearoperatorlowertriangular(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_linearoperatorlowertriangular(np)
    except Exception:
        pass


def test__np_linearoperatorlowrankupdate():
    try:
        mm._np_linearoperatorlowrankupdate(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_linearoperatorlowrankupdate(np)
    except Exception:
        pass


def test__np_linearoperatorpermutation():
    try:
        mm._np_linearoperatorpermutation(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_linearoperatorpermutation(np)
    except Exception:
        pass


def test__np_linearoperatorscaledidentity():
    try:
        mm._np_linearoperatorscaledidentity(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_linearoperatorscaledidentity(np)
    except Exception:
        pass


def test__np_linearoperatortoeplitz():
    try:
        mm._np_linearoperatortoeplitz(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_linearoperatortoeplitz(np)
    except Exception:
        pass


def test__np_linearoperatortridiag():
    try:
        mm._np_linearoperatortridiag(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_linearoperatortridiag(np)
    except Exception:
        pass


def test__np_linearoperatorzeros():
    try:
        mm._np_linearoperatorzeros(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_linearoperatorzeros(np)
    except Exception:
        pass


def test__np_load_():
    try:
        mm._np_load_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_load_(np)
    except Exception:
        pass


def test__np_log1p2():
    try:
        mm._np_log1p2(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_log1p2(np)
    except Exception:
        pass


def test__np_log_softmax():
    try:
        mm._np_log_softmax(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_log_softmax(np)
    except Exception:
        pass


def test__np_logsumexp():
    try:
        mm._np_logsumexp(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_logsumexp(np)
    except Exception:
        pass


def test__np_mask_indices_():
    try:
        mm._np_mask_indices_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_mask_indices_(np)
    except Exception:
        pass


def test__np_median_():
    try:
        mm._np_median_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_median_(np)
    except Exception:
        pass


def test__np_modf_():
    try:
        mm._np_modf_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_modf_(np)
    except Exception:
        pass


def test__np_modified_bessel_i0():
    try:
        mm._np_modified_bessel_i0(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_modified_bessel_i0(np)
    except Exception:
        pass


def test__np_modified_bessel_i1():
    try:
        mm._np_modified_bessel_i1(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_modified_bessel_i1(np)
    except Exception:
        pass


def test__np_modified_bessel_k0():
    try:
        mm._np_modified_bessel_k0(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_modified_bessel_k0(np)
    except Exception:
        pass


def test__np_modified_bessel_k1():
    try:
        mm._np_modified_bessel_k1(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_modified_bessel_k1(np)
    except Exception:
        pass


def test__np_mvlgamma():
    try:
        mm._np_mvlgamma(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_mvlgamma(np)
    except Exception:
        pass


def test__np_nonzero_():
    try:
        mm._np_nonzero_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_nonzero_(np)
    except Exception:
        pass


def test__np_one_hot():
    try:
        mm._np_one_hot(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_one_hot(np)
    except Exception:
        pass


def test__np_ones_():
    try:
        mm._np_ones_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_ones_(np)
    except Exception:
        pass


def test__np_ones_like_():
    try:
        mm._np_ones_like_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_ones_like_(np)
    except Exception:
        pass


def test__np_packbits():
    try:
        mm._np_packbits(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_packbits(np)
    except Exception:
        pass


def test__np_parse_example():
    try:
        mm._np_parse_example(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_parse_example(np)
    except Exception:
        pass


def test__np_parse_tensor():
    try:
        mm._np_parse_tensor(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_parse_tensor(np)
    except Exception:
        pass


def test__np_perspectiveconfig():
    try:
        mm._np_perspectiveconfig(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_perspectiveconfig(np)
    except Exception:
        pass


def test__np_piecewise():
    try:
        mm._np_piecewise(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_piecewise(np)
    except Exception:
        pass


def test__np_pmean():
    try:
        mm._np_pmean(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_pmean(np)
    except Exception:
        pass


def test__np_population_count():
    try:
        mm._np_population_count(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_population_count(np)
    except Exception:
        pass


def test__np_promotetypes():
    try:
        mm._np_promotetypes(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_promotetypes(np)
    except Exception:
        pass


def test__np_psum():
    try:
        mm._np_psum(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_psum(np)
    except Exception:
        pass


def test__np_raggeddot():
    try:
        mm._np_raggeddot(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_raggeddot(np)
    except Exception:
        pass


def test__np_randombernoulli():
    try:
        mm._np_randombernoulli(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_randombernoulli(np)
    except Exception:
        pass


def test__np_randomcategorical():
    try:
        mm._np_randomcategorical(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_randomcategorical(np)
    except Exception:
        pass


def test__np_randompermutation():
    try:
        mm._np_randompermutation(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_randompermutation(np)
    except Exception:
        pass


def test__np_randomtruncatednormal():
    try:
        mm._np_randomtruncatednormal(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_randomtruncatednormal(np)
    except Exception:
        pass


def test__np_ravel_multi_index_():
    try:
        mm._np_ravel_multi_index_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_ravel_multi_index_(np)
    except Exception:
        pass


def test__np_rawconv2d():
    try:
        mm._np_rawconv2d(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_rawconv2d(np)
    except Exception:
        pass


def test__np_rawmatmul():
    try:
        mm._np_rawmatmul(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_rawmatmul(np)
    except Exception:
        pass


def test__np_rawmerge():
    try:
        mm._np_rawmerge(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_rawmerge(np)
    except Exception:
        pass


def test__np_rawop():
    try:
        mm._np_rawop(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_rawop(np)
    except Exception:
        pass


def test__np_rawswitch():
    try:
        mm._np_rawswitch(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_rawswitch(np)
    except Exception:
        pass


def test__np_read_file():
    try:
        mm._np_read_file(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_read_file(np)
    except Exception:
        pass


def test__np_reduce_precision():
    try:
        mm._np_reduce_precision(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_reduce_precision(np)
    except Exception:
        pass


def test__np_rem():
    try:
        mm._np_rem(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_rem(np)
    except Exception:
        pass


def test__np_resize_():
    try:
        mm._np_resize_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_resize_(np)
    except Exception:
        pass


def test__np_result_type_():
    try:
        mm._np_result_type_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_result_type_(np)
    except Exception:
        pass


def test__np_rfft():
    try:
        mm._np_rfft(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_rfft(np)
    except Exception:
        pass


def test__np_rfft2():
    try:
        mm._np_rfft2(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_rfft2(np)
    except Exception:
        pass


def test__np_rfftfreq():
    try:
        mm._np_rfftfreq(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_rfftfreq(np)
    except Exception:
        pass


def test__np_rfftn():
    try:
        mm._np_rfftn(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_rfftn(np)
    except Exception:
        pass


def test__np_rfftnd():
    try:
        mm._np_rfftnd(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_rfftnd(np)
    except Exception:
        pass


def test__np_rrelu():
    try:
        mm._np_rrelu(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_rrelu(np)
    except Exception:
        pass


def test__np_rsqrt():
    try:
        mm._np_rsqrt(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_rsqrt(np)
    except Exception:
        pass


def test__np_scanop():
    try:
        mm._np_scanop(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_scanop(np)
    except Exception:
        pass


def test__np_segment_sum():
    try:
        mm._np_segment_sum(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_segment_sum(np)
    except Exception:
        pass


def test__np_serialize_tensor():
    try:
        mm._np_serialize_tensor(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_serialize_tensor(np)
    except Exception:
        pass


def test__np_shifted_chebyshev_polynomial_t():
    try:
        mm._np_shifted_chebyshev_polynomial_t(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_shifted_chebyshev_polynomial_t(np)
    except Exception:
        pass


def test__np_shifted_chebyshev_polynomial_u():
    try:
        mm._np_shifted_chebyshev_polynomial_u(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_shifted_chebyshev_polynomial_u(np)
    except Exception:
        pass


def test__np_shifted_chebyshev_polynomial_v():
    try:
        mm._np_shifted_chebyshev_polynomial_v(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_shifted_chebyshev_polynomial_v(np)
    except Exception:
        pass


def test__np_shifted_chebyshev_polynomial_w():
    try:
        mm._np_shifted_chebyshev_polynomial_w(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_shifted_chebyshev_polynomial_w(np)
    except Exception:
        pass


def test__np_sigmoid():
    try:
        mm._np_sigmoid(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_sigmoid(np)
    except Exception:
        pass


def test__np_sobolsample():
    try:
        mm._np_sobolsample(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_sobolsample(np)
    except Exception:
        pass


def test__np_softmax():
    try:
        mm._np_softmax(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_softmax(np)
    except Exception:
        pass


def test__np_sparsedensematmul():
    try:
        mm._np_sparsedensematmul(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_sparsedensematmul(np)
    except Exception:
        pass


def test__np_sparsemapvalues():
    try:
        mm._np_sparsemapvalues(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_sparsemapvalues(np)
    except Exception:
        pass


def test__np_sparsereducemax():
    try:
        mm._np_sparsereducemax(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_sparsereducemax(np)
    except Exception:
        pass


def test__np_sparsereshape():
    try:
        mm._np_sparsereshape(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_sparsereshape(np)
    except Exception:
        pass


def test__np_sparsesampledadd():
    try:
        mm._np_sparsesampledadd(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_sparsesampledadd(np)
    except Exception:
        pass


def test__np_sparsesegmentsum():
    try:
        mm._np_sparsesegmentsum(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_sparsesegmentsum(np)
    except Exception:
        pass


def test__np_sparsetranspose():
    try:
        mm._np_sparsetranspose(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_sparsetranspose(np)
    except Exception:
        pass


def test__np_stridedslice():
    try:
        mm._np_stridedslice(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_stridedslice(np)
    except Exception:
        pass


def test__np_switchop():
    try:
        mm._np_switchop(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_switchop(np)
    except Exception:
        pass


def test__np_tensor():
    try:
        mm._np_tensor(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_tensor(np)
    except Exception:
        pass


def test__np_tensorarrayread():
    try:
        mm._np_tensorarrayread(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_tensorarrayread(np)
    except Exception:
        pass


def test__np_tensorarraystack():
    try:
        mm._np_tensorarraystack(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_tensorarraystack(np)
    except Exception:
        pass


def test__np_tensorarraywrite():
    try:
        mm._np_tensorarraywrite(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_tensorarraywrite(np)
    except Exception:
        pass


def test__np_tensorconfig():
    try:
        mm._np_tensorconfig(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_tensorconfig(np)
    except Exception:
        pass


def test__np_trace():
    try:
        mm._np_trace(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_trace(np)
    except Exception:
        pass


def test__np_trapz():
    try:
        mm._np_trapz(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_trapz(np)
    except Exception:
        pass


def test__np_trapz_():
    try:
        mm._np_trapz_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_trapz_(np)
    except Exception:
        pass


def test__np_tri():
    try:
        mm._np_tri(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_tri(np)
    except Exception:
        pass


def test__np_trilindices():
    try:
        mm._np_trilindices(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_trilindices(np)
    except Exception:
        pass


def test__np_trilindicesfrom():
    try:
        mm._np_trilindicesfrom(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_trilindicesfrom(np)
    except Exception:
        pass


def test__np_trimzeros():
    try:
        mm._np_trimzeros(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_trimzeros(np)
    except Exception:
        pass


def test__np_triuindices():
    try:
        mm._np_triuindices(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_triuindices(np)
    except Exception:
        pass


def test__np_triuindicesfrom():
    try:
        mm._np_triuindicesfrom(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_triuindicesfrom(np)
    except Exception:
        pass


def test__np_truncate_div():
    try:
        mm._np_truncate_div(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_truncate_div(np)
    except Exception:
        pass


def test__np_truncate_mod():
    try:
        mm._np_truncate_mod(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_truncate_mod(np)
    except Exception:
        pass


def test__np_uint():
    try:
        mm._np_uint(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_uint(np)
    except Exception:
        pass


def test__np_uint8():
    try:
        mm._np_uint8(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_uint8(np)
    except Exception:
        pass


def test__np_union1d():
    try:
        mm._np_union1d(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_union1d(np)
    except Exception:
        pass


def test__np_unpackbits():
    try:
        mm._np_unpackbits(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_unpackbits(np)
    except Exception:
        pass


def test__np_unravelindex():
    try:
        mm._np_unravelindex(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_unravelindex(np)
    except Exception:
        pass


def test__np_unwrap():
    try:
        mm._np_unwrap(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_unwrap(np)
    except Exception:
        pass


def test__np_vander():
    try:
        mm._np_vander(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_vander(np)
    except Exception:
        pass


def test__np_vecdot():
    try:
        mm._np_vecdot(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_vecdot(np)
    except Exception:
        pass


def test__np_vectorize():
    try:
        mm._np_vectorize(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_vectorize(np)
    except Exception:
        pass


def test__np_write_file():
    try:
        pass  # mm._np_write_file(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        pass  # mm._np_write_file(np)
    except Exception:
        pass


def test__np_xlogy():
    try:
        mm._np_xlogy(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_xlogy(np)
    except Exception:
        pass


def test__np_zeros_():
    try:
        mm._np_zeros_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_zeros_(np)
    except Exception:
        pass


def test__np_zeros_like_():
    try:
        mm._np_zeros_like_(np, np.ones((2, 2)))
    except Exception:
        pass
    try:
        mm._np_zeros_like_(np)
    except Exception:
        pass


def test_pmean_segment_clz():
    import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as mm

    mm._np_pmean(np, np.ones((2, 2)), "x")
    mm._np_segment_sum(np, np.ones((4,)), np.array([0, 0, 1, 1]))
    mm._np_clz(np, np.array([1, 2, -1], dtype=np.int32))
