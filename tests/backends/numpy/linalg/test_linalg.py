from ml_switcheroo_compiler.backends.numpy.eager.linalg_advanced import _build_einsum_equation


def test_build_einsum_equation_loop() -> None:
    # Need a batch dimension to hit loop inside _build_einsum_equation
    a_ndim = 3
    b_ndim = 3
    # batch = (0, 0), contracting = (2, 1)
    dimension_numbers = (((2,), (1,)), ((0,), (0,)))

    a_dims, b_dims, out_dims = _build_einsum_equation(a_ndim, b_ndim, dimension_numbers)
    # just check execution succeeds
    assert a_dims
