"""Linear algebra operations."""

from __future__ import annotations


from typing import TYPE_CHECKING

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.linalg.frontend import _emit_linalg_node


@register_op("Cholesky")
class Cholesky(OpDef):
    """Cholesky Operation Definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            object: The shape.
        """
        return ()


@register_op("Svd")
class Svd(OpDef):
    """Svd Operation Definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            object: The shape.
        """
        return ()


@register_op("Qr")
class Qr(OpDef):
    """Qr Operation Definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            object: The shape.
        """
        return ()


@register_op("Inv")
class Inv(OpDef):
    """Inv Operation Definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            object: The shape.
        """
        return ()


@register_op("Det")
class Det(OpDef):
    """Det Operation Definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            object: The shape.
        """
        return ()


@register_op("Slogdet")
class Slogdet(OpDef):
    """Slogdet Operation Definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            object: The shape.
        """
        return ()


@register_op("Eigh")
class Eigh(OpDef):
    """Eigh Operation Definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            object: The shape.
        """
        return ()


@register_op("Eigvalsh")
class Eigvalsh(OpDef):
    """Eigvalsh Operation Definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            object: The shape.
        """
        return ()


@register_op("Solve")
class Solve(OpDef):
    """Solve Operation Definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            object: The shape.
        """
        return ()


@register_op("TriangularSolve")
class TriangularSolve(OpDef):
    """TriangularSolve Operation Definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            object: The shape.
        """
        return ()


@register_op("Lu")
class Lu(OpDef):
    """Lu Operation Definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            object: The shape.
        """
        return ()


@register_op("LuFactor")
class LuFactor(OpDef):
    """LuFactor Operation Definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            object: The shape.
        """
        return ()


@register_op("LuSolve")
class LuSolve(OpDef):
    """LuSolve Operation Definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            object: The shape.
        """
        return ()


@register_op("Norm")
class Norm(OpDef):
    """Norm Operation Definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            object: The shape.
        """
        return ()


@register_op("MatrixExponential")
class MatrixExponential(OpDef):
    """MatrixExponential Operation Definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            object: The shape.
        """
        return ()


@register_op("Cross")
class Cross(OpDef):
    """Cross Operation Definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            object: The shape.
        """
        return ()


@register_op("PowerIteration")
class PowerIteration(OpDef):
    """Power Iteration Operation Definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape for Power Iteration.

        Args:
            *args (object): The positional arguments.
            **kwargs (object): The keyword arguments.

        Returns:
            object: The tuple containing output shapes and dtypes.
        """
        in_shape = args[0].shape
        v_shape = in_shape[:-2] + (in_shape[-1],)  # pragma: no cover
        u_shape = in_shape[:-2] + (in_shape[-2],)  # pragma: no cover
        sigma_shape = in_shape[:-2]  # pragma: no cover
        return (v_shape, u_shape, sigma_shape), (args[0].dtype,) * 3  # pragma: no cover


if TYPE_CHECKING:
    pass


def cholesky(input: Tensor) -> Tensor:
    """Computes the Cholesky decomposition of a symmetric/Hermitian positive-definite.

    matrix

    Args:
        input (Tensor): The input symmetric/Hermitian positive-definite matrix

    Returns:
    Tensor: The lower-triangular or upper-triangular Cholesky factor
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Cholesky", input.data)
        return Tensor(
            backend.array(data), TensorConfig(backend.array(data).shape, input.dtype, input.device)
        )
    return _emit_linalg_node("Cholesky", [input], {}, [input.shape], [input.dtype])


def svd(
    input: Tensor,
    full_matrices: bool = True,
    compute_uv: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    """Computes the Singular Value Decomposition (SVD) of a matrix.

    Args:
        input (Tensor): The input matrix of shape (..., M, N)
        full_matrices (bool): If True, matrices U and Vh have shapes (..., M, M)
        and (..., N, N). Otherwise, shapes are (..., M, K) and (..., K, N)
        where K = min(M, N). Defaults to True
        compute_uv (bool): Whether to compute U and Vh in addition to S. Defaults to
        True

    Returns:
    tuple[Tensor, Tensor, Tensor]: A tuple containing:
        - U (Tensor): Left singular vectors
        - S (Tensor): Singular values
        - Vh (Tensor): Right singular vectors (conjugate transposed)
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        u, s, vh = backend.execute_op(
            "Svd",
            input.data,
            full_matrices=full_matrices,
            compute_uv=compute_uv,
        )
        return (
            Tensor(u, TensorConfig(u.shape, input.dtype, input.device)),
            Tensor(s, TensorConfig(s.shape, input.dtype, input.device)),
            Tensor(vh, TensorConfig(vh.shape, input.dtype, input.device)),
        )
    return _emit_linalg_node(
        "Svd",
        [input],
        {"full_matrices": full_matrices, "compute_uv": compute_uv},
        [input.shape, input.shape[:-1], input.shape],
        [input.dtype] * 3,
    )


def qr(input: Tensor, mode: str = "reduced") -> tuple[Tensor, Tensor]:
    """Computes the QR decomposition of a matrix.

    Args:
        input (Tensor): The input matrix
        mode (str): Specifies the mode of decomposition ('reduced', 'complete',
        'r', or 'raw'). Defaults to 'reduced'

    Returns:
    tuple[Tensor, Tensor]: A tuple containing:
        - Q (Tensor): The orthonormal matrix
        - R (Tensor): The upper-triangular matrix
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        q, r = backend.execute_op("Qr", input.data, mode=mode)
        return (
            Tensor(q, TensorConfig(q.shape, input.dtype, input.device)),
            Tensor(r, TensorConfig(r.shape, input.dtype, input.device)),
        )
    return _emit_linalg_node(
        "Qr", [input], {"mode": mode}, [input.shape, input.shape], [input.dtype] * 2
    )


def inv(input: Tensor) -> Tensor:
    """Computes the multiplicative inverse of a square matrix.

    Args:
        input (Tensor): The square matrix to invert

    Returns:
    Tensor: The multiplicative inverse of the input matrix
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Inv", input.data)
        return Tensor(
            backend.array(data), TensorConfig(backend.array(data).shape, input.dtype, input.device)
        )
    return _emit_linalg_node("Inv", [input], {}, [input.shape], [input.dtype])


def pinv(input: Tensor, rcond: float = 1e-15) -> Tensor:
    """Computes the Moore-Penrose pseudo-inverse of a matrix.

    Args:
        input (Tensor): The matrix to invert
        rcond (float): Cutoff for small singular values. Defaults to 1e-15

    Returns:
    Tensor: The pseudo-inverse of the input matrix
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Pinv", input.data, rcond=rcond)
        return Tensor(data, TensorConfig(data.shape, input.dtype, input.device))
    return _emit_linalg_node("Pinv", [input], {"rcond": rcond}, [input.shape], [input.dtype])


def det(input: Tensor) -> Tensor:
    """Computes the determinant of a square matrix.

    Args:
        input (Tensor): The square matrix

    Returns:
    Tensor: The determinant of the input matrix
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Det", input.data)
        return Tensor(
            backend.array(data), TensorConfig(backend.array(data).shape, input.dtype, input.device)
        )
    return _emit_linalg_node("Det", [input], {}, [()], [input.dtype])


def slogdet(input: Tensor) -> tuple[Tensor, Tensor]:
    """Computes the sign and natural logarithm of the determinant of a square matrix.

    Args:
        input (Tensor): The square matrix

    Returns:
    tuple[Tensor, Tensor]: A tuple containing:
        - sign (Tensor): A number representing the sign of the determinant
    - logdet (Tensor): The natural logarithm of the absolute value of the
    determinant
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        sign, logdet = backend.execute_op("Slogdet", input.data)
        return (
            Tensor(
                backend.array(sign),
                TensorConfig(backend.array(sign).shape, input.dtype, input.device),
            ),
            Tensor(
                backend.array(logdet),
                TensorConfig(backend.array(logdet).shape, input.dtype, input.device),
            ),
        )
    return _emit_linalg_node("Slogdet", [input], {}, [(), ()], [input.dtype] * 2)


def eigh(input: Tensor, UPLO: str = "L") -> tuple[Tensor, Tensor]:
    """Computes the eigenvalues and eigenvectors of a complex Hermitian or real symmetric.

    matrix

    Args:
        input (Tensor): The symmetric or Hermitian matrix
        UPLO (str): Specifies whether the calculation is done with the lower ('L')
        or upper ('U') triangular part of the matrix. Defaults to 'L'

    Returns:
    tuple[Tensor, Tensor]: A tuple containing:
        - eigenvalues (Tensor): The eigenvalues in ascending order
        - eigenvectors (Tensor): The column eigenvectors
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        w, v = backend.execute_op("Eigh", input.data, UPLO=UPLO)
        return (
            Tensor(w, TensorConfig(w.shape, input.dtype, input.device)),
            Tensor(v, TensorConfig(v.shape, input.dtype, input.device)),
        )
    return _emit_linalg_node(
        "Eigh", [input], {"UPLO": UPLO}, [input.shape[:-1], input.shape], [input.dtype] * 2
    )


def eigvalsh(input: Tensor, UPLO: str = "L") -> Tensor:
    """Computes the eigenvalues of a complex Hermitian or real symmetric matrix.

    Args:
        input (Tensor): The symmetric or Hermitian matrix
        UPLO (str): Specifies whether the calculation is done with the lower ('L')
        or upper ('U') triangular part of the matrix. Defaults to 'L'

    Returns:
    Tensor: The eigenvalues in ascending order
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Eigvalsh", input.data, UPLO=UPLO)
        return Tensor(
            backend.array(data), TensorConfig(backend.array(data).shape, input.dtype, input.device)
        )
    return _emit_linalg_node("Eigvalsh", [input], {"UPLO": UPLO}, [input.shape[:-1]], [input.dtype])


def matrix_power(input: Tensor, n: int) -> Tensor:
    """Raises a square matrix to the integer power `n`.

    Args:
        input (Tensor): The square matrix
        n (int): The exponent

    Returns:
    Tensor: The matrix raised to the power `n`
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("MatrixPower", input.data, n)
        return Tensor(data, TensorConfig(data.shape, input.dtype, input.device))
    return _emit_linalg_node("MatrixPower", [input], {"n": n}, [input.shape], [input.dtype])


def solve(a: Tensor, b: Tensor) -> Tensor:
    """Solves a linear matrix equation, or system of linear scalar equations.

    Args:
        a (Tensor): Coefficient matrix
        b (Tensor): Ordinate or 'dependent variable' values

    Returns:
    Tensor: Solution to the system of linear equations
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op(
            "Solve",
            (a.data if hasattr(a, "device") else a),
            (b.data if hasattr(b, "device") else b),
        )
        return Tensor(data, TensorConfig(data.shape, a.dtype, a.device))
    return _emit_linalg_node("Solve", [a, b], {}, [b.shape], [a.dtype])


def solve_triangular(
    a: Tensor,
    b: Tensor,
    lower: bool = False,
    unit_diagonal: bool = False,
) -> Tensor:
    """Solves the equation `a x = b` for `x`, assuming `a` is a triangular matrix.

    Args:
        a (Tensor): Triangular coefficient matrix
        b (Tensor): Right-hand side matrix or vector
        lower (bool): Use only data contained in the lower triangle of a. Default is to use upper triangle.
        unit_diagonal (bool): If True, diagonal elements of a are assumed to be 1.

    Returns:
    Tensor: The solution matrix `x`
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op(
            "TriangularSolve",
            a.data,
            b.data,
            lower=lower,
            unit_diagonal=unit_diagonal,
        )
        return Tensor(data, TensorConfig(data.shape, a.dtype, a.device))
    return _emit_linalg_node(
        "TriangularSolve",
        [a, b],
        {"lower": lower, "unit_diagonal": unit_diagonal},
        [b.shape],
        [a.dtype],
    )


def lu(a: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Computes the LU decomposition of a matrix.

    Args:
        a (Tensor): The input matrix to decompose

    Returns:
    tuple[Tensor, Tensor, Tensor]: The LU decomposition components (P, L, U)
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        p, l_mat, u = backend.execute_op("Lu", a.data)
        return (
            Tensor(p, TensorConfig(p.shape, a.dtype, a.device)),
            Tensor(l_mat, TensorConfig(l_mat.shape, a.dtype, a.device)),
            Tensor(u, TensorConfig(u.shape, a.dtype, a.device)),
        )
    return _emit_linalg_node("Lu", [a], {}, [a.shape, a.shape, a.shape], [a.dtype] * 3)


def lu_factor(a: Tensor) -> tuple[Tensor, Tensor]:
    """Computes pivoted LU decomposition of a matrix for use in `lu_solve`.

    Args:
        a (Tensor): The input matrix to decompose

    Returns:
    tuple[Tensor, Tensor]: A tuple (lu, piv) containing the LU factorization and pivot indices
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        lu, piv = backend.execute_op("LuFactor", a.data)
        piv_dtype = a.dtype
        return (
            Tensor(lu, TensorConfig(lu.shape, a.dtype, a.device)),
            Tensor(piv, TensorConfig(piv.shape, piv_dtype, a.device)),
        )

    piv_shape = a.shape[:-1]
    return _emit_linalg_node("LuFactor", [a], {}, [a.shape, piv_shape], [a.dtype, a.dtype])


def lu_solve(lu_and_piv: tuple[Tensor, Tensor], b: Tensor) -> Tensor:
    """Solve an equation system, a x = b, given the LU factorization of a.

    Args:
        lu_and_piv (tuple[Tensor, Tensor]): LU factorization and pivot indices from lu_factor.
        b (Tensor): Right-hand side matrix or vector.

    Returns:
    Tensor: The solution matrix `x`
    """
    lu, piv = lu_and_piv
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("LuSolve", lu.data, piv.data, b.data)
        return Tensor(data, TensorConfig(data.shape, b.dtype, b.device))
    return _emit_linalg_node("LuSolve", [lu, piv, b], {}, [b.shape], [b.dtype])


def norm(
    x: Tensor,
    ord: int | str | None = None,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Tensor:
    """Matrix or vector norm.

    Args:
        x (Tensor): Input tensor.
        ord (int | str | None): Order of the norm.
        axis (int | tuple[int, ...] | None): If axis is an integer, it specifies the axis of x along which to compute the vector norms.
        keepdims (bool): If True, the axes which are reduced are left in the result as dimensions with size one.

    Returns:
    Tensor: Norm of the matrix or vector(s).
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Norm", x.data, ord=ord, axis=axis, keepdims=keepdims)

        # Calculate shape
        if axis is None:
            out_shape = () if not keepdims else tuple(1 for _ in x.shape)
        else:
            axes = (axis,) if isinstance(axis, int) else axis
            out_shape = tuple(
                1 if i in axes and keepdims else s
                for i, s in enumerate(x.shape)
                if i not in axes or keepdims
            )

        return Tensor(data, TensorConfig(out_shape, x.dtype, x.device))

    # Simple shape calculation for tracing
    if axis is None:
        out_shape = () if not keepdims else tuple(1 for _ in x.shape)
    else:
        axes = (axis,) if isinstance(axis, int) else axis
        out_shape = tuple(
            1 if i in axes and keepdims else s
            for i, s in enumerate(x.shape)
            if i not in axes or keepdims
        )

    return _emit_linalg_node(
        "Norm", [x], {"ord": ord, "axis": axis, "keepdims": keepdims}, [out_shape], [x.dtype]
    )


def matrix_exponential(a: Tensor) -> Tensor:
    """Compute the matrix exponential of a square matrix.

    Args:
        a (Tensor): The input square matrix.

    Returns:
    Tensor: The matrix exponential.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("MatrixExponential", a.data)
        return Tensor(data, TensorConfig(data.shape, a.dtype, a.device))
    return _emit_linalg_node("MatrixExponential", [a], {}, [a.shape], [a.dtype])


def _power_iteration_eager(
    input: Tensor,
    num_iters: int,
    u: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Execute power iteration eagerly."""
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()

    v_data, u_data, sigma_data = backend.execute_op(
        "PowerIteration",
        input.data,
        num_iters=num_iters,
        u=u.data if u is not None else None,
    )

    return (
        Tensor(v_data, TensorConfig(v_data.shape, input.dtype, input.device)),
        Tensor(u_data, TensorConfig(u_data.shape, input.dtype, input.device)),
        Tensor(sigma_data, TensorConfig(sigma_data.shape, input.dtype, input.device)),
    )


def power_iteration(
    input: Tensor,
    num_iters: int = 1,
    u: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Computes the dominant singular value and vectors using power iteration.

    Args:
        input (Tensor): The input matrix of shape (..., M, N)
        num_iters (int): The number of iterations to perform. Defaults to 1
        u (Tensor | None): Optional initial estimate for the left singular vector
            of shape (..., M, 1). If None, a uniform vector of ones is used.

    Returns:
    tuple[Tensor, Tensor, Tensor]: A tuple containing:
        - v (Tensor): Right singular vector estimate
        - u (Tensor): Left singular vector estimate
        - sigma (Tensor): Spectral norm estimate
    """
    if config.eager_mode:
        return _power_iteration_eager(input, num_iters, u)

    inputs = [input]
    if u is not None:
        inputs.append(u)

    in_shape = input.shape
    v_shape = in_shape[:-2] + (in_shape[-1],)
    u_shape = in_shape[:-2] + (in_shape[-2],)
    sigma_shape = in_shape[:-2]

    return _emit_linalg_node(
        "PowerIteration",
        inputs,
        {"num_iters": num_iters},
        [v_shape, u_shape, sigma_shape],
        [input.dtype, input.dtype, input.dtype],
    )
