"""Core abstractions and logic definitions for linear_operator.py."""

from __future__ import annotations


class LinearOperator:
    """LinearOperator mock."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialize."""
        self.args = args
        self.kwargs = kwargs


class LinearOperatorAdjoint(LinearOperator):
    """Configuration class for linear operator adjoint."""

    def __init__(self, operator: LinearOperator, **kwargs: object) -> None:
        """Initialize."""
        super().__init__(operator=operator, **kwargs)


class LinearOperatorBlockDiag(LinearOperator):
    """Configuration class for linear operator block diag."""

    def __init__(self, operators: list[LinearOperator], **kwargs: object) -> None:
        """Initialize."""
        super().__init__(operators=operators, **kwargs)


class LinearOperatorBlockLowerTriangular(LinearOperator):
    """Configuration class for linear operator block lower triangular."""

    def __init__(self, operators: list[list[LinearOperator]], **kwargs: object) -> None:
        """Initialize."""
        super().__init__(operators=operators, **kwargs)


class LinearOperatorCirculant(LinearOperator):
    """Configuration class for linear operator circulant."""

    def __init__(self, spectrum: object, **kwargs: object) -> None:
        """Initialize."""
        super().__init__(spectrum=spectrum, **kwargs)


class LinearOperatorCirculant2D(LinearOperator):
    """Configuration class for linear operator circulant2 d."""

    def __init__(self, spectrum: object, **kwargs: object) -> None:
        """Initialize."""
        super().__init__(spectrum=spectrum, **kwargs)


class LinearOperatorCirculant3D(LinearOperator):
    """Configuration class for linear operator circulant3 d."""

    def __init__(self, spectrum: object, **kwargs: object) -> None:
        """Initialize."""
        super().__init__(spectrum=spectrum, **kwargs)


class LinearOperatorComposition(LinearOperator):
    """Configuration class for linear operator composition."""

    def __init__(self, operators: list[LinearOperator], **kwargs: object) -> None:
        """Initialize."""
        super().__init__(operators=operators, **kwargs)


class LinearOperatorDiag(LinearOperator):
    """Configuration class for linear operator diag."""

    def __init__(self, diag: object, **kwargs: object) -> None:
        """Initialize."""
        super().__init__(diag=diag, **kwargs)


class LinearOperatorFullMatrix(LinearOperator):
    """Configuration class for linear operator full matrix."""

    def __init__(self, matrix: object, **kwargs: object) -> None:
        """Initialize."""
        super().__init__(matrix=matrix, **kwargs)


class LinearOperatorHouseholder(LinearOperator):
    """Configuration class for linear operator householder."""

    def __init__(self, reflection_axis: object, **kwargs: object) -> None:
        """Initialize."""
        super().__init__(reflection_axis=reflection_axis, **kwargs)


class LinearOperatorIdentity(LinearOperator):
    """Configuration class for linear operator identity."""

    def __init__(self, num_rows: int, **kwargs: object) -> None:
        """Initialize."""
        super().__init__(num_rows=num_rows, **kwargs)


class LinearOperatorInversion(LinearOperator):
    """Configuration class for linear operator inversion."""

    def __init__(self, operator: LinearOperator, **kwargs: object) -> None:
        """Initialize."""
        super().__init__(operator=operator, **kwargs)


class LinearOperatorKronecker(LinearOperator):
    """Configuration class for linear operator kronecker."""

    def __init__(self, operators: list[LinearOperator], **kwargs: object) -> None:
        """Initialize."""
        super().__init__(operators=operators, **kwargs)


class LinearOperatorLowRankUpdate(LinearOperator):
    """Configuration class for linear operator low rank update."""

    def __init__(self, base_operator: LinearOperator, u: object, **kwargs: object) -> None:
        """Initialize."""
        super().__init__(base_operator=base_operator, u=u, **kwargs)


class LinearOperatorLowerTriangular(LinearOperator):
    """Configuration class for linear operator lower triangular."""

    def __init__(self, tril: object, **kwargs: object) -> None:
        """Initialize."""
        super().__init__(tril=tril, **kwargs)


class LinearOperatorPermutation(LinearOperator):
    """Configuration class for linear operator permutation."""

    def __init__(self, perm: object, **kwargs: object) -> None:
        """Initialize."""
        super().__init__(perm=perm, **kwargs)


class LinearOperatorScaledIdentity(LinearOperator):
    """Configuration class for linear operator scaled identity."""

    def __init__(self, num_rows: int, multiplier: object, **kwargs: object) -> None:
        """Initialize."""
        super().__init__(num_rows=num_rows, multiplier=multiplier, **kwargs)


class LinearOperatorToeplitz(LinearOperator):
    """Configuration class for linear operator toeplitz."""

    def __init__(self, col: object, row: object, **kwargs: object) -> None:
        """Initialize."""
        super().__init__(col=col, row=row, **kwargs)


class LinearOperatorTridiag(LinearOperator):
    """Configuration class for linear operator tridiag."""

    def __init__(self, diagonals: object, **kwargs: object) -> None:
        """Initialize."""
        super().__init__(diagonals=diagonals, **kwargs)


class LinearOperatorZeros(LinearOperator):
    """Configuration class for linear operator zeros."""

    def __init__(self, num_rows: int, num_cols: int, **kwargs: object) -> None:
        """Initialize."""
        super().__init__(num_rows=num_rows, num_cols=num_cols, **kwargs)
