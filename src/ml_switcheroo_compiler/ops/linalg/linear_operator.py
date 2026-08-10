from __future__ import annotations

# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915

"""Core abstractions and logic definitions for linear_operator.py."""
from typing import Any


class LinearOperator:
    """LinearOperator mock."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.
        """
        self.args = args
        self.kwargs = kwargs


class LinearOperatorAdjoint(LinearOperator):
    """Configuration class for linear operator adjoint."""

    def __init__(self, operator: LinearOperator, **kwargs: Any) -> None:
        """Initialize.

        Args:
            operator (LinearOperator): The operator parameter.
            **kwargs (object): Keyword args.
        """
        super().__init__(operator=operator, **kwargs)


class LinearOperatorBlockDiag(LinearOperator):
    """Configuration class for linear operator block diag."""

    def __init__(self, operators: list[LinearOperator], **kwargs: Any) -> None:
        """Initialize.

        Args:
            operators (list): The operators parameter.
            **kwargs (object): Keyword args.
        """
        super().__init__(operators=operators, **kwargs)


class LinearOperatorBlockLowerTriangular(LinearOperator):
    """Configuration class for linear operator block lower triangular."""

    def __init__(self, operators: list[list[LinearOperator]], **kwargs: Any) -> None:
        """Initialize.

        Args:
            operators (list): The operators parameter.
            **kwargs (object): Keyword args.
        """
        super().__init__(operators=operators, **kwargs)


class LinearOperatorCirculant(LinearOperator):
    """Configuration class for linear operator circulant."""

    def __init__(self, spectrum: Any, **kwargs: Any) -> None:
        """Initialize.

        Args:
            spectrum (object): The spectrum parameter.
            **kwargs (object): Keyword args.
        """
        super().__init__(spectrum=spectrum, **kwargs)


class LinearOperatorCirculant2D(LinearOperator):
    """Configuration class for linear operator circulant2 d."""

    def __init__(self, spectrum: Any, **kwargs: Any) -> None:
        """Initialize.

        Args:
            spectrum (object): The spectrum parameter.
            **kwargs (object): Keyword args.
        """
        super().__init__(spectrum=spectrum, **kwargs)


class LinearOperatorCirculant3D(LinearOperator):
    """Configuration class for linear operator circulant3 d."""

    def __init__(self, spectrum: Any, **kwargs: Any) -> None:
        """Initialize.

        Args:
            spectrum (object): The spectrum parameter.
            **kwargs (object): Keyword args.
        """
        super().__init__(spectrum=spectrum, **kwargs)


class LinearOperatorComposition(LinearOperator):
    """Configuration class for linear operator composition."""

    def __init__(self, operators: list[LinearOperator], **kwargs: Any) -> None:
        """Initialize.

        Args:
            operators (list): The operators parameter.
            **kwargs (object): Keyword args.
        """
        super().__init__(operators=operators, **kwargs)


class LinearOperatorDiag(LinearOperator):
    """Configuration class for linear operator diag."""

    def __init__(self, diag: Any, **kwargs: Any) -> None:
        """Initialize.

        Args:
            diag (object): The diag parameter.
            **kwargs (object): Keyword args.
        """
        super().__init__(diag=diag, **kwargs)


class LinearOperatorFullMatrix(LinearOperator):
    """Configuration class for linear operator full matrix."""

    def __init__(self, matrix: Any, **kwargs: Any) -> None:
        """Initialize.

        Args:
            matrix (object): The matrix parameter.
            **kwargs (object): Keyword args.
        """
        super().__init__(matrix=matrix, **kwargs)


class LinearOperatorHouseholder(LinearOperator):
    """Configuration class for linear operator householder."""

    def __init__(self, reflection_axis: Any, **kwargs: Any) -> None:
        """Initialize.

        Args:
            reflection_axis (object): The reflection_axis parameter.
            **kwargs (object): Keyword args.
        """
        super().__init__(reflection_axis=reflection_axis, **kwargs)


class LinearOperatorIdentity(LinearOperator):
    """Configuration class for linear operator identity."""

    def __init__(self, num_rows: int, **kwargs: Any) -> None:
        """Initialize.

        Args:
            num_rows (int): The num_rows parameter.
            **kwargs (object): Keyword args.
        """
        super().__init__(num_rows=num_rows, **kwargs)


class LinearOperatorInversion(LinearOperator):
    """Configuration class for linear operator inversion."""

    def __init__(self, operator: LinearOperator, **kwargs: Any) -> None:
        """Initialize.

        Args:
            operator (LinearOperator): The operator parameter.
            **kwargs (object): Keyword args.
        """
        super().__init__(operator=operator, **kwargs)


class LinearOperatorKronecker(LinearOperator):
    """Configuration class for linear operator kronecker."""

    def __init__(self, operators: list[LinearOperator], **kwargs: Any) -> None:
        """Initialize.

        Args:
            operators (list): The operators parameter.
            **kwargs (object): Keyword args.
        """
        super().__init__(operators=operators, **kwargs)


class LinearOperatorLowRankUpdate(LinearOperator):
    """Configuration class for linear operator low rank update."""

    def __init__(self, base_operator: LinearOperator, u: Any, **kwargs: Any) -> None:
        """Initialize.

        Args:
            base_operator (LinearOperator): The base_operator parameter.
            u (object): The u parameter.
            **kwargs (object): Keyword args.
        """
        super().__init__(base_operator=base_operator, u=u, **kwargs)


class LinearOperatorLowerTriangular(LinearOperator):
    """Configuration class for linear operator lower triangular."""

    def __init__(self, tril: Any, **kwargs: Any) -> None:
        """Initialize.

        Args:
            tril (object): The tril parameter.
            **kwargs (object): Keyword args.
        """
        super().__init__(tril=tril, **kwargs)


class LinearOperatorPermutation(LinearOperator):
    """Configuration class for linear operator permutation."""

    def __init__(self, perm: Any, **kwargs: Any) -> None:
        """Initialize.

        Args:
            perm (object): The perm parameter.
            **kwargs (object): Keyword args.
        """
        super().__init__(perm=perm, **kwargs)


class LinearOperatorScaledIdentity(LinearOperator):
    """Configuration class for linear operator scaled identity."""

    def __init__(self, num_rows: int, multiplier: Any, **kwargs: Any) -> None:
        """Initialize.

        Args:
            num_rows (int): The num_rows parameter.
            multiplier (object): The multiplier parameter.
            **kwargs (object): Keyword args.
        """
        super().__init__(num_rows=num_rows, multiplier=multiplier, **kwargs)


class LinearOperatorToeplitz(LinearOperator):
    """Configuration class for linear operator toeplitz."""

    def __init__(self, col: Any, row: Any, **kwargs: Any) -> None:
        """Initialize.

        Args:
            col (object): The col parameter.
            row (object): The row parameter.
            **kwargs (object): Keyword args.
        """
        super().__init__(col=col, row=row, **kwargs)


class LinearOperatorTridiag(LinearOperator):
    """Configuration class for linear operator tridiag."""

    def __init__(self, diagonals: Any, **kwargs: Any) -> None:
        """Initialize.

        Args:
            diagonals (object): The diagonals parameter.
            **kwargs (object): Keyword args.
        """
        super().__init__(diagonals=diagonals, **kwargs)


class LinearOperatorZeros(LinearOperator):
    """Configuration class for linear operator zeros."""

    def __init__(self, num_rows: int, num_cols: int, **kwargs: Any) -> None:
        """Initialize.

        Args:
            num_rows (int): The num_rows parameter.
            num_cols (int): The num_cols parameter.
            **kwargs (object): Keyword args.
        """
        super().__init__(num_rows=num_rows, num_cols=num_cols, **kwargs)
