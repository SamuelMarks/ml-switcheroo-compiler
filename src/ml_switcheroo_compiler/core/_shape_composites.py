"""Composite tensor, sparse, ragged, and linear-operator operand protocols for Shape Layer 2."""

from __future__ import annotations

import typing
from collections.abc import Sequence

from ml_switcheroo_compiler.core._shape_types import OpValue
from ml_switcheroo_compiler.core.dtype import DType

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.registry import EagerTensorProtocol


@typing.runtime_checkable
class TensorOperandProtocol(typing.Protocol):
    """Shared structural protocol for tensor/array operator operands."""

    shape: tuple[int, ...]
    dtype: DType | str
    ndim: int
    size: int
    data: OpValue

    def to_numpy(self) -> EagerTensorProtocol:
        """Convert operand to a NumPy array.

        Returns:
            EagerTensorProtocol: NumPy array representation.
        """
        ...

    def numpy(self) -> EagerTensorProtocol:
        """Return NumPy array representation.

        Returns:
            EagerTensorProtocol: NumPy array representation.
        """
        ...


@typing.runtime_checkable
class RaggedOperandProtocol(typing.Protocol):
    """Shared structural protocol for ragged tensor operator operands."""

    values: OpValue
    row_splits: OpValue
    nested_row_splits: Sequence[OpValue]
    shape: tuple[int | None, ...]
    dtype: DType | str


@typing.runtime_checkable
class SparseOperandProtocol(typing.Protocol):
    """Shared structural protocol for sparse tensor operator operands."""

    indices: OpValue
    values: OpValue
    dense_shape: tuple[int, ...]
    shape: tuple[int, ...]
    dtype: DType | str

    def tocoo(self) -> SparseOperandProtocol:
        """Convert sparse operand to COO format.

        Returns:
            SparseOperandProtocol: COO sparse representation.
        """
        ...

    def toarray(self) -> EagerTensorProtocol:
        """Convert sparse operand to dense array.

        Returns:
            EagerTensorProtocol: Dense array representation.
        """
        ...

    def todense(self) -> EagerTensorProtocol:
        """Convert sparse operand to dense tensor.

        Returns:
            EagerTensorProtocol: Dense tensor representation.
        """
        ...


@typing.runtime_checkable
class TensorArrayOperandProtocol(typing.Protocol):
    """Shared structural protocol for dynamic TensorArray operator operands."""

    def size(self) -> int | OpValue:
        """Return number of elements in the TensorArray.

        Returns:
            int | OpValue: TensorArray size.
        """
        ...

    def read(self, index: int | OpValue) -> OpValue:
        """Read element at `index`.

        Args:
            index (int | OpValue): Target slot index.

        Returns:
            OpValue: Element tensor.
        """
        ...

    def write(self, index: int | OpValue, value: OpValue) -> TensorArrayOperandProtocol:
        """Write `value` at `index`.

        Args:
            index (int | OpValue): Target slot index.
            value (OpValue): Element tensor to write.

        Returns:
            TensorArrayOperandProtocol: Updated TensorArray.
        """
        ...

    def stack(self) -> OpValue:
        """Stack all elements in the TensorArray along axis 0.

        Returns:
            OpValue: Stacked tensor.
        """
        ...

    def unstack(self, value: OpValue) -> TensorArrayOperandProtocol:
        """Unstack `value` along axis 0 into the TensorArray.

        Args:
            value (OpValue): Tensor to unstack.

        Returns:
            TensorArrayOperandProtocol: Populated TensorArray.
        """
        ...

    def gather(self, indices: OpValue) -> OpValue:
        """Gather elements at `indices`.

        Args:
            indices (OpValue): Slot indices to gather.

        Returns:
            OpValue: Gathered tensor.
        """
        ...

    def scatter(self, indices: OpValue, value: OpValue) -> TensorArrayOperandProtocol:
        """Scatter `value` into `indices`.

        Args:
            indices (OpValue): Target slot indices.
            value (OpValue): Values to scatter.

        Returns:
            TensorArrayOperandProtocol: Updated TensorArray.
        """
        ...


@typing.runtime_checkable
class LinearOperatorOperandProtocol(typing.Protocol):
    """Shared structural protocol for structured LinearOperator operands."""

    shape: tuple[int, ...]
    dtype: DType | str
    H: OpValue
    T: OpValue
    A: OpValue

    def matmul(self, x: OpValue, adjoint: bool = False, adjoint_arg: bool = False) -> OpValue:
        """Multiply operator by matrix `x`.

        Args:
            x (OpValue): Right-hand matrix operand.
            adjoint (bool): Take adjoint of this operator.
            adjoint_arg (bool): Take adjoint of `x`.

        Returns:
            OpValue: Matrix product result.
        """
        ...

    def matvec(self, x: OpValue, adjoint: bool = False) -> OpValue:
        """Multiply operator by vector `x`.

        Args:
            x (OpValue): Right-hand vector operand.
            adjoint (bool): Take adjoint of this operator.

        Returns:
            OpValue: Matrix-vector product result.
        """
        ...

    def solve(self, rhs: OpValue, adjoint: bool = False, adjoint_arg: bool = False) -> OpValue:
        """Solve linear system `Op @ X = rhs`.

        Args:
            rhs (OpValue): Right-hand side matrix.
            adjoint (bool): Solve with adjoint of this operator.
            adjoint_arg (bool): Take adjoint of `rhs`.

        Returns:
            OpValue: Solution tensor.
        """
        ...

    def solvevec(self, rhs: OpValue, adjoint: bool = False) -> OpValue:
        """Solve linear system `Op @ x = rhs` for vector `rhs`.

        Args:
            rhs (OpValue): Right-hand side vector.
            adjoint (bool): Solve with adjoint of this operator.

        Returns:
            OpValue: Solution vector.
        """
        ...

    def to_dense(self) -> OpValue:
        """Materialize dense matrix representation.

        Returns:
            OpValue: Dense matrix tensor.
        """
        ...

    def adjoint(self) -> LinearOperatorOperandProtocol:
        """Return conjugate transpose LinearOperator.

        Returns:
            LinearOperatorOperandProtocol: Adjoint operator.
        """
        ...

    def diag_part(self) -> OpValue:
        """Extract main diagonal part.

        Returns:
            OpValue: Diagonal vector tensor.
        """
        ...

    def trace(self) -> OpValue:
        """Compute matrix trace.

        Returns:
            OpValue: Trace scalar tensor.
        """
        ...

    def det(self) -> OpValue:
        """Compute matrix determinant.

        Returns:
            OpValue: Determinant scalar tensor.
        """
        ...

    def log_abs_determinant(self) -> OpValue:
        """Compute log absolute determinant.

        Returns:
            OpValue: Log abs determinant tensor.
        """
        ...

    def eigvals(self) -> OpValue:
        """Compute eigenvalues of operator.

        Returns:
            OpValue: Eigenvalues tensor.
        """
        ...

    def cond(self) -> OpValue:
        """Compute condition number of operator.

        Returns:
            OpValue: Condition number tensor.
        """
        ...

    def cholesky(self) -> LinearOperatorOperandProtocol:
        """Compute Cholesky factor LinearOperator.

        Returns:
            LinearOperatorOperandProtocol: Cholesky factor operator.
        """
        ...

    def inverse(self) -> LinearOperatorOperandProtocol:
        """Compute inverse LinearOperator.

        Returns:
            LinearOperatorOperandProtocol: Inverse operator.
        """
        ...
