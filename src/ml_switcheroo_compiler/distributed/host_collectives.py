"""Host-level CPU collective communication primitives for distributed execution."""

from collections.abc import Sequence
from typing import Optional

import numpy as np


class HostCollectiveCommunicator:
    """CPU/host reference implementation of distributed collective primitives."""

    def __init__(self, world_size: int = 1, rank: int = 0) -> None:
        """Initialize HostCollectiveCommunicator.

        Args:
            world_size (int): Total number of participating ranks.
            rank (int): Identifier for current rank.
        """
        self.world_size: int = world_size
        self.rank: int = rank

    def all_reduce(
        self,
        data: np.ndarray,
        op: str = "SUM",
        all_ranks_data: Optional[Sequence[np.ndarray]] = None,
    ) -> np.ndarray:
        """Perform an all-reduce operation across ranks.

        Args:
            data (np.ndarray): Tensor payload for current rank.
            op (str): Reduction operator string ('SUM', 'MAX', 'MIN', 'PROD').
            all_ranks_data (Optional[Sequence[np.ndarray]]): Sequence of inputs from all ranks.

        Returns:
            np.ndarray: Reduced output tensor.
        """
        if all_ranks_data is None:
            return np.copy(data)

        stack: np.ndarray = np.stack(list(all_ranks_data), axis=0)
        norm_op: str = op.upper()
        if norm_op == "SUM":
            return np.sum(stack, axis=0)
        elif norm_op == "MAX":
            return np.max(stack, axis=0)
        elif norm_op == "MIN":
            return np.min(stack, axis=0)
        elif norm_op == "PROD":
            return np.prod(stack, axis=0)
        return np.sum(stack, axis=0)

    def all_gather(
        self,
        data: np.ndarray,
        axis: int = 0,
        all_ranks_data: Optional[Sequence[np.ndarray]] = None,
    ) -> np.ndarray:
        """Gather tensors from all ranks along a specified axis.

        Args:
            data (np.ndarray): Tensor payload for current rank.
            axis (int): Axis along which to concatenate gathered tensors.
            all_ranks_data (Optional[Sequence[np.ndarray]]): Sequence of inputs from all ranks.

        Returns:
            np.ndarray: Concatenated gathered output tensor.
        """
        if all_ranks_data is None:
            return np.copy(data)
        return np.concatenate(list(all_ranks_data), axis=axis)

    def reduce_scatter(
        self,
        data: np.ndarray,
        op: str = "SUM",
        scatter_dim: int = 0,
        all_ranks_data: Optional[Sequence[np.ndarray]] = None,
    ) -> np.ndarray:
        """Reduce tensors across ranks then scatter segments to individual ranks.

        Args:
            data (np.ndarray): Input tensor.
            op (str): Reduction operator string ('SUM', 'MAX', 'MIN', 'PROD').
            scatter_dim (int): Dimension along which to partition and scatter.
            all_ranks_data (Optional[Sequence[np.ndarray]]): Sequence of inputs from all ranks.

        Returns:
            np.ndarray: Scatting chunk for the current rank.
        """
        reduced: np.ndarray = self.all_reduce(data, op=op, all_ranks_data=all_ranks_data)
        chunks: list[np.ndarray] = np.split(reduced, self.world_size, axis=scatter_dim)
        return chunks[self.rank]

    def all_to_all(
        self,
        data: np.ndarray,
        scatter_dim: int = 0,
        gather_dim: int = 0,
        all_ranks_data: Optional[Sequence[np.ndarray]] = None,
    ) -> np.ndarray:
        """Perform all-to-all communication among ranks.

        Args:
            data (np.ndarray): Input tensor for current rank.
            scatter_dim (int): Axis along which inputs are split.
            gather_dim (int): Axis along which collected outputs are concatenated.
            all_ranks_data (Optional[Sequence[np.ndarray]]): Sequence of inputs from all ranks.

        Returns:
            np.ndarray: Evaluated all-to-all output tensor.
        """
        if all_ranks_data is None:
            return np.copy(data)
        split_ranks: list[list[np.ndarray]] = [np.split(rank_arr, self.world_size, axis=scatter_dim) for rank_arr in all_ranks_data]
        rank_slices: list[np.ndarray] = [split_ranks[r][self.rank] for r in range(self.world_size)]
        return np.concatenate(rank_slices, axis=gather_dim)

    def broadcast(
        self,
        data: np.ndarray,
        root: int = 0,
        all_ranks_data: Optional[Sequence[np.ndarray]] = None,
    ) -> np.ndarray:
        """Broadcast tensor from root rank to all ranks.

        Args:
            data (np.ndarray): Input tensor for current rank.
            root (int): Rank originating the broadcast.
            all_ranks_data (Optional[Sequence[np.ndarray]]): Sequence of inputs from all ranks.

        Returns:
            np.ndarray: Broadcasted tensor matching root rank payload.
        """
        if all_ranks_data is None:
            return np.copy(data)
        return np.copy(all_ranks_data[root])
