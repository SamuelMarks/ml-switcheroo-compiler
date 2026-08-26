"""Module frontend_segment.py."""

from __future__ import annotations

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915

"""Frontend reductions ops."""


from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.base import dispatch_eager

from .frontend_utils import _emit_reduction_node


def _emit_segment_op(
    op_type: str,
    data: Tensor,
    segment_ids: Tensor,
    num_segments: int | None = None,
):
    """Evaluate _emit_segment_op operation.

    Args:
        op_type (str): The op_type parameter.
        data (Tensor): The data parameter.
        segment_ids (Tensor): The segment_ids parameter.
        num_segments (object): The num_segments parameter.

    Returns:
        Tensor: Result.
    """
    inputs = [data, segment_ids]
    attributes = {}
    if num_segments is not None:
        attributes["num_segments"] = num_segments

    return _emit_reduction_node(
        op_type,
        inputs,
        attributes,
        (),  # Placeholder shape
        data.dtype,
    )


@dispatch_eager("SegmentSum")
def segment_sum(data: Tensor, segment_ids: Tensor, num_segments: int | None = None):
    """Compute the sum of tensor elements grouped by segment_ids.

    Args:
        data (Tensor): The data tensor
        segment_ids (Tensor): The segment indices
        num_segments (int | None): The number of segments. Optional

    Returns:
        Tensor: The segmented sum tensor
    """
    return _emit_segment_op("SegmentSum", data, segment_ids, num_segments)


@dispatch_eager("SegmentMax")
def segment_max(data: Tensor, segment_ids: Tensor, num_segments: int | None = None):
    """Compute the max of tensor elements grouped by segment_ids.

    Args:
        data (Tensor): The data tensor
        segment_ids (Tensor): The segment indices
        num_segments (int | None): The number of segments. Optional

    Returns:
        Tensor: The segmented max tensor
    """
    return _emit_segment_op("SegmentMax", data, segment_ids, num_segments)


@dispatch_eager("SegmentMean")
def segment_mean(data: Tensor, segment_ids: Tensor, num_segments: int | None = None):
    """Compute the mean of tensor elements grouped by segment_ids.

    Args:
        data (Tensor): The data tensor
        segment_ids (Tensor): The segment indices
        num_segments (int | None): The number of segments. Optional

    Returns:
        Tensor: The segmented mean tensor
    """
    return _emit_segment_op("SegmentMean", data, segment_ids, num_segments)


@dispatch_eager("SegmentMin")
def segment_min(data: Tensor, segment_ids: Tensor, num_segments: int | None = None):
    """Compute the min of tensor elements grouped by segment_ids.

    Args:
        data (Tensor): The data tensor
        segment_ids (Tensor): The segment indices
        num_segments (int | None): The number of segments. Optional

    Returns:
        Tensor: The segmented min tensor
    """
    return _emit_segment_op("SegmentMin", data, segment_ids, num_segments)


@dispatch_eager("SegmentProd")
def segment_prod(data: Tensor, segment_ids: Tensor, num_segments: int | None = None):
    """Compute the prod of tensor elements grouped by segment_ids.

    Args:
        data (Tensor): The data tensor
        segment_ids (Tensor): The segment indices
        num_segments (int | None): The number of segments. Optional

    Returns:
        Tensor: The segmented prod tensor
    """
    return _emit_segment_op("SegmentProd", data, segment_ids, num_segments)


@dispatch_eager("UnsortedSegmentMax")
def unsorted_segment_max(data: Tensor, segment_ids: Tensor, num_segments: int | None = None):
    """Compute the max of tensor elements grouped by unsorted segment_ids.

    Args:
        data (Tensor): The data tensor
        segment_ids (Tensor): The segment indices
        num_segments (int | None): The number of segments. Optional

    Returns:
        Tensor: The segmented max tensor
    """
    return _emit_segment_op("UnsortedSegmentMax", data, segment_ids, num_segments)


@dispatch_eager("UnsortedSegmentMean")
def unsorted_segment_mean(data: Tensor, segment_ids: Tensor, num_segments: int | None = None):
    """Compute the mean of tensor elements grouped by unsorted segment_ids.

    Args:
        data (Tensor): The data tensor
        segment_ids (Tensor): The segment indices
        num_segments (int | None): The number of segments. Optional

    Returns:
        Tensor: The segmented mean tensor
    """
    return _emit_segment_op("UnsortedSegmentMean", data, segment_ids, num_segments)


@dispatch_eager("UnsortedSegmentMin")
def unsorted_segment_min(data: Tensor, segment_ids: Tensor, num_segments: int | None = None):
    """Compute the min of tensor elements grouped by unsorted segment_ids.

    Args:
        data (Tensor): The data tensor
        segment_ids (Tensor): The segment indices
        num_segments (int | None): The number of segments. Optional

    Returns:
        Tensor: The segmented min tensor
    """
    return _emit_segment_op("UnsortedSegmentMin", data, segment_ids, num_segments)


@dispatch_eager("UnsortedSegmentProd")
def unsorted_segment_prod(data: Tensor, segment_ids: Tensor, num_segments: int | None = None):
    """Compute the prod of tensor elements grouped by unsorted segment_ids.

    Args:
        data (Tensor): The data tensor
        segment_ids (Tensor): The segment indices
        num_segments (int | None): The number of segments. Optional

    Returns:
        Tensor: The segmented prod tensor
    """
    return _emit_segment_op("UnsortedSegmentProd", data, segment_ids, num_segments)


@dispatch_eager("UnsortedSegmentSqrtN")
def unsorted_segment_sqrt_n(data: Tensor, segment_ids: Tensor, num_segments: int | None = None):
    """Compute the sqrt_n of tensor elements grouped by unsorted segment_ids.

    Args:
        data (Tensor): The data tensor
        segment_ids (Tensor): The segment indices
        num_segments (int | None): The number of segments. Optional

    Returns:
        Tensor: The segmented sqrt_n tensor
    """
    return _emit_segment_op("UnsortedSegmentSqrtN", data, segment_ids, num_segments)


@dispatch_eager("UnsortedSegmentSum")
def unsorted_segment_sum(data: Tensor, segment_ids: Tensor, num_segments: int | None = None):
    """Compute the sum of tensor elements grouped by unsorted segment_ids.

    Args:
        data (Tensor): The data tensor
        segment_ids (Tensor): The segment indices
        num_segments (int | None): The number of segments. Optional

    Returns:
        Tensor: The segmented sum tensor
    """
    return _emit_segment_op("UnsortedSegmentSum", data, segment_ids, num_segments)
