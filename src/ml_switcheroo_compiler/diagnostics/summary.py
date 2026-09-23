# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Module summary.py."""

"""Summary generation utilities for TensorBoard integration."""

from ml_switcheroo_compiler.core.tensor import Tensor


def write_raw_pb(pb_data: bytes, logdir: str) -> None:
    """Write raw protobuf data to a file.

    Args:
        pb_data (bytes): The pb_data parameter.
        logdir (str): The logdir parameter.
    """
    import os

    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, "events.out.tfevents.pb"), "ab") as f:
        f.write(pb_data)


def encode_image(tensor: Tensor) -> bytes:
    """Encode an image tensor into PNG byte format suitable for summary.image.

    Args:
        tensor (Tensor): The image tensor to encode (2D, 3D, or 4D).

    Returns:
        bytes: PNG encoded image bytes starting with valid PNG header.
    """
    import io

    import numpy as np
    from PIL import Image

    raw_data: object = getattr(tensor, "data", tensor)
    arr: np.ndarray = np.asarray(raw_data)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[:, :, 0]

    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
    else:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    img = Image.fromarray(arr)
    buf: io.BytesIO = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
