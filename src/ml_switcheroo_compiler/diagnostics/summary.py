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
    """Encode an image tensor into a format suitable for summary.image.

    Args:
        tensor (Tensor): The tensor parameter.

    Returns:
        bytes: Result.
    """
    # Placeholder for image encoding
    return b"encoded_image_data"
