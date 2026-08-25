# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Module pb_utils.py."""

"""Minimal Protobuf Encoder."""


def encode_varint(value: int) -> bytes:
    """Encode an integer as a protobuf varint.

    Args:
        value (int): The value parameter.

    Returns:
        bytes: Result.
    """
    if value < 0:
        value += 1 << 64
    res: object = bytearray()
    while True:
        byte: object = value & 0x7F
        value >>= 7
        if value:
            res.append(byte | 0x80)
        else:
            res.append(byte)
            break
    return bytes(res)


class ProtobufWriter:
    """Provide a minimal writer for protobuf binary format."""

    def __init__(self) -> None:
        """Initialize ProtobufWriter."""
        self.data = bytearray()

    def add_varint(self, tag: int, value: int) -> None:
        """Add a varint field.

        Args:
            tag (int): The tag parameter.
            value (int): The value parameter.
        """
        self.data.extend(encode_varint((tag << 3) | 0))
        self.data.extend(encode_varint(value))

    def add_bytes(self, tag: int, value: bytes) -> None:
        """Add a length-delimited bytes field.

        Args:
            tag (int): The tag parameter.
            value (bytes): The value parameter.
        """
        self.data.extend(encode_varint((tag << 3) | 2))
        self.data.extend(encode_varint(len(value)))
        self.data.extend(value)

    def add_string(self, tag: int, value: str) -> None:
        """Add a length-delimited string field.

        Args:
            tag (int): The tag parameter.
            value (str): The value parameter.
        """
        self.add_bytes(tag, value.encode("utf-8"))

    def add_message(self, tag: int, writer: "ProtobufWriter") -> None:
        """Add a nested message field.

        Args:
            tag (int): The tag parameter.
            writer (object): The writer parameter.
        """
        self.add_bytes(tag, writer.get_bytes())

    def get_bytes(self) -> bytes:
        """Get the encoded bytes.

        Returns:
        bytes: Result.
        """
        return bytes(self.data)
