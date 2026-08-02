"""Minimal Protobuf Encoder."""


def encode_varint(value: int) -> bytes:
    """Encode an integer as a protobuf varint."""
    if value < 0:
        value += 1 << 64
    res = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            res.append(byte | 0x80)
        else:
            res.append(byte)
            break
    return bytes(res)


class ProtobufWriter:
    """A minimal writer for protobuf binary format."""

    def __init__(self) -> None:
        """Initialize ProtobufWriter."""
        self.data = bytearray()

    def add_varint(self, tag: int, value: int) -> None:
        """Add a varint field."""
        self.data.extend(encode_varint((tag << 3) | 0))
        self.data.extend(encode_varint(value))

    def add_bytes(self, tag: int, value: bytes) -> None:
        """Add a length-delimited bytes field."""
        self.data.extend(encode_varint((tag << 3) | 2))
        self.data.extend(encode_varint(len(value)))
        self.data.extend(value)

    def add_string(self, tag: int, value: str) -> None:
        """Add a length-delimited string field."""
        self.add_bytes(tag, value.encode("utf-8"))

    def add_message(self, tag: int, writer: "ProtobufWriter") -> None:
        """Add a nested message field."""
        self.add_bytes(tag, writer.get_bytes())

    def get_bytes(self) -> bytes:
        """Get the encoded bytes."""
        return bytes(self.data)
