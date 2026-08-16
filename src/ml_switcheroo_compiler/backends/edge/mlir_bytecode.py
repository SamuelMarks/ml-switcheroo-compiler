"""MLIR Bytecode Encoder."""

import struct
from typing import Any


class MLIRBytecodeEncoder:
    """Lightweight pure-Python MLIR Bytecode encoder."""

    MAGIC = b"ML\xefR"
    VERSION = 1

    SECTION_STRING = 0
    SECTION_DIALECT = 1
    SECTION_ATTR_TYPE = 2
    SECTION_IR = 3
    SECTION_RESOURCE = 4

    def __init__(self) -> None:
        """Initialize encoder."""
        self.strings: list[str] = []
        self.string_map: dict[str, int] = {}
        self.dialects: list[str] = []
        self.ops: list[dict[str, Any]] = []

    def _add_string(self, s: str) -> int:
        """Add a string."""
        if s not in self.string_map:
            self.string_map[s] = len(self.strings)
            self.strings.append(s)
        return self.string_map[s]

    def add_dialect(self, dialect: str) -> None:
        """Add dialect."""
        self._add_string(dialect)
        if dialect not in self.dialects:
            self.dialects.append(dialect)

    def add_op(self, op_name: str, args: list[str], rets: list[str]) -> None:
        """Add op."""
        self._add_string(op_name)
        self.ops.append({"name": op_name, "args": args, "rets": rets})

    def _encode_varint(self, value: int) -> bytes:
        """Encode a varint."""
        """Encode unsigned varint."""
        result = bytearray()
        while True:
            byte = value & 0x7F
            value >>= 7
            if value:
                result.append(byte | 0x80)
            else:
                result.append(byte)
                break
        return bytes(result)

    def _encode_section(self, section_id: int, payload: bytes) -> bytes:
        """Encode a section."""
        return struct.pack("<B", section_id) + self._encode_varint(len(payload)) + payload

    def _encode_string_section(self) -> bytes:
        """Encode string section."""
        payload = self._encode_varint(len(self.strings))
        for s in self.strings:
            payload += s.encode("utf-8") + b"\x00"
        return self._encode_section(self.SECTION_STRING, payload)

    def _encode_dialect_section(self) -> bytes:
        """Encode dialect section."""
        payload = self._encode_varint(len(self.dialects))
        for d in self.dialects:
            payload += self._encode_varint(self._add_string(d))
        return self._encode_section(self.SECTION_DIALECT, payload)

    def _encode_ir_section(self) -> bytes:
        """Encode IR section."""
        # Simple IR section encoding mock
        payload = self._encode_varint(len(self.ops))  # region 0 ops count
        for op in self.ops:
            payload += self._encode_varint(self._add_string(op["name"]))
            payload += self._encode_varint(len(op["args"]))
            payload += self._encode_varint(len(op["rets"]))
        return self._encode_section(self.SECTION_IR, payload)

    def encode(self) -> bytes:
        """Encode the MLIR bytecode."""
        output = bytearray()
        output += self.MAGIC
        output += struct.pack("<B", self.VERSION)

        # Producer string
        output += self._add_string("ml_switcheroo_compiler").to_bytes(1, "little")  # fake producer

        output += self._encode_string_section()
        output += self._encode_dialect_section()
        output += self._encode_ir_section()

        # Add alignment padding to 4 bytes if needed, omitted for simplicity
        return bytes(output)
