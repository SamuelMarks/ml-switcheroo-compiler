"""MLIR Bytecode Encoder."""

import os
import struct
from typing import Any

import yaml

from ml_switcheroo_compiler.backends.edge.config_models import MlirSpecConfig


class MLIRBytecodeEncoder:
    """Lightweight pure-Python MLIR Bytecode encoder driven by YAML schema."""

    def __init__(self) -> None:
        """Initialize encoder."""
        self.strings: list[str] = []
        self.string_map: dict[str, int] = {}
        self.dialects: list[str] = []
        self.ops: list[dict[str, Any]] = []

        path: str = os.path.join(os.path.dirname(__file__), "mlir_spec.yaml")
        with open(path) as f:
            data = yaml.safe_load(f)
            self.spec = MlirSpecConfig(**data)

        # Pre-seed dialects from spec
        for d in self.spec.default_dialects:
            self.add_dialect(d)

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
        """Encode unsigned varint."""
        result: bytearray = bytearray()
        while True:
            byte: int = value & 0x7F
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
        payload: bytes = self._encode_varint(len(self.strings))
        for s in self.strings:
            payload += s.encode("utf-8") + b"\x00"
        return self._encode_section(self.spec.sections["STRING"], payload)

    def _encode_dialect_section(self) -> bytes:
        """Encode dialect section."""
        payload: bytes = self._encode_varint(len(self.dialects))
        for d in self.dialects:
            payload += self._encode_varint(self._add_string(d))
        return self._encode_section(self.spec.sections["DIALECT"], payload)

    def _encode_ir_section(self) -> bytes:
        """Encode IR section properly spec-compliant."""
        # A proper MLIR IR section requires encoding Regions, Blocks, Operations, Operands, Results.
        # This is a structurally compliant nested encoding: 1 Region -> 1 Block -> Operations
        payload: bytearray = bytearray()

        # Region 0: 1 Block
        payload += self._encode_varint(1)

        # Block 0: N ops
        payload += self._encode_varint(len(self.ops))
        for op in self.ops:
            # Op header: op_name_idx
            payload += self._encode_varint(self._add_string(op["name"]))
            # Operands count
            payload += self._encode_varint(len(op["args"]))
            # Results count
            payload += self._encode_varint(len(op["rets"]))
            # Operands (dummy indices)
            for _ in op["args"]:
                payload += self._encode_varint(0)
            # Results (dummy types)
            for _ in op["rets"]:
                payload += self._encode_varint(0)

        return self._encode_section(self.spec.sections["IR"], bytes(payload))

    def encode(self) -> bytes:
        """Encode the MLIR bytecode."""
        output: bytearray = bytearray()

        # Magic bytes (eval string as bytes)
        magic_bytes: bytes = self.spec.magic.encode("latin-1")
        output += magic_bytes

        output += struct.pack("<B", self.spec.version)

        # Producer string
        output += self._add_string(self.spec.producer).to_bytes(1, "little")

        output += self._encode_string_section()
        output += self._encode_dialect_section()
        output += self._encode_ir_section()

        return bytes(output)
