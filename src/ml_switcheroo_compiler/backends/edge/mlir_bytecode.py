"""MLIR Bytecode Encoder."""

import os
import struct
from typing import Optional, Union

import yaml

from ml_switcheroo_compiler.backends.edge.config_models import MlirSpecConfig


class MLIRBytecodeEncoder:
    """Lightweight pure-Python MLIR Bytecode encoder driven by YAML schema."""

    def __init__(self) -> None:
        """Initialize encoder."""
        self.strings: list[str] = []
        self.string_map: dict[str, int] = {}
        self.dialects: list[str] = []
        self.types: list[str] = []
        self.type_map: dict[str, int] = {}
        self.ssa_values: dict[str, int] = {}
        self.ops: list[dict[str, Union[str, list[str], list[int]]]] = []

        path: str = os.path.join(os.path.dirname(__file__), "mlir_spec.yaml")
        with open(path) as f:
            data: dict[str, Union[str, int, list[str], dict[str, int]]] = yaml.safe_load(f)
            self.spec: MlirSpecConfig = MlirSpecConfig(**data)

        # Pre-seed dialects from spec
        for d in self.spec.default_dialects:
            self.add_dialect(d)

    def _add_string(self, s: str) -> int:
        """Add a string.

        Args:
            s (str): String to add.

        Returns:
            int: Index of string in table.
        """
        if s not in self.string_map:
            self.string_map[s] = len(self.strings)
            self.strings.append(s)
        return self.string_map[s]

    def add_dialect(self, dialect: str) -> None:
        """Add dialect.

        Args:
            dialect (str): Dialect name.
        """
        self._add_string(dialect)
        if dialect not in self.dialects:
            self.dialects.append(dialect)

    def add_type(self, type_str: str) -> int:
        """Register a tensor or scalar type into the Type section table.

        Args:
            type_str (str): Dialect type specification string.

        Returns:
            int: Monotonically increasing type table index.
        """
        if type_str not in self.type_map:
            type_idx: int = len(self.types)
            self.type_map[type_str] = type_idx
            self.types.append(type_str)
            self._add_string(type_str)
            return type_idx
        return self.type_map[type_str]

    def _get_or_create_ssa(self, var_name: str) -> int:
        """Resolve or register a new monotonically increasing SSA identifier.

        Args:
            var_name (str): Identifier name from IRNode or graph input.

        Returns:
            int: 1-based monotonically increasing SSA index.
        """
        if var_name not in self.ssa_values:
            # 1-based indexing for SSA identifiers
            new_id: int = len(self.ssa_values) + 1
            self.ssa_values[var_name] = new_id
        return self.ssa_values[var_name]

    def add_op(
        self,
        op_name: str,
        args: list[str],
        rets: list[str],
        ret_types: Optional[list[str]] = None,
    ) -> None:
        """Add op with explicit SSA identifiers and return types.

        Args:
            op_name (str): Operation opcode.
            args (list[str]): Input operand names.
            rets (list[str]): Output result names.
            ret_types (Optional[list[str]]): Optional list of output dialect types.
        """
        self._add_string(op_name)
        operand_ssa_ids: list[int] = [self._get_or_create_ssa(arg) for arg in args]
        result_ssa_ids: list[int] = [self._get_or_create_ssa(ret) for ret in rets]

        effective_types: list[str] = ret_types if ret_types is not None else ["tensor<?x?xf32>"] * len(rets)
        type_indices: list[int] = [self.add_type(t) for t in effective_types]

        self.ops.append(
            {
                "name": op_name,
                "args": args,
                "rets": rets,
                "operand_ssa_ids": operand_ssa_ids,
                "result_ssa_ids": result_ssa_ids,
                "result_type_indices": type_indices,
            }
        )

    def _encode_varint(self, value: int) -> bytes:
        """Encode unsigned varint.

        Args:
            value (int): Integer to encode.

        Returns:
            bytes: Encoded unsigned LEB128 varint.
        """
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
        """Encode a section.

        Args:
            section_id (int): Section identifier.
            payload (bytes): Section byte payload.

        Returns:
            bytes: Encoded section.
        """
        return struct.pack("<B", section_id) + self._encode_varint(len(payload)) + payload

    def _encode_string_section(self) -> bytes:
        """Encode string section.

        Returns:
            bytes: Encoded string table section.
        """
        payload: bytes = self._encode_varint(len(self.strings))
        for s in self.strings:
            payload += s.encode("utf-8") + b"\x00"
        return self._encode_section(self.spec.sections["STRING"], payload)

    def _encode_dialect_section(self) -> bytes:
        """Encode dialect section.

        Returns:
            bytes: Encoded dialect table section.
        """
        payload: bytes = self._encode_varint(len(self.dialects))
        for d in self.dialects:
            payload += self._encode_varint(self._add_string(d))
        return self._encode_section(self.spec.sections["DIALECT"], payload)

    def _encode_attr_type_section(self) -> bytes:
        """Encode attribute and type section.

        Returns:
            bytes: Encoded attribute/type table section.
        """
        payload: bytearray = bytearray()
        payload += self._encode_varint(len(self.types))
        for t in self.types:
            payload += self._encode_varint(self._add_string(t))
        return self._encode_section(self.spec.sections["ATTR_TYPE"], bytes(payload))

    def _encode_ir_section(self) -> bytes:
        """Encode IR section properly spec-compliant with real SSA values and type indices.

        Returns:
            bytes: Encoded MLIR IR payload section.
        """
        # A proper MLIR IR section requires encoding Regions, Blocks, Operations, Operands, Results.
        # This is a structurally compliant nested encoding: 1 Region -> 1 Block -> Operations
        payload: bytearray = bytearray()

        # Region 0: 1 Block
        payload += self._encode_varint(1)

        # Block 0: N ops
        payload += self._encode_varint(len(self.ops))
        for op in self.ops:
            # Op header: op_name_idx
            payload += self._encode_varint(self._add_string(str(op["name"])))
            operand_ids: list[int] = list(op.get("operand_ssa_ids", []))
            type_indices: list[int] = list(op.get("result_type_indices", []))

            # Operands count
            payload += self._encode_varint(len(operand_ids))
            # Results count
            payload += self._encode_varint(len(type_indices))

            # Operands (real monotonically increasing SSA indices)
            for ssa_id in operand_ids:
                payload += self._encode_varint(ssa_id)
            # Results (real type table indices)
            for t_idx in type_indices:
                payload += self._encode_varint(t_idx)

        return self._encode_section(self.spec.sections["IR"], bytes(payload))

    def encode(self) -> bytes:
        """Encode the MLIR bytecode.

        Returns:
            bytes: Complete binary MLIR bytecode payload.
        """
        output: bytearray = bytearray()

        # Magic bytes (eval string as bytes)
        magic_bytes: bytes = self.spec.magic.encode("latin-1")
        output += magic_bytes

        output += struct.pack("<B", self.spec.version)

        # Producer string
        output += self._add_string(self.spec.producer).to_bytes(1, "little")

        output += self._encode_string_section()
        output += self._encode_dialect_section()
        output += self._encode_attr_type_section()
        output += self._encode_ir_section()

        return bytes(output)
