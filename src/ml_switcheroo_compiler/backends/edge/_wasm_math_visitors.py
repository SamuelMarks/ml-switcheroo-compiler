"""Elementary unary, binary, trigonometric, and reduction visitors for WASM."""

from __future__ import annotations

from ml_switcheroo_ir import IRNode

from ml_switcheroo_compiler.backends.edge._wasm_simd_loops import _WasmSimdLoops


class _WasmMathVisitors(_WasmSimdLoops):
    """WASM visitor methods for elementary math and reduction operations."""

    def visit_Trig(
        self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int
    ) -> None:
        """Generate Trig (Sin/Cos) WASM.

        Args:
            node: Input `node` argument (node).
            op_type: Input `op_type` argument (op type).
            clean_id: Input `clean_id` argument (clean id).
            inputs: Input `inputs` argument (inputs).
            shape: Input `shape` argument (shape).
            nelem: Input `nelem` argument (nelem).
        """
        self._generate_vector_unrolled_op(node, op_type, clean_id, inputs, shape, nelem)

    def visit_Exp(
        self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int
    ) -> None:
        """Generate Exp WASM.

        Args:
            node: Input `node` argument (node).
            op_type: Input `op_type` argument (op type).
            clean_id: Input `clean_id` argument (clean id).
            inputs: Input `inputs` argument (inputs).
            shape: Input `shape` argument (shape).
            nelem: Input `nelem` argument (nelem).
        """
        self._generate_vector_unrolled_op(node, op_type, clean_id, inputs, shape, nelem)

    def visit_Log(
        self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int
    ) -> None:
        """Generate Log WASM.

        Args:
            node: Input `node` argument (node).
            op_type: Input `op_type` argument (op type).
            clean_id: Input `clean_id` argument (clean id).
            inputs: Input `inputs` argument (inputs).
            shape: Input `shape` argument (shape).
            nelem: Input `nelem` argument (nelem).
        """
        self._generate_vector_unrolled_op(node, op_type, clean_id, inputs, shape, nelem)

    def visit_Tanh(
        self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int
    ) -> None:
        """Generate Tanh WASM.

        Args:
            node: Input `node` argument (node).
            op_type: Input `op_type` argument (op type).
            clean_id: Input `clean_id` argument (clean id).
            inputs: Input `inputs` argument (inputs).
            shape: Input `shape` argument (shape).
            nelem: Input `nelem` argument (nelem).
        """
        self._generate_vector_unrolled_op(node, op_type, clean_id, inputs, shape, nelem)

    def visit_Sigmoid(
        self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int
    ) -> None:
        """Generate Sigmoid WASM.

        Args:
            node: Input `node` argument (node).
            op_type: Input `op_type` argument (op type).
            clean_id: Input `clean_id` argument (clean id).
            inputs: Input `inputs` argument (inputs).
            shape: Input `shape` argument (shape).
            nelem: Input `nelem` argument (nelem).
        """
        self._generate_vector_unrolled_op(node, op_type, clean_id, inputs, shape, nelem)

    def visit_ReduceSum(
        self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int
    ) -> None:
        """Generate ReduceSum WASM.

        Args:
            node (IRNode): The IR node.
            op_type (str): Operation type.
            clean_id (str): Sanitized identifier.
            inputs (list[str]): Input identifiers.
            shape (list[int]): Shape dimensions.
            nelem (int): Total number of elements.
        """
        from ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider import get_wasm_template

        template: dict[str, str] = get_wasm_template("ReduceSum")
        in0 = inputs[0] if len(inputs) > 0 else "dummy"

        body = template.get("body", "").format(clean_id=clean_id, nelem=nelem, in0=in0)
        for line in body.split("\n"):
            if line.strip() or line == "":
                self.add_line(f"  {line}")

    def visit_Add(
        self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int
    ) -> None:
        """Generate Add WASM SIMD.

        Args:
            node (IRNode): The IR node.
            op_type (str): Operation type.
            clean_id (str): Sanitized identifier.
            inputs (list[str]): Input identifiers.
            shape (list[int]): Shape dimensions.
            nelem (int): Total number of elements.
        """
        self._generate_binary_simd_op(node, op_type, clean_id, inputs, shape, nelem)

    def visit_Sub(
        self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int
    ) -> None:
        """Generate Sub WASM SIMD.

        Args:
            node (IRNode): The IR node.
            op_type (str): Operation type.
            clean_id (str): Sanitized identifier.
            inputs (list[str]): Input identifiers.
            shape (list[int]): Shape dimensions.
            nelem (int): Total number of elements.
        """
        self._generate_binary_simd_op(node, op_type, clean_id, inputs, shape, nelem)

    def visit_Mul(
        self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int
    ) -> None:
        """Generate Mul WASM SIMD.

        Args:
            node (IRNode): The IR node.
            op_type (str): Operation type.
            clean_id (str): Sanitized identifier.
            inputs (list[str]): Input identifiers.
            shape (list[int]): Shape dimensions.
            nelem (int): Total number of elements.
        """
        self._generate_binary_simd_op(node, op_type, clean_id, inputs, shape, nelem)

    def visit_Div(
        self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int
    ) -> None:
        """Generate Div WASM SIMD.

        Args:
            node (IRNode): The IR node.
            op_type (str): Operation type.
            clean_id (str): Sanitized identifier.
            inputs (list[str]): Input identifiers.
            shape (list[int]): Shape dimensions.
            nelem (int): Total number of elements.
        """
        self._generate_binary_simd_op(node, op_type, clean_id, inputs, shape, nelem)

    def visit_Min(
        self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int
    ) -> None:
        """Generate Min WASM SIMD.

        Args:
            node (IRNode): The IR node.
            op_type (str): Operation type.
            clean_id (str): Sanitized identifier.
            inputs (list[str]): Input identifiers.
            shape (list[int]): Shape dimensions.
            nelem (int): Total number of elements.
        """
        self._generate_binary_simd_op(node, op_type, clean_id, inputs, shape, nelem)

    def visit_Max(
        self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int
    ) -> None:
        """Generate Max WASM SIMD.

        Args:
            node (IRNode): The IR node.
            op_type (str): Operation type.
            clean_id (str): Sanitized identifier.
            inputs (list[str]): Input identifiers.
            shape (list[int]): Shape dimensions.
            nelem (int): Total number of elements.
        """
        self._generate_binary_simd_op(node, op_type, clean_id, inputs, shape, nelem)

    def visit_Abs(
        self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int
    ) -> None:
        """Generate Abs WASM SIMD.

        Args:
            node (IRNode): The IR node.
            op_type (str): Operation type.
            clean_id (str): Sanitized identifier.
            inputs (list[str]): Input identifiers.
            shape (list[int]): Shape dimensions.
            nelem (int): Total number of elements.
        """
        self._generate_vector_unrolled_op(node, op_type, clean_id, inputs, shape, nelem)

    def visit_Neg(
        self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int
    ) -> None:
        """Generate Neg WASM SIMD.

        Args:
            node (IRNode): The IR node.
            op_type (str): Operation type.
            clean_id (str): Sanitized identifier.
            inputs (list[str]): Input identifiers.
            shape (list[int]): Shape dimensions.
            nelem (int): Total number of elements.
        """
        self._generate_vector_unrolled_op(node, op_type, clean_id, inputs, shape, nelem)

    def visit_Sqrt(
        self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int
    ) -> None:
        """Generate Sqrt WASM SIMD.

        Args:
            node (IRNode): The IR node.
            op_type (str): Operation type.
            clean_id (str): Sanitized identifier.
            inputs (list[str]): Input identifiers.
            shape (list[int]): Shape dimensions.
            nelem (int): Total number of elements.
        """
        self._generate_vector_unrolled_op(node, op_type, clean_id, inputs, shape, nelem)

    def visit_Relu(
        self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int
    ) -> None:
        """Generate Relu WASM SIMD.

        Args:
            node (IRNode): The IR node.
            op_type (str): Operation type.
            clean_id (str): Sanitized identifier.
            inputs (list[str]): Input identifiers.
            shape (list[int]): Shape dimensions.
            nelem (int): Total number of elements.
        """
        self._generate_vector_unrolled_op(node, op_type, clean_id, inputs, shape, nelem)

    def visit_Ceil(
        self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int
    ) -> None:
        """Generate Ceil WASM SIMD.

        Args:
            node (IRNode): The IR node.
            op_type (str): Operation type.
            clean_id (str): Sanitized identifier.
            inputs (list[str]): Input identifiers.
            shape (list[int]): Shape dimensions.
            nelem (int): Total number of elements.
        """
        self._generate_vector_unrolled_op(node, op_type, clean_id, inputs, shape, nelem)

    def visit_Floor(
        self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int
    ) -> None:
        """Generate Floor WASM SIMD.

        Args:
            node (IRNode): The IR node.
            op_type (str): Operation type.
            clean_id (str): Sanitized identifier.
            inputs (list[str]): Input identifiers.
            shape (list[int]): Shape dimensions.
            nelem (int): Total number of elements.
        """
        self._generate_vector_unrolled_op(node, op_type, clean_id, inputs, shape, nelem)

    def visit_Round(
        self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int
    ) -> None:
        """Generate Round WASM SIMD.

        Args:
            node (IRNode): The IR node.
            op_type (str): Operation type.
            clean_id (str): Sanitized identifier.
            inputs (list[str]): Input identifiers.
            shape (list[int]): Shape dimensions.
            nelem (int): Total number of elements.
        """
        self._generate_vector_unrolled_op(node, op_type, clean_id, inputs, shape, nelem)

    def visit_Sin(
        self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int
    ) -> None:
        """Generate Sin WASM SIMD.

        Args:
            node (IRNode): The IR node.
            op_type (str): Operation type.
            clean_id (str): Sanitized identifier.
            inputs (list[str]): Input identifiers.
            shape (list[int]): Shape dimensions.
            nelem (int): Total number of elements.
        """
        self._generate_vector_unrolled_op(node, op_type, clean_id, inputs, shape, nelem)

    def visit_Cos(
        self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int
    ) -> None:
        """Generate Cos WASM SIMD.

        Args:
            node (IRNode): The IR node.
            op_type (str): Operation type.
            clean_id (str): Sanitized identifier.
            inputs (list[str]): Input identifiers.
            shape (list[int]): Shape dimensions.
            nelem (int): Total number of elements.
        """
        self._generate_vector_unrolled_op(node, op_type, clean_id, inputs, shape, nelem)
