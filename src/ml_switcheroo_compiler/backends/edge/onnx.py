"""WebGPU, WASM, and ONNX Target Emission."""

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator


class ONNXCodeGenerator(BaseGenerator):
    """Emit ONNX Protobuf payload."""

    def visit(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Visit a node and return the code string.

        Args:
            node (object): The IR node
            input_vars (list[str]): The input variable names
            **kwargs (object): Additional attributes

        Returns:
            str: The generated code
        """
        return "onnx_op"

    def generate(self) -> str:
        """Evaluate generate.

        Returns:
            str: The computed result.
        """
        return "/* ONNX Generated Code */"
