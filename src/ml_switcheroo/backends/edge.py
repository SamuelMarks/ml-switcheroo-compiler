"""WebGPU, WASM, and ONNX Target Emission."""

from ml_switcheroo.backends.base_generator import BaseGenerator


class WebGPUCodeGenerator(BaseGenerator):
    """Emit WebGPU WGSL and JS orchestrator."""

    def _dispatch_op_template(
        self, op_instance: object, *args: object, **kwargs: object
    ) -> str:
        """Docstring."""
        return "wgsl_op"

    def generate(self) -> str:
        """Docstring."""
        return "/* WGSL WebGPU Generated Code */"


class WebGLCodeGenerator(BaseGenerator):
    """Emit WebGL GLSL fallback and JS orchestrator."""

    def _dispatch_op_template(
        self, op_instance: object, *args: object, **kwargs: object
    ) -> str:
        """Docstring."""
        return "glsl_op"

    def generate(self) -> str:
        """Docstring."""
        return "/* GLSL WebGL Generated Code */"


class WasmCodeGenerator(BaseGenerator):
    """Emit WASM SIMD C++ mapping."""

    def _dispatch_op_template(
        self, op_instance: object, *args: object, **kwargs: object
    ) -> str:
        """Docstring."""
        return "wasm_op"

    def generate(self) -> str:
        """Docstring."""
        return "/* WASM SIMD Generated Code */"


class ONNXCodeGenerator(BaseGenerator):
    """Emit ONNX Protobuf payload."""

    def _dispatch_op_template(
        self, op_instance: object, *args: object, **kwargs: object
    ) -> str:
        """Docstring."""
        return "onnx_op"

    def generate(self) -> str:
        """Docstring."""
        return "/* ONNX Generated Code */"
