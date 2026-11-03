"""Data Parallel NumPy (dpnp) code generator targeting Intel SYCL hardware."""

from __future__ import annotations

from ml_switcheroo_compiler.backends.base_generator import PythonStringGenerator
from ml_switcheroo_compiler.backends.common.generator_mixins import get_shared_ast_visitors
from ml_switcheroo_compiler.backends.dpnp.eager import execute_op as _execute_op
from ml_switcheroo_compiler.backends.dpnp.types import (
    array as _array,
)
from ml_switcheroo_compiler.backends.dpnp.types import (
    asarray as _asarray,
)
from ml_switcheroo_compiler.backends.dpnp.types import (
    asnumpy as _asnumpy,
)
from ml_switcheroo_compiler.backends.dpnp.types import (
    empty as _empty,
)
from ml_switcheroo_compiler.backends.dpnp.types import (
    from_numpy as _from_numpy,
)
from ml_switcheroo_compiler.backends.dpnp.types import (
    full as _full,
)
from ml_switcheroo_compiler.backends.dpnp.types import (
    get_device as _get_device,
)
from ml_switcheroo_compiler.backends.dpnp.types import (
    get_usm_type as _get_usm_type,
)
from ml_switcheroo_compiler.backends.dpnp.types import (
    is_sycl_array as _is_sycl_array,
)
from ml_switcheroo_compiler.backends.dpnp.types import (
    item as _item,
)
from ml_switcheroo_compiler.backends.dpnp.types import (
    ones as _ones,
)
from ml_switcheroo_compiler.backends.dpnp.types import (
    to_numpy as _to_numpy,
)
from ml_switcheroo_compiler.backends.dpnp.types import (
    zeros as _zeros,
)
from ml_switcheroo_compiler.backends.eager_registry import EagerValue, as_tensor
from ml_switcheroo_compiler.backends.registry import register_backend
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


@register_backend("dpnp")
class DPNPGenerator(PythonStringGenerator):
    """Generate hardware-accelerated Python/DPNP code targeting Intel SYCL devices."""

    def __init__(
        self,
        graph: IRGraph,
        device: str = "auto",
        sycl_queue: EagerValue | None = None,
    ) -> None:
        """Initialize the DPNP code generator.

        Args:
            graph (IRGraph): The computation graph to compile.
            device (str): SYCL target device ('auto', 'cpu', 'gpu', 'fpga').
            sycl_queue (object | None): Optional pre-initialized SYCL queue.
        """
        super().__init__(graph)
        self.device: str = device
        self.sycl_queue: EagerValue | None = sycl_queue
        self.visitors.extend([*get_shared_ast_visitors(generator=self)])

    def get_fallback_prefix(self) -> str:
        """Get the library prefix string used when emitting DPNP operations.

        Returns:
            str: The prefix 'dpnp'.
        """
        return "dpnp"

    def get_helper_functions(self) -> list[str]:
        """Get helper functions required for code emission.

        Returns:
            list[str]: Empty list of helper functions.
        """
        res: list[str] = []
        return res

    def generate(self) -> str:
        """Generate the complete DPNP script with SYCL queue target.

        Returns:
            str: Generated Python script using DPNP.
        """
        self.code = [self.header]
        self.add_line("import dpnp")
        if self.device != "auto":
            self.add_line(f"# Target SYCL Device: {self.device}")
        if self.sycl_queue is not None:
            self.add_line("# Active SYCL Queue configuration attached")
        self.add_line("")
        self.add_line(f"def {self._func_name}(args):")
        self.indent_level += 1
        self._generate_body("args")
        self.indent_level -= 1
        return "\n".join(self.code)

    def visit_Add(self, node: IRNode, input_vars: list[str], **kwargs: EagerValue) -> str:
        """Emit DPNP addition operation.

        Args:
            node (IRNode): Target node.
            input_vars (list[str]): Input operand names.
            **kwargs (EagerValue): Extra attributes.

        Returns:
            str: Code string.
        """
        del node, kwargs
        return f"dpnp.add({', '.join(input_vars)})"

    def visit_Sub(self, node: IRNode, input_vars: list[str], **kwargs: EagerValue) -> str:
        """Emit DPNP subtraction operation.

        Args:
            node (IRNode): Target node.
            input_vars (list[str]): Input operand names.
            **kwargs (EagerValue): Extra attributes.

        Returns:
            str: Code string.
        """
        del node, kwargs
        return f"dpnp.subtract({', '.join(input_vars)})"

    def visit_Mul(self, node: IRNode, input_vars: list[str], **kwargs: EagerValue) -> str:
        """Emit DPNP multiplication operation.

        Args:
            node (IRNode): Target node.
            input_vars (list[str]): Input operand names.
            **kwargs (EagerValue): Extra attributes.

        Returns:
            str: Code string.
        """
        del node, kwargs
        return f"dpnp.multiply({', '.join(input_vars)})"

    def visit_Div(self, node: IRNode, input_vars: list[str], **kwargs: EagerValue) -> str:
        """Emit DPNP division operation.

        Args:
            node (IRNode): Target node.
            input_vars (list[str]): Input operand names.
            **kwargs (EagerValue): Extra attributes.

        Returns:
            str: Code string.
        """
        del node, kwargs
        return f"dpnp.divide({', '.join(input_vars)})"

    def visit_MatMul(self, node: IRNode, input_vars: list[str], **kwargs: EagerValue) -> str:
        """Emit DPNP matrix multiplication operation.

        Args:
            node (IRNode): Target node.
            input_vars (list[str]): Input operand names.
            **kwargs (EagerValue): Extra attributes.

        Returns:
            str: Code string.
        """
        del node, kwargs
        return f"dpnp.matmul({', '.join(input_vars)})"

    def generic_visit(self, node: IRNode, input_vars: list[str], **kwargs: EagerValue) -> str:
        """Fallback for generic nodes emitting DPNP operations.

        Args:
            node (IRNode): The node to process.
            input_vars (list[str]): Input variable names.
            **kwargs (EagerValue): Extra attributes.

        Returns:
            str: Generated code string.
        """
        return super().generic_visit(node, input_vars, **kwargs)

    def _compile_aot_impl(self, graph: IRGraph, **kwargs: EagerValue) -> EagerValue:
        """Compile IRGraph into an executable callable targeting Intel SYCL.

        Args:
            graph (IRGraph): Target computational graph.
            **kwargs (EagerValue): Optional compilation options.

        Returns:
            EagerValue: Execution callable.
        """
        from ml_switcheroo_compiler.core.tensor import Tensor
        from ml_switcheroo_compiler.interpreter.evaluator import evaluate_graph

        generator_cls = type(self)

        def aot_dpnp_runner(*w_args: EagerValue, **w_kw: EagerValue) -> EagerValue:
            """Execute graph targeting DPNP device.

            Args:
                *w_args (EagerValue): Input tensor values.
                **w_kw (EagerValue): Keyword inputs.

            Returns:
                EagerValue: Output results.
            """
            input_nodes = [n for n in graph.nodes.values() if n.op_type == "Input"]
            inputs: dict[str, EagerValue] = {}
            for i, inp_node in enumerate(input_nodes):
                if i < len(w_args):
                    arg_val = w_args[i]
                    inputs[inp_node.id] = arg_val.data if isinstance(arg_val, Tensor) else arg_val
            inputs.update(w_kw)
            evaluated = evaluate_graph(graph, inputs=inputs, backend=generator_cls)
            if graph.outputs:
                if len(graph.outputs) == 1:
                    return evaluated.get(graph.outputs[0])
                return tuple(evaluated.get(out_id) for out_id in graph.outputs)
            return as_tensor(evaluated)

        return aot_dpnp_runner

    @staticmethod
    def zeros(
        shape: tuple[int, ...],
        dtype: str | None = None,
        device: str = "auto",
        usm_type: str = "device",
        sycl_queue: EagerValue | None = None,
    ) -> EagerValue:
        """Create a DPNP array of zeros.

        Args:
            shape (tuple[int, ...]): Target shape.
            dtype (str | None): Optional dtype name.
            device (str): SYCL target device.
            usm_type (str): USM memory type.
            sycl_queue (EagerValue | None): Optional SYCL queue.

        Returns:
            EagerValue: Zero-filled DPNP array.
        """
        return _zeros(shape, dtype=dtype, device=device, usm_type=usm_type, sycl_queue=sycl_queue)

    @staticmethod
    def ones(
        shape: tuple[int, ...],
        dtype: str | None = None,
        device: str = "auto",
        usm_type: str = "device",
        sycl_queue: EagerValue | None = None,
    ) -> EagerValue:
        """Create a DPNP array of ones.

        Args:
            shape (tuple[int, ...]): Target shape.
            dtype (str | None): Optional dtype name.
            device (str): SYCL target device.
            usm_type (str): USM memory type.
            sycl_queue (EagerValue | None): Optional SYCL queue.

        Returns:
            EagerValue: One-filled DPNP array.
        """
        return _ones(shape, dtype=dtype, device=device, usm_type=usm_type, sycl_queue=sycl_queue)

    @staticmethod
    def empty(
        shape: tuple[int, ...],
        dtype: str | None = None,
        device: str = "auto",
        usm_type: str = "device",
        sycl_queue: EagerValue | None = None,
    ) -> EagerValue:
        """Allocate an uninitialized DPNP array.

        Args:
            shape (tuple[int, ...]): Target shape.
            dtype (str | None): Optional dtype name.
            device (str): SYCL target device.
            usm_type (str): USM memory type.
            sycl_queue (EagerValue | None): Optional SYCL queue.

        Returns:
            EagerValue: Uninitialized DPNP array.
        """
        return _empty(shape, dtype=dtype, device=device, usm_type=usm_type, sycl_queue=sycl_queue)

    @staticmethod
    def full(
        shape: tuple[int, ...],
        fill_value: EagerValue = 0.0,
        dtype: str | None = None,
        device: str = "auto",
        usm_type: str = "device",
        sycl_queue: EagerValue | None = None,
    ) -> EagerValue:
        """Create a DPNP array filled with scalar value.

        Args:
            shape (tuple[int, ...]): Target shape.
            fill_value (EagerValue): Scalar fill value.
            dtype (str | None): Optional dtype name.
            device (str): SYCL target device.
            usm_type (str): USM memory type.
            sycl_queue (EagerValue | None): Optional SYCL queue.

        Returns:
            EagerValue: Value-filled DPNP array.
        """
        return _full(shape, fill_value=fill_value, dtype=dtype, device=device, usm_type=usm_type, sycl_queue=sycl_queue)

    @staticmethod
    def array(
        data: EagerValue,
        dtype: EagerValue | None = None,
        device: str = "auto",
        usm_type: str = "device",
        sycl_queue: EagerValue | None = None,
    ) -> EagerValue:
        """Create a DPNP array from data sequence.

        Args:
            data (EagerValue): Input data.
            dtype (EagerValue | None): Optional dtype name.
            device (str): SYCL target device.
            usm_type (str): USM memory type.
            sycl_queue (EagerValue | None): Optional SYCL queue.

        Returns:
            EagerValue: DPNP array.
        """
        dt_str = str(dtype) if dtype is not None else None
        return _array(data, dtype=dt_str, device=device, usm_type=usm_type, sycl_queue=sycl_queue)

    @staticmethod
    def asarray(
        data: EagerValue,
        dtype: str | None = None,
        device: str = "auto",
        usm_type: str = "device",
        sycl_queue: EagerValue | None = None,
    ) -> EagerValue:
        """Convert input to DPNP USM format.

        Args:
            data (EagerValue): Input data.
            dtype (str | None): Optional dtype name.
            device (str): SYCL target device.
            usm_type (str): USM memory type.
            sycl_queue (EagerValue | None): Optional SYCL queue.

        Returns:
            EagerValue: DPNP array.
        """
        return _asarray(data, dtype=dtype, device=device, usm_type=usm_type, sycl_queue=sycl_queue)

    @staticmethod
    def asnumpy(data: EagerValue) -> EagerValue:
        """Transfer DPNP USM array back to host ndarray.

        Args:
            data (EagerValue): Target array buffer.

        Returns:
            EagerValue: Host ndarray buffer.
        """
        return _asnumpy(data)

    @staticmethod
    def to_numpy(data: EagerValue) -> EagerValue:
        """Convert DPNP array to host array.

        Args:
            data (EagerValue): Target array buffer.

        Returns:
            EagerValue: Host ndarray buffer.
        """
        return _to_numpy(data)

    @staticmethod
    def from_numpy(
        data: EagerValue,
        device: str = "auto",
        usm_type: str = "device",
        sycl_queue: EagerValue | None = None,
    ) -> EagerValue:
        """Transfer host array to DPNP SYCL USM buffer.

        Args:
            data (EagerValue): Source host array.
            device (str): Target SYCL device.
            usm_type (str): Target USM type.
            sycl_queue (EagerValue | None): Optional SYCL queue.

        Returns:
            EagerValue: DPNP array.
        """
        return _from_numpy(data, device=device, usm_type=usm_type, sycl_queue=sycl_queue)

    @staticmethod
    def get_device(data: EagerValue) -> str:
        """Inspect the device of a DPNP array buffer.

        Args:
            data (EagerValue): Array buffer.

        Returns:
            str: Device identifier string.
        """
        return _get_device(data)

    @staticmethod
    def get_usm_type(data: EagerValue) -> str | None:
        """Inspect the USM allocation type of a DPNP array buffer.

        Args:
            data (EagerValue): Array buffer.

        Returns:
            str | None: USM type or None.
        """
        return _get_usm_type(data)

    @staticmethod
    def is_sycl_array(data: EagerValue) -> bool:
        """Determine whether an object is a DPNP/SYCL array buffer.

        Args:
            data (EagerValue): Array candidate.

        Returns:
            bool: True if DPNP/SYCL array buffer.
        """
        return _is_sycl_array(data)

    @staticmethod
    def item(data: EagerValue) -> float:
        """Extract scalar item value from single-element tensor.

        Args:
            data (EagerValue): Target data.

        Returns:
            float: Extracted scalar value.
        """
        return _item(data)

    @staticmethod
    def execute_op(op_type: str, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Execute operation eagerly on DPNP backend.

        Args:
            op_type (str): Operation name string.
            *args (EagerValue): Positional arguments.
            **kwargs (EagerValue): Keyword arguments.

        Returns:
            EagerValue: Evaluated DPNP array.
        """
        return _execute_op(op_type, *args, **kwargs)
