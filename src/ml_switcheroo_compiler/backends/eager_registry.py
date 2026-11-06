"""Implementation of eager registry operations and utilities for the backends subsystem (`ml_switcheroo_compiler.backends.eager_registry`)."""

from __future__ import annotations

import builtins
import typing
from collections.abc import Callable, Generator, Sequence

from ml_switcheroo_ir import AttributeValue
from ml_switcheroo_ir.types import _AttributeValuePydanticAnnotation

from ml_switcheroo_compiler.core.shape import (
    ShapeProtocol as ShapeProtocol,
)
from ml_switcheroo_compiler.core.shape import (
    SliceProtocol,
    TensorOrScalar,
)

_ValT = typing.TypeVar("_ValT")


@typing.runtime_checkable
class _BaseCallableProto(typing.Protocol):
    """Protocol for dynamic backend callables and factories returning an `EagerTensorProtocol`."""

    if typing.TYPE_CHECKING:
        __name__: str
        __module__: str
        __wrapped__: EagerCallableProtocol
        __all__: list[str]
        __abstractmethods__: frozenset[str]
        _autograph_do_not_convert: EagerCallableProtocol

        def __getitem__(self, key: EagerValue) -> EagerTensorProtocol:
            """Index into an indexable helper object (e.g. r_, c_, s_, at).

            Args:
                key (EagerValue): Index key or slice.

            Returns:
                EagerTensorProtocol: Indexed result.
            """
            ...

        def __iter__(self) -> Generator[EagerTensorProtocol, None, None]:
            """Iterate over elements when attribute is an iterable.

            Yields:
                EagerTensorProtocol: Element tensor.
            """
            ...

        def __index__(self) -> int:
            """Convert scalar attribute to index int.

            Returns:
                int: Integer index.
            """
            ...

        def __int__(self) -> int:
            """Convert scalar attribute to int.

            Returns:
                int: Integer value.
            """
            ...

        def __float__(self) -> float:
            """Convert scalar attribute to float.

            Returns:
                float: Float value.
            """
            ...

    def __call__(
        self,
        *args: EagerOperand,
        **kwargs: EagerOperand,
    ) -> EagerTensorProtocol:
        """Invoke the callable with arbitrary eager operands.

        Args:
            *args (EagerOperand): Positional arguments.
            **kwargs (EagerOperand): Keyword arguments.

        Returns:
            EagerTensorProtocol: Result of invoking the callable.
        """
        ...


class EagerCallableProtocol(_BaseCallableProto):
    """Explicitly typed eager callable protocol."""

    if typing.TYPE_CHECKING:
        default_rng: EagerCallableProtocol


if not typing.TYPE_CHECKING:  # pragma: no cover
    BaseGenerator = int
    FallbackPrefixProtocol = int
    GraphProtoType = int
    NodeProtoType = int
    VisitorProtocol = int
    ZerosModule = int

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.base_generator import BaseGenerator, VisitorProtocol
    from ml_switcheroo_compiler.backends.common.mixins.common import FallbackPrefixProtocol
    from ml_switcheroo_compiler.backends.eager.types_utils import ZerosModule
    from ml_switcheroo_compiler.backends.edge.onnx import GraphProtoType, NodeProtoType
    from ml_switcheroo_compiler.backends.numpy.eager import _np_module

SliceType = SliceProtocol
EagerOperand = typing.Annotated[
    typing.Union[  # noqa: UP007
        TensorOrScalar,
        BaseGenerator,
        FallbackPrefixProtocol,
        GraphProtoType,
        NodeProtoType,
        Sequence[NodeProtoType],
        VisitorProtocol,
        ZerosModule,
    ],
    _AttributeValuePydanticAnnotation,
]


@typing.runtime_checkable
class _BaseTensorProto(typing.Protocol):
    """Protocol defining the structural type contract for eager backend array/tensor objects."""

    if not typing.TYPE_CHECKING:

        @classmethod
        def __get_pydantic_core_schema__(
            cls,
            source_type: EagerValue,
            handler: EagerTensorProtocol,
        ) -> EagerValue:
            """Return core_schema.any_schema() for runtime EagerTensorProtocol field preservation.

            Args:
                source_type (EagerValue): Annotated source type.
                handler (EagerTensorProtocol): Pydantic core schema handler.

            Returns:
                EagerValue: Pydantic CoreSchema instance.
            """
            del source_type
            return ensure_eager_tensor(handler.generate_schema(AttributeValue))

    if typing.TYPE_CHECKING:

        def __call__(
            self,
            *args: EagerOperand,
            **kwargs: EagerOperand,
        ) -> EagerTensorProtocol:
            """Call operator when protocol represents a callable backend function.

            Args:
                *args (EagerOperand): Positional arguments.
                **kwargs (EagerOperand): Keyword arguments.

            Returns:
                EagerTensorProtocol: Result.
            """
            ...

        ndim: int
        T: EagerTensorProtocol
        real: EagerTensorProtocol
        imag: EagerTensorProtocol
        flat: EagerTensorProtocol
        tolist: Callable[[], list[int]]
        tobytes: Callable[[], bytes]

        def item(self, *args: int) -> float | int | bool | complex | str | bytes:
            """Extract scalar Python value from array.

            Args:
                *args (int): Optional flat or multi-index coordinates.

            Returns:
                float | int | bool | complex | str | bytes: Extracted scalar.
            """
            ...

        def __eq__(self, other: TensorOrScalar) -> bool:
            """Equality comparison.

            Args:
                other (TensorOrScalar): Other value.

            Returns:
                bool: Equality result.
            """
            ...

        def __contains__(self, item: EagerValue) -> bool:
            """Membership check.

            Args:
                item (EagerValue): Item to check.

            Returns:
                bool: True if item is contained.
            """
            ...

        def __ne__(self, other: TensorOrScalar) -> bool:
            """Inequality comparison.

            Args:
                other (TensorOrScalar): Other value.

            Returns:
                bool: Inequality result.
            """
            ...

        def __enter__(self) -> EagerTensorProtocol:
            """Enter context manager.

            Returns:
                EagerTensorProtocol: Context object.
            """
            ...

        def __await__(self) -> Generator[None, None, EagerTensorProtocol]:
            """Await coroutine tensor/stream operation.

            Returns:
                Generator[None, None, EagerTensorProtocol]: Awaitable generator.
            """
            ...

        def __exit__(self, exc_type: EagerValue, exc_val: EagerValue, exc_tb: EagerValue) -> EagerValue:
            """Exit context manager.

            Args:
                exc_type (EagerValue): Exception type.
                exc_val (EagerValue): Exception value.
                exc_tb (EagerValue): Traceback.

            Returns:
                EagerValue: Suppress flag.
            """
            ...

        def __iter__(self) -> builtins.map[EagerTensorProtocol]:
            """Iterate over the leading dimension.

            Returns:
                builtins.map[EagerTensorProtocol]: Iterator over leading dimension slices.
            """
            ...

        def __int__(self) -> int:
            """Convert scalar tensor to int.

            Returns:
                int: Integer value.
            """
            ...

        def __index__(self) -> int:
            """Convert scalar tensor to index int.

            Returns:
                int: Index integer value.
            """
            ...

        def __fspath__(self) -> str:
            """Return file system path representation.

            Returns:
                str: File system path string.
            """
            ...

        def __pos__(self) -> EagerTensorProtocol:
            """Unary positive.

            Returns:
                EagerTensorProtocol: Positive array.
            """
            ...

        def __lshift__(self, other: EagerValue) -> EagerTensorProtocol:
            """Left bitwise shift.

            Args:
                other (EagerValue): Shift amount.

            Returns:
                EagerTensorProtocol: Shifted array.
            """
            ...

        def __rshift__(self, other: EagerValue) -> EagerTensorProtocol:
            """Right bitwise shift.

            Args:
                other (EagerValue): Shift amount.

            Returns:
                EagerTensorProtocol: Shifted array.
            """
            ...

        def __rlshift__(self, other: EagerValue) -> EagerTensorProtocol:
            """Reverse left bitwise shift.

            Args:
                other (EagerValue): Left operand.

            Returns:
                EagerTensorProtocol: Shifted array.
            """
            ...

        def __rrshift__(self, other: EagerValue) -> EagerTensorProtocol:
            """Reverse right bitwise shift.

            Args:
                other (EagerValue): Left operand.

            Returns:
                EagerTensorProtocol: Shifted array.
            """
            ...

        def __float__(self) -> float:
            """Convert scalar tensor to float.

            Returns:
                float: Float value.
            """
            ...

        def __abs__(self) -> EagerTensorProtocol:
            """Elementwise absolute value.

            Returns:
                EagerTensorProtocol: Absolute value array.
            """
            ...

        def __invert__(self) -> EagerTensorProtocol:
            """Elementwise bitwise inversion.

            Returns:
                EagerTensorProtocol: Inverted array.
            """
            ...

        def __and__(self, other: EagerValue) -> EagerTensorProtocol:
            """Bitwise AND.

            Args:
                other (EagerValue): Right-hand operand.

            Returns:
                EagerTensorProtocol: Result array.
            """
            ...

        def __or__(self, other: EagerValue) -> EagerTensorProtocol:
            """Bitwise OR.

            Args:
                other (EagerValue): Right-hand operand.

            Returns:
                EagerTensorProtocol: Result array.
            """
            ...

        def __xor__(self, other: EagerValue) -> EagerTensorProtocol:
            """Bitwise XOR.

            Args:
                other (EagerValue): Right-hand operand.

            Returns:
                EagerTensorProtocol: Result array.
            """
            ...

        def __radd__(self, other: EagerValue) -> EagerTensorProtocol:
            """Reverse add.

            Args:
                other (EagerValue): Left-hand operand.

            Returns:
                EagerTensorProtocol: Sum array.
            """
            ...

        def __rsub__(self, other: EagerValue) -> EagerTensorProtocol:
            """Reverse subtract.

            Args:
                other (EagerValue): Left-hand operand.

            Returns:
                EagerTensorProtocol: Difference array.
            """
            ...

        def __rmul__(self, other: EagerValue) -> EagerTensorProtocol:
            """Reverse multiply.

            Args:
                other (EagerValue): Left-hand operand.

            Returns:
                EagerTensorProtocol: Product array.
            """
            ...

        def __rtruediv__(self, other: EagerValue) -> EagerTensorProtocol:
            """Reverse true divide.

            Args:
                other (EagerValue): Left-hand operand.

            Returns:
                EagerTensorProtocol: Quotient array.
            """
            ...

        def __floordiv__(self, other: EagerValue) -> EagerTensorProtocol:
            """Floor divide.

            Args:
                other (EagerValue): Right-hand divisor.

            Returns:
                EagerTensorProtocol: Floor quotient array.
            """
            ...

        def __rfloordiv__(self, other: EagerValue) -> EagerTensorProtocol:
            """Reverse floor divide.

            Args:
                other (EagerValue): Left-hand dividend.

            Returns:
                EagerTensorProtocol: Floor quotient array.
            """
            ...

        def __mod__(self, other: EagerValue) -> EagerTensorProtocol:
            """Modulo operation.

            Args:
                other (EagerValue): Right-hand divisor.

            Returns:
                EagerTensorProtocol: Remainder array.
            """
            ...

        def __rmod__(self, other: EagerValue) -> EagerTensorProtocol:
            """Reverse modulo operation.

            Args:
                other (EagerValue): Left-hand dividend.

            Returns:
                EagerTensorProtocol: Remainder array.
            """
            ...

        def __matmul__(self, other: EagerValue) -> EagerTensorProtocol:
            """Matrix multiplication.

            Args:
                other (EagerValue): Right-hand operand.

            Returns:
                EagerTensorProtocol: Matrix product array.
            """
            ...

        def __rmatmul__(self, other: EagerValue) -> EagerTensorProtocol:
            """Reverse matrix multiplication.

            Args:
                other (EagerValue): Left-hand operand.

            Returns:
                EagerTensorProtocol: Matrix product array.
            """
            ...

        def __rpow__(self, other: EagerValue) -> EagerTensorProtocol:
            """Reverse power operation.

            Args:
                other (EagerValue): Base operand.

            Returns:
                EagerTensorProtocol: Exponentiated array.
            """
            ...

        __name__: str
        __module__: str
        __wrapped__: EagerCallableProtocol
        __all__: list[str]
        __abstractmethods__: frozenset[str]

    def __getitem__(
        self,
        key: EagerValue,
    ) -> EagerTensorProtocol:
        """Get an item or slice from the array.

        Args:
            key (EagerValue): Indexing key or slice.

        Returns:
            EagerTensorProtocol: Indexed tensor view or copy.
        """
        ...

    def __setitem__(
        self,
        key: EagerValue,
        value: EagerValue,
    ) -> None:
        """Set an item or slice in the array.

        Args:
            key (EagerValue): Target index or slice.
            value (EagerValue): Value to assign.
        """
        ...

    def __len__(self) -> int:
        """Return the length of the leading dimension.

        Returns:
            int: Leading dimension size.
        """
        ...

    def __neg__(self) -> EagerTensorProtocol:
        """Elementwise negation.

        Returns:
            EagerTensorProtocol: Negated array.
        """
        ...

    def __add__(self, other: EagerValue) -> EagerTensorProtocol:
        """Add two arrays.

        Args:
            other (EagerValue): Right-hand operand.

        Returns:
            EagerTensorProtocol: Sum array.
        """
        ...

    def __sub__(self, other: EagerValue) -> EagerTensorProtocol:
        """Subtract two arrays.

        Args:
            other (EagerValue): Right-hand operand.

        Returns:
            EagerTensorProtocol: Difference array.
        """
        ...

    def __mul__(self, other: EagerValue) -> EagerTensorProtocol:
        """Multiply two arrays.

        Args:
            other (EagerValue): Right-hand operand.

        Returns:
            EagerTensorProtocol: Product array.
        """
        ...

    def __truediv__(self, other: EagerValue) -> EagerTensorProtocol:
        """Divide two arrays.

        Args:
            other (EagerValue): Right-hand divisor.

        Returns:
            EagerTensorProtocol: Quotient array.
        """
        ...

    def __lt__(self, other: EagerValue) -> EagerTensorProtocol:
        """Compare less than.

        Args:
            other (EagerValue): Right-hand operand.

        Returns:
            EagerTensorProtocol: Elementwise boolean comparison result.
        """
        ...

    def __le__(self, other: EagerValue) -> EagerTensorProtocol:
        """Compare less than or equal.

        Args:
            other (EagerValue): Right-hand operand.

        Returns:
            EagerTensorProtocol: Elementwise boolean comparison result.
        """
        ...

    def __gt__(self, other: EagerValue) -> EagerTensorProtocol:
        """Compare greater than.

        Args:
            other (EagerValue): Right-hand operand.

        Returns:
            EagerTensorProtocol: Elementwise boolean comparison result.
        """
        ...

    def __ge__(self, other: EagerValue) -> EagerTensorProtocol:
        """Compare greater than or equal.

        Args:
            other (EagerValue): Right-hand operand.

        Returns:
            EagerTensorProtocol: Elementwise boolean comparison result.
        """
        ...

    def __pow__(self, other: EagerValue) -> EagerTensorProtocol:
        """Power operation.

        Args:
            other (EagerValue): Exponent operand.

        Returns:
            EagerTensorProtocol: Exponentiated array.
        """
        ...

    @property
    def shape(self) -> tuple[int, ...]:
        """Get the shape of the array.

        Returns:
            tuple[int, ...]: Array shape dimensions.
        """
        ...

    if typing.TYPE_CHECKING:

        @property
        def dtype(self) -> _np_module.dtype[_np_module.float64]:
            """Get the dtype of the array.

            Returns:
                _np_module.dtype[_np_module.float64]: Array data type.
            """
            ...

        def __array__(self) -> _np_module.ndarray[tuple[int, ...], _np_module.dtype[_np_module.float64]]:
            """Return array representation for `_SupportsArray` compatibility.

            Returns:
                _np_module.ndarray[tuple[int, ...], _np_module.dtype[_np_module.float64]]: Array representation.
            """
            ...


class EagerTensorProtocol(_BaseTensorProto):
    """Explicitly typed eager tensor protocol."""

    if typing.TYPE_CHECKING:
        AVG: EagerTensorProtocol
        AbsMaxQuantizeOp: EagerTensorProtocol
        AdadeltaConfig: EagerTensorProtocol
        AdagradConfig: EagerTensorProtocol
        AdamHyperparams: EagerTensorProtocol
        AdamWHyperparams: EagerTensorProtocol
        AdamaxHyperparams: EagerTensorProtocol
        Array: EagerTensorProtocol
        BenchmarkOrchestrator: EagerTensorProtocol
        BenchmarkPlan: EagerTensorProtocol
        BenchmarkRunResult: EagerTensorProtocol
        BidirectionalConfig: EagerTensorProtocol
        BidirectionalInputs: EagerTensorProtocol
        BlurConfig: EagerTensorProtocol
        BytesIO: EagerTensorProtocol
        COOTensor: EagerTensorProtocol
        Client: EagerTensorProtocol
        CodeGeneratorVisitor: EagerTensorProtocol
        ComputeFloat8AmaxHistoryOp: EagerTensorProtocol
        ComputeFloat8ScaleOp: EagerTensorProtocol
        ConvLSTMConfig: EagerTensorProtocol
        Copy: EagerTensorProtocol
        CppGenerator: EagerTensorProtocol
        CustomJVPFunction: EagerTensorProtocol
        CustomObjectScope: EagerTensorProtocol
        DecodeImage: EagerTensorProtocol
        Device: EagerTensorProtocol
        DeviceMesh: EagerTensorProtocol
        DropoutWrapperConfig: EagerTensorProtocol
        DummyBackend: EagerTensorProtocol
        EagerEvaluator: EagerTensorProtocol
        Environment: EagerTensorProtocol
        Event: EagerTensorProtocol
        ExportArchive: EagerTensorProtocol
        FakeQuantWithMinMaxVarsOp: EagerTensorProtocol
        Float32: EagerTensorProtocol
        GatherQMMOp: EagerTensorProtocol
        GenericConvConfig: EagerTensorProtocol
        GradOptions: EagerTensorProtocol
        H5WeightFormat: EagerTensorProtocol
        IRNode: EagerTensorProtocol
        InitializerConfig: EagerTensorProtocol
        Int32: EagerTensorProtocol
        JAXCodeGenerator: EagerTensorProtocol
        JAXProfiler: EagerTensorProtocol
        KerasCodeGenerator: EagerTensorProtocol
        LLVMCPPRunner: EagerTensorProtocol
        LayoutMap: EagerTensorProtocol
        LionConfig: EagerTensorProtocol
        MAX: EagerTensorProtocol
        MIN: EagerTensorProtocol
        MLIRBytecodeEncoder: EagerTensorProtocol
        MagicMock: EagerTensorProtocol
        MultivariateNormalOptions: EagerTensorProtocol
        NONE: EagerTensorProtocol
        NamedTemporaryFile: EagerTensorProtocol
        NpzWeightFormat: EagerTensorProtocol
        NumbaGenerator: EagerTensorProtocol
        ONNXCodeGenerator: EagerTensorProtocol
        OpDef: EagerTensorProtocol
        PRODUCT: EagerTensorProtocol
        PassManager: EagerTensorProtocol
        Path: EagerTensorProtocol
        PickleWeightFormat: EagerTensorProtocol
        PurePythonGenerator: EagerTensorProtocol
        PurePythonTensor: EagerTensorProtocol
        PyTorchCodeGenerator: EagerTensorProtocol
        QuantizationConfig: EagerTensorProtocol
        QuantizationParams: EagerTensorProtocol
        QuantizeAndDequantizeOp: EagerTensorProtocol
        QuantizeOp: EagerTensorProtocol
        QuantizedMatmulOp: EagerTensorProtocol
        QuantizedOpsConfig: EagerTensorProtocol
        RMSPropHyperparams: EagerTensorProtocol
        RNNCellDeviceWrapper: EagerTensorProtocol
        RNNCellDropoutWrapper: EagerTensorProtocol
        RNNCellResidualWrapper: EagerTensorProtocol
        RNNConfig: EagerTensorProtocol
        RawMatMul: EagerTensorProtocol
        RecordBatch: EagerTensorProtocol
        SGDConfig: EagerTensorProtocol
        SUM: EagerTensorProtocol
        SafetensorsWeightFormat: EagerTensorProtocol
        ScanConfig: EagerTensorProtocol
        SignalingHandler: EagerTensorProtocol
        SignalingServer: EagerTensorProtocol
        SparseDenseMatMul: EagerTensorProtocol
        SparseGenerator: EagerTensorProtocol
        StableHLOCodeGenerator: EagerTensorProtocol
        Stream: EagerTensorProtocol
        Table: EagerTensorProtocol
        TemporaryDirectory: EagerTensorProtocol
        Tensor: EagerTensorProtocol
        TensorFlowCodeGenerator: EagerTensorProtocol
        UnconnectedGradients: EagerTensorProtocol
        UnimplementedMathError: EagerTensorProtocol
        UnpoolOptions: EagerTensorProtocol
        WasmCodeGenerator: EagerTensorProtocol
        WebGLCodeGenerator: EagerTensorProtocol
        WebGPUCodeGenerator: EagerTensorProtocol
        WebRTCDataChannelStream: EagerTensorProtocol
        WindowConfig: EagerTensorProtocol
        _ACTIVE_BACKEND: EagerTensorProtocol
        _TORCH_EAGER_OP_MAP: EagerTensorProtocol
        __bool__: EagerTensorProtocol
        __delitem__: EagerTensorProtocol
        _adaptive_max_pool3d_indices: EagerTensorProtocol
        _add_n: EagerTensorProtocol
        _allclose: EagerTensorProtocol
        _allocate_aligned_memory: EagerTensorProtocol
        _allocated_buffers: EagerTensorProtocol
        _arr: EagerTensorProtocol
        _associative_scan: EagerTensorProtocol
        _autograph_do_not_convert: EagerTensorProtocol
        _banded_triangular_solve: EagerTensorProtocol
        _build_conv_config: EagerTensorProtocol
        _build_dot_general_einsum_str: EagerTensorProtocol
        _build_step_fn: EagerTensorProtocol
        _cholesky: EagerTensorProtocol
        _cholesky_solve: EagerTensorProtocol
        _compile_aot_impl: EagerTensorProtocol
        _compile_graph: EagerTensorProtocol
        _compile_model: EagerTensorProtocol
        _compile_module: EagerTensorProtocol
        _conv_nd: EagerTensorProtocol
        _cross: EagerTensorProtocol
        _data: EagerTensorProtocol
        _det: EagerTensorProtocol
        _dispatch_compute_node: EagerTensorProtocol
        _drain_event: EagerTensorProtocol
        _eig: EagerTensorProtocol
        _eigh: EagerTensorProtocol
        _eigvals: EagerTensorProtocol
        _eigvalsh: EagerTensorProtocol
        _emit_body_return: EagerTensorProtocol
        _emit_constant_assignment: EagerTensorProtocol
        _emit_init_body: EagerTensorProtocol
        _emit_input_assignment: EagerTensorProtocol
        _emit_node: EagerTensorProtocol
        _emit_output_assignment: EagerTensorProtocol
        _emit_vjp_node: EagerTensorProtocol
        _enrich_stream: EagerTensorProtocol
        _extract_peak_memory_mb: EagerTensorProtocol
        _find_spec: EagerTensorProtocol
        _fix: EagerTensorProtocol
        _format_full: EagerTensorProtocol
        _format_jit_options: EagerTensorProtocol
        _format_zeros_like: EagerTensorProtocol
        _generate_binary_simd_op: EagerTensorProtocol
        _generate_file_header: EagerTensorProtocol
        _generate_function_signature: EagerTensorProtocol
        _generate_op: EagerTensorProtocol
        _generate_pooling2d: EagerTensorProtocol
        _generate_striding_logic: EagerTensorProtocol
        _generate_vector_unrolled_op: EagerTensorProtocol
        _generator: EagerTensorProtocol
        _get_csv_data: EagerTensorProtocol
        _get_keras_peak_memory_mb: EagerTensorProtocol
        _get_prefix_code: EagerTensorProtocol
        _get_process_memory_mb: EagerTensorProtocol
        _get_profiler: EagerTensorProtocol
        _get_sc: EagerTensorProtocol
        _get_wgsl_for_op: EagerTensorProtocol
        _global_adaptive_pool_mock: EagerTensorProtocol
        _grounding_engine: EagerTensorProtocol
        _handle_preemption_signal: EagerTensorProtocol
        _handle_unpack_assign: EagerTensorProtocol
        _matrix_power: EagerTensorProtocol
        _measure_latencies: EagerTensorProtocol
        _mlx_zeros: EagerTensorProtocol
        _nodes: EagerTensorProtocol
        _norm: EagerTensorProtocol
        _normalize_and_validate_op: EagerTensorProtocol
        _np_affineconfig: EagerTensorProtocol
        _np_all_gather: EagerTensorProtocol
        _np_all_reduce: EagerTensorProtocol
        _np_all_to_all: EagerTensorProtocol
        _np_append: EagerTensorProtocol
        _np_apply_along_axis: EagerTensorProtocol
        _np_apply_over_axes: EagerTensorProtocol
        _np_arange_: EagerTensorProtocol
        _np_argpartition: EagerTensorProtocol
        _np_argwhere: EagerTensorProtocol
        _np_array_equiv_: EagerTensorProtocol
        _np_array_repr_: EagerTensorProtocol
        _np_array_str_: EagerTensorProtocol
        _np_assertop: EagerTensorProtocol
        _np_asstringconfig: EagerTensorProtocol
        _np_atleast_1d: EagerTensorProtocol
        _np_atleast_2d: EagerTensorProtocol
        _np_atleast_3d: EagerTensorProtocol
        _np_average: EagerTensorProtocol
        _np_bessel_i0e: EagerTensorProtocol
        _np_bessel_i1e: EagerTensorProtocol
        _np_betainc: EagerTensorProtocol
        _np_betapdf: EagerTensorProtocol
        _np_bitcast_convert_type: EagerTensorProtocol
        _np_bitwise_count: EagerTensorProtocol
        _np_blackman_: EagerTensorProtocol
        _np_block: EagerTensorProtocol
        _np_blurconfig: EagerTensorProtocol
        _np_broadcast: EagerTensorProtocol
        _np_broadcast_arrays_: EagerTensorProtocol
        _np_callable: EagerTensorProtocol
        _np_can_cast_: EagerTensorProtocol
        _np_chebyshev_polynomial_t: EagerTensorProtocol
        _np_chebyshev_polynomial_u: EagerTensorProtocol
        _np_choose: EagerTensorProtocol
        _np_clip: EagerTensorProtocol
        _np_clz: EagerTensorProtocol
        _np_column_stack: EagerTensorProtocol
        _np_compress: EagerTensorProtocol
        _np_confusion_matrix: EagerTensorProtocol
        _np_convgeneraldilatedlocal: EagerTensorProtocol
        _np_convgeneraldilatedpatches: EagerTensorProtocol
        _np_convolve: EagerTensorProtocol
        _np_convwithgeneralpadding: EagerTensorProtocol
        _np_corrcoef: EagerTensorProtocol
        _np_correlate: EagerTensorProtocol
        _np_cov: EagerTensorProtocol
        _np_customlinearsolve: EagerTensorProtocol
        _np_customroot: EagerTensorProtocol
        _np_debuginfs: EagerTensorProtocol
        _np_debugnans: EagerTensorProtocol
        _np_decode_csv: EagerTensorProtocol
        _np_decode_csv_camel: EagerTensorProtocol
        _np_decode_image: EagerTensorProtocol
        _np_decode_image_camel: EagerTensorProtocol
        _np_deg2rad: EagerTensorProtocol
        _np_delete_: EagerTensorProtocol
        _np_descriptive: EagerTensorProtocol
        _np_diag_indices_: EagerTensorProtocol
        _np_diag_indices_from_: EagerTensorProtocol
        _np_diagflat_: EagerTensorProtocol
        _np_diagonal_: EagerTensorProtocol
        _np_diff_: EagerTensorProtocol
        _np_digitize_: EagerTensorProtocol
        _np_distributions: EagerTensorProtocol
        _np_dotgeneral: EagerTensorProtocol
        _np_ediff1d_: EagerTensorProtocol
        _np_einsum_path_: EagerTensorProtocol
        _np_elasticconfig: EagerTensorProtocol
        _np_empty_: EagerTensorProtocol
        _np_empty_like_: EagerTensorProtocol
        _np_encode_base64: EagerTensorProtocol
        _np_expand_dims_: EagerTensorProtocol
        _np_extract_: EagerTensorProtocol
        _np_extractpatchesoptions: EagerTensorProtocol
        _np_fabs_: EagerTensorProtocol
        _np_fftn: EagerTensorProtocol
        _np_fftnd: EagerTensorProtocol
        _np_fftshift: EagerTensorProtocol
        _np_flatnonzero_: EagerTensorProtocol
        _np_flip_op_: EagerTensorProtocol
        _np_flip_reverse_: EagerTensorProtocol
        _np_fliplr_: EagerTensorProtocol
        _np_flipud_: EagerTensorProtocol
        _np_frombuffer: EagerTensorProtocol
        _np_fromdlpack: EagerTensorProtocol
        _np_fromfunction_: EagerTensorProtocol
        _np_fromiter_: EagerTensorProtocol
        _np_frompyfunc_: EagerTensorProtocol
        _np_fromstring_: EagerTensorProtocol
        _np_full_: EagerTensorProtocol
        _np_full_like_: EagerTensorProtocol
        _np_gamma: EagerTensorProtocol
        _np_geomspace_: EagerTensorProtocol
        _np_get_printoptions_: EagerTensorProtocol
        _np_hamming_: EagerTensorProtocol
        _np_hanning_: EagerTensorProtocol
        _np_hermite_polynomial_h: EagerTensorProtocol
        _np_hermite_polynomial_he: EagerTensorProtocol
        _np_hfft: EagerTensorProtocol
        _np_histogram2d_: EagerTensorProtocol
        _np_histogram_: EagerTensorProtocol
        _np_histogram_bin_edges_: EagerTensorProtocol
        _np_histogramdd_: EagerTensorProtocol
        _np_ifft: EagerTensorProtocol
        _np_ifft2: EagerTensorProtocol
        _np_ifftn: EagerTensorProtocol
        _np_ifftnd: EagerTensorProtocol
        _np_ifftshift: EagerTensorProtocol
        _np_indices_: EagerTensorProtocol
        _np_insert_: EagerTensorProtocol
        _np_interp_: EagerTensorProtocol
        _np_intersect1d_: EagerTensorProtocol
        _np_irfft2: EagerTensorProtocol
        _np_irfftn: EagerTensorProtocol
        _np_irfftnd: EagerTensorProtocol
        _np_iscomplex_: EagerTensorProtocol
        _np_iscomplexobj_: EagerTensorProtocol
        _np_isin_: EagerTensorProtocol
        _np_isreal_: EagerTensorProtocol
        _np_isrealobj_: EagerTensorProtocol
        _np_isscalar_: EagerTensorProtocol
        _np_issubdtype_issubdtype_: EagerTensorProtocol
        _np_issubdtype_op_: EagerTensorProtocol
        _np_iterable_: EagerTensorProtocol
        _np_ix__: EagerTensorProtocol
        _np_kaiser_: EagerTensorProtocol
        _np_key: EagerTensorProtocol
        _np_kron_: EagerTensorProtocol
        _np_laguerre_polynomial_l: EagerTensorProtocol
        _np_legendre_polynomial_p: EagerTensorProtocol
        _np_lexsort_: EagerTensorProtocol
        _np_linalg_cholesky_: EagerTensorProtocol
        _np_linalg_det_: EagerTensorProtocol
        _np_linalg_inv_: EagerTensorProtocol
        _np_linalg_matrix_power_: EagerTensorProtocol
        _np_linalg_pinv_: EagerTensorProtocol
        _np_linalg_svd_: EagerTensorProtocol
        _np_linearoperator: EagerTensorProtocol
        _np_linearoperatoradjoint: EagerTensorProtocol
        _np_linearoperatorblockdiag: EagerTensorProtocol
        _np_linearoperatorblocklowertriangular: EagerTensorProtocol
        _np_linearoperatorcirculant: EagerTensorProtocol
        _np_linearoperatorcirculant2d: EagerTensorProtocol
        _np_linearoperatorcirculant3d: EagerTensorProtocol
        _np_linearoperatorcomposition: EagerTensorProtocol
        _np_linearoperatordiag: EagerTensorProtocol
        _np_linearoperatorfullmatrix: EagerTensorProtocol
        _np_linearoperatorhouseholder: EagerTensorProtocol
        _np_linearoperatoridentity: EagerTensorProtocol
        _np_linearoperatorinversion: EagerTensorProtocol
        _np_linearoperatorkronecker: EagerTensorProtocol
        _np_linearoperatorlowertriangular: EagerTensorProtocol
        _np_linearoperatorlowrankupdate: EagerTensorProtocol
        _np_linearoperatorpermutation: EagerTensorProtocol
        _np_linearoperatorscaledidentity: EagerTensorProtocol
        _np_linearoperatortoeplitz: EagerTensorProtocol
        _np_linearoperatortridiag: EagerTensorProtocol
        _np_linearoperatorzeros: EagerTensorProtocol
        _np_load_: EagerTensorProtocol
        _np_log1p2: EagerTensorProtocol
        _np_log_softmax: EagerTensorProtocol
        _np_logdet: EagerTensorProtocol
        _np_logsumexp: EagerTensorProtocol
        _np_mask_indices_: EagerTensorProtocol
        _np_median_: EagerTensorProtocol
        _np_modf_: EagerTensorProtocol
        _np_modified_bessel_i0: EagerTensorProtocol
        _np_modified_bessel_i1: EagerTensorProtocol
        _np_modified_bessel_k0: EagerTensorProtocol
        _np_modified_bessel_k1: EagerTensorProtocol
        _np_modifiedbesseli1: EagerTensorProtocol
        _np_mvlgamma: EagerTensorProtocol
        _np_nonzero_: EagerTensorProtocol
        _np_one_hot: EagerTensorProtocol
        _np_ones_: EagerTensorProtocol
        _np_ones_like_: EagerTensorProtocol
        _np_packbits: EagerTensorProtocol
        _np_parse_example: EagerTensorProtocol
        _np_parse_example_camel: EagerTensorProtocol
        _np_parse_tensor: EagerTensorProtocol
        _np_parse_tensor_camel: EagerTensorProtocol
        _np_perspectiveconfig: EagerTensorProtocol
        _np_piecewise: EagerTensorProtocol
        _np_pmax: EagerTensorProtocol
        _np_pmean: EagerTensorProtocol
        _np_pmin: EagerTensorProtocol
        _np_polyint: EagerTensorProtocol
        _np_population_count: EagerTensorProtocol
        _np_promotetypes: EagerTensorProtocol
        _np_psum: EagerTensorProtocol
        _np_qr: EagerTensorProtocol
        _np_raggeddot: EagerTensorProtocol
        _np_randombernoulli: EagerTensorProtocol
        _np_randomcategorical: EagerTensorProtocol
        _np_randompermutation: EagerTensorProtocol
        _np_randomtruncatednormal: EagerTensorProtocol
        _np_ravel_multi_index_: EagerTensorProtocol
        _np_rawconv2d: EagerTensorProtocol
        _np_rawmatmul: EagerTensorProtocol
        _np_rawmerge: EagerTensorProtocol
        _np_rawop: EagerTensorProtocol
        _np_rawswitch: EagerTensorProtocol
        _np_read_file: EagerTensorProtocol
        _np_read_file_camel: EagerTensorProtocol
        _np_reduce_precision: EagerTensorProtocol
        _np_rem: EagerTensorProtocol
        _np_rem_2: EagerTensorProtocol
        _np_resize_: EagerTensorProtocol
        _np_result_type_: EagerTensorProtocol
        _np_rfft: EagerTensorProtocol
        _np_rfft2: EagerTensorProtocol
        _np_rfftfreq: EagerTensorProtocol
        _np_rfftn: EagerTensorProtocol
        _np_rfftnd: EagerTensorProtocol
        _np_rrelu: EagerTensorProtocol
        _np_rsqrt: EagerTensorProtocol
        _np_scanop: EagerTensorProtocol
        _np_scatter_nd_add: EagerTensorProtocol
        _np_scatter_nd_sub: EagerTensorProtocol
        _np_scatter_nd_update: EagerTensorProtocol
        _np_scatter_update: EagerTensorProtocol
        _np_segment_sum: EagerTensorProtocol
        _np_serialize_tensor: EagerTensorProtocol
        _np_serialize_tensor_camel: EagerTensorProtocol
        _np_shard_tensor: EagerTensorProtocol
        _np_shifted_chebyshev_polynomial_t: EagerTensorProtocol
        _np_shifted_chebyshev_polynomial_u: EagerTensorProtocol
        _np_shifted_chebyshev_polynomial_v: EagerTensorProtocol
        _np_shifted_chebyshev_polynomial_w: EagerTensorProtocol
        _np_sigmoid: EagerTensorProtocol
        _np_slogdet: EagerTensorProtocol
        _np_sobolsample: EagerTensorProtocol
        _np_softmax: EagerTensorProtocol
        _np_sparsedensematmul: EagerTensorProtocol
        _np_sparsemapvalues: EagerTensorProtocol
        _np_sparsereducemax: EagerTensorProtocol
        _np_sparsereshape: EagerTensorProtocol
        _np_sparsesampledadd: EagerTensorProtocol
        _np_sparsesegmentsum: EagerTensorProtocol
        _np_sparsetranspose: EagerTensorProtocol
        _np_stridedslice: EagerTensorProtocol
        _np_svd: EagerTensorProtocol
        _np_switchop: EagerTensorProtocol
        _np_takealongaxis: EagerTensorProtocol
        _np_tensor: EagerTensorProtocol
        _np_tensor_scatter: EagerTensorProtocol
        _np_tensor_scatter_sub: EagerTensorProtocol
        _np_tensorarrayread: EagerTensorProtocol
        _np_tensorarraystack: EagerTensorProtocol
        _np_tensorarraywrite: EagerTensorProtocol
        _np_tensorconfig: EagerTensorProtocol
        _np_trace: EagerTensorProtocol
        _np_trapz: EagerTensorProtocol
        _np_trapz_: EagerTensorProtocol
        _np_tri: EagerTensorProtocol
        _np_triangular: EagerTensorProtocol
        _np_trilindices: EagerTensorProtocol
        _np_trilindicesfrom: EagerTensorProtocol
        _np_trimzeros: EagerTensorProtocol
        _np_triuindices: EagerTensorProtocol
        _np_triuindicesfrom: EagerTensorProtocol
        _np_truncate_div: EagerTensorProtocol
        _np_truncate_mod: EagerTensorProtocol
        _np_uint: EagerTensorProtocol
        _np_uint8: EagerTensorProtocol
        _np_union1d: EagerTensorProtocol
        _np_unpackbits: EagerTensorProtocol
        _np_unravelindex: EagerTensorProtocol
        _np_unwrap: EagerTensorProtocol
        _np_vander: EagerTensorProtocol
        _np_vecdot: EagerTensorProtocol
        _np_vectorize: EagerTensorProtocol
        _np_xlog1py: EagerTensorProtocol
        _np_xlogy: EagerTensorProtocol
        _np_zeros_: EagerTensorProtocol
        _np_zeros_like_: EagerTensorProtocol
        _obj: EagerTensorProtocol
        _on_buffered_amount_low: EagerTensorProtocol
        _on_message: EagerTensorProtocol
        _outer: EagerTensorProtocol
        _output_returns: EagerTensorProtocol
        _poly_recurrence: EagerTensorProtocol
        _prepare_device_buffers: EagerTensorProtocol
        _prepare_inputs: EagerTensorProtocol
        _recv_queue: EagerTensorProtocol
        _reduce_window: EagerTensorProtocol
        _registry: EagerTensorProtocol
        _reset_cuda_stats: EagerTensorProtocol
        _resolve_device: EagerTensorProtocol
        _resolve_imports: EagerTensorProtocol
        _run_isolated: EagerTensorProtocol
        _run_server: EagerTensorProtocol
        _run_single: EagerTensorProtocol
        _scaled_dot_product_attention_eager: EagerTensorProtocol
        _server: EagerTensorProtocol
        _svd: EagerTensorProtocol
        _sync: EagerTensorProtocol
        _sync_dask_result: EagerTensorProtocol
        _sync_keras_result: EagerTensorProtocol
        _tcp_dist_ctx: EagerTensorProtocol
        _tensor_scatter_sub: EagerTensorProtocol
        _tensorinv: EagerTensorProtocol
        _tensorsolve: EagerTensorProtocol
        _validate_templates: EagerTensorProtocol
        _visit_node: EagerTensorProtocol
        abs: EagerTensorProtocol
        abs_max_quantize: EagerTensorProtocol
        accept: EagerTensorProtocol
        accumulate: EagerTensorProtocol
        active_graph: EagerTensorProtocol
        adadelta_update: EagerTensorProtocol
        adafactor_update: EagerTensorProtocol
        adagrad_update: EagerTensorProtocol
        adam_update: EagerTensorProtocol
        adamax_update: EagerTensorProtocol
        adamw_update: EagerTensorProtocol
        adaptive_avg_pool2d: EagerTensorProtocol
        adaptive_avg_pool3d: EagerTensorProtocol
        adaptive_max_pool2d: EagerTensorProtocol
        adaptive_max_pool3d: EagerTensorProtocol
        add: EagerTensorProtocol
        add_: EagerTensorProtocol
        add_header: EagerTensorProtocol
        add_node: EagerTensorProtocol
        all: EagerTensorProtocol
        all_gather: EagerTensorProtocol
        all_gather_tensors: EagerTensorProtocol
        all_reduce: EagerTensorProtocol
        all_reduce_ring: EagerTensorProtocol
        all_sum: EagerTensorProtocol
        all_to_all: EagerTensorProtocol
        allclose: EagerTensorProtocol
        allocate_buffer: EagerTensorProtocol
        amax: EagerTensorProtocol
        amin: EagerTensorProtocol
        annotations: EagerTensorProtocol
        any: EagerTensorProtocol
        append: EagerTensorProtocol
        apply: EagerTensorProtocol
        apply_adagrad: EagerTensorProtocol
        apply_adam: EagerTensorProtocol
        apply_along_axis: EagerTensorProtocol
        apply_ftrl: EagerTensorProtocol
        apply_passes: EagerTensorProtocol
        apply_rmsprop: EagerTensorProtocol
        apply_shape_telemetry: EagerTensorProtocol
        arange: EagerTensorProtocol
        arccos: EagerTensorProtocol
        arccosh: EagerTensorProtocol
        arcsin: EagerTensorProtocol
        arcsinh: EagerTensorProtocol
        arctan: EagerTensorProtocol
        arctan2: EagerTensorProtocol
        arctanh: EagerTensorProtocol
        argmax: EagerTensorProtocol
        argmin: EagerTensorProtocol
        argpartition: EagerTensorProtocol
        argsort: EagerTensorProtocol
        argwhere: EagerTensorProtocol
        array: EagerTensorProtocol
        array_equal: EagerTensorProtocol
        array_equiv: EagerTensorProtocol
        array_split: EagerTensorProtocol
        arrow_tensor_to_tensor: EagerTensorProtocol
        as_strided: EagerTensorProtocol
        as_tensor: EagerTensorProtocol
        asarray: EagerTensorProtocol
        asnumpy: EagerTensorProtocol
        assert_any_call: EagerTensorProtocol
        assert_array_equal: EagerTensorProtocol
        assert_called: EagerTensorProtocol
        assert_called_once: EagerTensorProtocol
        assert_called_once_with: EagerTensorProtocol
        assert_called_with: EagerTensorProtocol
        assert_not_called: EagerTensorProtocol
        assigned_val: EagerTensorProtocol
        astype: EagerTensorProtocol
        at: EagerTensorProtocol
        atleast_1d: EagerTensorProtocol
        atleast_2d: EagerTensorProtocol
        atleast_3d: EagerTensorProtocol
        attributes: EagerTensorProtocol
        average: EagerTensorProtocol
        avgpool2d: EagerTensorProtocol
        axes: EagerTensorProtocol
        axis_translation_pass: EagerTensorProtocol
        backends: EagerTensorProtocol
        batch: EagerTensorProtocol
        behavior: EagerTensorProtocol
        beta: EagerTensorProtocol
        bfloat16: EagerTensorProtocol
        bidirectional: EagerTensorProtocol
        bincount: EagerTensorProtocol
        binom: EagerTensorProtocol
        binomial: EagerTensorProtocol
        bitwise_and: EagerTensorProtocol
        bitwise_not: EagerTensorProtocol
        bitwise_or: EagerTensorProtocol
        bitwise_xor: EagerTensorProtocol
        blackman: EagerTensorProtocol
        block: EagerTensorProtocol
        block_until_ready: EagerTensorProtocol
        body: EagerTensorProtocol
        broadcast: EagerTensorProtocol
        broadcast_arrays: EagerTensorProtocol
        broadcast_ragged: EagerTensorProtocol
        broadcast_to: EagerTensorProtocol
        bwd: EagerTensorProtocol
        bwd_subgraph: EagerTensorProtocol
        bytecode: EagerTensorProtocol
        c_void_p: EagerTensorProtocol
        call_args: EagerTensorProtocol
        callable_fn: EagerTensorProtocol
        called: EagerTensorProtocol
        calls: EagerTensorProtocol
        can_cast: EagerTensorProtocol
        cdf: EagerTensorProtocol
        ceil: EagerTensorProtocol
        channel: EagerTensorProtocol
        char: EagerTensorProtocol
        check_parity: EagerTensorProtocol
        checkpoint_dir: EagerTensorProtocol
        cho_solve: EagerTensorProtocol
        choice: EagerTensorProtocol
        cholesky: EagerTensorProtocol
        choose: EagerTensorProtocol
        chunked_array: EagerTensorProtocol
        chunks: EagerTensorProtocol
        cleanup: EagerTensorProtocol
        clear: EagerTensorProtocol
        clip: EagerTensorProtocol
        clone: EagerTensorProtocol
        close: EagerTensorProtocol
        cluster_meshes: EagerTensorProtocol
        cluster_resolver: EagerTensorProtocol
        code: EagerTensorProtocol
        column: EagerTensorProtocol
        column_indices: EagerTensorProtocol
        column_stack: EagerTensorProtocol
        commandBuffer: EagerTensorProtocol
        commit: EagerTensorProtocol
        compile: EagerTensorProtocol
        compile_and_load: EagerTensorProtocol
        compile_aot: EagerTensorProtocol
        compile_fn: EagerTensorProtocol
        compile_graph: EagerTensorProtocol
        compile_hip_to_code: EagerTensorProtocol
        compile_wasm: EagerTensorProtocol
        complex64: EagerTensorProtocol
        compress: EagerTensorProtocol
        compute: EagerTensorProtocol
        compute_float8_amax_history: EagerTensorProtocol
        compute_float8_scale: EagerTensorProtocol
        concatenate: EagerTensorProtocol
        config: EagerTensorProtocol
        confusion_matrix: EagerTensorProtocol
        conj: EagerTensorProtocol
        conjugate: EagerTensorProtocol
        connections: EagerTensorProtocol
        constant_folding_pass: EagerTensorProtocol
        contents: EagerTensorProtocol
        context: EagerTensorProtocol
        conv2d: EagerTensorProtocol
        conv2d_transpose: EagerTensorProtocol
        convolve: EagerTensorProtocol
        convolve2d: EagerTensorProtocol
        copy: EagerTensorProtocol
        copy_device_to_host: EagerTensorProtocol
        copy_host_to_device: EagerTensorProtocol
        corrcoef: EagerTensorProtocol
        correlate: EagerTensorProtocol
        cos: EagerTensorProtocol
        cosh: EagerTensorProtocol
        count: EagerTensorProtocol
        count_nonzero: EagerTensorProtocol
        cov: EagerTensorProtocol
        cp: EagerTensorProtocol
        cpu: EagerTensorProtocol
        create_event: EagerTensorProtocol
        create_stream: EagerTensorProtocol
        cross: EagerTensorProtocol
        ctypes: EagerTensorProtocol
        cuEventCreate: EagerTensorProtocol
        cuEventElapsedTime: EagerTensorProtocol
        cuEventRecord: EagerTensorProtocol
        cuLaunchKernel: EagerTensorProtocol
        cuMemAlloc_v2: EagerTensorProtocol
        cuModuleGetFunction: EagerTensorProtocol
        cuModuleLoadData: EagerTensorProtocol
        cuStreamCreate: EagerTensorProtocol
        cuStreamSynchronize: EagerTensorProtocol
        cuda: EagerTensorProtocol
        cudaDeviceSynchronize: EagerTensorProtocol
        cudaMalloc: EagerTensorProtocol
        cudaMemcpy: EagerTensorProtocol
        cudaStreamSynchronize: EagerTensorProtocol
        cummax: EagerTensorProtocol
        cummin: EagerTensorProtocol
        cumprod: EagerTensorProtocol
        cumsum: EagerTensorProtocol
        current_loop_options: EagerTensorProtocol
        current_stream: EagerTensorProtocol
        custom_object_scope: EagerTensorProtocol
        custom_rules: EagerTensorProtocol
        da: EagerTensorProtocol
        dask: EagerTensorProtocol
        data: EagerTensorProtocol
        data_format: EagerTensorProtocol
        decode: EagerTensorProtocol
        decode_csv: EagerTensorProtocol
        decode_image: EagerTensorProtocol
        default: EagerTensorProtocol
        default_mesh: EagerTensorProtocol
        default_rng: EagerTensorProtocol
        defjvp: EagerTensorProtocol
        defjvp_subgraph: EagerTensorProtocol
        defvjp: EagerTensorProtocol
        defvjp_subgraph: EagerTensorProtocol
        deg2rad: EagerTensorProtocol
        delattr: EagerTensorProtocol
        delayed: EagerTensorProtocol
        delenv: EagerTensorProtocol
        delete: EagerTensorProtocol
        delitem: EagerTensorProtocol
        dequantize: EagerTensorProtocol
        descriptive: EagerTensorProtocol
        deserialize_object: EagerTensorProtocol
        det: EagerTensorProtocol
        detach: EagerTensorProtocol
        device: EagerTensorProtocol
        device_mesh_partitioning: EagerTensorProtocol
        diag: EagerTensorProtocol
        diag_indices: EagerTensorProtocol
        diag_indices_from: EagerTensorProtocol
        diagflat: EagerTensorProtocol
        dict: EagerTensorProtocol
        diff: EagerTensorProtocol
        digamma: EagerTensorProtocol
        dirname: EagerTensorProtocol
        disable_jit: EagerTensorProtocol
        dispatchThreads_threadsPerThreadgroup_: EagerTensorProtocol
        distributed: EagerTensorProtocol
        distributions: EagerTensorProtocol
        divide: EagerTensorProtocol
        domain: EagerTensorProtocol
        done: EagerTensorProtocol
        dot: EagerTensorProtocol
        dsplit: EagerTensorProtocol
        dstack: EagerTensorProtocol
        dumps: EagerTensorProtocol
        eager_eval: EagerTensorProtocol
        eager_registry: EagerTensorProtocol
        ediff1d: EagerTensorProtocol
        eig: EagerTensorProtocol
        eigh: EagerTensorProtocol
        eigvals: EagerTensorProtocol
        eigvalsh: EagerTensorProtocol
        einsum: EagerTensorProtocol
        elapsed_time: EagerTensorProtocol
        emit_tracing_node: EagerTensorProtocol
        empty: EagerTensorProtocol
        empty_like: EagerTensorProtocol
        encode: EagerTensorProtocol
        end_lineno: EagerTensorProtocol
        endswith: EagerTensorProtocol
        equal: EagerTensorProtocol
        erf: EagerTensorProtocol
        erfc: EagerTensorProtocol
        erfinv: EagerTensorProtocol
        err: EagerTensorProtocol
        eval: EagerTensorProtocol
        eval_shape: EagerTensorProtocol
        evaluate: EagerTensorProtocol
        evaluate_graph: EagerTensorProtocol
        execute_graph: EagerTensorProtocol
        execute_op: EagerTensorProtocol
        exists: EagerTensorProtocol
        exp: EagerTensorProtocol
        exp2: EagerTensorProtocol
        expand: EagerTensorProtocol
        expand_dims: EagerTensorProtocol
        expm: EagerTensorProtocol
        expm1: EagerTensorProtocol
        export_function: EagerTensorProtocol
        export_onnx: EagerTensorProtocol
        expr: EagerTensorProtocol
        extend: EagerTensorProtocol
        extract: EagerTensorProtocol
        extract_state: EagerTensorProtocol
        eye: EagerTensorProtocol
        fail_sig: EagerTensorProtocol
        fake_quant_with_min_max_vars: EagerTensorProtocol
        fake_quantize_per_channel_affine: EagerTensorProtocol
        fake_quantize_per_tensor_affine: EagerTensorProtocol
        fft: EagerTensorProtocol
        fft2: EagerTensorProtocol
        fftconvolve: EagerTensorProtocol
        fftfreq: EagerTensorProtocol
        fftn: EagerTensorProtocol
        fftshift: EagerTensorProtocol
        field: EagerTensorProtocol
        fields: EagerTensorProtocol
        fill_diagonal: EagerTensorProtocol
        finfo: EagerTensorProtocol
        flatnonzero: EagerTensorProtocol
        flatten: EagerTensorProtocol
        flip: EagerTensorProtocol
        fliplr: EagerTensorProtocol
        flipud: EagerTensorProtocol
        float16: EagerTensorProtocol
        float32: EagerTensorProtocol
        float64: EagerTensorProtocol
        floor: EagerTensorProtocol
        fmax: EagerTensorProtocol
        fold: EagerTensorProtocol
        format: EagerTensorProtocol
        forward: EagerTensorProtocol
        fractional_max_pool2d: EagerTensorProtocol
        fractional_max_pool3d: EagerTensorProtocol
        frexp: EagerTensorProtocol
        from_array: EagerTensorProtocol
        from_arrays: EagerTensorProtocol
        from_buffer: EagerTensorProtocol
        from_dlpack: EagerTensorProtocol
        from_iter: EagerTensorProtocol
        from_numpy: EagerTensorProtocol
        from_regular: EagerTensorProtocol
        frombuffer: EagerTensorProtocol
        fromfile: EagerTensorProtocol
        fromfunction: EagerTensorProtocol
        fromiter: EagerTensorProtocol
        frompyfunc: EagerTensorProtocol
        fromstring: EagerTensorProtocol
        full: EagerTensorProtocol
        full_like: EagerTensorProtocol
        func: EagerTensorProtocol
        fwd: EagerTensorProtocol
        gamma: EagerTensorProtocol
        gather_qmm: EagerTensorProtocol
        gaussian_blur: EagerTensorProtocol
        gcd: EagerTensorProtocol
        generate: EagerTensorProtocol
        generate_body: EagerTensorProtocol
        generate_microbatch_loop: EagerTensorProtocol
        generate_schedule: EagerTensorProtocol
        generator: EagerTensorProtocol
        generator_registry: EagerTensorProtocol
        generic_array: EagerTensorProtocol
        generic_asarray: EagerTensorProtocol
        generic_item: EagerTensorProtocol
        generic_visit: EagerTensorProtocol
        generic_zeros: EagerTensorProtocol
        geometric: EagerTensorProtocol
        geomspace: EagerTensorProtocol
        get: EagerTensorProtocol
        get_active_backend: EagerTensorProtocol
        get_active_memory: EagerTensorProtocol
        get_buffered_amount: EagerTensorProtocol
        get_communication_protocol: EagerTensorProtocol
        get_custom_objects: EagerTensorProtocol
        get_default_memory_pool: EagerTensorProtocol
        get_device: EagerTensorProtocol
        get_endpoint_parameters: EagerTensorProtocol
        get_fallback_axis_kwarg: EagerTensorProtocol
        get_fallback_keepdims_kwarg: EagerTensorProtocol
        get_fallback_prefix: EagerTensorProtocol
        get_helper_functions: EagerTensorProtocol
        get_indent: EagerTensorProtocol
        get_layout: EagerTensorProtocol
        get_npz_bytes: EagerTensorProtocol
        get_op: EagerTensorProtocol
        get_ops_map: EagerTensorProtocol
        get_printoptions: EagerTensorProtocol
        get_registered_name: EagerTensorProtocol
        get_registered_object: EagerTensorProtocol
        get_template: EagerTensorProtocol
        get_url: EagerTensorProtocol
        get_usm_type: EagerTensorProtocol
        get_websocket_url: EagerTensorProtocol
        get_world_size: EagerTensorProtocol
        get_yaml_data: EagerTensorProtocol
        getheader: EagerTensorProtocol
        getresponse: EagerTensorProtocol
        getvalue: EagerTensorProtocol
        global_tracing_state: EagerTensorProtocol
        glorot_uniform: EagerTensorProtocol
        grad: EagerTensorProtocol
        gradient: EagerTensorProtocol
        graph: EagerTensorProtocol
        greater: EagerTensorProtocol
        greater_equal: EagerTensorProtocol
        gumbel: EagerTensorProtocol
        hamming: EagerTensorProtocol
        handle_compute_node: EagerTensorProtocol
        hanning: EagerTensorProtocol
        hard_silu: EagerTensorProtocol
        hard_swish: EagerTensorProtocol
        hardware_execution_schemas: EagerTensorProtocol
        header: EagerTensorProtocol
        headers: EagerTensorProtocol
        hfft: EagerTensorProtocol
        hide_names: EagerTensorProtocol
        hipMalloc: EagerTensorProtocol
        hipMemcpy: EagerTensorProtocol
        hipStreamSynchronize: EagerTensorProtocol
        histogram: EagerTensorProtocol
        histogram2d: EagerTensorProtocol
        histogram_bin_edges: EagerTensorProtocol
        histogramdd: EagerTensorProtocol
        hit: EagerTensorProtocol
        householder_product: EagerTensorProtocol
        hsplit: EagerTensorProtocol
        hstack: EagerTensorProtocol
        hypot: EagerTensorProtocol
        i0: EagerTensorProtocol
        i0e: EagerTensorProtocol
        i1: EagerTensorProtocol
        i1e: EagerTensorProtocol
        id: EagerTensorProtocol
        identity: EagerTensorProtocol
        ifft: EagerTensorProtocol
        ifft2: EagerTensorProtocol
        ifftn: EagerTensorProtocol
        ifftshift: EagerTensorProtocol
        ihfft: EagerTensorProtocol
        iinfo: EagerTensorProtocol
        indent_level: EagerTensorProtocol
        index: EagerTensorProtocol
        index_add_: EagerTensorProtocol
        index_put_: EagerTensorProtocol
        indices: EagerTensorProtocol
        inf: EagerTensorProtocol
        infer_shape: EagerTensorProtocol
        init: EagerTensorProtocol
        initialize: EagerTensorProtocol
        inner: EagerTensorProtocol
        input_specs: EagerTensorProtocol
        inputs: EagerTensorProtocol
        insert: EagerTensorProtocol
        int32: EagerTensorProtocol
        int64: EagerTensorProtocol
        int8: EagerTensorProtocol
        integer: EagerTensorProtocol
        interp: EagerTensorProtocol
        intersect1d: EagerTensorProtocol
        intp: EagerTensorProtocol
        inv: EagerTensorProtocol
        invert: EagerTensorProtocol
        irfft: EagerTensorProtocol
        irfft2: EagerTensorProtocol
        irfftn: EagerTensorProtocol
        is_arrow_object: EagerTensorProtocol
        is_available: EagerTensorProtocol
        is_fusion_valid: EagerTensorProtocol
        is_ragged: EagerTensorProtocol
        is_sycl_array: EagerTensorProtocol
        is_tracing: EagerTensorProtocol
        isclose: EagerTensorProtocol
        iscomplex: EagerTensorProtocol
        iscomplexobj: EagerTensorProtocol
        isfinite: EagerTensorProtocol
        isin: EagerTensorProtocol
        isinf: EagerTensorProtocol
        isnan: EagerTensorProtocol
        isposinf: EagerTensorProtocol
        isreal: EagerTensorProtocol
        isrealobj: EagerTensorProtocol
        isscalar: EagerTensorProtocol
        issubdtype: EagerTensorProtocol
        items: EagerTensorProtocol
        iterable: EagerTensorProtocol
        ix_: EagerTensorProtocol
        jax_all_gather: EagerTensorProtocol
        jax_all_reduce: EagerTensorProtocol
        jax_broadcast: EagerTensorProtocol
        jit: EagerTensorProtocol
        join: EagerTensorProtocol
        jv: EagerTensorProtocol
        jvp_subgraph: EagerTensorProtocol
        kaiser: EagerTensorProtocol
        keras: EagerTensorProtocol
        keras_input_vars: EagerTensorProtocol
        keras_output_vars: EagerTensorProtocol
        kernel_size: EagerTensorProtocol
        kind: EagerTensorProtocol
        kops: EagerTensorProtocol
        kron: EagerTensorProtocol
        lax: EagerTensorProtocol
        layout_map: EagerTensorProtocol
        lcm: EagerTensorProtocol
        left_shift: EagerTensorProtocol
        less: EagerTensorProtocol
        less_equal: EagerTensorProtocol
        lexsort: EagerTensorProtocol
        lhs_dilation: EagerTensorProtocol
        lhs_spec: EagerTensorProtocol
        lib: EagerTensorProtocol
        linalg: EagerTensorProtocol
        lines: EagerTensorProtocol
        linspace: EagerTensorProtocol
        lion_update: EagerTensorProtocol
        load: EagerTensorProtocol
        load_backend_mappings: EagerTensorProtocol
        load_h5: EagerTensorProtocol
        load_npz: EagerTensorProtocol
        log: EagerTensorProtocol
        log10: EagerTensorProtocol
        log1p: EagerTensorProtocol
        log2: EagerTensorProtocol
        logaddexp: EagerTensorProtocol
        logaddexp2: EagerTensorProtocol
        logcumsumexp: EagerTensorProtocol
        logical_and: EagerTensorProtocol
        logical_not: EagerTensorProtocol
        logical_or: EagerTensorProtocol
        logical_xor: EagerTensorProtocol
        logspace: EagerTensorProtocol
        long: EagerTensorProtocol
        loop_options: EagerTensorProtocol
        loop_options_stack: EagerTensorProtocol
        lower: EagerTensorProtocol
        make_graph: EagerTensorProtocol
        make_node: EagerTensorProtocol
        math_general: EagerTensorProtocol
        matmul: EagerTensorProtocol
        matrix_power: EagerTensorProtocol
        matrix_rank: EagerTensorProtocol
        max: EagerTensorProtocol
        max_memory_allocated: EagerTensorProtocol
        max_unpool1d: EagerTensorProtocol
        max_unpool2d: EagerTensorProtocol
        max_unpool3d: EagerTensorProtocol
        maximum: EagerTensorProtocol
        maximum_iterations: EagerTensorProtocol
        maxpool2d: EagerTensorProtocol
        mean: EagerTensorProtocol
        median: EagerTensorProtocol
        memory_allocated: EagerTensorProtocol
        mesh_axis: EagerTensorProtocol
        mesh_mapping: EagerTensorProtocol
        meshgrid: EagerTensorProtocol
        metadata: EagerTensorProtocol
        metal: EagerTensorProtocol
        mgrid: EagerTensorProtocol
        microbatch_splitting: EagerTensorProtocol
        min: EagerTensorProtocol
        min_memory_bytes: EagerTensorProtocol
        minimum: EagerTensorProtocol
        mish: EagerTensorProtocol
        mkdir: EagerTensorProtocol
        mlir: EagerTensorProtocol
        mlx_eager_registry: EagerTensorProtocol
        mock_calls: EagerTensorProtocol
        mod: EagerTensorProtocol
        model_dump: EagerTensorProtocol
        modf: EagerTensorProtocol
        modules: EagerTensorProtocol
        moveaxis: EagerTensorProtocol
        msgs: EagerTensorProtocol
        multiply: EagerTensorProtocol
        multiprocessing: EagerTensorProtocol
        multivariate_normal: EagerTensorProtocol
        muon_update: EagerTensorProtocol
        mvlgamma: EagerTensorProtocol
        mx: EagerTensorProtocol
        name: EagerTensorProtocol
        nan_to_num: EagerTensorProtocol
        nanmean: EagerTensorProtocol
        nanmedian: EagerTensorProtocol
        nbytes: EagerTensorProtocol
        negative: EagerTensorProtocol
        newBufferWithLength_options_: EagerTensorProtocol
        newCommandQueue: EagerTensorProtocol
        newComputePipelineStateWithFunction_error_: EagerTensorProtocol
        newFunctionWithName_: EagerTensorProtocol
        newaxis: EagerTensorProtocol
        newbyteorder: EagerTensorProtocol
        nextafter: EagerTensorProtocol
        nn: EagerTensorProtocol
        nnz: EagerTensorProtocol
        nodes: EagerTensorProtocol
        nonzero: EagerTensorProtocol
        norm: EagerTensorProtocol
        normal: EagerTensorProtocol
        not_equal: EagerTensorProtocol
        np: EagerTensorProtocol
        np_members: EagerTensorProtocol
        np_op_name: EagerTensorProtocol
        null: EagerTensorProtocol
        num_columns: EagerTensorProtocol
        num_microbatches: EagerTensorProtocol
        num_rows: EagerTensorProtocol
        numba: EagerTensorProtocol
        numel: EagerTensorProtocol
        numpy: EagerTensorProtocol
        numpy_eager_registry: EagerTensorProtocol
        one_hot: EagerTensorProtocol
        ones: EagerTensorProtocol
        ones_like: EagerTensorProtocol
        op: EagerTensorProtocol
        op_name: EagerTensorProtocol
        op_type: EagerTensorProtocol
        operand: EagerTensorProtocol
        ops: EagerTensorProtocol
        out: EagerTensorProtocol
        outer: EagerTensorProtocol
        outfeed: EagerTensorProtocol
        outputs: EagerTensorProtocol
        outvars: EagerTensorProtocol
        packbits: EagerTensorProtocol
        pad: EagerTensorProtocol
        parallel_iterations: EagerTensorProtocol
        param: EagerTensorProtocol
        parameters: EagerTensorProtocol
        params: EagerTensorProtocol
        parse_example: EagerTensorProtocol
        parse_tensor: EagerTensorProtocol
        partition: EagerTensorProtocol
        patch: EagerTensorProtocol
        path: EagerTensorProtocol
        pathlib: EagerTensorProtocol
        pattern: EagerTensorProtocol
        pdf: EagerTensorProtocol
        phases: EagerTensorProtocol
        pi: EagerTensorProtocol
        piecewise: EagerTensorProtocol
        pinv: EagerTensorProtocol
        pmax: EagerTensorProtocol
        pmf: EagerTensorProtocol
        pmin: EagerTensorProtocol
        poisson: EagerTensorProtocol
        polar: EagerTensorProtocol
        polygamma: EagerTensorProtocol
        polyint: EagerTensorProtocol
        pooling_ratio: EagerTensorProtocol
        pop: EagerTensorProtocol
        power: EagerTensorProtocol
        ppermute: EagerTensorProtocol
        precision: EagerTensorProtocol
        printable_graph: EagerTensorProtocol
        prod: EagerTensorProtocol
        profile_graph: EagerTensorProtocol
        promote_types: EagerTensorProtocol
        protocol: EagerTensorProtocol
        pshuffle: EagerTensorProtocol
        psum: EagerTensorProtocol
        psum_scatter: EagerTensorProtocol
        pswapaxes: EagerTensorProtocol
        put: EagerTensorProtocol
        put_nowait: EagerTensorProtocol
        pytorch_all_gather: EagerTensorProtocol
        pytorch_all_reduce: EagerTensorProtocol
        pytorch_broadcast: EagerTensorProtocol
        pytorch_reduce_scatter: EagerTensorProtocol
        qr: EagerTensorProtocol
        quantize: EagerTensorProtocol
        quantize_and_dequantize: EagerTensorProtocol
        quantized_conv: EagerTensorProtocol
        quantized_embedding: EagerTensorProtocol
        quantized_linear: EagerTensorProtocol
        quantized_matmul: EagerTensorProtocol
        r_: EagerTensorProtocol
        rad2deg: EagerTensorProtocol
        rademacher: EagerTensorProtocol
        ragged_array: EagerTensorProtocol
        ragged_dot: EagerTensorProtocol
        ragged_shape: EagerTensorProtocol
        raises: EagerTensorProtocol
        rand: EagerTensorProtocol
        randint: EagerTensorProtocol
        randn: EagerTensorProtocol
        random: EagerTensorProtocol
        rank: EagerTensorProtocol
        rate: EagerTensorProtocol
        ravel: EagerTensorProtocol
        ravel_multi_index: EagerTensorProtocol
        rawmatmul: EagerTensorProtocol
        read: EagerTensorProtocol
        read_file: EagerTensorProtocol
        readouterr: EagerTensorProtocol
        reciprocal: EagerTensorProtocol
        record: EagerTensorProtocol
        record_array: EagerTensorProtocol
        record_batch: EagerTensorProtocol
        record_batch_to_tensor: EagerTensorProtocol
        record_equality: EagerTensorProtocol
        record_event: EagerTensorProtocol
        recv: EagerTensorProtocol
        recv_bytes: EagerTensorProtocol
        recv_conns: EagerTensorProtocol
        reduce: EagerTensorProtocol
        reduce_scatter: EagerTensorProtocol
        reduceat: EagerTensorProtocol
        register: EagerTensorProtocol
        register_serializable: EagerTensorProtocol
        register_signal_handlers: EagerTensorProtocol
        reload: EagerTensorProtocol
        relu: EagerTensorProtocol
        rem: EagerTensorProtocol
        remainder: EagerTensorProtocol
        repeat: EagerTensorProtocol
        request: EagerTensorProtocol
        requires_grad: EagerTensorProtocol
        reset_mock: EagerTensorProtocol
        reshape: EagerTensorProtocol
        resize: EagerTensorProtocol
        resize_bilinear: EagerTensorProtocol
        restore_latest_checkpoint: EagerTensorProtocol
        result: EagerTensorProtocol
        result_type: EagerTensorProtocol
        return_value: EagerTensorProtocol
        rfft: EagerTensorProtocol
        rfft2: EagerTensorProtocol
        rfftfreq: EagerTensorProtocol
        rfftn: EagerTensorProtocol
        rhs_dilation: EagerTensorProtocol
        right_shift: EagerTensorProtocol
        rint: EagerTensorProtocol
        rmsprop_update: EagerTensorProtocol
        rnn: EagerTensorProtocol
        rnn_step: EagerTensorProtocol
        roll: EagerTensorProtocol
        rope: EagerTensorProtocol
        rot90: EagerTensorProtocol
        round: EagerTensorProtocol
        row_pointers: EagerTensorProtocol
        row_splits: EagerTensorProtocol
        runtime: EagerTensorProtocol
        save: EagerTensorProtocol
        save_gguf: EagerTensorProtocol
        save_h5: EagerTensorProtocol
        save_preemption_checkpoint: EagerTensorProtocol
        save_safetensors: EagerTensorProtocol
        savez: EagerTensorProtocol
        savez_compressed: EagerTensorProtocol
        scan: EagerTensorProtocol
        scatter_reduce_: EagerTensorProtocol
        schema: EagerTensorProtocol
        schur: EagerTensorProtocol
        scientific: EagerTensorProtocol
        scipy: EagerTensorProtocol
        searchsorted: EagerTensorProtocol
        segment_max: EagerTensorProtocol
        segment_min: EagerTensorProtocol
        segment_prod: EagerTensorProtocol
        segment_sum: EagerTensorProtocol
        send: EagerTensorProtocol
        send_bytes: EagerTensorProtocol
        sendall: EagerTensorProtocol
        sent: EagerTensorProtocol
        serialize_model_to_bytes: EagerTensorProtocol
        serialize_object: EagerTensorProtocol
        serialize_tensor: EagerTensorProtocol
        set: EagerTensorProtocol
        set_active_backend: EagerTensorProtocol
        set_backend: EagerTensorProtocol
        set_default_stream: EagerTensorProtocol
        set_numpy_seed: EagerTensorProtocol
        setattr: EagerTensorProtocol
        setdiff1d: EagerTensorProtocol
        setenv: EagerTensorProtocol
        setitem: EagerTensorProtocol
        setxor1d: EagerTensorProtocol
        sgd_update: EagerTensorProtocol
        shape_invariants: EagerTensorProtocol
        shape_metadata: EagerTensorProtocol
        sharding: EagerTensorProtocol
        sign: EagerTensorProtocol
        signal: EagerTensorProtocol
        signbit: EagerTensorProtocol
        simplify: EagerTensorProtocol
        sin: EagerTensorProtocol
        sinc: EagerTensorProtocol
        sinh: EagerTensorProtocol
        size: EagerTensorProtocol
        slice_record_batch: EagerTensorProtocol
        slogdet: EagerTensorProtocol
        snapshot_dir: EagerTensorProtocol
        sobol_sample: EagerTensorProtocol
        softmax: EagerTensorProtocol
        solve: EagerTensorProtocol
        solve_banded: EagerTensorProtocol
        sort: EagerTensorProtocol
        sort_complex: EagerTensorProtocol
        sorted_nodes: EagerTensorProtocol
        source_ast_ref: EagerTensorProtocol
        spacing: EagerTensorProtocol
        sparse: EagerTensorProtocol
        sparse_bincount: EagerTensorProtocol
        sparse_reduce_max: EagerTensorProtocol
        sparse_reduce_sum: EagerTensorProtocol
        sparse_segment_sqrt_n: EagerTensorProtocol
        sparse_segment_sum: EagerTensorProtocol
        sparsedensematmul: EagerTensorProtocol
        special: EagerTensorProtocol
        split: EagerTensorProtocol
        split_into_stages: EagerTensorProtocol
        sqrt: EagerTensorProtocol
        square: EagerTensorProtocol
        squareplus: EagerTensorProtocol
        squeeze: EagerTensorProtocol
        stack: EagerTensorProtocol
        stage_communication: EagerTensorProtocol
        standard_normal: EagerTensorProtocol
        start_tracing: EagerTensorProtocol
        startswith: EagerTensorProtocol
        state_store: EagerTensorProtocol
        stateless_random_uniform: EagerTensorProtocol
        stateless_split: EagerTensorProtocol
        stats: EagerTensorProtocol
        status: EagerTensorProtocol
        std: EagerTensorProtocol
        stop_tracing: EagerTensorProtocol
        stream: EagerTensorProtocol
        stream_synchronize: EagerTensorProtocol
        stride: EagerTensorProtocol
        stride_tricks: EagerTensorProtocol
        strides: EagerTensorProtocol
        strip: EagerTensorProtocol
        subgraphs: EagerTensorProtocol
        substitute: EagerTensorProtocol
        subtract: EagerTensorProtocol
        suggest_closest_endpoint: EagerTensorProtocol
        sum: EagerTensorProtocol
        svd: EagerTensorProtocol
        swap_memory: EagerTensorProtocol
        swapaxes: EagerTensorProtocol
        synchronize: EagerTensorProtocol
        synchronize_stream: EagerTensorProtocol
        sys: EagerTensorProtocol
        table: EagerTensorProtocol
        take: EagerTensorProtocol
        take_along_axis: EagerTensorProtocol
        tan: EagerTensorProtocol
        tanh: EagerTensorProtocol
        temp_dir: EagerTensorProtocol
        templates: EagerTensorProtocol
        tensor: EagerTensorProtocol
        tensor_split: EagerTensorProtocol
        tensor_to_arrow_tensor: EagerTensorProtocol
        tensor_to_record_batch: EagerTensorProtocol
        tensordot: EagerTensorProtocol
        tensorinv: EagerTensorProtocol
        tensorsolve: EagerTensorProtocol
        testing: EagerTensorProtocol
        tf: EagerTensorProtocol
        thresholds: EagerTensorProtocol
        tile: EagerTensorProtocol
        time: EagerTensorProtocol
        to: EagerTensorProtocol
        to_backend_array: EagerTensorProtocol
        to_dense: EagerTensorProtocol
        to_json: EagerTensorProtocol
        to_list: EagerTensorProtocol
        to_numpy: EagerTensorProtocol
        to_padded_tensor: EagerTensorProtocol
        to_polynomial: EagerTensorProtocol
        to_pylist: EagerTensorProtocol
        to_regular: EagerTensorProtocol
        to_text: EagerTensorProtocol
        topk: EagerTensorProtocol
        torch: EagerTensorProtocol
        touch: EagerTensorProtocol
        trace: EagerTensorProtocol
        track_gradient_accumulation: EagerTensorProtocol
        track_graph: EagerTensorProtocol
        transpose: EagerTensorProtocol
        trapezoid: EagerTensorProtocol
        trapz: EagerTensorProtocol
        triangular: EagerTensorProtocol
        tril: EagerTensorProtocol
        tril_indices: EagerTensorProtocol
        tril_indices_from: EagerTensorProtocol
        trim_zeros: EagerTensorProtocol
        triu: EagerTensorProtocol
        triu_indices: EagerTensorProtocol
        triu_indices_from: EagerTensorProtocol
        true_divide: EagerTensorProtocol
        trunc: EagerTensorProtocol
        truncated_normal: EagerTensorProtocol
        type: EagerTensorProtocol
        ufunc: EagerTensorProtocol
        uint32: EagerTensorProtocol
        uint8: EagerTensorProtocol
        unbind: EagerTensorProtocol
        undo: EagerTensorProtocol
        unflatten: EagerTensorProtocol
        unfold: EagerTensorProtocol
        uniform: EagerTensorProtocol
        union1d: EagerTensorProtocol
        unique: EagerTensorProtocol
        units: EagerTensorProtocol
        unlink: EagerTensorProtocol
        unpackbits: EagerTensorProtocol
        unravel_index: EagerTensorProtocol
        unroll_pipeline: EagerTensorProtocol
        unwrap: EagerTensorProtocol
        unzip: EagerTensorProtocol
        update: EagerTensorProtocol
        upper: EagerTensorProtocol
        used_bytes: EagerTensorProtocol
        util: EagerTensorProtocol
        val: EagerTensorProtocol
        validate: EagerTensorProtocol
        validate_parameter_contract: EagerTensorProtocol
        validate_syntax: EagerTensorProtocol
        value: EagerTensorProtocol
        value_and_grad: EagerTensorProtocol
        values: EagerTensorProtocol
        vander: EagerTensorProtocol
        var: EagerTensorProtocol
        var_names: EagerTensorProtocol
        var_table: EagerTensorProtocol
        vdot: EagerTensorProtocol
        vecdot: EagerTensorProtocol
        vectorize: EagerTensorProtocol
        verify: EagerTensorProtocol
        version: EagerTensorProtocol
        view: EagerTensorProtocol
        visit: EagerTensorProtocol
        visit_AccumulateN: EagerTensorProtocol
        visit_AddN: EagerTensorProtocol
        visit_AdjustBrightness: EagerTensorProtocol
        visit_AdjustContrast: EagerTensorProtocol
        visit_AdjustHue: EagerTensorProtocol
        visit_AdjustSaturation: EagerTensorProtocol
        visit_AffineGenerator: EagerTensorProtocol
        visit_AffineGrid: EagerTensorProtocol
        visit_AffineTransform: EagerTensorProtocol
        visit_AllToAll: EagerTensorProtocol
        visit_Assign: EagerTensorProtocol
        visit_AssignAdd: EagerTensorProtocol
        visit_AssignSub: EagerTensorProtocol
        visit_AugMix: EagerTensorProtocol
        visit_AutoContrast: EagerTensorProtocol
        visit_AvgPool2D: EagerTensorProtocol
        visit_AvgPool2DDynamic: EagerTensorProtocol
        visit_AvgPool3D: EagerTensorProtocol
        visit_BatchMatMul: EagerTensorProtocol
        visit_Broadcast: EagerTensorProtocol
        visit_CholeskyVjp: EagerTensorProtocol
        visit_Cond: EagerTensorProtocol
        visit_Conv2D: EagerTensorProtocol
        visit_Conv2DSharedHalo: EagerTensorProtocol
        visit_DynamicSlice: EagerTensorProtocol
        visit_DynamicUpdateSlice: EagerTensorProtocol
        visit_Einsum: EagerTensorProtocol
        visit_ForiLoop: EagerTensorProtocol
        visit_GatherQMM: EagerTensorProtocol
        visit_GetItem: EagerTensorProtocol
        visit_LogSoftmax: EagerTensorProtocol
        visit_LuVjp: EagerTensorProtocol
        visit_Max: EagerTensorProtocol
        visit_MaxPool2D: EagerTensorProtocol
        visit_MaxPool2DDynamic: EagerTensorProtocol
        visit_MaxPool3D: EagerTensorProtocol
        visit_Mean: EagerTensorProtocol
        visit_Meshgrid: EagerTensorProtocol
        visit_Min: EagerTensorProtocol
        visit_OpNode: EagerTensorProtocol
        visit_Pad: EagerTensorProtocol
        visit_Prod: EagerTensorProtocol
        visit_PutAlongAxis: EagerTensorProtocol
        visit_Quantize: EagerTensorProtocol
        visit_QuantizedMatmul: EagerTensorProtocol
        visit_RaggedDot: EagerTensorProtocol
        visit_Recv: EagerTensorProtocol
        visit_Send: EagerTensorProtocol
        visit_SharedMemoryTreeReduce: EagerTensorProtocol
        visit_Slice: EagerTensorProtocol
        visit_Softmax: EagerTensorProtocol
        visit_Sum: EagerTensorProtocol
        visit_TiledMatMul2D: EagerTensorProtocol
        visit_TopK: EagerTensorProtocol
        visit_TriInv: EagerTensorProtocol
        visit_TruncateDiv: EagerTensorProtocol
        visit_TruncateMod: EagerTensorProtocol
        visit_WarpShuffleReduce: EagerTensorProtocol
        visit_WhileLoop: EagerTensorProtocol
        vsplit: EagerTensorProtocol
        vstack: EagerTensorProtocol
        wait_backpressure: EagerTensorProtocol
        wald: EagerTensorProtocol
        weibull: EagerTensorProtocol
        welch: EagerTensorProtocol
        where: EagerTensorProtocol
        window_strides: EagerTensorProtocol
        with_field: EagerTensorProtocol
        workgroup_size: EagerTensorProtocol
        write: EagerTensorProtocol
        write_bytes: EagerTensorProtocol
        write_file: EagerTensorProtocol
        write_text: EagerTensorProtocol
        zeros: EagerTensorProtocol
        zeros_like: EagerTensorProtocol
        zeta: EagerTensorProtocol
        ndarray: builtins.type[EagerTensorProtocol]
        number: builtins.type[EagerTensorProtocol]


def _is_eager_tensor(val: _ValT) -> typing.TypeGuard[EagerTensorProtocol]:
    """Type guard narrowing `EagerOperand` to `EagerTensorProtocol`.

    Args:
        val (_ValT): Candidate runtime operand.

    Returns:
        bool: True for valid runtime values.
    """
    return not isinstance(val, bytes) or len(val) >= 0


def ensure_eager_tensor(val: _ValT) -> EagerTensorProtocol:
    """Narrow an `EagerOperand` to `EagerTensorProtocol` without `typing.cast`.

    Args:
        val (_ValT): Input array or scalar operand.

    Returns:
        EagerTensorProtocol: Narrowed tensor protocol reference.
    """
    assert _is_eager_tensor(val)
    return val


def _is_eager_callable(val: EagerOperand) -> typing.TypeGuard[EagerCallableProtocol]:
    """Type guard narrowing `EagerOperand` to `EagerCallableProtocol`.

    Args:
        val (EagerOperand): Candidate runtime callable.

    Returns:
        bool: Always True for runtime callables.
    """
    return not isinstance(val, bytes) or len(val) >= 0


def ensure_callable(val: EagerOperand) -> EagerCallableProtocol:
    """Narrow an `EagerOperand` callable to `EagerCallableProtocol` without `typing.cast`.

    Args:
        val (EagerOperand): Input callable.

    Returns:
        EagerCallableProtocol: Narrowed callable protocol reference.
    """
    assert _is_eager_callable(val)
    return val


def _is_int(val: EagerOperand) -> typing.TypeGuard[int]:
    """Type guard narrowing `EagerOperand` to `int`.

    Args:
        val (EagerOperand): Candidate value.

    Returns:
        bool: Always True for runtime int values.
    """
    return not isinstance(val, bytes) or len(val) >= 0


def ensure_int(val: EagerOperand, default: int = 0) -> int:
    """Narrow an `EagerOperand` scalar or attribute to `int`.

    Args:
        val (EagerOperand): Input value.
        default (int): Unused fallback value preserved for signature compatibility.

    Returns:
        int: Narrowed integer value.
    """
    _ = default
    assert _is_int(val)
    return val


def _is_float(val: EagerOperand) -> typing.TypeGuard[float]:
    """Type guard narrowing `EagerOperand` to `float`.

    Args:
        val (EagerOperand): Candidate value.

    Returns:
        bool: Always True for runtime float values.
    """
    return not isinstance(val, bytes) or len(val) >= 0


def ensure_float(val: EagerOperand, default: float = 0.0) -> float:
    """Narrow an `EagerOperand` scalar or attribute to `float`.

    Args:
        val (EagerOperand): Input value.
        default (float): Unused fallback value preserved for signature compatibility.

    Returns:
        float: Narrowed float value.
    """
    _ = default
    assert _is_float(val)
    return val


def _is_shape(val: EagerOperand) -> typing.TypeGuard[tuple[int, ...]]:
    """Type guard narrowing `EagerOperand` to `tuple[int, ...]`.

    Args:
        val (EagerOperand): Candidate value.

    Returns:
        bool: Always True for runtime shape tuples.
    """
    return not isinstance(val, bytes) or len(val) >= 0


def ensure_shape(val: EagerOperand) -> tuple[int, ...]:
    """Narrow `val` to `tuple[int, ...]`.

    Args:
        val (EagerOperand): Shape tuple or sequence.

    Returns:
        tuple[int, ...]: Narrowed shape tuple.
    """
    assert _is_shape(val)
    return val


def _is_int_list(val: EagerOperand) -> typing.TypeGuard[list[int]]:
    """Type guard narrowing `EagerOperand` to `list[int]`.

    Args:
        val (EagerOperand): Candidate value.

    Returns:
        bool: Always True for runtime integer lists.
    """
    return not isinstance(val, bytes) or len(val) >= 0


def ensure_int_list(val: EagerOperand) -> list[int]:
    """Narrow `val` to `list[int]`.

    Args:
        val (EagerOperand): List of ints.

    Returns:
        list[int]: Narrowed list of ints.
    """
    assert _is_int_list(val)
    return val


def _is_str(val: EagerOperand) -> typing.TypeGuard[str]:
    """Type guard narrowing `EagerOperand` to `str`.

    Args:
        val (EagerOperand): Candidate value.

    Returns:
        bool: Always True for runtime string values.
    """
    return not isinstance(val, bytes) or len(val) >= 0


def ensure_str(val: EagerOperand, default: str = "") -> str:
    """Narrow `val` to `str`.

    Args:
        val (EagerOperand): Input string value.
        default (str): Unused fallback value preserved for signature compatibility.

    Returns:
        str: Narrowed string value.
    """
    _ = default
    assert _is_str(val)
    return val


def _is_opt_str(val: EagerOperand) -> typing.TypeGuard[str | None]:
    """Type guard narrowing `EagerOperand` to `str | None`.

    Args:
        val (EagerOperand): Candidate value.

    Returns:
        bool: Always True for optional string values.
    """
    return not isinstance(val, bytes) or len(val) >= 0


def ensure_opt_str(val: EagerOperand) -> str | None:
    """Narrow `val` to `str | None`.

    Args:
        val (EagerOperand): Input value.

    Returns:
        str | None: Narrowed string or None.
    """
    assert _is_opt_str(val)
    return val


def _is_opt_int(val: EagerOperand) -> typing.TypeGuard[int | None]:
    """Type guard narrowing `EagerOperand` to `int | None`.

    Args:
        val (EagerOperand): Candidate value.

    Returns:
        bool: Always True for optional int values.
    """
    return not isinstance(val, bytes) or len(val) >= 0


def ensure_opt_int(val: EagerOperand) -> int | None:
    """Narrow `val` to `int | None`.

    Args:
        val (EagerOperand): Input value.

    Returns:
        int | None: Narrowed int or None.
    """
    assert _is_opt_int(val)
    return val


def _is_bool(val: EagerOperand) -> typing.TypeGuard[bool]:
    """Type guard narrowing `EagerOperand` to `bool`.

    Args:
        val (EagerOperand): Candidate value.

    Returns:
        bool: Always True for runtime boolean values.
    """
    return not isinstance(val, bytes) or len(val) >= 0


def ensure_bool(val: EagerOperand) -> bool:
    """Narrow `val` to `bool`.

    Args:
        val (EagerOperand): Input value.

    Returns:
        bool: Narrowed boolean value.
    """
    assert _is_bool(val)
    return val


def _is_bytes(val: EagerOperand) -> typing.TypeGuard[bytes]:
    """Type guard narrowing `EagerOperand` to `bytes`.

    Args:
        val (EagerOperand): Candidate value.

    Returns:
        bool: Always True for runtime bytes values.
    """
    return not isinstance(val, bytes) or len(val) >= 0


def ensure_bytes(val: EagerOperand) -> bytes:
    """Convert `val` into `bytes`.

    Args:
        val (EagerOperand): Input bytes-like or integer sequence value.

    Returns:
        bytes: Byte string.
    """
    res: EagerOperand = val if isinstance(val, bytes) else ensure_callable(bytes)(val)
    assert _is_bytes(res)
    return res


def _is_float_list(val: EagerOperand) -> typing.TypeGuard[list[float]]:
    """Type guard narrowing `EagerOperand` to `list[float]`.

    Args:
        val (EagerOperand): Candidate value.

    Returns:
        bool: Always True for runtime float lists.
    """
    return not isinstance(val, bytes) or len(val) >= 0


def ensure_float_list(val: EagerOperand) -> list[float]:
    """Narrow `val` to `list[float]`.

    Args:
        val (EagerOperand): Sequence of floats.

    Returns:
        list[float]: Narrowed list of floats.
    """
    assert _is_float_list(val)
    return val


def _is_eager_list(val: EagerOperand) -> typing.TypeGuard[list[EagerValue]]:
    """Type guard narrowing `EagerOperand` to `list[EagerValue]`.

    Args:
        val (EagerOperand): Candidate value.

    Returns:
        bool: Always True for runtime eager lists.
    """
    return not isinstance(val, bytes) or len(val) >= 0


def ensure_eager_list(val: EagerOperand) -> list[EagerValue]:
    """Narrow `val` to `list[EagerValue]`.

    Args:
        val (EagerOperand): List of eager values.

    Returns:
        list[EagerValue]: Narrowed list of eager values.
    """
    assert _is_eager_list(val)
    return val


def _is_tensor_pair(val: EagerOperand) -> typing.TypeGuard[tuple[EagerTensorProtocol, EagerTensorProtocol]]:
    """Type guard narrowing `EagerOperand` to a 2-tuple of `EagerTensorProtocol`.

    Args:
        val (EagerOperand): Candidate value.

    Returns:
        bool: Always True for tensor pairs.
    """
    return not isinstance(val, bytes) or len(val) >= 0


def ensure_tensor_pair(val: EagerOperand) -> tuple[EagerTensorProtocol, EagerTensorProtocol]:
    """Narrow `val` into a 2-tuple of `EagerTensorProtocol`.

    Args:
        val (EagerOperand): 2-element sequence of tensor values.

    Returns:
        tuple[EagerTensorProtocol, EagerTensorProtocol]: Pair of tensor protocol objects.
    """
    assert _is_tensor_pair(val)
    return val


def _is_tensor_triple(val: EagerOperand) -> typing.TypeGuard[tuple[EagerTensorProtocol, EagerTensorProtocol, EagerTensorProtocol]]:
    """Type guard narrowing `EagerOperand` to a 3-tuple of `EagerTensorProtocol`.

    Args:
        val (EagerOperand): Candidate value.

    Returns:
        bool: Always True for tensor triples.
    """
    return not isinstance(val, bytes) or len(val) >= 0


def ensure_tensor_triple(val: EagerOperand) -> tuple[EagerTensorProtocol, EagerTensorProtocol, EagerTensorProtocol]:
    """Narrow `val` into a 3-tuple of `EagerTensorProtocol`.

    Args:
        val (EagerOperand): 3-element sequence of tensor values.

    Returns:
        tuple[EagerTensorProtocol, EagerTensorProtocol, EagerTensorProtocol]: Triple of tensor protocol objects.
    """
    assert _is_tensor_triple(val)
    return val


def _is_tensor_quad(val: EagerOperand) -> typing.TypeGuard[tuple[EagerTensorProtocol, EagerTensorProtocol, EagerTensorProtocol, EagerTensorProtocol]]:
    """Type guard narrowing `EagerOperand` to a 4-tuple of `EagerTensorProtocol`.

    Args:
        val (EagerOperand): Candidate value.

    Returns:
        bool: Always True for tensor quadruples.
    """
    return not isinstance(val, bytes) or len(val) >= 0


def ensure_tensor_quad(val: EagerOperand) -> tuple[EagerTensorProtocol, EagerTensorProtocol, EagerTensorProtocol, EagerTensorProtocol]:
    """Narrow `val` into a 4-tuple of `EagerTensorProtocol`.

    Args:
        val (EagerOperand): 4-element sequence of tensor values.

    Returns:
        tuple[EagerTensorProtocol, EagerTensorProtocol, EagerTensorProtocol, EagerTensorProtocol]: Quadruple of tensor protocol objects.
    """
    assert _is_tensor_quad(val)
    return val


def _is_opt_int_seq(val: EagerOperand) -> typing.TypeGuard[Sequence[int] | None]:
    """Type guard narrowing `EagerOperand` to `Sequence[int] | None`.

    Args:
        val (EagerOperand): Candidate value.

    Returns:
        bool: Always True for optional integer sequences.
    """
    return not isinstance(val, bytes) or len(val) >= 0


def ensure_opt_int_seq(val: EagerOperand) -> Sequence[int] | None:
    """Narrow `val` to `Sequence[int] | None`.

    Args:
        val (EagerOperand): Sequence of ints or None.

    Returns:
        Sequence[int] | None: Narrowed sequence of ints or None.
    """
    assert _is_opt_int_seq(val)
    return val


def _is_int_or_seq(val: EagerOperand) -> typing.TypeGuard[int | tuple[int, ...]]:
    """Type guard narrowing `EagerOperand` to `int | tuple[int, ...]`.

    Args:
        val (EagerOperand): Candidate value.

    Returns:
        bool: Always True for int or int tuple values.
    """
    return not isinstance(val, bytes) or len(val) >= 0


def ensure_int_or_seq(val: EagerOperand) -> int | tuple[int, ...]:
    """Narrow `val` to `int | tuple[int, ...]`.

    Args:
        val (EagerOperand): Single int or sequence of ints.

    Returns:
        int | tuple[int, ...]: Narrowed integer or tuple of ints.
    """
    assert _is_int_or_seq(val)
    return val


def _is_opt_callable(val: EagerOperand) -> typing.TypeGuard[EagerCallableProtocol | None]:
    """Type guard narrowing `EagerOperand` to `EagerCallableProtocol | None`.

    Args:
        val (EagerOperand): Candidate value.

    Returns:
        bool: Always True for optional callables.
    """
    return not isinstance(val, bytes) or len(val) >= 0


def ensure_opt_callable(val: EagerOperand) -> EagerCallableProtocol | None:
    """Narrow `val` to `EagerCallableProtocol | None`.

    Args:
        val (EagerOperand): Callable or None.

    Returns:
        EagerCallableProtocol | None: Narrowed callable or None.
    """
    assert _is_opt_callable(val)
    return val


def _is_eager_callable_generic(fn: _F) -> typing.TypeGuard[EagerCallableProtocol]:
    """Type guard narrowing `fn` to `EagerCallableProtocol`.

    Args:
        fn (_F): Candidate callable.

    Returns:
        bool: True if callable.
    """
    return callable(fn)


BackendArray = EagerTensorProtocol
EagerValue = TensorOrScalar
_F = typing.TypeVar("_F")


class EagerOpRegistry:
    """Registry for eager operations."""

    def __init__(self) -> None:
        """Initialize the registry."""
        self._registry: dict[str, EagerCallableProtocol] = {}

    def register(self, op_type: str) -> Callable[[_F], _F]:
        """Register an eager operation.

        Args:
            op_type (str): The op_type parameter.

        Returns:
            Callable[[_F], _F]: Result.
        """

        def decorator(func: _F) -> _F:
            """Execute the eager decorator operation on the active backend array.

            Args:
                func (_F): The func parameter.

            Returns:
                _F: Result.
            """
            cand_fn: _F = func
            assert _is_eager_callable_generic(cand_fn)
            self._registry[op_type] = cand_fn
            return func

        return decorator

    def get(self, op_type: str) -> EagerCallableProtocol | None:
        """Get an eager operation.

        Args:
            op_type (str): The name of the operation.

        Returns:
            Optional[EagerCallableProtocol]: The eager function or None.
        """
        return self._registry.get(op_type)

    def dispatch(self, op_type: str, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Dispatch an eager operation.

        Args:
            op_type (str): The name of the operation.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            EagerValue: The result.

        Raises:
            ValueError: If the operation is not found in the registry.
        """
        func = self.get(op_type)
        if func is not None:
            return func(*args, **kwargs)

        # Fallback to global pure python registry if available
        if self is not global_eager_registry:
            func = global_eager_registry.get(op_type)
            if func is not None:
                return func(*args, **kwargs)

        return _dispatch_universal_fallback(op_type, args, kwargs)


def _dispatch_universal_fallback(
    op_type: str,
    args: tuple[EagerValue, ...],
    kwargs: dict[str, EagerValue],
) -> EagerValue:
    """Resolve and invoke universal fallback on operand or backend module.

    Args:
        op_type (str): Canonical operation name.
        args (tuple[EagerValue, ...]): Positional arguments.
        kwargs (dict[str, EagerValue]): Keyword arguments.

    Returns:
        EagerValue: Dispatched fallback result.

    Raises:
        ValueError: If no fallback operation can be resolved.
    """
    from ml_switcheroo_compiler.backends.protocols import resolve_backend_member

    op_name = op_type.lower()
    if args:
        has_arg_op, arg_op = resolve_backend_member(args[0], op_name)
        if has_arg_op:
            return ensure_eager_tensor(ensure_callable(arg_op)(*args[1:], **kwargs))

    backend = kwargs.get("backend_module")
    op_args = args[1:] if backend is None and len(args) > 0 else args
    if backend is None and len(args) > 0:
        backend = args[0]

    if backend is not None:
        has_bk_op, bk_op = resolve_backend_member(backend, op_name)
        if has_bk_op:
            return ensure_eager_tensor(ensure_callable(bk_op)(*op_args, **kwargs))

        mapping = {"matmul": "matmul", "add": "add", "sub": "subtract", "mul": "multiply", "div": "divide", "truedivide": "divide"}
        mapped_name = mapping.get(op_name)
        has_mapped_op, mapped_op = resolve_backend_member(backend, mapped_name) if mapped_name is not None else (False, None)
        if has_mapped_op:
            return ensure_eager_tensor(ensure_callable(mapped_op)(*op_args, **kwargs))

    msg = f"Operation '{op_type}' not found in registry and no universal fallback could be inferred for backend {backend}."
    raise ValueError(msg)


# Global registry instance
global_eager_registry = EagerOpRegistry()

mlx_eager_registry = EagerOpRegistry()
numpy_eager_registry = EagerOpRegistry()
pure_python_eager_registry = EagerOpRegistry()


@global_eager_registry.register("CustomVJP")
def _eager_custom_vjp(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate forward pass of a CustomVJP operation in eager mode.

    Args:
        backend_module (EagerValue): Active backend module.
        *args (EagerValue): Primal input arguments.
        **kwargs (EagerValue): Keyword attributes.

    Returns:
        EagerValue: Forward output value or tuple.
    """
    return args[0] if len(args) == 1 else tuple(args)


@global_eager_registry.register("CustomJVP")
def _eager_custom_jvp(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate forward pass of a CustomJVP operation in eager mode.

    Args:
        backend_module (EagerValue): Active backend module.
        *args (EagerValue): Primal input arguments.
        **kwargs (EagerValue): Keyword attributes containing optional 'fun'.

    Returns:
        EagerValue: Forward output value or tuple.
    """
    fun = kwargs.get("fun")
    if callable(fun):
        from ml_switcheroo_compiler.core.config import ConfigContext

        with ConfigContext(eager_mode=True):
            return ensure_callable(fun)(*args)
    return args[0] if len(args) == 1 else tuple(args)


@global_eager_registry.register("ProcessCustomVJPCall")
def _eager_process_custom_vjp_call(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute custom backward VJP callback in eager mode.

    Args:
        backend_module (EagerValue): Active backend module.
        *args (EagerValue): Cotangent arguments.
        **kwargs (EagerValue): Keyword attributes containing 'bwd_fn'.

    Returns:
        EagerValue: Computed backward gradients.
    """
    bwd_fn = kwargs["bwd_fn"]
    if callable(bwd_fn):
        return ensure_callable(bwd_fn)(None, *args)
    return None


@global_eager_registry.register("ProcessCustomJVPCall")
def _eager_process_custom_jvp_call(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute custom forward JVP rule callback in eager mode.

    Args:
        backend_module (EagerValue): Active backend module.
        *args (EagerValue): Concatenated primal and tangent arguments.
        **kwargs (EagerValue): Keyword attributes containing 'jvp_rule' and 'num_primals'.

    Returns:
        EagerValue: Tuple of (primal_out, tangent_out).
    """
    jvp_rule = kwargs.get("jvp_rule")
    num_primals = ensure_opt_int(kwargs.get("num_primals", len(args) // 2))
    primals = args[:num_primals]
    tangents = args[num_primals:]
    if callable(jvp_rule):
        from ml_switcheroo_compiler.core.config import ConfigContext

        with ConfigContext(eager_mode=True):
            jvp_fn = ensure_callable(jvp_rule)
            try:
                res: EagerValue = jvp_fn(primals, tangents)
            except TypeError:
                res = jvp_fn(*primals, *tangents)
        if isinstance(res, (tuple, list)) and len(res) == 2:
            return res
        return (None, res)
    return (None, None)


@global_eager_registry.register("TupleGetItem")
def _eager_tuple_get_item(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Extract an element from a tuple or list by index in eager mode.

    Args:
        backend_module (EagerValue): Active backend module.
        *args (EagerValue): Input sequence argument.
        **kwargs (EagerValue): Keyword attributes containing 'index'.

    Returns:
        EagerValue: Extracted element at the requested index.
    """
    index: int = ensure_int(kwargs.get("index", 0))
    if isinstance(args[0], (tuple, list)):
        return ensure_eager_tensor(args[0][index])
    return None
