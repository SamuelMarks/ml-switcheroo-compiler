"""Ahead-of-time (AOT) and JIT function compilation pipeline."""

from __future__ import annotations

import logging
import typing
from collections.abc import Iterable

from ml_switcheroo_ir import LogicalGraph

from ml_switcheroo_compiler.backends.registry import BackendRegistry
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import AttrValue, TensorOrScalar
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.tracing.builder import TracingNodeBuilder
from ml_switcheroo_compiler.tracing.state import global_tracing_state
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor, TracerTape
from ml_switcheroo_compiler.transforms.pass_manager import PassManager

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.core.shape import VariadicCallableProtocol

_COMPILATION_CACHE: dict[str, VariadicCallableProtocol] = {}


def _fallback_eager(
    fn: TensorOrScalar,
    args: tuple[TensorOrScalar, ...],
    kw: dict[str, AttrValue],
) -> TensorOrScalar:
    """Execute `fn` in eager mode as a fallback when compilation is bypassed.

    Args:
        fn (TensorOrScalar): Target function to evaluate.
        args (tuple[TensorOrScalar, ...]): Positional call arguments.
        kw (dict[str, AttrValue]): Keyword call arguments.

    Returns:
        TensorOrScalar: Result of eager evaluation.
    """
    was_eager = config.eager_mode
    config.eager_mode = True
    try:
        call_fn: VariadicCallableProtocol = fn  # type: ignore[assignment]
        return call_fn(*args, **kw)
    finally:
        config.eager_mode = was_eager


def _build_signature_key(fn: TensorOrScalar, backend: str, args: TensorOrScalar) -> str:
    """Build a deterministic cache key from function identity, target backend, and argument specs.

    Args:
        fn (TensorOrScalar): Function being compiled.
        backend (str): Target backend identifier.
        args (TensorOrScalar): Positional arguments tuple.

    Returns:
        str: Signature cache key string.
    """
    sig_parts = []
    args_iter: Iterable[TensorOrScalar] = args if isinstance(args, Iterable) else ()
    for a in args_iter:
        if hasattr(a, "shape") and hasattr(a, "dtype"):
            sig_parts.append(f"T({getattr(a, 'shape', ())},{getattr(a, 'dtype', '')})")
        else:
            sig_parts.append(f"S({type(a).__name__})")
    code_sig = getattr(fn, "__code__", None)
    return f"{getattr(fn, '__name__', '')}_{id(fn)}_{id(code_sig)}_{backend}_" + "_".join(sig_parts)


def _prepare_proxy_args(args: tuple[TensorOrScalar, ...]) -> list[TensorOrScalar]:
    """Wrap tensor arguments in `ProxyTensor` instances and emit `Input` nodes in the active graph.

    Args:
        args (tuple[TensorOrScalar, ...]): Raw call positional arguments.

    Returns:
        list[TensorOrScalar]: Proxy-wrapped arguments ready for symbolic tracing.
    """
    proxy_args: list[TensorOrScalar] = []
    for i, a in enumerate(args):
        if hasattr(a, "shape") and hasattr(a, "dtype"):
            arg_id = f"arg_{i}"
            shape = getattr(a, "shape", ())
            raw_dt = getattr(a, "dtype", DType.Float32)
            if isinstance(raw_dt, DType):
                dtype = raw_dt
                dt_str = raw_dt.value
            else:
                dt_str = getattr(raw_dt, "value", str(raw_dt))
                if dt_str.startswith("DType."):
                    dt_str = dt_str.split(".", 1)[1]
                if dt_str in DType._value2member_map_:
                    dtype = DType(dt_str)
                else:
                    dtype = DType.Float32
                    dt_str = "float32"
            proxy = ProxyTensor(id=arg_id, shape=shape, dtype=dt_str)
            device = getattr(a, "device", "cpu")
            proxy_args.append(Tensor(proxy, TensorConfig(proxy.shape, dtype, device)))
            TracingNodeBuilder.create_tracing_logical_node("Input", [], {}, shape)

            last_node = list(getattr(global_tracing_state.active_graph, "nodes").values())[-1]
            old_id = last_node.id
            del getattr(global_tracing_state.active_graph, "nodes")[old_id]
            last_node.id = arg_id
            getattr(global_tracing_state.active_graph, "nodes")[arg_id] = last_node
        else:
            proxy_args.append(a)
    return proxy_args


def _capture_outputs(out: TensorOrScalar) -> None:
    """Record output tensor node IDs in the active tracing graph.

    Args:
        out (TensorOrScalar): Traced function return value or sequence of tensors.
    """
    if isinstance(out, Tensor):
        out_id, _ = TracingNodeBuilder.extract_from_tensor(out)
        out_node_id = TracingNodeBuilder.create_tracing_logical_node("Output", [out_id], {}, getattr(out, "shape", ()))
        getattr(global_tracing_state.active_graph, "outputs").append(out_node_id)
    elif isinstance(out, (list, tuple)):
        out_ids = []
        for x in out:
            if isinstance(x, Tensor):
                oid, _ = TracingNodeBuilder.extract_from_tensor(x)
                out_ids.append(oid)
        if out_ids:
            out_node_id = TracingNodeBuilder.create_tracing_logical_node("Output", out_ids, {}, ())
            getattr(global_tracing_state.active_graph, "outputs").append(out_node_id)


def _get_namespace(backend: str, generator_cls: TensorOrScalar) -> dict[str, TensorOrScalar]:
    """Construct the execution namespace dictionary for evaluating generated backend code.

    Args:
        backend (str): Target backend name.
        generator_cls (TensorOrScalar): Backend generator class.

    Returns:
        dict[str, TensorOrScalar]: Namespace dictionary mapping the backend module.
    """
    namespace: dict[str, TensorOrScalar] = {}
    if hasattr(generator_cls, "get_module"):
        namespace[backend] = getattr(generator_cls, "get_module")()
    elif backend == "numpy":
        from ml_switcheroo_compiler.backends.registry import BackendRegistry

        namespace[backend] = BackendRegistry.get("numpy").get_module()
    return namespace


def _run_optimization_pipeline(graph: LogicalGraph) -> LogicalGraph:
    """Run the standard middle-end optimization pass pipeline on a traced graph.

    Args:
        graph (LogicalGraph): Unoptimized traced logical graph.

    Returns:
        LogicalGraph: Optimized logical graph.
    """
    from ml_switcheroo_compiler.transforms.passes.constant_folding import constant_folding_pass
    from ml_switcheroo_compiler.transforms.passes.cse import cse_pass
    from ml_switcheroo_compiler.transforms.passes.dce import dce_pass
    from ml_switcheroo_compiler.transforms.passes.dtype_inference import dtype_inference_pass
    from ml_switcheroo_compiler.transforms.passes.rematerialization import rematerialization_pass
    from ml_switcheroo_compiler.transforms.passes.shape_inference import shape_inference_pass

    pm = PassManager()
    pm.add_pass(shape_inference_pass)
    pm.add_pass(dtype_inference_pass)
    pm.add_pass(constant_folding_pass)
    pm.add_pass(cse_pass)
    pm.add_pass(rematerialization_pass)
    pm.add_pass(dce_pass)
    return pm.run(graph)


def _trace_function_graph(
    fn: TensorOrScalar,
    args: tuple[TensorOrScalar, ...],
    kw: dict[str, AttrValue],
) -> LogicalGraph | None:
    """Trace a function call with proxy arguments and optimize the resulting graph.

    Args:
        fn (TensorOrScalar): Target function to trace.
        args (tuple[TensorOrScalar, ...]): Positional arguments.
        kw (dict[str, AttrValue]): Keyword arguments.

    Returns:
        LogicalGraph | None: Optimized graph if tracing succeeded with outputs, else None.
    """
    tape = TracerTape()
    tape.start_tracing()
    proxy_args = _prepare_proxy_args(args)

    try:
        call_fn: VariadicCallableProtocol = fn  # type: ignore[assignment]
        out = call_fn(*proxy_args, **kw)
        _capture_outputs(out)
    except Exception:
        tape.stop_tracing()
        return None

    graph = tape.stop_tracing()
    if not graph or not graph.outputs:
        return None
    return _run_optimization_pipeline(graph)


def _try_static_compile_aot(
    generator_cls: TensorOrScalar,
    graph: LogicalGraph,
    kwargs: dict[str, AttrValue],
) -> VariadicCallableProtocol | None:
    """Attempt static or class-level `compile_aot` on the generator class.

    Args:
        generator_cls (TensorOrScalar): Backend generator class.
        graph (LogicalGraph): Optimized logical graph.
        kwargs (dict[str, AttrValue]): Compilation options.

    Returns:
        VariadicCallableProtocol | None: Compiled callable if available, else None.
    """
    import inspect

    from ml_switcheroo_compiler.backends.base_generator import BaseGenerator

    if not hasattr(generator_cls, "compile_aot"):
        return None
    try:
        static_compile = inspect.getattr_static(generator_cls, "compile_aot")
        if isinstance(static_compile, (staticmethod, classmethod)) or not (isinstance(generator_cls, type) and issubclass(generator_cls, BaseGenerator)):
            compiled_fn = getattr(generator_cls, "compile_aot")(graph, **kwargs)
            if callable(compiled_fn):
                res_fn: VariadicCallableProtocol = compiled_fn
                return res_fn
    except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
        logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
    return None


def _build_generated_entrypoint(
    code: str,
    namespace: dict[str, TensorOrScalar],
    backend: str,
    strict: bool,
) -> VariadicCallableProtocol | None:
    """Execute generated source code and wrap its entrypoint callable.

    Args:
        code (str): Generated Python code string.
        namespace (dict[str, TensorOrScalar]): Execution namespace dictionary.
        backend (str): Target backend name.
        strict (bool): Whether to raise `CompilationError` when no entrypoint exists.

    Returns:
        VariadicCallableProtocol | None: Wrapped entrypoint callable, or None if not found and non-strict.

    Raises:
        CompilationError: If strict mode is enabled and no executable entrypoint is found.
    """
    exec(code, namespace)
    if "apply_model" in namespace:

        def apply_wrapper(*w_args: TensorOrScalar, **w_kw: TensorOrScalar) -> TensorOrScalar:
            """Invoke the generated `apply_model` entrypoint.

            Args:
                *w_args (TensorOrScalar): Positional arguments.
                **w_kw (TensorOrScalar): Keyword arguments.

            Returns:
                TensorOrScalar: Evaluated model output.
            """
            entry_fn: VariadicCallableProtocol = namespace["apply_model"]  # type: ignore[assignment]
            return entry_fn({}, *w_args, **w_kw)

        return apply_wrapper

    if "evaluate" in namespace:

        def eval_wrapper(*w_args: TensorOrScalar, **w_kw: TensorOrScalar) -> TensorOrScalar:
            """Invoke the generated `evaluate` entrypoint.

            Args:
                *w_args (TensorOrScalar): Positional arguments.
                **w_kw (TensorOrScalar): Keyword arguments.

            Returns:
                TensorOrScalar: Evaluated graph output.
            """
            del w_kw
            numpy_args = [a.data if isinstance(a, Tensor) else a for a in w_args]
            entry_fn: VariadicCallableProtocol = namespace["evaluate"]  # type: ignore[assignment]
            return entry_fn(numpy_args)

        return eval_wrapper

    if strict:
        from ml_switcheroo_compiler.core.errors import CompilationError

        raise CompilationError(f"No executable entrypoint found in generated code for backend '{backend}'")
    return None


def _execute_generated_code(
    code: str,
    generator_cls: TensorOrScalar,
    backend: str,
    key: str,
    strict: bool,
    fn: TensorOrScalar,
    args: tuple[TensorOrScalar, ...],
    kw: dict[str, AttrValue],
) -> TensorOrScalar:
    """Build executable entrypoint from generated code and invoke with arguments.

    Args:
        code (str): Generated source code string.
        generator_cls (TensorOrScalar): Backend generator class.
        backend (str): Target backend name.
        key (str): Cache signature key.
        strict (bool): Whether strict compilation mode is enabled.
        fn (TensorOrScalar): Original function for eager fallback.
        args (tuple[TensorOrScalar, ...]): Call positional arguments.
        kw (dict[str, AttrValue]): Call keyword arguments.

    Returns:
        TensorOrScalar: Result of executing the compiled function or eager fallback.

    Raises:
        CompilationError: If strict mode is enabled and execution fails.
    """
    from ml_switcheroo_compiler.core.errors import CompilationError

    namespace = _get_namespace(backend, generator_cls)

    try:
        compiled_fn = _build_generated_entrypoint(code, namespace, backend, strict)
        if compiled_fn is None:
            return _fallback_eager(fn, args, kw)
        res = compiled_fn(*args, **kw)
        _COMPILATION_CACHE[key] = compiled_fn
        return res
    except CompilationError:
        raise
    except Exception as e:
        if strict:
            raise CompilationError(f"Execution of compiled code failed: {e}") from e
        return _fallback_eager(fn, args, kw)


def _compile_with_generator_instance(
    generator_cls: TensorOrScalar,
    graph: LogicalGraph,
    backend: str,
    key: str,
    kwargs: dict[str, AttrValue],
    fn: TensorOrScalar,
    args: tuple[TensorOrScalar, ...],
    kw: dict[str, AttrValue],
) -> TensorOrScalar:
    """Instantiate backend generator, compile or exec generated code, and invoke with args.

    Args:
        generator_cls (TensorOrScalar): Backend generator class.
        graph (LogicalGraph): Optimized logical graph.
        backend (str): Target backend name.
        key (str): Cache signature key.
        kwargs (dict[str, AttrValue]): Compilation options.
        fn (TensorOrScalar): Original function for eager fallback.
        args (tuple[TensorOrScalar, ...]): Call positional arguments.
        kw (dict[str, AttrValue]): Call keyword arguments.

    Returns:
        TensorOrScalar: Result of executing the compiled function or eager fallback.

    Raises:
        CompilationError: If strict mode is enabled and compilation or execution fails.
    """
    from ml_switcheroo_compiler.core.errors import CompilationError

    strict = bool(kwargs.get("strict", False))

    try:
        gen_ctor: VariadicCallableProtocol = generator_cls  # type: ignore[assignment]
        generator = gen_ctor(graph)
    except Exception as e:
        if strict:
            raise CompilationError(f"Failed to instantiate generator '{backend}': {e}") from e
        return _fallback_eager(fn, args, kw)

    if hasattr(generator, "compile_aot"):
        try:
            res_aot = getattr(generator, "compile_aot")(graph, **kwargs)
            if callable(res_aot):
                aot_fn: VariadicCallableProtocol = res_aot
                _COMPILATION_CACHE[key] = aot_fn
                return aot_fn(*args, **kw)
        except NotImplementedError:
            logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
        except Exception as e:
            if strict:
                raise CompilationError(f"AOT compilation failed: {e}") from e
            return _fallback_eager(fn, args, kw)

    code = str(getattr(generator, "generate")())
    return _execute_generated_code(code, generator_cls, backend, key, strict, fn, args, kw)


def compile_function(fn: TensorOrScalar, backend: str = "numpy", **kwargs: AttrValue) -> TensorOrScalar:
    """Compile a Python function into an optimized backend execution callable.

    Args:
        fn (TensorOrScalar): Function to trace and compile.
        backend (str): Target backend name.
        **kwargs (AttrValue): Additional compilation options (e.g., `strict=True`).

    Returns:
        TensorOrScalar: Compiled callable wrapper.
    """

    def compiled_wrapper(*args: TensorOrScalar, **kw: AttrValue) -> TensorOrScalar:
        """Trace and compile `fn` on first call for a given input signature.

        Args:
            *args (TensorOrScalar): Positional arguments for `fn`.
            **kw (AttrValue): Keyword arguments for `fn`.

        Returns:
            TensorOrScalar: Evaluated output of the compiled (or fallback eager) function.
        """
        key = _build_signature_key(fn, backend, args)
        if key in _COMPILATION_CACHE:
            return _COMPILATION_CACHE[key](*args, **kw)

        graph = _trace_function_graph(fn, args, kw)
        if graph is None:
            return _fallback_eager(fn, args, kw)

        try:
            generator_cls = BackendRegistry.get(backend)
        except ValueError as e:
            if kwargs.get("strict", False):
                from ml_switcheroo_compiler.core.errors import CompilationError

                raise CompilationError(f"Target backend '{backend}' not found: {e}") from e
            return _fallback_eager(fn, args, kw)

        static_fn = _try_static_compile_aot(generator_cls, graph, kwargs)
        if static_fn is not None:
            _COMPILATION_CACHE[key] = static_fn
            return static_fn(*args, **kw)

        return _compile_with_generator_instance(generator_cls, graph, backend, key, kwargs, fn, args, kw)

    return compiled_wrapper
