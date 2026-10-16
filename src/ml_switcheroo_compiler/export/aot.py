"""aot module."""

from __future__ import annotations

import logging
from typing import Optional, TypeVar, Union, cast  # noqa: F401

from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue

T = TypeVar("T")

# ruff: noqa: E402

from collections.abc import Callable  # noqa: F401

from ml_switcheroo_ir import LogicalGraph

from ml_switcheroo_compiler.backends.registry import BackendRegistry
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.tracing.builder import TracingNodeBuilder
from ml_switcheroo_compiler.tracing.state import global_tracing_state
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor, TracerTape
from ml_switcheroo_compiler.transforms.pass_manager import PassManager

# Cache for compiled artifacts
_COMPILATION_CACHE = {}


def _fallback_eager(
    fn: Callable[..., T],
    args: tuple[T, ...],
    kw: dict[str, T],
) -> T:
    """Execute function eagerly as a fallback.

    Args:
        fn: The original function.
        args: Positional arguments.
        kw: Keyword arguments.

    Returns: Tensor: Result of eager evaluation.
    """
    was_eager = config.eager_mode
    config.eager_mode = True
    try:
        return fn(*args, **kw)
    finally:
        config.eager_mode = was_eager


def _build_signature_key(fn: EagerValue, backend: str, args: EagerValue) -> str:
    """Build a cache key for the compilation signature.

    Args:
        fn (Callable): The fn parameter.
        backend (str): The backend parameter.
        args (tuple): The args parameter.

    Returns:
        str: Result.
    """
    sig_parts = []
    for a in getattr(args, "__iter__")():
        if hasattr(a, "shape") and hasattr(a, "dtype"):
            sig_parts.append(f"T({getattr(a, 'shape', ())},{getattr(a, 'dtype', '')})")
        else:
            sig_parts.append(f"S({type(a).__name__})")
    code_sig = getattr(fn, "__code__", None)
    return f"{getattr(fn, '__name__', '')}_{id(fn)}_{id(code_sig)}_{backend}_" + "_".join(sig_parts)


def _prepare_proxy_args(args: tuple[T, ...]) -> list[T | Tensor]:
    """Prepare proxy arguments for tracing.

    Args:
        args (tuple): The args parameter.

    Returns:
        list: Result.
    """
    proxy_args: list[T | Tensor] = []
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

            # Overwrite id to match
            last_node = list(getattr(global_tracing_state.active_graph, "nodes").values())[-1]
            old_id = last_node.id
            del getattr(global_tracing_state.active_graph, "nodes")[old_id]
            last_node.id = arg_id
            getattr(global_tracing_state.active_graph, "nodes")[arg_id] = last_node
        else:
            proxy_args.append(a)
    return proxy_args


def _capture_outputs(out: EagerValue) -> None:
    """Capture outputs in the active tracing graph.

    Args:
        out (T): The out parameter.
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


def _get_namespace(backend: str, generator_cls: type) -> EagerValue:
    """Get the namespace for executing generated code.

    Args:
        backend (str): The backend parameter.
        generator_cls (type): The generator_cls parameter.

    Returns:
        dict: Result.
    """
    namespace = {}
    if hasattr(generator_cls, "get_module"):
        namespace[backend] = generator_cls.get_module()
    elif backend == "numpy":
        from ml_switcheroo_compiler.backends.registry import BackendRegistry

        namespace[backend] = BackendRegistry.get("numpy").get_module()
    return namespace


def _run_optimization_pipeline(graph: LogicalGraph) -> LogicalGraph:
    """Run the standard optimization pass pipeline on a traced graph.

    Args:
        graph (LogicalGraph): The traced logical graph.

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


def _trace_function_graph(fn: Callable[..., T], args: tuple[T, ...], kw: dict[str, T]) -> Optional[LogicalGraph]:
    """Trace a function call with proxy arguments and optimize the resulting graph.

    Args:
        fn (Callable[..., T]): Target function to trace.
        args (tuple[T, ...]): Positional arguments.
        kw (dict[str, T]): Keyword arguments.

    Returns:
        Optional[LogicalGraph]: Optimized graph if tracing succeeded with outputs, else None.
    """
    tape = TracerTape()
    tape.start_tracing()
    proxy_args = _prepare_proxy_args(args)

    try:
        out = fn(*proxy_args, **kw)
        _capture_outputs(cast("EagerValue", out))
    except Exception:
        tape.stop_tracing()
        return None

    graph = tape.stop_tracing()
    if not graph or not graph.outputs:
        return None
    return _run_optimization_pipeline(graph)


def _try_static_compile_aot(
    generator_cls: type,
    graph: LogicalGraph,
    kwargs: dict[str, Union[bool, int, str, None]],
) -> Optional[Callable[..., T]]:
    """Attempt static or class-level compile_aot on the generator class.

    Args:
        generator_cls (type): Backend generator class.
        graph (LogicalGraph): Optimized logical graph.
        kwargs (dict[str, Union[bool, int, str, None]]): Compilation options.

    Returns:
        Optional[Callable[..., T]]: Compiled callable if available, else None.
    """
    import inspect

    from ml_switcheroo_compiler.backends.base_generator import BaseGenerator

    if not hasattr(generator_cls, "compile_aot"):
        return None
    try:
        static_compile = inspect.getattr_static(generator_cls, "compile_aot")
        if isinstance(static_compile, (staticmethod, classmethod)) or not (isinstance(generator_cls, type) and issubclass(generator_cls, BaseGenerator)):
            compiled_fn = generator_cls.compile_aot(graph, **kwargs)
            if callable(compiled_fn):
                return cast(Optional[Callable[..., T]], compiled_fn)
    except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
        logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
    return None


def _build_generated_entrypoint(
    code: str,
    namespace: dict[str, Callable[..., T]],
    backend: str,
    strict: bool,
) -> Optional[Callable[..., T]]:
    """Execute generated source code and wrap its entrypoint callable.

    Args:
        code (str): Generated Python code string.
        namespace (dict[str, Callable[..., T]]): Execution namespace dictionary.
        backend (str): Target backend name.
        strict (bool): Whether to raise CompilationError when no entrypoint exists.

    Returns:
        Optional[Callable[..., T]]: Wrapped entrypoint callable, or None if not found and non-strict.

    Raises:
        CompilationError: If strict mode is enabled and no executable entrypoint is found.
    """
    exec(code, namespace)
    if "apply_model" in namespace:

        def apply_wrapper(*w_args: T, **w_kw: T) -> T:
            """Wrap to invoke the generated apply_model function.

            Args:
                *w_args (T): Positional arguments.
                **w_kw (T): Keyword arguments.

            Returns:
                T: The result.
            """
            return namespace["apply_model"]({}, *w_args, **w_kw)

        return apply_wrapper

    if "evaluate" in namespace:

        def eval_wrapper(*w_args: T, **w_kw: T) -> T:
            """Wrap to invoke the generated evaluate function.

            Args:
                *w_args (T): Positional arguments.
                **w_kw (T): Keyword arguments.

            Returns:
                T: The result.
            """
            del w_kw
            numpy_args = [a.data if isinstance(a, Tensor) else a for a in w_args]
            return namespace["evaluate"](numpy_args)

        return eval_wrapper

    if strict:
        from ml_switcheroo_compiler.core.errors import CompilationError

        raise CompilationError(f"No executable entrypoint found in generated code for backend '{backend}'")
    return None


def _execute_generated_code(
    code: str,
    generator_cls: type,
    backend: str,
    key: str,
    strict: bool,
    fn: Callable[..., T],
    args: tuple[T, ...],
    kw: dict[str, T],
) -> T:
    """Build executable entrypoint from generated code and invoke with arguments.

    Args:
        code (str): Generated source code string.
        generator_cls (type): Backend generator class.
        backend (str): Target backend name.
        key (str): Cache signature key.
        strict (bool): Whether strict compilation mode is enabled.
        fn (Callable[..., T]): Original function for eager fallback.
        args (tuple[T, ...]): Call positional arguments.
        kw (dict[str, T]): Call keyword arguments.

    Returns:
        T: Result of executing the compiled function or eager fallback.

    Raises:
        CompilationError: If strict mode is enabled and execution fails.
    """
    from ml_switcheroo_compiler.core.errors import CompilationError

    namespace = _get_namespace(backend, generator_cls)

    try:
        compiled_fn: Callable[..., EagerValue] | None = cast("Callable[..., EagerTensorProtocol]", _build_generated_entrypoint)(code, namespace, backend, strict)
        if compiled_fn is None:
            return _fallback_eager(fn, args, kw)
        res = compiled_fn(*args, **kw)
        _COMPILATION_CACHE[key] = compiled_fn
        return cast(T, res)
    except CompilationError:
        raise
    except Exception as e:
        if strict:
            raise CompilationError(f"Execution of compiled code failed: {e}") from e
        return _fallback_eager(fn, args, kw)


def _compile_with_generator_instance(
    generator_cls: type,
    graph: LogicalGraph,
    backend: str,
    key: str,
    kwargs: dict[str, Union[bool, int, str, None]],
    fn: Callable[..., T],
    args: tuple[T, ...],
    kw: dict[str, T],
) -> T:
    """Instantiate backend generator, compile or exec generated code, and invoke with args.

    Args:
        generator_cls (type): Backend generator class.
        graph (LogicalGraph): Optimized logical graph.
        backend (str): Target backend name.
        key (str): Cache signature key.
        kwargs (dict[str, Union[bool, int, str, None]]): Compilation options.
        fn (Callable[..., T]): Original function for eager fallback.
        args (tuple[T, ...]): Call positional arguments.
        kw (dict[str, T]): Call keyword arguments.

    Returns:
        T: Result of executing the compiled function or eager fallback.

    Raises:
        CompilationError: If strict mode is enabled and compilation or execution fails.
    """
    from ml_switcheroo_compiler.core.errors import CompilationError

    strict = bool(kwargs.get("strict", False))

    try:
        generator = generator_cls(graph)
    except Exception as e:
        if strict:
            raise CompilationError(f"Failed to instantiate generator '{backend}': {e}") from e
        return _fallback_eager(fn, args, kw)

    if hasattr(generator, "compile_aot"):
        try:
            res_aot = generator.compile_aot(graph, **kwargs)
            if callable(res_aot):
                _COMPILATION_CACHE[key] = res_aot
                return cast(T, res_aot(*args, **kw))
        except NotImplementedError:
            logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
        except Exception as e:
            if strict:
                raise CompilationError(f"AOT compilation failed: {e}") from e
            return _fallback_eager(fn, args, kw)

    code = str(generator.generate())
    return _execute_generated_code(code, generator_cls, backend, key, strict, fn, args, kw)


def compile_function(fn: EagerValue, backend: str = "numpy", **kwargs: EagerValue) -> EagerValue:
    """Compiles a function functionally, intended for torch.compile integrations.

    Args:
        fn: The function to compile.
        backend: The target execution backend.
        kwargs: Additional compilation options.

    Returns:
        Callable: The compiled function.
    """

    def compiled_wrapper(*args: EagerValue, **kw: EagerValue) -> EagerValue:
        """Wrap that traces and compiles the function on first call.

        Args:
            *args (T): Positional arguments for the function.
            **kw (T): Keyword arguments for the function.

        Returns: Tensor: The result of the compiled (or eager) function.
        """
        key = _build_signature_key(fn, backend, args)
        if key in _COMPILATION_CACHE:
            return _COMPILATION_CACHE[key](*args, **kw)

        graph = cast("Callable[..., EagerTensorProtocol]", _trace_function_graph)(fn, args, kw)
        if graph is None:
            return cast(EagerValue, cast("Callable[..., EagerTensorProtocol]", _fallback_eager)(fn, args, kw))

        try:
            generator_cls = BackendRegistry.get(backend)
        except ValueError as e:
            if kwargs.get("strict", False):
                from ml_switcheroo_compiler.core.errors import CompilationError

                raise CompilationError(f"Target backend '{backend}' not found: {e}") from e
            return cast(EagerValue, cast("Callable[..., EagerTensorProtocol]", _fallback_eager)(fn, args, kw))

        static_fn = cast("Callable[..., Optional[Callable[..., EagerValue]]]", _try_static_compile_aot)(generator_cls, graph, kwargs)
        if static_fn is not None:
            _COMPILATION_CACHE[key] = static_fn
            return static_fn(*args, **kw)

        return cast(EagerValue, cast("Callable[..., EagerTensorProtocol]", _compile_with_generator_instance)(generator_cls, graph, backend, key, kwargs, fn, args, kw))

    return compiled_wrapper
