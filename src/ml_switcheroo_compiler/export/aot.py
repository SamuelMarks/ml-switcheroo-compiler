# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Ahead-of-Time compilation hooks for frontend integrations."""

from collections.abc import Callable

from ml_switcheroo_compiler.backends.registry import BackendRegistry
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.tracing.builder import TracingNodeBuilder
from ml_switcheroo_compiler.tracing.state import global_tracing_state
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor, TracerTape
from ml_switcheroo_compiler.transforms.pass_manager import PassManager

# Cache for compiled artifacts
_COMPILATION_CACHE: dict[str, Callable[..., object]] = {}


def _fallback_eager(fn: Callable[..., object], args: tuple[object, ...], kw: dict[str, object]) -> object:
    """Execute function eagerly as a fallback.

    Args:
        fn: The original function.
        args: Positional arguments.
        kw: Keyword arguments.

    Returns: object: Result of eager evaluation.
    """
    was_eager: object = config.eager_mode
    config.eager_mode = True
    try:
        return fn(*args, **kw)
    finally:
        config.eager_mode = was_eager


def _build_signature_key(fn: Callable[..., object], backend: str, args: tuple[object, ...]) -> str:
    """Build a cache key for the compilation signature.

    Args:
        fn (Callable): The fn parameter.
        backend (str): The backend parameter.
        args (tuple): The args parameter.

    Returns:
        str: Result.
    """
    sig_parts: object = []
    for a in args:
        if hasattr(a, "shape") and hasattr(a, "dtype"):
            sig_parts.append(f"T({getattr(a, 'shape', ())},{getattr(a, 'dtype', '')})")
        else:
            sig_parts.append(f"S({type(a).__name__})")
    return f"{id(fn)}_{backend}_" + "_".join(sig_parts)


def _prepare_proxy_args(args: tuple[object, ...]) -> list[object]:
    """Prepare proxy arguments for tracing.

    Args:
        args (tuple): The args parameter.

    Returns:
        list: Result.
    """
    proxy_args: object = []
    for i, a in enumerate(args):
        if hasattr(a, "shape") and hasattr(a, "dtype"):
            arg_id: object = f"arg_{i}"
            shape: object = getattr(a, "shape", ())
            proxy: object = ProxyTensor(id=arg_id, shape=shape, dtype=str(getattr(a, "dtype", "")))
            dtype: object = getattr(a, "dtype", DType.Float32)
            device: object = getattr(a, "device", "cpu")
            proxy_args.append(Tensor(proxy, TensorConfig(proxy.shape, dtype, device)))
            TracingNodeBuilder.create_tracing_logical_node("Input", [], {}, shape)

            # Overwrite id to match
            last_node: object = list(global_tracing_state.active_graph.nodes.values())[-1]
            old_id: object = last_node.id
            del global_tracing_state.active_graph.nodes[old_id]
            last_node.id = arg_id
            global_tracing_state.active_graph.nodes[arg_id] = last_node
        else:
            proxy_args.append(a)
    return proxy_args


def _capture_outputs(out: object) -> None:
    """Capture outputs in the active tracing graph.

    Args:
        out (object): The out parameter.
    """
    if isinstance(out, Tensor):
        out_id, _ = TracingNodeBuilder.extract_from_tensor(out)
        out_node_id: object = TracingNodeBuilder.create_tracing_logical_node("Output", [out_id], {}, getattr(out, "shape", ()))
        global_tracing_state.active_graph.outputs.append(out_node_id)
    elif isinstance(out, (list, tuple)):
        out_ids: object = []
        for x in out:
            if isinstance(x, Tensor):
                oid, _ = TracingNodeBuilder.extract_from_tensor(x)
                out_ids.append(oid)
        if out_ids:
            out_node_id: object = TracingNodeBuilder.create_tracing_logical_node("Output", out_ids, {}, ())
            global_tracing_state.active_graph.outputs.append(out_node_id)


def _get_namespace(backend: str, generator_cls: type) -> dict[str, object]:
    """Get the namespace for executing generated code.

    Args:
        backend (str): The backend parameter.
        generator_cls (type): The generator_cls parameter.

    Returns:
        dict: Result.
    """
    namespace: object = {}
    if hasattr(generator_cls, "get_module"):
        namespace[backend] = generator_cls.get_module()
    elif backend == "numpy":
        import numpy

        namespace[backend] = numpy
    return namespace


def compile_function(fn: Callable[..., object], backend: str = "numpy", **kwargs: object) -> Callable[..., object]:
    """Compiles a function functionally, intended for torch.compile integrations.

    Args:
        fn: The function to compile.
        backend: The target execution backend.
        kwargs: Additional compilation options.

    Returns:
        Callable: The compiled function.
    """

    def compiled_wrapper(*args: object, **kw: object) -> object:
        """Wrap that traces and compiles the function on first call.

        Args:
            *args (object): Positional arguments for the function.
            **kw (object): Keyword arguments for the function.

        Returns: object: The result of the compiled (or eager) function.
        """
        key: object = _build_signature_key(fn, backend, args)
        if key in _COMPILATION_CACHE:
            return _COMPILATION_CACHE[key](*args, **kw)

        tape: object = TracerTape()
        tape.start_tracing()
        proxy_args: object = _prepare_proxy_args(args)

        try:
            out: object = fn(*proxy_args, **kw)
            _capture_outputs(out)
        except Exception:
            tape.stop_tracing()
            return _fallback_eager(fn, args, kw)

        graph: object = tape.stop_tracing()
        if not graph or not graph.outputs:
            return _fallback_eager(fn, args, kw)

        pm: object = PassManager()
        from ml_switcheroo_compiler.transforms.passes.constant_folding import constant_folding_pass
        from ml_switcheroo_compiler.transforms.passes.cse import cse_pass
        from ml_switcheroo_compiler.transforms.passes.dce import dce_pass
        from ml_switcheroo_compiler.transforms.passes.dtype_inference import dtype_inference_pass
        from ml_switcheroo_compiler.transforms.passes.rematerialization import rematerialization_pass
        from ml_switcheroo_compiler.transforms.passes.shape_inference import shape_inference_pass

        pm.add_pass(shape_inference_pass)
        pm.add_pass(dtype_inference_pass)
        pm.add_pass(constant_folding_pass)
        pm.add_pass(cse_pass)
        pm.add_pass(rematerialization_pass)
        pm.add_pass(dce_pass)

        graph: object = pm.run(graph)

        try:
            generator_cls: object = BackendRegistry.get(backend)
        except ValueError:
            return _fallback_eager(fn, args, kw)

        if hasattr(generator_cls, "compile_aot"):
            compiled_fn: object = generator_cls.compile_aot(graph, **kwargs)
            _COMPILATION_CACHE[key] = compiled_fn
            return compiled_fn(*args, **kw)

        generator: object = generator_cls(graph)
        code: object = generator.generate()
        namespace: object = _get_namespace(backend, generator_cls)

        try:
            exec(code, namespace)  # noqa: S102
            if "apply_model" in namespace:

                def apply_wrapper(*w_args: object, **w_kw: object) -> object:
                    """Wrap to invoke the generated apply_model function.

                    Args:
                        *w_args (object): Positional arguments.
                        **w_kw (object): Keyword arguments.

                    Returns: object: The result.
                    """
                    return namespace["apply_model"]({}, *w_args, **w_kw)

                compiled_fn: object = apply_wrapper
            elif "evaluate" in namespace:

                def eval_wrapper(*w_args: object, **w_kw: object) -> object:
                    """Wrap to invoke the generated evaluate function.

                    Args:
                        *w_args (object): Positional arguments.
                        **w_kw (object): Keyword arguments.

                    Returns: object: The result.
                    """
                    numpy_args: object = [a.data if isinstance(a, Tensor) else a for a in w_args]
                    return namespace["evaluate"](numpy_args)

                compiled_fn: object = eval_wrapper
            else:
                return _fallback_eager(fn, args, kw)

            res: object = compiled_fn(*args, **kw)
            _COMPILATION_CACHE[key] = compiled_fn
            return res
        except Exception:
            return _fallback_eager(fn, args, kw)

    return compiled_wrapper
