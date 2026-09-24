# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Foreign module integration."""

from ml_switcheroo_compiler.ops.base import OpDef, register_op


@register_op("ForeignCall")
class ForeignCall(OpDef):
    """Universal ForeignCall op for external graphs/modules."""

    def infer_shape(self, *args, **kwargs) -> tuple[int, ...]:
        """Infer the shape of the output for ForeignCall via introspection.

        Args:
            *args (object): Positional arguments (callable, inputs, ...).
            **kwargs (object): Keyword arguments including optional 'output_shape'.

        Returns:
            tuple[int, ...]: The inferred shape.
        """
        if "output_shape" in kwargs and kwargs["output_shape"] is not None:
            shape_val = kwargs["output_shape"]
            return tuple(shape_val) if isinstance(shape_val, (list, tuple)) else (shape_val,)

        if args:
            first = args[0]
            if hasattr(first, "shape"):
                return tuple(first.shape)
            if hasattr(first, "output_shape"):
                s = first.output_shape
                return tuple(s) if isinstance(s, (list, tuple)) else (s,)
            if hasattr(first, "output_shapes"):
                s = first.output_shapes
                return tuple(s[0]) if isinstance(s, (list, tuple)) and len(s) > 0 else ()
            if hasattr(first, "subgraph") and hasattr(first.subgraph, "outputs"):
                out_nodes = [first.subgraph.nodes.get(out) for out in first.subgraph.outputs if out in first.subgraph.nodes]
                if out_nodes and hasattr(out_nodes[0], "shape_metadata") and out_nodes[0].shape_metadata:
                    return tuple(out_nodes[0].shape_metadata)

            import inspect
            import typing

            if callable(first):
                try:
                    hints = typing.get_type_hints(first)
                    if "return" in hints:
                        ret_type = hints["return"]
                        args_tuple = typing.get_args(ret_type)
                        if args_tuple:
                            return tuple(args_tuple)
                except Exception:
                    pass
                try:
                    sig = inspect.signature(first)
                    ret_anno = sig.return_annotation
                    if hasattr(ret_anno, "__args__"):
                        return tuple(ret_anno.__args__)
                except Exception:
                    pass

        return ()
