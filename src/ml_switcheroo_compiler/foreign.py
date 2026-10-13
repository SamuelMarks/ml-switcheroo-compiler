# ruff: noqa: E402, E501, F403
"""Foreign module integration."""

from ml_switcheroo_compiler.ops.base import OpDef, register_op


@register_op("ForeignCall")
class ForeignCall(OpDef):
    """Universal ForeignCall op for external graphs/modules."""

    def infer_shape(self, *args, **kwargs) -> tuple[int, ...]:  # noqa: C901, PLR0911, PLR0912, PLR0915
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
                        if args_tuple and all(isinstance(x, int) or x is int for x in args_tuple):
                            return tuple(args_tuple)
                except (TypeError, NameError) as e:
                    import logging

                    logging.getLogger(__name__).warning("Failed to infer return type hints for %s: %s", first, e)
                try:
                    sig = inspect.signature(first)
                    ret_anno = sig.return_annotation
                    if hasattr(ret_anno, "__args__") and all(isinstance(x, int) or x is int for x in ret_anno.__args__):
                        return tuple(ret_anno.__args__)
                except (ValueError, TypeError) as e:
                    from ml_switcheroo_compiler.core.errors import ShapeInferenceError

                    raise ShapeInferenceError(f"Failed to infer shape from signature for {first}: {e}") from e

                try:
                    import ast

                    lines = inspect.getsource(first).split("\n")
                    if lines:
                        spaces = len(lines[0]) - len(lines[0].lstrip())
                        src_code = "\n".join(line[spaces:] if len(line) >= spaces else line for line in lines)
                    else:
                        src_code = ""
                    tree = ast.parse(src_code)
                    for n in ast.walk(tree):
                        if isinstance(n, ast.Return) and n.value is not None:
                            if isinstance(n.value, ast.Tuple):
                                dims: list[int] = []
                                for elt in n.value.elts:
                                    if isinstance(elt, ast.Constant) and isinstance(elt.value, int):
                                        dims.append(elt.value)
                                if dims and len(dims) == len(n.value.elts):
                                    return tuple(dims)
                except (OSError, TypeError, SyntaxError) as e:
                    import logging

                    logging.getLogger(__name__).debug("AST introspection failed for %s: %s", first, e)

            if len(args) > 1 and hasattr(args[1], "shape"):
                return tuple(args[1].shape)

        return ()
