"""Operation utilities."""


def get_source_inputs(tensor: object) -> list[object]:
    """Returns the list of input tensors that a tensor depends on."""
    # This is a dummy implementation if no node history.
    if hasattr(tensor, "_keras_history"):
        node = tensor._keras_history.node
        if not node.operation.inputs:
            return [tensor]
        res = []
        for inp in node.operation.inputs:
            res.extend(get_source_inputs(inp))
        return res
    return [tensor]


def compute_shape_propagation(name: str, shape: tuple, args: tuple, kwargs: dict) -> object:
    """Compute shape propagation."""
    if name == "reshape":
        shape = args[1]
    elif name == "transpose":
        axes = kwargs.get("axes", args[1] if len(args) > 1 else None)
        if axes is not None:
            shape = tuple(shape[i] for i in axes)
        else:
            shape = tuple(reversed(shape))
    elif name == "expand_dims":
        axis = kwargs.get("axis", args[1] if len(args) > 1 else -1)
        if axis < 0:
            axis += len(shape) + 1
        shape = tuple(shape[:axis]) + (1,) + tuple(shape[axis:])
    elif name == "squeeze":
        axis = kwargs.get("axis", args[1] if len(args) > 1 else None)
        if axis is not None:
            if isinstance(axis, int):
                axis = [axis]
            shape = tuple(s for i, s in enumerate(shape) if i not in axis)
        else:
            shape = tuple(s for s in shape if s != 1)
    elif name == "split":
        num_or_size_splits = args[1]
        axis = kwargs.get("axis", args[2] if len(args) > 2 else 0)
        if isinstance(num_or_size_splits, int):
            sub_shape = list(shape)
            sub_shape[axis] = (
                sub_shape[axis] // num_or_size_splits if sub_shape[axis] is not None else None
            )
            sub_shape = tuple(sub_shape)
            return [sub_shape for _ in range(num_or_size_splits)]
    elif name == "mean":
        axis = kwargs.get("axis", args[1] if len(args) > 1 else None)
        keepdims = kwargs.get("keepdims", False)
        if axis is not None:
            if isinstance(axis, int):
                axis = [axis]
            if not keepdims:
                shape = tuple(
                    s for i, s in enumerate(shape) if i not in axis and (i - len(shape)) not in axis
                )
            else:
                shape = tuple(
                    1 if (i in axis or (i - len(shape)) in axis) else s for i, s in enumerate(shape)
                )
    return shape
