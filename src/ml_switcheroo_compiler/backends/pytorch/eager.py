"""Backend utilities."""


def execute_op(cls: type, op_type: str, *args: object, **kwargs: object) -> object:
    """Execute execute_op.

    Args:
        cls (Any): The class.
        op_type (Any): Argument op_type.
        *args (Any): Argument *args.
        **kwargs (Any): Argument **kwargs.

    Returns:
    Any: The result.
    """
    import torch

    op_map = {
        "Add": torch.add,
        "Subtract": torch.sub,
        "Multiply": torch.mul,
        "TrueDivide": torch.div,
        "Exp": torch.exp,
        "Log": torch.log,
        "Matmul": torch.matmul,
        "Sin": torch.sin,
        "Cos": torch.cos,
        "Sum": torch.sum,
        "Mean": torch.mean,
        "Max": torch.max,
        "Min": torch.min,
        "Reshape": torch.reshape,
        "Transpose": torch.transpose,
        "Equal": torch.eq,
        "NotEqual": torch.ne,
        "Greater": torch.gt,
        "Less": torch.lt,
        "Negative": torch.neg,
    }

    if op_type in op_map:
        func = op_map[op_type]
        return func(*args, **kwargs)

    # Handle BroadcastTo separately
    if op_type == "BroadcastTo":
        return args[0].expand(kwargs["shape"])

    try:
        func = getattr(torch, op_type.lower())
        return func(*args, **kwargs)
    except AttributeError:
        msg = f"Operation '{op_type}' is not supported by torch backend."
        raise NotImplementedError(msg) from None
