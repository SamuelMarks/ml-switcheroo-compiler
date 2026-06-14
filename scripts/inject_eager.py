"""Inject eager operations into backends."""


def inject(filepath: str, backend_name: str, mod_name: str) -> None:
    """Inject ops into a backend file.

    Args:
        filepath (str): The filepath of the backend to modify.
        backend_name (str): The name of the backend.
        mod_name (str): The name of the underlying module.
    """
    with open(filepath) as f:
        content = f.read()

    if "def execute_op" in content:
        return

    injection = f"""
    @classmethod
    def execute_op(cls, op_type: str, *args: object, **kwargs: object) -> object:
        import {mod_name}
        try:
            func = getattr({mod_name}, op_type.lower())
            return func(*args, **kwargs)
        except AttributeError:
            pass

        op_map = {{
            "Add": getattr({mod_name}, "add", None),
            "Subtract": getattr({mod_name}, "subtract", None),
            "Multiply": getattr({mod_name}, "multiply", None),
            "TrueDivide": getattr({mod_name}, "divide", getattr({mod_name}, "true_divide", None)),
            "Exp": getattr({mod_name}, "exp", None),
            "Log": getattr({mod_name}, "log", None),
            "Matmul": getattr({mod_name}, "matmul", None),
            "Sin": getattr({mod_name}, "sin", None),
            "Cos": getattr({mod_name}, "cos", None),
            "Sum": getattr({mod_name}, "sum", None),
            "Mean": getattr({mod_name}, "mean", None),
            "Max": getattr({mod_name}, "max", None),
            "Min": getattr({mod_name}, "min", None),
            "Reshape": getattr({mod_name}, "reshape", None),
            "Transpose": getattr({mod_name}, "transpose", None),
            "Equal": getattr({mod_name}, "equal", None),
            "NotEqual": getattr({mod_name}, "not_equal", None),
            "Greater": getattr({mod_name}, "greater", None),
            "Less": getattr({mod_name}, "less", None),
            "Negative": getattr({mod_name}, "negative", None),
        }}

        if op_type in op_map and op_map[op_type] is not None:
            return op_map[op_type](*args, **kwargs)

        if op_type == "BroadcastTo":
            return getattr({mod_name}, "broadcast_to")(*args, **kwargs)

        msg = f"Operation '{{op_type}}' is not supported by {backend_name} backend."
        raise NotImplementedError(msg)

    @classmethod
    def zeros(cls, shape: tuple[int, ...]) -> object:
        import {mod_name}
        return {mod_name}.zeros(shape)

    @classmethod
    def array(cls, data: object) -> object:
        import {mod_name}
        try:
            return {mod_name}.array(data)
        except AttributeError:
            return {mod_name}.convert_to_tensor(data)

    @classmethod
    def asarray(cls, data: object) -> object:
        import {mod_name}
        try:
            return {mod_name}.asarray(data)
        except AttributeError:
            return {mod_name}.convert_to_tensor(data)

    @classmethod
    def item(cls, data: object) -> float:
        import {mod_name}
        try:
            return float({mod_name}.asarray(data).item())
        except AttributeError:
            return float(data)
"""
    with open(filepath, "a") as f:
        f.write(injection)


inject("src/ml_switcheroo_compiler/backends/jax.py", "jax", "jax.numpy")
inject("src/ml_switcheroo_compiler/backends/mlx.py", "mlx", "mlx.core")
inject("src/ml_switcheroo_compiler/backends/keras.py", "keras", "keras.ops")
inject("src/ml_switcheroo_compiler/backends/tensorflow.py", "tensorflow", "tensorflow.math")
