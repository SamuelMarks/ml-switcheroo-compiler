"""Auto-generated high-fidelity type stubs for cupy from live module introspection."""
# ruff: noqa: E501

class Tensor:
    shape: tuple[int, ...]
    dtype: str
    def __init__(self, *args: object, **kwargs: object) -> None: ...

def __getattr__(name: str) -> Tensor: ...
