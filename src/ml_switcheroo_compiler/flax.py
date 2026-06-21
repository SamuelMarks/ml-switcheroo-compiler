"""Flax (NNX) frontend compatibility layer."""

from __future__ import annotations


from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


@dataclass(init=False)
class Module:
    """Base class for all neural network modules in NNX."""

    def __init__(self, **kwargs: object) -> None:
        """Initialize the module.

        Args:
            **kwargs: Additional keyword arguments.
        """
        for k, v in kwargs.items():
            setattr(self, k, v)


@dataclass(init=False)
class GraphDef:
    """Represents the static structure of a Module graph."""


@dataclass(init=False)
class State(dict):
    """A nested dictionary structure for nnx variables."""


@dataclass
class Variable:
    """A base class representing a stateful variable in the framework."""

    def __init__(self, value: object, **kwargs: object) -> None:
        """Initialize.

        Args:
            value (object): The value parameter for the operation.
            **kwargs: Additional keyword arguments.
        """
        self.value = value


@dataclass(init=False)
class Param(Variable):
    """A variable representing a trainable parameter."""


@dataclass(init=False)
class BatchStat(Variable):
    """A variable representing non-trainable state."""


@dataclass(init=False)
class Rng(Variable):
    """A variable holding a random number generator stream."""


@dataclass(init=False)
class Dense(Module):
    """A standard linear transformation layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        **kwargs: object,
    ) -> None:
        """Initialize.

        Args:
            in_features (int): The in_features parameter for the operation.
            out_features (int): The out_features parameter for the operation.
            use_bias (bool): The use_bias parameter for the operation.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias


@dataclass(init=False)
class Linear(Dense):
    """Alias/variant of standard linear (dense) layer."""


@dataclass(init=False)
class LinearGeneral(Module):
    """A general linear transformation layer."""

    def __init__(
        self,
        in_features: int | Sequence[int],
        out_features: int | Sequence[int],
        **kwargs: object,
    ) -> None:
        """Initialize.

        Args:
            in_features (int | Sequence[int]): The in_features parameter for the operation.
            out_features (int | Sequence[int]): The out_features parameter for the operation.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features


@dataclass(init=False)
class Einsum(Module):
    """A module that performs a linear transformation using an einsum equation."""

    def __init__(self, einsum_str: str, kernel_shape: tuple[int, ...], **kwargs: object) -> None:
        """Initialize.

        Args:
            einsum_str (str): The einsum_str parameter for the operation.
            kernel_shape (tuple[int, ...]): The kernel_shape parameter for the operation.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.einsum_str = einsum_str
        self.kernel_shape = kernel_shape


@dataclass(init=False)
class LoRA(Module):
    """Low-Rank Adaptation injection module."""


@dataclass(init=False)
class LoRALinear(Module):
    """A pre-configured Linear layer with LoRA adapters."""

    def __init__(self, in_features: int, out_features: int, **kwargs: object) -> None:
        """Initialize.

        Args:
            in_features (int): The in_features parameter for the operation.
            out_features (int): The out_features parameter for the operation.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features


@dataclass
class ConvConfig:
    """Configuration for a convolutional layer.

    Attributes:
        in_features: The in_features parameter for the operation.
        out_features: The out_features parameter for the operation.
        kernel_size: The kernel_size parameter for the operation.
        strides: The strides parameter for the operation.
        padding: The padding parameter for the operation.
    """

    in_features: int
    out_features: int
    kernel_size: tuple[int, ...]
    strides: tuple[int, ...] | None = None
    padding: str | tuple[tuple[int, int], ...] = "VALID"


@dataclass(init=False)
class Conv(Module):
    """A general n-dimensional convolutional layer."""

    def __init__(
        self,
        config: ConvConfig,
        **kwargs: object,
    ) -> None:
        """Initialize.

        Args:
            config (ConvConfig): The configuration for the convolution layer.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.config = config
        self.in_features = config.in_features
        self.out_features = config.out_features
        self.kernel_size = config.kernel_size
        self.strides = config.strides
        self.padding = config.padding


@dataclass(init=False)
class ConvTranspose(Module):
    """A general n-dimensional transposed convolution."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: tuple[int, ...],
        **kwargs: object,
    ) -> None:
        """Initialize.

        Args:
            in_features (int): The in_features parameter for the operation.
            out_features (int): The out_features parameter for the operation.
            kernel_size (tuple[int, ...]): The kernel_size parameter for the operation.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size


@dataclass(init=False)
class Embed(Module):
    """A simple lookup table that stores embeddings of a fixed dictionary."""

    def __init__(self, num_embeddings: int, features: int, **kwargs: object) -> None:
        """Initialize.

        Args:
            num_embeddings (int): The num_embeddings parameter for the operation.
            features (int): The features parameter for the operation.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.num_embeddings = num_embeddings
        self.features = features


@dataclass(init=False)
class MultiHeadAttention(Module):
    """Standard Multi-Head Attention implementation."""

    def __init__(self, num_heads: int, qkv_features: int, **kwargs: object) -> None:
        """Initialize.

        Args:
            num_heads (int): The num_heads parameter for the operation.
            qkv_features (int): The qkv_features parameter for the operation.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.qkv_features = qkv_features


@dataclass(init=False)
class MultiHeadDotProductAttention(Module):
    """Core attention kernel without projection layers."""

    def __init__(self, num_heads: int, qkv_features: int, **kwargs: object) -> None:
        """Initialize.

        Args:
            num_heads (int): The num_heads parameter for the operation.
            qkv_features (int): The qkv_features parameter for the operation.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.qkv_features = qkv_features


@dataclass(init=False)
class BatchNorm(Module):
    """Batch Normalization layer."""

    def __init__(
        self,
        num_features: int,
        use_running_average: bool = False,
        **kwargs: object,
    ) -> None:
        """Initialize.

        Args:
            num_features (int): The num_features parameter for the operation.
            use_running_average (bool): The use_running_average parameter for the operation.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.num_features = num_features
        self.use_running_average = use_running_average


@dataclass(init=False)
class LayerNorm(Module):
    """Layer Normalization layer."""

    def __init__(self, num_features: int, reduction_axes: int = -1, **kwargs: object) -> None:
        """Initialize.

        Args:
            num_features (int): The num_features parameter for the operation.
            reduction_axes (int): The reduction_axes parameter for the operation.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.num_features = num_features
        self.reduction_axes = reduction_axes


@dataclass(init=False)
class RMSNorm(Module):
    """Root Mean Square Normalization layer."""

    def __init__(self, num_features: int, **kwargs: object) -> None:
        """Initialize.

        Args:
            num_features (int): The num_features parameter for the operation.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.num_features = num_features


@dataclass(init=False)
class Dropout(Module):
    """A dropout layer."""

    def __init__(self, rate: float, rng_collection: str = "dropout", **kwargs: object) -> None:
        """Initialize.

        Args:
            rate (float): The rate parameter for the operation.
            rng_collection (str): The rng_collection parameter for the operation.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.rate = rate
        self.rng_collection = rng_collection


@dataclass(init=False)
class Sequential(Module):
    """Applies a sequence of modules sequentially."""

    def __init__(self, *layers: Module, **kwargs: object) -> None:
        """Initialize.

        Args:
            *layers: Additional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.layers = layers


@dataclass(init=False)
class List(Module):
    """A module that holds a list of sub-modules."""

    def __init__(self, modules: Iterable[Module], **kwargs: object) -> None:
        """Initialize.

        Args:
            modules (Iterable[Module]): The modules parameter for the operation.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.modules = list(modules)


@dataclass(init=False)
class Dict(Module):
    """A module that holds a dictionary of sub-modules."""

    def __init__(self, modules: dict[str, Module], **kwargs: object) -> None:
        """Initialize.

        Args:
            modules (dict[str, Module]): The modules parameter for the operation.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.modules = modules


@dataclass(init=False)
class Jit(Module):
    """JIT-compiles the execution of a sub-module."""

    def __init__(self, module_constructor: Callable[..., Module], **kwargs: object) -> None:
        """Initialize.

        Args:
            module_constructor (Callable[..., Module]): The module_constructor parameter.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.module_constructor = module_constructor


@dataclass(init=False)
class Vmap(Module):
    """Vectorizes the execution of a sub-module."""

    def __init__(self, module_constructor: Callable[..., Module], **kwargs: object) -> None:
        """Initialize.

        Args:
            module_constructor (Callable[..., Module]): The module_constructor parameter.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.module_constructor = module_constructor


@dataclass(init=False)
class Scan(Module):
    """Loops over a sequence of inputs, maintaining module state."""

    def __init__(self, module_constructor: Callable[..., Module], **kwargs: object) -> None:
        """Initialize.

        Args:
            module_constructor (Callable[..., Module]): The module_constructor parameter.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.module_constructor = module_constructor


@dataclass(init=False)
class Remat(Module):
    """Checkpoints a sub-module to save memory during backprop."""

    def __init__(self, module_constructor: Callable[..., Module], **kwargs: object) -> None:
        """Initialize.

        Args:
            module_constructor (Callable[..., Module]): The module_constructor parameter.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.module_constructor = module_constructor


@dataclass(init=False)
class Pmap(Module):
    """Parallelizes module execution across multiple devices."""

    def __init__(self, module_constructor: Callable[..., Module], **kwargs: object) -> None:
        """Initialize.

        Args:
            module_constructor (Callable[..., Module]): The module_constructor parameter.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.module_constructor = module_constructor
