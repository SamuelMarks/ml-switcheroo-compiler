"""RNN operations."""

from __future__ import annotations

import typing
from collections.abc import Callable, Sequence  # noqa: F401

import pydantic

from ml_switcheroo_compiler.backends.registry import EagerCallableProtocol
from ml_switcheroo_compiler.core.config import config as global_config
from ml_switcheroo_compiler.core.shape import TensorOrScalar, TypedVariadicCallable
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.binary import add, multiply
from ml_switcheroo_compiler.ops.control_flow import scan as cf_scan
from ml_switcheroo_compiler.ops.nn.dropout import dropout
from ml_switcheroo_compiler.ops.shape import (
    concatenate,
    permute,
    stack,
    unstack,
)
from ml_switcheroo_compiler.ops.shape import flip as cf_reverse


class _PositionalRNNModel(pydantic.BaseModel):
    """Base RNN configuration model supporting positional and keyword initialization."""

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    def __init__(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> None:
        """Initialize model from positional or keyword arguments."""
        super().__init__(**{**dict(zip(self.__class__.model_fields.keys(), args)), **kwargs})


class RNNConfig(_PositionalRNNModel):
    """Configuration for RNN ops."""

    time_major: bool = False
    go_backwards: bool = False
    unroll: bool = False
    return_all_outputs: bool = True


class BidirectionalInputs(_PositionalRNNModel):
    """Inputs for bidirectional RNN."""

    forward_inputs: Tensor
    backward_inputs: Tensor
    forward_initial_state: tuple[Tensor, ...]
    backward_initial_state: tuple[Tensor, ...]


class BidirectionalConfig(_PositionalRNNModel):
    """Configuration for bidirectional RNN."""

    merge_mode: str | None = "concat"
    time_major: bool = False
    unroll: bool = False


class RNNWeights(_PositionalRNNModel):
    """Weights for RNN cell."""

    kernel: Tensor
    recurrent_kernel: Tensor
    bias: Tensor | None = None


class ConvLSTMConfig(_PositionalRNNModel):
    """Configuration for ConvLSTM cell."""

    strides: int | Sequence[int] = 1
    padding: str = "SAME"
    data_format: str = "channels_last"


class ScanConfig(_PositionalRNNModel):
    """Configuration for scan."""

    length: int | None = None
    reverse: bool = False
    unroll: bool = False


def scan(
    f: TensorOrScalar,
    init: tuple[Tensor, ...],
    xs: Tensor,
    config: ScanConfig | None = None,
) -> tuple[tuple[Tensor, ...], Tensor]:
    """Scan loop construct.

    Args:
        f (TensorOrScalar): The f parameter.
        init (tuple): The init parameter.
        xs (Tensor): The xs parameter.
        config (Optional): The config parameter.

    Returns:
        tuple: Result.
    """
    conf = config if config is not None else ScanConfig()
    fn_callable = typing.cast(TypedVariadicCallable[TensorOrScalar], f)

    if global_config.eager_mode or conf.unroll:
        xs_unstacked = unstack(xs, axis=0)

        if conf.reverse:
            xs_unstacked = list(typing.cast("EagerCallableProtocol", reversed)(xs_unstacked))

        carry = init
        ys = []

        for x in getattr(xs_unstacked, "__iter__")():
            carry, y = getattr(fn_callable(carry, x), "__iter__")()
            ys.append(y)

        return typing.cast(tuple[tuple[Tensor, ...], Tensor], (carry, stack(ys, axis=0)))
    else:
        if conf.reverse:
            xs = cf_reverse(xs, (0,))

        carry, y = typing.cast("TypedVariadicCallable[tuple[tuple[Tensor, ...], Tensor]]", cf_scan)(fn_callable, init, xs, conf.length)

        return carry, y


def bidirectional(
    inputs: BidirectionalInputs,
    cell_fn: TensorOrScalar,
    config: BidirectionalConfig | None = None,
) -> tuple[Tensor, tuple[Tensor, ...], tuple[Tensor, ...]]:
    """Bidirectional RNN wrapper.

    Args:
        inputs (BidirectionalInputs): The bidirectional inputs.
        cell_fn (TensorOrScalar): The RNN cell function.
        config (Optional[BidirectionalConfig]): Configuration.
        config (Optional[ScanConfig]): Configuration for scan.

    Returns:
        tuple[Tensor, tuple[Tensor, ...], tuple[Tensor, ...]]:
            Merged output sequence, forward final states, backward final states.
    """
    conf = config if config is not None else BidirectionalConfig()
    forward_out, forward_state = typing.cast("TypedVariadicCallable[tuple[Tensor, tuple[Tensor, ...]]]", rnn)(
        inputs.forward_inputs,
        inputs.forward_initial_state,
        cell_fn,
        config=RNNConfig(time_major=conf.time_major, unroll=conf.unroll, go_backwards=False),
    )

    backward_out, backward_state = typing.cast("TypedVariadicCallable[tuple[Tensor, tuple[Tensor, ...]]]", rnn)(
        inputs.backward_inputs,
        inputs.backward_initial_state,
        cell_fn,
        config=RNNConfig(time_major=conf.time_major, unroll=conf.unroll, go_backwards=False),
    )

    conf = config if config is not None else BidirectionalConfig()
    if conf.merge_mode == "concat":
        merged_out = concatenate([forward_out, backward_out], axis=-1)
    elif conf.merge_mode == "sum":
        merged_out = add(forward_out, backward_out)
    elif conf.merge_mode == "mul":
        merged_out = multiply(forward_out, backward_out)
    elif conf.merge_mode == "ave":
        merged_out = multiply(add(forward_out, backward_out), 0.5)
    else:
        # None
        merged_out = (forward_out, backward_out)

    return typing.cast(tuple[Tensor, tuple[Tensor, ...], tuple[Tensor, ...]], (merged_out, forward_state, backward_state))


def _permute_time_major(inputs: Tensor) -> TensorOrScalar:
    """Swap batch and time dimensions.

    Args:
        inputs (Tensor): The inputs parameter.

    Returns:
        Tensor: Result.
    """
    dims = list(range(len(inputs.shape)))
    dims[0], dims[1] = 1, 0
    return permute(inputs, tuple(dims))


def rnn(
    inputs: Tensor,
    initial_state: tuple[Tensor, ...],
    cell_fn: TensorOrScalar,
    config: RNNConfig | None = None,
) -> tuple[Tensor, tuple[Tensor, ...]]:
    """Define base recurrent loop evaluation.

    Args:
        inputs (Tensor): The inputs parameter.
        initial_state (tuple): The initial_state parameter.
        cell_fn (TensorOrScalar): The cell_fn parameter.
        config (Optional): The config parameter.

    Returns:
        tuple: Result.
    """
    conf = config if config is not None else RNNConfig()
    cell_callable = typing.cast(TypedVariadicCallable[TensorOrScalar], cell_fn)
    if not conf.time_major:
        inputs = typing.cast(Tensor, _permute_time_major(inputs))

    def scan_fn(carry: Tensor, x: Tensor) -> TensorOrScalar:
        """Compute the scan fn tensor operation.

        Args:
            carry (Tensor): The carry parameter.
            x (Tensor): The x parameter.

        Returns:
            tuple: Result.
        """
        out, new_carry = getattr(cell_callable(x, carry), "__iter__")()
        return new_carry, out

    final_state, outputs = scan(
        scan_fn,
        initial_state,
        inputs,
        config=ScanConfig(reverse=conf.go_backwards, unroll=conf.unroll),
    )
    if not conf.return_all_outputs:
        outputs = outputs[-1] if conf.time_major else outputs[:, -1]

    if not conf.time_major:
        outputs = typing.cast(Tensor, _permute_time_major(outputs))

    return outputs, final_state


class RNNCellDeviceWrapper:
    """RNNCellDeviceWrapper."""

    def __init__(self, cell: TensorOrScalar, device: TensorOrScalar, **kwargs: TensorOrScalar) -> None:
        """Init.

        Args:
            cell (TensorOrScalar): The cell parameter.
            device (TensorOrScalar): The device parameter.
            **kwargs (TensorOrScalar): Keyword args.
        """
        self._cell = cell
        self._device = device

    def __call__(self, inputs: TensorOrScalar, state: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Call.

        Args:
        inputs (TensorOrScalar): The inputs parameter.
        state (TensorOrScalar): The state parameter.
        **kwargs (TensorOrScalar): Keyword args.

        Returns:
        tuple: Result.
        """
        return typing.cast(TensorOrScalar, typing.cast("EagerCallableProtocol", self._cell)(inputs, state, **kwargs))


class DropoutWrapperConfig(pydantic.BaseModel):
    """Configuration for RNNCellDropoutWrapper."""

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    input_keep_prob: float = 1.0
    output_keep_prob: float = 1.0
    state_keep_prob: float = 1.0
    variational_recurrent: bool = False
    input_size: int | None = None
    dtype: TensorOrScalar = None
    seed: int | None = None
    dropout_state_filter_visitor: TensorOrScalar = None


class RNNCellDropoutWrapper:
    """Wrap that adds dropout to input and/or output of the given cell."""

    def __init__(
        self,
        cell: TensorOrScalar,
        config: DropoutWrapperConfig | None = None,
        **kwargs: TensorOrScalar,
    ) -> None:
        """Initialize the RNNCellDropoutWrapper.

        Args:
            cell (TensorOrScalar): The RNN cell to wrap.
            config (Optional[DropoutWrapperConfig]): Configuration for dropout.
            kwargs (TensorOrScalar): Additional keyword arguments.
        """
        self._cell = cell
        self._config = config if config is not None else DropoutWrapperConfig()

    def __call__(self, inputs: Tensor, state: tuple[Tensor, ...], **kwargs: TensorOrScalar) -> tuple[Tensor, tuple[Tensor, ...]]:
        """Run the cell with dropout.

        Args:
            inputs (Tensor): Input tensor.
            state (tuple[Tensor, ...]): Current state.
            kwargs (TensorOrScalar): Additional keyword arguments.

        Returns:
            tuple[Tensor, tuple[Tensor, ...]]: Output tensor and new state.
        """
        if self._config.input_keep_prob < 1.0:
            inputs = typing.cast(Tensor, dropout(inputs, rate=1.0 - self._config.input_keep_prob))
        out, new_state = getattr(typing.cast("EagerCallableProtocol", self._cell)(inputs, state, **kwargs), "__iter__")()
        if self._config.output_keep_prob < 1.0:
            out = dropout(out, rate=1.0 - self._config.output_keep_prob)
        return out, new_state


class RNNCellResidualWrapper:
    """RNNCellResidualWrapper."""

    def __init__(self, cell: TensorOrScalar, residual_fn: TensorOrScalar = None, **kwargs: TensorOrScalar) -> None:
        """Init.

        Args:
            cell (TensorOrScalar): The cell parameter.
            residual_fn (TensorOrScalar): The residual_fn parameter.
            **kwargs (TensorOrScalar): Keyword args.
        """
        self._cell = cell
        self._residual_fn = residual_fn

    def __call__(self, inputs: TensorOrScalar, state: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Call.

        Args:
            inputs (TensorOrScalar): The inputs parameter.
            state (TensorOrScalar): The state parameter.
            **kwargs (TensorOrScalar): Keyword args.

        Returns:
            tuple: Result.
        """
        out, new_state = getattr(typing.cast("EagerCallableProtocol", self._cell)(inputs, state, **kwargs), "__iter__")()
        if self._residual_fn is not None:
            out = typing.cast("EagerCallableProtocol", self._residual_fn)(inputs, out)
        else:
            out = add(inputs, out)
        return out, new_state
