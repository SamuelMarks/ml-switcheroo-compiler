"""RNN operations."""

from dataclasses import dataclass
from typing import Callable, Optional, cast  # noqa: F401

from ml_switcheroo_compiler.backends.registry import EagerTensorProtocol, EagerValue
from ml_switcheroo_compiler.core.config import config as global_config
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


@dataclass
class RNNConfig:
    """Configuration for RNN ops."""

    time_major: bool = False
    go_backwards: bool = False
    unroll: bool = False
    return_all_outputs: bool = True


@dataclass
class BidirectionalInputs:
    """Inputs for bidirectional RNN."""

    forward_inputs: Tensor
    backward_inputs: Tensor
    forward_initial_state: tuple[Tensor, ...]
    backward_initial_state: tuple[Tensor, ...]


@dataclass
class BidirectionalConfig:
    """Configuration for bidirectional RNN."""

    merge_mode: str = "concat"
    time_major: bool = False
    unroll: bool = False


@dataclass
class RNNWeights:
    """Weights for RNN cell."""

    kernel: Tensor
    recurrent_kernel: Tensor
    bias: Optional[Tensor] = None


@dataclass
class ConvLSTMConfig:
    """Configuration for ConvLSTM cell."""

    strides: int = 1
    padding: str = "SAME"
    data_format: str = "channels_last"


@dataclass
class ScanConfig:
    """Configuration for scan."""

    length: Optional[int] = None
    reverse: bool = False
    unroll: bool = False


def scan(
    f: Callable[..., EagerValue],
    init: tuple[Tensor, ...],
    xs: Tensor,
    config: Optional[ScanConfig] = None,
) -> tuple[tuple[Tensor, ...], Tensor]:
    """Scan loop construct.

    Args:
        f (Any): The f parameter.
        init (tuple): The init parameter.
        xs (Tensor): The xs parameter.
        config (Optional): The config parameter.

    Returns:
        tuple: Result.
    """
    conf = config if config is not None else ScanConfig()

    if global_config.eager_mode or conf.unroll:
        xs_unstacked = unstack(xs, axis=0)

        if conf.reverse:
            xs_unstacked = list(cast("Callable[..., EagerTensorProtocol]", reversed)(xs_unstacked))

        carry = init
        ys = []

        for x in getattr(xs_unstacked, "__iter__")():
            carry, y = getattr(f(carry, x), "__iter__")()
            ys.append(y)

        return cast(tuple[tuple[Tensor, ...], Tensor], (carry, stack(ys, axis=0)))
    else:
        if conf.reverse:
            xs = cf_reverse(xs, (0,))

        carry, y = cast("Callable[..., tuple[tuple[Tensor, ...], Tensor]]", cf_scan)(f, init, xs, conf.length)

        return carry, y


def bidirectional(
    inputs: BidirectionalInputs,
    cell_fn: EagerValue,
    config: Optional[BidirectionalConfig] = None,
) -> tuple[Tensor, tuple[Tensor, ...], tuple[Tensor, ...]]:
    """Bidirectional RNN wrapper.

    Args:
        inputs (BidirectionalInputs): The bidirectional inputs.
        cell_fn (Any): The RNN cell function.
        config (Optional[BidirectionalConfig]): Configuration.
        config (Optional[ScanConfig]): Configuration for scan.

    Returns:
        tuple[Tensor, tuple[Tensor, ...], tuple[Tensor, ...]]:
            Merged output sequence, forward final states, backward final states.
    """
    conf = config if config is not None else BidirectionalConfig()
    forward_out, forward_state = cast("Callable[..., tuple[Tensor, tuple[Tensor, ...]]]", rnn)(
        inputs.forward_inputs,
        inputs.forward_initial_state,
        cell_fn,
        config=RNNConfig(time_major=conf.time_major, unroll=conf.unroll, go_backwards=False),
    )

    backward_out, backward_state = cast("Callable[..., tuple[Tensor, tuple[Tensor, ...]]]", rnn)(
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

    return cast(tuple[Tensor, tuple[Tensor, ...], tuple[Tensor, ...]], (merged_out, forward_state, backward_state))


def _permute_time_major(inputs: Tensor) -> EagerValue:
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
    cell_fn: Callable[..., EagerValue],
    config: Optional[RNNConfig] = None,
) -> tuple[Tensor, tuple[Tensor, ...]]:
    """Define base recurrent loop evaluation.

    Args:
        inputs (Tensor): The inputs parameter.
        initial_state (tuple): The initial_state parameter.
        cell_fn (Any): The cell_fn parameter.
        config (Optional): The config parameter.

    Returns:
        tuple: Result.
    """
    conf = config if config is not None else RNNConfig()
    if not conf.time_major:
        inputs = cast(Tensor, _permute_time_major(inputs))

    def scan_fn(carry: Tensor, x: Tensor) -> EagerValue:
        """Evaluate scan_fn operation.

        Args:
            carry (Tensor): The carry parameter.
            x (Tensor): The x parameter.

        Returns:
            tuple: Result.
        """
        out, new_carry = getattr(cell_fn(x, carry), "__iter__")()
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
        outputs = cast(Tensor, _permute_time_major(outputs))

    return outputs, final_state


class RNNCellDeviceWrapper:
    """RNNCellDeviceWrapper."""

    def __init__(self, cell: EagerValue, device: EagerValue, **kwargs: EagerValue) -> None:
        """Init.

        Args:
            cell (Any): The cell parameter.
            device (Any): The device parameter.
            **kwargs (Any): Keyword args.
        """
        self._cell = cell
        self._device = device

    def __call__(self, inputs: EagerValue, state: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Call.

        Args:
        inputs (Any): The inputs parameter.
        state (Any): The state parameter.
        **kwargs (Any): Keyword args.

        Returns:
        tuple: Result.
        """
        return cast(EagerValue, cast("Callable[..., EagerTensorProtocol]", self._cell)(inputs, state, **kwargs))


@dataclass
class DropoutWrapperConfig:
    """Configuration for RNNCellDropoutWrapper."""

    input_keep_prob: float = 1.0
    output_keep_prob: float = 1.0
    state_keep_prob: float = 1.0
    variational_recurrent: bool = False
    input_size: Optional[int] = None
    dtype = None
    seed: Optional[int] = None
    dropout_state_filter_visitor = None


class RNNCellDropoutWrapper:
    """Wrap that adds dropout to input and/or output of the given cell."""

    def __init__(
        self,
        cell: EagerValue,
        config: Optional[DropoutWrapperConfig] = None,
        **kwargs: EagerValue,
    ) -> None:
        """Initialize the RNNCellDropoutWrapper.

        Args:
            cell (Any): The RNN cell to wrap.
            config (Optional[DropoutWrapperConfig]): Configuration for dropout.
            kwargs (Any): Additional keyword arguments.
        """
        self._cell = cell
        self._config = config if config is not None else DropoutWrapperConfig()

    def __call__(self, inputs: Tensor, state: tuple[Tensor, ...], **kwargs: EagerValue) -> tuple[Tensor, tuple[Tensor, ...]]:
        """Run the cell with dropout.

        Args:
            inputs (Tensor): Input tensor.
            state (tuple[Tensor, ...]): Current state.
            kwargs (Any): Additional keyword arguments.

        Returns:
            tuple[Tensor, tuple[Tensor, ...]]: Output tensor and new state.
        """
        if self._config.input_keep_prob < 1.0:
            inputs = cast(Tensor, dropout(inputs, rate=1.0 - self._config.input_keep_prob))
        out, new_state = getattr(cast("Callable[..., EagerTensorProtocol]", self._cell)(inputs, state, **kwargs), "__iter__")()
        if self._config.output_keep_prob < 1.0:
            out = dropout(out, rate=1.0 - self._config.output_keep_prob)
        return out, new_state


class RNNCellResidualWrapper:
    """RNNCellResidualWrapper."""

    def __init__(self, cell: EagerValue, residual_fn: EagerValue = None, **kwargs: EagerValue) -> None:
        """Init.

        Args:
            cell (Any): The cell parameter.
            residual_fn (Any): The residual_fn parameter.
            **kwargs (Any): Keyword args.
        """
        self._cell = cell
        self._residual_fn = residual_fn

    def __call__(self, inputs: EagerValue, state: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Call.

        Args:
            inputs (Any): The inputs parameter.
            state (Any): The state parameter.
            **kwargs (Any): Keyword args.

        Returns:
            tuple: Result.
        """
        out, new_state = getattr(cast("Callable[..., EagerTensorProtocol]", self._cell)(inputs, state, **kwargs), "__iter__")()
        if self._residual_fn is not None:
            out = cast("Callable[..., EagerTensorProtocol]", self._residual_fn)(inputs, out)
        else:
            out = add(inputs, out)
        return out, new_state
