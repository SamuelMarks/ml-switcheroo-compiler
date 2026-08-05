"""RNN operations."""

from dataclasses import dataclass
from typing import Optional

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
    f: object,
    init: tuple[Tensor, ...],
    xs: Tensor,
    config: Optional[ScanConfig] = None,
) -> tuple[tuple[Tensor, ...], Tensor]:
    """Scan loop construct.

    Args:
        f (object): The f parameter.
        init (tuple): The init parameter.
        xs (Tensor): The xs parameter.
        config (Optional): The config parameter.

    Returns:
        tuple: Result.
    """
    conf = config if config is not None else ScanConfig()

    if global_config.eager_mode or conf.unroll:
        xs_unstacked = unstack(xs, dim=0)

        if conf.reverse:
            xs_unstacked = list(reversed(xs_unstacked))

        carry = init
        ys = []

        for x in xs_unstacked:
            carry, y = f(carry, x)
            ys.append(y)

        return carry, stack(ys, dim=0)
    else:
        if conf.reverse:
            xs = cf_reverse(xs, (0,))

        carry, y = cf_scan(f, init, xs, conf.length)

        return carry, y


def bidirectional(
    inputs: BidirectionalInputs,
    cell_fn: object,
    config: Optional[BidirectionalConfig] = None,
) -> tuple[Tensor, tuple[Tensor, ...], tuple[Tensor, ...]]:
    """Bidirectional RNN wrapper.

    Args:
        inputs (BidirectionalInputs): The bidirectional inputs.
        cell_fn (object): The RNN cell function.
        config (Optional[BidirectionalConfig]): Configuration.
        config (Optional[ScanConfig]): Configuration for scan.

    Returns:
        tuple[Tensor, tuple[Tensor, ...], tuple[Tensor, ...]]:
            Merged output sequence, forward final states, backward final states.
    """
    conf = config if config is not None else BidirectionalConfig()
    forward_out, forward_state = rnn(
        inputs.forward_inputs,
        inputs.forward_initial_state,
        cell_fn,
        config=RNNConfig(time_major=conf.time_major, unroll=conf.unroll, go_backwards=False),
    )

    backward_out, backward_state = rnn(
        inputs.backward_inputs,
        inputs.backward_initial_state,
        cell_fn,
        config=RNNConfig(time_major=conf.time_major, unroll=conf.unroll, go_backwards=False),
    )

    conf = config if config is not None else BidirectionalConfig()
    if conf.merge_mode == "concat":
        merged_out = concatenate([forward_out, backward_out], dim=-1)
    elif conf.merge_mode == "sum":
        merged_out = add(forward_out, backward_out)
    elif conf.merge_mode == "mul":
        merged_out = multiply(forward_out, backward_out)
    elif conf.merge_mode == "ave":
        merged_out = multiply(add(forward_out, backward_out), 0.5)
    else:
        # None
        merged_out = (forward_out, backward_out)

    return merged_out, forward_state, backward_state


def _permute_time_major(inputs: Tensor) -> Tensor:
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
    cell_fn: object,
    config: Optional[RNNConfig] = None,
) -> tuple[Tensor, tuple[Tensor, ...]]:
    """Define base recurrent loop evaluation.

    Args:
        inputs (Tensor): The inputs parameter.
        initial_state (tuple): The initial_state parameter.
        cell_fn (object): The cell_fn parameter.
        config (Optional): The config parameter.

    Returns:
        tuple: Result.
    """
    conf = config if config is not None else RNNConfig()
    if not conf.time_major:
        inputs = _permute_time_major(inputs)

    def scan_fn(carry: Tensor, x: Tensor) -> tuple[Tensor, Tensor]:
        """Evaluate scan_fn operation.

        Args:
            carry (Tensor): The carry parameter.
            x (Tensor): The x parameter.

        Returns:
            tuple: Result.
        """
        out, new_carry = cell_fn(x, carry)
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
        outputs = _permute_time_major(outputs)

    return outputs, final_state


class RNNCellDeviceWrapper:
    """RNNCellDeviceWrapper."""

    def __init__(self, cell: object, device: object, **kwargs: object) -> None:
        """Init.

        Args:
            cell (object): The cell parameter.
            device (object): The device parameter.
            **kwargs (object): Keyword args.
        """
        self._cell = cell
        self._device = device

    def __call__(self, inputs: object, state: object, **kwargs: object) -> tuple:
        """Call.

        Args:
        inputs (object): The inputs parameter.
        state (object): The state parameter.
        **kwargs (object): Keyword args.

        Returns:
        tuple: Result.
        """
        return self._cell(inputs, state, **kwargs)


@dataclass
class DropoutWrapperConfig:
    """Configuration for RNNCellDropoutWrapper."""

    input_keep_prob: float = 1.0
    output_keep_prob: float = 1.0
    state_keep_prob: float = 1.0
    variational_recurrent: bool = False
    input_size: Optional[int] = None
    dtype: Optional[object] = None
    seed: Optional[int] = None
    dropout_state_filter_visitor: Optional[object] = None


class RNNCellDropoutWrapper:
    """Wrap that adds dropout to input and/or output of the given cell."""

    def __init__(
        self,
        cell: object,
        config: Optional[DropoutWrapperConfig] = None,
        **kwargs: object,
    ) -> None:
        """Initialize the RNNCellDropoutWrapper.

        Args:
            cell (object): The RNN cell to wrap.
            config (Optional[DropoutWrapperConfig]): Configuration for dropout.
            kwargs (object): Additional keyword arguments.
        """
        self._cell = cell
        self._config = config if config is not None else DropoutWrapperConfig()

    def __call__(self, inputs: Tensor, state: tuple[Tensor, ...], **kwargs: object) -> tuple[Tensor, tuple[Tensor, ...]]:
        """Run the cell with dropout.

        Args:
            inputs (Tensor): Input tensor.
            state (tuple[Tensor, ...]): Current state.
            kwargs (object): Additional keyword arguments.

        Returns:
            tuple[Tensor, tuple[Tensor, ...]]: Output tensor and new state.
        """
        if self._config.input_keep_prob < 1.0:
            inputs = dropout(inputs, rate=1.0 - self._config.input_keep_prob)
        out, new_state = self._cell(inputs, state, **kwargs)
        if self._config.output_keep_prob < 1.0:
            out = dropout(out, rate=1.0 - self._config.output_keep_prob)
        return out, new_state


class RNNCellResidualWrapper:
    """RNNCellResidualWrapper."""

    def __init__(self, cell: object, residual_fn: object = None, **kwargs: object) -> None:
        """Init.

        Args:
            cell (object): The cell parameter.
            residual_fn (object): The residual_fn parameter.
            **kwargs (object): Keyword args.
        """
        self._cell = cell
        self._residual_fn = residual_fn

    def __call__(self, inputs: object, state: object, **kwargs: object) -> tuple:
        """Call.

        Args:
            inputs (object): The inputs parameter.
            state (object): The state parameter.
            **kwargs (object): Keyword args.

        Returns:
            tuple: Result.
        """
        out, new_state = self._cell(inputs, state, **kwargs)
        if self._residual_fn is not None:
            out = self._residual_fn(inputs, out)
        else:
            out = add(inputs, out)
        return out, new_state
