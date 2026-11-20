"""RNN operations."""

from __future__ import annotations

from collections.abc import Callable, Sequence  # noqa: F401

import pydantic

from ml_switcheroo_compiler.backends.registry import coerce_operand_list, require_callable
from ml_switcheroo_compiler.core.config import config as global_config
from ml_switcheroo_compiler.core.shape import OpValue, is_variadic_callable
from ml_switcheroo_compiler.core.tensor import Tensor, ensure_tensor
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


def _unpack_pair(val: OpValue) -> tuple[OpValue, OpValue]:
    """Unpack a 2-element sequence result into a pair.

    Args:
        val (OpValue): 2-element sequence result.

    Returns:
        tuple[OpValue, OpValue]: Unpacked `(first, second)` pair.
    """
    assert isinstance(val, (tuple, list))
    return val[0], val[1]


class _PositionalRNNModel(pydantic.BaseModel):
    """Base RNN configuration model supporting positional and keyword initialization."""

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    def __init__(self, *args: OpValue, **kwargs: OpValue) -> None:
        """Initialize model from positional or keyword arguments.

        Args:
            *args (OpValue): Positional field values.
            **kwargs (OpValue): Keyword field values.
        """
        field_names = tuple(type(self).model_fields.keys())
        pos_map = {field_names[i]: args[i] for i in range(min(len(field_names), len(args)))}
        super().__init__(**{**pos_map, **kwargs})


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
    f: OpValue,
    init: Tensor | tuple[Tensor, ...],
    xs: Tensor,
    config: ScanConfig | None = None,
) -> tuple[tuple[Tensor, ...] | OpValue, Tensor]:
    """Scan loop construct.

    Args:
        f (OpValue): Input `f` argument (f).
        init (Tensor | tuple[Tensor, ...]): Input `init` argument (init).
        xs (Tensor): Input `xs` argument (xs).
        config (ScanConfig | None): Input `config` argument (config).

    Returns:
        tuple: Computed `tuple` output.
    """
    conf = config if config is not None else ScanConfig()
    fn_callable = f

    if global_config.eager_mode or conf.unroll:
        xs_list = coerce_operand_list(unstack(xs, axis=0))

        if conf.reverse:
            xs_list = list(reversed(xs_list))

        carry: OpValue = init
        ys: list[Tensor] = []

        for x in xs_list:
            _callable_fn_1: OpValue = fn_callable
            assert is_variadic_callable(_callable_fn_1)
            carry, y = _unpack_pair(_callable_fn_1(carry, x))
            ys.append(ensure_tensor(y))

        carry_tup: tuple[Tensor, ...] = tuple(ensure_tensor(t) for t in coerce_operand_list(carry))
        return (carry_tup, ensure_tensor(stack(ys, axis=0)))
    else:
        if conf.reverse:
            xs = cf_reverse(xs, (0,))

        def _step_fn(c: tuple[Tensor, ...], x_elem: Tensor) -> tuple[tuple[Tensor, ...], Tensor]:
            """Invoke RNN step function.

            Args:
                c (tuple[Tensor, ...]): Carry state tuple.
                x_elem (Tensor): Current input slice.

            Returns:
                tuple[tuple[Tensor, ...], Tensor]: Updated carry and step output.
            """
            _callable_fn_2: OpValue = fn_callable
            assert is_variadic_callable(_callable_fn_2)
            res_step = _callable_fn_2(c, x_elem)
            c_next, y_next = _unpack_pair(res_step)
            c_tup: tuple[Tensor, ...] = (
                tuple(ensure_tensor(t) for t in c_next)
                if isinstance(c_next, (list, tuple))
                else (ensure_tensor(c_next),)
            )
            return c_tup, ensure_tensor(y_next)

        scan_res = cf_scan(_step_fn, init, xs, conf.length)
        s_carry, s_y = _unpack_pair(scan_res)
        return s_carry, ensure_tensor(s_y)


def bidirectional(
    inputs: BidirectionalInputs,
    cell_fn: OpValue,
    config: BidirectionalConfig | None = None,
) -> tuple[Tensor | tuple[Tensor, Tensor], tuple[Tensor, ...], tuple[Tensor, ...]]:
    """Bidirectional RNN wrapper.

    Args:
        inputs (BidirectionalInputs): The bidirectional inputs.
        cell_fn (OpValue): The RNN cell function.
        config (Optional[BidirectionalConfig]): Configuration.
        config (Optional[ScanConfig]): Configuration for scan.

    Returns:
        tuple[Tensor | tuple[Tensor, Tensor], tuple[Tensor, ...], tuple[Tensor, ...]]:
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
    merged_out: Tensor | tuple[Tensor, Tensor]
    if conf.merge_mode == "concat":
        merged_out = ensure_tensor(concatenate([forward_out, backward_out], axis=-1))
    elif conf.merge_mode == "sum":
        merged_out = add(forward_out, backward_out)
    elif conf.merge_mode == "mul":
        merged_out = multiply(forward_out, backward_out)
    elif conf.merge_mode == "ave":
        merged_out = multiply(add(forward_out, backward_out), 0.5)
    else:
        # None
        merged_out = (forward_out, backward_out)

    return (merged_out, forward_state, backward_state)


def _permute_time_major(inputs: Tensor) -> OpValue:
    """Swap batch and time dimensions.

    Args:
        inputs (Tensor): Input `inputs` argument (inputs).

    Returns:
        Tensor: Computed `Tensor` output.
    """
    dims = list(range(len(inputs.shape)))
    dims[0], dims[1] = 1, 0
    return permute(inputs, tuple(dims))


def rnn(
    inputs: Tensor,
    initial_state: tuple[Tensor, ...],
    cell_fn: OpValue,
    config: RNNConfig | None = None,
) -> tuple[Tensor, tuple[Tensor, ...]]:
    """Define base recurrent loop evaluation.

    Args:
        inputs (Tensor): Input `inputs` argument (inputs).
        initial_state (tuple[Tensor, ...]): Input `initial_state` argument (initial state).
        cell_fn (OpValue): Input `cell_fn` argument (cell fn).
        config (RNNConfig | None): Input `config` argument (config).

    Returns:
        tuple: Computed `tuple` output.
    """
    conf = config if config is not None else RNNConfig()
    cell_callable = cell_fn
    if not conf.time_major:
        inputs = ensure_tensor(_permute_time_major(inputs))

    def scan_fn(carry: Tensor, x: Tensor) -> OpValue:
        """Compute the scan fn tensor operation.

        Args:
            carry (Tensor): Input `carry` argument (carry).
            x (Tensor): Input `x` argument (x).

        Returns:
            tuple: Computed `tuple` output.
        """
        out, new_carry = require_callable(_unpack_pair)(require_callable(cell_callable)(x, carry))
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
        outputs = ensure_tensor(_permute_time_major(outputs))

    final_state_tup: tuple[Tensor, ...] = (
        tuple(ensure_tensor(t) for t in final_state)
        if isinstance(final_state, (tuple, list))
        else (ensure_tensor(final_state),)
    )
    return outputs, final_state_tup


class RNNCellDeviceWrapper:
    """RNNCellDeviceWrapper."""

    def __init__(self, cell: OpValue, device: OpValue, **kwargs: OpValue) -> None:
        """Init.

        Args:
            cell (OpValue): Input `cell` argument (cell).
            device (OpValue): Input `device` argument (device).
            **kwargs (OpValue): Keyword args.
        """
        self._cell = cell
        self._device = device

    def __call__(self, inputs: OpValue, state: OpValue, **kwargs: OpValue) -> OpValue:
        """Call.

        Args:
        inputs (OpValue): Input `inputs` argument (inputs).
        state (OpValue): Input `state` argument (state).
        **kwargs (OpValue): Keyword args.

        Returns:
        tuple: Computed `tuple` output.
        """
        return require_callable(self._cell)(inputs, state, **kwargs)


class DropoutWrapperConfig(pydantic.BaseModel):
    """Configuration for RNNCellDropoutWrapper."""

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    input_keep_prob: float = 1.0
    output_keep_prob: float = 1.0
    state_keep_prob: float = 1.0
    variational_recurrent: bool = False
    input_size: int | None = None
    dtype: OpValue = None
    seed: int | None = None
    dropout_state_filter_visitor: OpValue = None


class RNNCellDropoutWrapper:
    """Wrap that adds dropout to input and/or output of the given cell."""

    def __init__(
        self,
        cell: OpValue,
        config: DropoutWrapperConfig | None = None,
        **kwargs: OpValue,
    ) -> None:
        """Initialize the RNNCellDropoutWrapper.

        Args:
            cell (OpValue): The RNN cell to wrap.
            config (Optional[DropoutWrapperConfig]): Configuration for dropout.
            kwargs (OpValue): Additional keyword arguments.
        """
        self._cell = cell
        self._config = config if config is not None else DropoutWrapperConfig()

    def __call__(
        self,
        inputs: Tensor | OpValue,
        state: Tensor | tuple[Tensor, ...] | OpValue,
        **kwargs: OpValue,
    ) -> OpValue:
        """Run the cell with dropout.

        Args:
            inputs (Tensor | OpValue): Input tensor.
            state (Tensor | tuple[Tensor, ...] | OpValue): Current state.
            kwargs (OpValue): Additional keyword arguments.

        Returns:
            tuple[Tensor, tuple[Tensor, ...]]: Output tensor and new state.
        """
        inp_t = inputs if isinstance(inputs, Tensor) else ensure_tensor(inputs)
        if self._config.input_keep_prob < 1.0:
            inp_t = ensure_tensor(dropout(inp_t, rate=1.0 - self._config.input_keep_prob))
        out, new_state = _unpack_pair(require_callable(self._cell)(inp_t, state, **kwargs))
        if self._config.output_keep_prob < 1.0:
            out = dropout(ensure_tensor(out), rate=1.0 - self._config.output_keep_prob)
        return out, new_state


class RNNCellResidualWrapper:
    """RNNCellResidualWrapper."""

    def __init__(self, cell: OpValue, residual_fn: OpValue = None, **kwargs: OpValue) -> None:
        """Init.

        Args:
            cell (OpValue): Input `cell` argument (cell).
            residual_fn (OpValue): Input `residual_fn` argument (residual fn).
            **kwargs (OpValue): Keyword args.
        """
        self._cell = cell
        self._residual_fn = residual_fn

    def __call__(self, inputs: OpValue, state: OpValue, **kwargs: OpValue) -> OpValue:
        """Call.

        Args:
            inputs (OpValue): Input `inputs` argument (inputs).
            state (OpValue): Input `state` argument (state).
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple: Computed `tuple` output.
        """
        out, new_state = _unpack_pair(require_callable(self._cell)(inputs, state, **kwargs))
        if self._residual_fn is not None:
            out = require_callable(self._residual_fn)(inputs, out)
        else:
            out = add(inputs, out)
        return out, new_state
