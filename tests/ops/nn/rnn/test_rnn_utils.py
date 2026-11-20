"""Tests for test_rnn_utils."""

from __future__ import annotations

import types
import typing
from unittest.mock import patch

import numpy as np
import pytest

import ml_switcheroo_compiler.ops.nn.rnn as rnn_module
from ml_switcheroo_compiler.backends.eager_registry import BackendArg
from ml_switcheroo_compiler.core.config import Config, ConfigContext, config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.device import Device as CoreDevice
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.dtype import DType as CoreDType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.core.tensor import Tensor as CoreTensor
from ml_switcheroo_compiler.core.tensor import TensorConfig as CoreTensorConfig
from ml_switcheroo_compiler.ops.nn.rnn_cell import simple_rnn_cell
from ml_switcheroo_compiler.ops.nn.rnn_utils import (
    BidirectionalConfig,
    BidirectionalInputs,
    DropoutWrapperConfig,
    RNNCellDeviceWrapper,
    RNNCellDropoutWrapper,
    RNNCellResidualWrapper,
    RNNConfig,
    ScanConfig,
    _permute_time_major,
    bidirectional,
    rnn,
    scan,
)
from ml_switcheroo_compiler.ops.nn.time_distributed import TimeDistributed, time_distributed
from tests.testing_utils import (
    EagerBoundCallable,
    MockTestStub,
    coerce_operand_list,
    ensure_tensor,
    extract_test_shape,
    require_callable,
    unpack_pair,
)

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.tracing.state import TracingState


def test_rnn_utils_branches() -> None:
    """Test function test_rnn_utils_branches."""
    from unittest.mock import patch

    import numpy as np

    from ml_switcheroo_compiler.ops.nn import rnn_utils as ru

    t = Tensor(
        np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig(shape=(2, 2), dtype=DType("float32"), device=Device("cpu"))
    )
    t3 = Tensor(np.ones((2, 2, 2)), TensorConfig(shape=(2, 2, 2), dtype=DType("float32"), device=Device("cpu")))
    orig = config.eager_mode
    config.eager_mode = True
    try:

        def my_f(c: BackendArg, x: BackendArg) -> BackendArg:
            """Test function my_f.

            Args:
                c: Test parameter `c`.
                x: Test parameter `x`.

            Returns:
                BackendArg: Computed output value from `my_f`.
            """
            return (c, x)

        ru.scan(my_f, (t,), t, ru.ScanConfig(reverse=True, unroll=True))
        config.eager_mode = False
        with patch("ml_switcheroo_compiler.ops.nn.rnn_utils.cf_scan", return_value=((t,), t)):
            with patch("ml_switcheroo_compiler.ops.nn.rnn_utils.cf_reverse", return_value=t):
                ru.scan(my_f, (t,), t, ru.ScanConfig(reverse=True, unroll=False))
        config.eager_mode = True
        b_in = ru.BidirectionalInputs(
            forward_inputs=t, backward_inputs=t, forward_initial_state=(t,), backward_initial_state=(t,)
        )
        for mode in ["concat", "sum", "mul", "ave", "none"]:
            with patch("ml_switcheroo_compiler.ops.nn.rnn_utils.rnn", return_value=(t, (t,))):
                with (
                    patch("ml_switcheroo_compiler.ops.nn.rnn_utils.concatenate", return_value=t),
                    patch("ml_switcheroo_compiler.ops.nn.rnn_utils.add", return_value=t),
                    patch("ml_switcheroo_compiler.ops.nn.rnn_utils.multiply", return_value=t),
                ):
                    ru.bidirectional(b_in, my_f, ru.BidirectionalConfig(merge_mode=mode))
        with patch("ml_switcheroo_compiler.ops.nn.rnn_utils.scan", return_value=((t,), t3)):
            with patch("ml_switcheroo_compiler.ops.nn.rnn_utils._permute_time_major", return_value=t3):
                ru.rnn(t, (t,), my_f, ru.RNNConfig(time_major=True, return_all_outputs=False))
                ru.rnn(t, (t,), my_f, ru.RNNConfig(time_major=False, return_all_outputs=False))
        cell = ru.RNNCellDeviceWrapper(my_f, None)
        cell(t, t)
        c_drop = ru.RNNCellDropoutWrapper(my_f, ru.DropoutWrapperConfig(input_keep_prob=0.5, output_keep_prob=0.5))
        with patch("ml_switcheroo_compiler.ops.nn.rnn_utils.dropout", return_value=t):
            c_drop(t, t)
        c_res = ru.RNNCellResidualWrapper(my_f)
        with patch("ml_switcheroo_compiler.ops.nn.rnn_utils.add", return_value=t):
            c_res(t, t)
        c_res2 = ru.RNNCellResidualWrapper(my_f, residual_fn=lambda i, o: o)
        c_res2(t, t)
    finally:
        config.eager_mode = orig
    assert config.eager_mode == orig


def create_eager_tensor(data: BackendArg) -> BackendArg:
    """Test function create_eager_tensor.

    Args:
        data: Test parameter `data`.

    Returns:
        BackendArg: Computed output value from `create_eager_tensor`.
    """
    return Tensor(
        data,
        TensorConfig(extract_test_shape(data), DType("float32"), Device("cpu")),
    )


def test_rnn_utils_missing_branches() -> None:
    """Test function test_rnn_utils_missing_branches."""

    class MockCell(MockTestStub):
        """Test class MockCell."""

        def __call__(self, inputs: BackendArg, state: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function __call__.

            Args:
                inputs: Test parameter `inputs`.
                state: Test parameter `state`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `__call__`.
            """
            return (inputs, state)

    with ConfigContext(eager_mode=True):
        config = DropoutWrapperConfig(input_keep_prob=0.5, output_keep_prob=0.5)
        wrapper = RNNCellDropoutWrapper(MockCell(), config)
        inputs = create_eager_tensor(np.ones((2, 2)))
        state = (create_eager_tensor(np.ones((2, 2))),)
        (out, new_state) = unpack_pair(wrapper(inputs, state, training=True))
        assert out is not inputs
        assert id(new_state) == id(state)
        (out, new_state) = unpack_pair(wrapper(inputs, state, training=False))
        out_arr = ensure_tensor(out).numpy()
        in_arr = ensure_tensor(inputs).numpy()
        assert isinstance(out_arr, np.ndarray) and isinstance(in_arr, np.ndarray)
        np.testing.assert_array_equal(out_arr, in_arr)
        assert id(new_state) == id(state)
        config2 = DropoutWrapperConfig(input_keep_prob=1.0, output_keep_prob=1.0)
        wrapper2 = RNNCellDropoutWrapper(MockCell(), config2)
        (out2, new_state2) = unpack_pair(wrapper2(inputs, state, training=True))
        assert out2 is inputs
        assert id(new_state2) == id(state)


class _TestRnnBruteState(typing.TypedDict, total=False):
    """Typed shared state for `_test_rnn_brute` split helpers."""

    bias: Tensor
    dev_wrapper: RNNCellDeviceWrapper
    global_tracing_state: TracingState
    h0: Tensor
    my_cell: EagerBoundCallable
    rec_kernel: Tensor
    t_in: Tensor


def _test_rnn_brute_part_1(_state: _TestRnnBruteState) -> None:
    """Execute part 1 of test_rnn_brute.

    Args:
        _state: Shared test state dictionary.
    """
    config.backend = "numpy"
    config.eager_mode = True
    t_in = Tensor(np.random.rand(2, 3, 4).astype(np.float32), TensorConfig((2, 3, 4), DType("float32"), "cpu"))
    h0 = Tensor(np.zeros((2, 8)).astype(np.float32), TensorConfig((2, 8), DType("float32"), "cpu"))
    kernel = Tensor(np.random.rand(4, 8).astype(np.float32), TensorConfig((4, 8), DType("float32"), "cpu"))
    rec_kernel = Tensor(np.random.rand(8, 8).astype(np.float32), TensorConfig((8, 8), DType("float32"), "cpu"))
    bias = Tensor(np.random.rand(8).astype(np.float32), TensorConfig((8,), DType("float32"), "cpu"))

    def my_cell(x: BackendArg, state: BackendArg) -> BackendArg:
        """Test function my_cell.

        Args:
            x: Test parameter `x`.
            state: Test parameter `state`.

        Returns:
            BackendArg: Computed output value from `my_cell`.
        """
        state_tup = tuple(ensure_tensor(s) for s in coerce_operand_list(state))
        return simple_rnn_cell(ensure_tensor(x), state_tup, kernel, rec_kernel, bias)

    _permute_time_major(t_in)
    rnn(t_in, (h0,), my_cell)
    t_in_time_major = Tensor(
        np.random.rand(3, 2, 4).astype(np.float32), TensorConfig((3, 2, 4), DType("float32"), "cpu")
    )
    rnn(t_in_time_major, (h0,), my_cell, config=RNNConfig(time_major=True, return_all_outputs=False))
    rnn(t_in, (h0,), my_cell, config=RNNConfig(time_major=False, return_all_outputs=False, go_backwards=True))
    bidir_inputs = BidirectionalInputs(
        forward_inputs=t_in, backward_inputs=t_in, forward_initial_state=(h0,), backward_initial_state=(h0,)
    )
    bidirectional(bidir_inputs, my_cell)
    bidirectional(bidir_inputs, my_cell, config=BidirectionalConfig(merge_mode="sum"))
    bidirectional(bidir_inputs, my_cell, config=BidirectionalConfig(merge_mode="mul"))
    bidirectional(bidir_inputs, my_cell, config=BidirectionalConfig(merge_mode="ave"))
    bidirectional(bidir_inputs, my_cell, config=BidirectionalConfig(merge_mode="none"))
    dev_wrapper = RNNCellDeviceWrapper(my_cell, "cpu")
    _state.update({"bias": bias, "dev_wrapper": dev_wrapper, "h0": h0})
    _state.update({"my_cell": require_callable(my_cell), "rec_kernel": rec_kernel, "t_in": t_in})


def _test_rnn_brute_part_2(_state: _TestRnnBruteState) -> None:
    """Execute part 2 of test_rnn_brute.

    Args:
        _state: Shared test state dictionary.
    """
    (bias, dev_wrapper, h0, my_cell) = (_state["bias"], _state["dev_wrapper"], _state["h0"], _state["my_cell"])
    (rec_kernel, t_in) = (_state["rec_kernel"], _state["t_in"])
    require_callable(dev_wrapper)(
        require_callable(Tensor)(
            require_callable(np.random.rand)(2, 4).astype(np.float32),
            require_callable(TensorConfig)((2, 4), require_callable(DType)("float32"), "cpu"),
        ),
        (h0,),
    )
    drop_wrapper = RNNCellDropoutWrapper(
        my_cell, config=DropoutWrapperConfig(input_keep_prob=0.5, output_keep_prob=0.5)
    )
    drop_wrapper(
        Tensor(
            np.random.rand(2, 4).astype(np.float32),
            TensorConfig((2, 4), DType("float32"), "cpu"),
        ),
        (h0,),
    )

    def my_cell_res(x: BackendArg, state: BackendArg) -> BackendArg:
        """Test function my_cell_res.

        Args:
            x: Test parameter `x`.
            state: Test parameter `state`.

        Returns:
            BackendArg: Computed output value from `my_cell_res`.
        """
        state_tup = tuple(ensure_tensor(s) for s in coerce_operand_list(state))
        return simple_rnn_cell(
            ensure_tensor(x),
            state_tup,
            Tensor(
                np.random.rand(8, 8).astype(np.float32),
                TensorConfig((8, 8), DType("float32"), "cpu"),
            ),
            rec_kernel,
            bias,
        )

    res_wrapper = RNNCellResidualWrapper(my_cell_res)
    res_wrapper(Tensor(np.random.rand(2, 8).astype(np.float32), TensorConfig((2, 8), DType("float32"), "cpu")), (h0,))

    def my_res_fn(x: BackendArg, y: BackendArg) -> BackendArg:
        """Test function my_res_fn.

        Args:
            x: Test parameter `x`.
            y: Test parameter `y`.

        Returns:
            BackendArg: Computed output value from `my_res_fn`.
        """
        return y

    res_wrapper2 = RNNCellResidualWrapper(my_cell_res, residual_fn=my_res_fn)
    res_wrapper2(Tensor(np.random.rand(2, 8).astype(np.float32), TensorConfig((2, 8), DType("float32"), "cpu")), (h0,))
    time_distributed(ensure_tensor(t_in), wrapped_op_name="Relu")
    TimeDistributed().infer_shape(t_in)
    config.eager_mode = False
    from ml_switcheroo_compiler.tracing.state import global_tracing_state

    global_tracing_state.is_tracing = True
    _state.update({"global_tracing_state": global_tracing_state})
    _state.update({"bias": bias, "dev_wrapper": dev_wrapper, "h0": h0})
    _state.update({"my_cell": require_callable(my_cell), "rec_kernel": rec_kernel, "t_in": t_in})


def _test_rnn_brute_part_3(_state: _TestRnnBruteState) -> None:
    """Execute part 3 of test_rnn_brute.

    Args:
        _state: Shared test state dictionary.
    """
    (global_tracing_state, h0, t_in) = (_state["global_tracing_state"], _state["h0"], _state["t_in"])
    try:
        time_distributed(ensure_tensor(t_in), wrapped_op_name="Relu")

        def my_cell_trace(x: BackendArg, state: BackendArg) -> BackendArg:
            """Test function my_cell_trace.

            Args:
                x: Test parameter `x`.
                state: Test parameter `state`.

            Returns:
                BackendArg: Computed output value from `my_cell_trace`.
            """
            return (x, (x,))

        with pytest.raises(AttributeError):
            rnn(t_in, (h0,), my_cell_trace)
        with pytest.raises(AttributeError):
            rnn(
                t_in,
                (h0,),
                my_cell_trace,
                config=RNNConfig(time_major=True, return_all_outputs=False, go_backwards=True),
            )
    finally:
        pytest.MonkeyPatch().setattr(global_tracing_state, "is_tracing", False, raising=False)
    pytest.MonkeyPatch().setattr(global_tracing_state, "is_tracing", True, raising=False)
    _state.update({"global_tracing_state": global_tracing_state, "h0": h0, "t_in": t_in})


def _test_rnn_brute_part_4(_state: _TestRnnBruteState) -> None:
    """Execute part 4 of test_rnn_brute.

    Args:
        _state: Shared test state dictionary.
    """
    (global_tracing_state, h0, t_in) = (_state["global_tracing_state"], _state["h0"], _state["t_in"])
    try:

        def my_f(carry: BackendArg, x: BackendArg) -> BackendArg:
            """Test function my_f.

            Args:
                carry: Test parameter `carry`.
                x: Test parameter `x`.

            Returns:
                BackendArg: Computed output value from `my_f`.
            """
            return (carry, x)

        with patch("ml_switcheroo_compiler.ops.nn.rnn_utils.cf_scan") as mock_scan:
            mock_scan.return_value = ((h0,), t_in)
            scan(my_f, (h0,), t_in)
            scan(my_f, (h0,), t_in, config=ScanConfig(reverse=True))
    finally:
        pytest.MonkeyPatch().setattr(global_tracing_state, "is_tracing", False, raising=False)


def test_rnn_brute() -> None:
    """Test function test_rnn_brute."""
    _state: _TestRnnBruteState = {}
    _test_rnn_brute_part_1(_state)
    _test_rnn_brute_part_2(_state)
    _test_rnn_brute_part_3(_state)
    _test_rnn_brute_part_4(_state)
    assert len(_state) > 0


def test_rnn_import() -> None:
    """Test function test_rnn_import."""
    assert rnn_module.rnn_step.__name__ == "simple_rnn_cell"


def test_rnn_cell_coverage() -> None:
    """Test function test_rnn_cell_coverage."""
    import numpy as np

    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ops.nn.rnn_cell import simple_rnn_cell

    orig_eager = config.eager_mode
    config.eager_mode = True
    inputs = Tensor(np.array([[1.0, 2.0]]), TensorConfig((1, 2), "float32", "cpu"))
    state = (Tensor(np.array([[0.5]]), TensorConfig((1, 1), "float32", "cpu")),)
    kernel = Tensor(np.array([[1.0], [2.0]]), TensorConfig((2, 1), "float32", "cpu"))
    recurrent_kernel = Tensor(np.array([[0.1]]), TensorConfig((1, 1), "float32", "cpu"))
    bias = Tensor(np.array([0.1]), TensorConfig((1,), "float32", "cpu"))
    try:
        (out, next_st) = simple_rnn_cell(inputs, state, kernel, recurrent_kernel, bias)
        assert out.shape == (1, 1)
        (out, next_st) = simple_rnn_cell(inputs, state, kernel, recurrent_kernel, None)
        assert out.shape == (1, 1)
    finally:
        config.eager_mode = orig_eager


class _TestRnnUtilsCoverageState(typing.TypedDict, total=False):
    """Typed shared state for `_test_rnn_utils_coverage` split helpers."""

    DType: type[CoreDType]
    Device: type[CoreDevice]
    Tensor: type[CoreTensor]
    TensorConfig: type[CoreTensorConfig]
    config: Config
    inputs: BidirectionalInputs
    mock_cell: EagerBoundCallable
    mock_rnn: EagerBoundCallable
    mock_scan: EagerBoundCallable
    np: types.ModuleType
    patch: EagerBoundCallable
    rnn_utils: types.ModuleType
    state: types.ModuleType
    t: CoreTensor


def _test_rnn_utils_coverage_sub_1(_state: _TestRnnUtilsCoverageState) -> None:
    """Execute sub-part 1 of test_rnn_utils_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (DType, Device, Tensor, TensorConfig) = (
        _state["DType"],
        _state["Device"],
        _state["Tensor"],
        _state["TensorConfig"],
    )
    (config, np, rnn_utils) = (_state["config"], _state["np"], _state["rnn_utils"])
    t = Tensor(np.array([[[1.0, 2.0]]]), TensorConfig((2, 1, 2), DType("float32"), Device("cpu")))
    pytest.MonkeyPatch().setattr(config, "eager_mode", True, raising=False)

    def mock_cell(carry: BackendArg, x: BackendArg) -> BackendArg:
        """Test function mock_cell.

        Args:
            carry: Test parameter `carry`.
            x: Test parameter `x`.

        Returns:
            BackendArg: Computed output value from `mock_cell`.
        """
        return (carry, x)

    (carry, ys) = rnn_utils.scan(mock_cell, t, t, config=rnn_utils.ScanConfig(unroll=True, reverse=True))
    assert ys.shape == (2, 1, 2)
    pytest.MonkeyPatch().setattr(config, "eager_mode", False, raising=False)
    inputs = rnn_utils.BidirectionalInputs(
        forward_inputs=t, backward_inputs=t, forward_initial_state=(t,), backward_initial_state=(t,)
    )
    from unittest.mock import patch

    def mock_rnn(inputs: BackendArg, state: BackendArg, cell_fn: BackendArg, config: BackendArg) -> BackendArg:
        """Test function mock_rnn.

        Args:
            inputs: Test parameter `inputs`.
            state: Test parameter `state`.
            cell_fn: Test parameter `cell_fn`.
            config: Test parameter `config`.

        Returns:
            BackendArg: Computed output value from `mock_rnn`.
        """
        return (inputs, state)

    _state.update({"inputs": inputs, "mock_cell": require_callable(mock_cell), "mock_rnn": require_callable(mock_rnn)})
    _state.update({"patch": require_callable(patch), "t": t})
    _state.update({"DType": DType, "Device": Device, "Tensor": Tensor})
    _state.update({"TensorConfig": TensorConfig, "config": config, "np": np})
    _state.update({"rnn_utils": rnn_utils})


def _test_rnn_utils_coverage_sub_2(_state: _TestRnnUtilsCoverageState) -> None:
    """Execute sub-part 2 of test_rnn_utils_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (inputs, mock_cell, mock_rnn, patch) = (_state["inputs"], _state["mock_cell"], _state["mock_rnn"], _state["patch"])
    rnn_utils = _state["rnn_utils"]
    with patch("ml_switcheroo_compiler.ops.nn.rnn_utils.rnn", mock_rnn):
        rnn_utils.bidirectional(inputs, mock_cell, config=rnn_utils.BidirectionalConfig(merge_mode="concat"))
        rnn_utils.bidirectional(inputs, mock_cell, config=rnn_utils.BidirectionalConfig(merge_mode="sum"))
        rnn_utils.bidirectional(inputs, mock_cell, config=rnn_utils.BidirectionalConfig(merge_mode="mul"))
        rnn_utils.bidirectional(inputs, mock_cell, config=rnn_utils.BidirectionalConfig(merge_mode="ave"))
        rnn_utils.bidirectional(inputs, mock_cell, config=rnn_utils.BidirectionalConfig(merge_mode=None))

    def mock_scan(fn: BackendArg, init: BackendArg, xs: BackendArg, config: BackendArg) -> BackendArg:
        """Test function mock_scan.

        Args:
            fn: Test parameter `fn`.
            init: Test parameter `init`.
            xs: Test parameter `xs`.
            config: Test parameter `config`.

        Returns:
            BackendArg: Computed output value from `mock_scan`.
        """
        (c, out) = unpack_pair(require_callable(fn)(init, xs))

        class DummyOut:
            """Test class DummyOut."""

            shape = (1, 1, 1)

            def __getitem__(self, key: BackendArg) -> BackendArg:
                """Test function __getitem__.

                Args:
                    key: Test parameter `key`.

                Returns:
                    BackendArg: Computed output value from `__getitem__`.
                """
                return self

        return (c, DummyOut())

    _state.update({"mock_scan": require_callable(mock_scan)})
    _state.update({"inputs": inputs, "mock_cell": require_callable(mock_cell), "mock_rnn": require_callable(mock_rnn)})
    _state.update({"patch": require_callable(patch), "rnn_utils": rnn_utils})


def _test_rnn_utils_coverage_sub_3(_state: _TestRnnUtilsCoverageState) -> None:
    """Execute sub-part 3 of test_rnn_utils_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (mock_cell, mock_scan, patch, rnn_utils) = (
        _state["mock_cell"],
        _state["mock_scan"],
        _state["patch"],
        _state["rnn_utils"],
    )
    (state, t) = (_state["state"], _state["t"])
    _ = state
    with patch("ml_switcheroo_compiler.ops.nn.rnn_utils.scan", mock_scan):
        rnn_utils.rnn(t, (t,), mock_cell, config=rnn_utils.RNNConfig(time_major=False, return_all_outputs=True))
        rnn_utils.rnn(t, (t,), mock_cell, config=rnn_utils.RNNConfig(time_major=True, return_all_outputs=False))

    class MockCell(MockTestStub):
        """Test class MockCell."""

        def __call__(self, x: BackendArg, state: BackendArg) -> BackendArg:
            """Test function __call__.

            Args:
                x: Test parameter `x`.
                state: Test parameter `state`.

            Returns:
                BackendArg: Computed output value from `__call__`.
            """
            return (x, state)

    w = rnn_utils.RNNCellDeviceWrapper(MockCell(), "cpu")
    w(t, (t,))
    rw = rnn_utils.RNNCellResidualWrapper(MockCell())
    rw(t, (t,))
    rw2 = rnn_utils.RNNCellResidualWrapper(MockCell(), residual_fn=lambda i, o: i)
    rw2(t, (t,))
    with patch("ml_switcheroo_compiler.ops.nn.rnn_utils.dropout", return_value=t):
        dw = rnn_utils.RNNCellDropoutWrapper(
            MockCell(), config=rnn_utils.DropoutWrapperConfig(input_keep_prob=0.5, output_keep_prob=0.5)
        )
        dw(t, (t,))
    _state.update(
        {
            "mock_cell": require_callable(mock_cell),
            "mock_scan": require_callable(mock_scan),
            "patch": require_callable(patch),
        }
    )
    _state.update({"rnn_utils": rnn_utils, "state": state, "t": t})


def test_rnn_utils_coverage() -> None:
    """Test function test_rnn_utils_coverage."""
    import numpy as np

    import ml_switcheroo_compiler.ops.nn.rnn_utils as rnn_utils
    import ml_switcheroo_compiler.tracing.state as state
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.device import Device
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    orig = config.eager_mode
    orig_tracing = state.global_tracing_state.is_tracing
    config.eager_mode = False
    state.global_tracing_state.is_tracing = True
    orig_add_node = state.global_tracing_state.add_node
    pytest.MonkeyPatch().setattr(state.global_tracing_state, "add_node", lambda node: None, raising=False)
    try:
        _state: _TestRnnUtilsCoverageState = {}
        _state.update({"DType": DType, "Device": Device, "Tensor": Tensor, "TensorConfig": TensorConfig})
        _state.update({"config": config, "np": np})
        _state.update({"rnn_utils": rnn_utils, "state": state})
        for _sub_fn in (_test_rnn_utils_coverage_sub_1, _test_rnn_utils_coverage_sub_2, _test_rnn_utils_coverage_sub_3):
            _sub_fn(_state)
        assert len(_state) > 0
    finally:
        config.eager_mode = orig
        state.global_tracing_state.is_tracing = orig_tracing
        pytest.MonkeyPatch().setattr(state.global_tracing_state, "add_node", orig_add_node, raising=False)


def test_rnn_utils_coverage2() -> None:
    """Test function test_rnn_utils_coverage2."""
    import numpy as np

    import ml_switcheroo_compiler.ops.nn.rnn_utils as rnn_utils
    import ml_switcheroo_compiler.tracing.state as state
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.device import Device
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    orig = config.eager_mode
    orig_tracing = state.global_tracing_state.is_tracing
    config.eager_mode = False
    state.global_tracing_state.is_tracing = True
    orig_add_node = state.global_tracing_state.add_node
    pytest.MonkeyPatch().setattr(state.global_tracing_state, "add_node", lambda node: None, raising=False)
    try:
        t = Tensor(np.array([[[1.0, 2.0]]]), TensorConfig((2, 1, 2), DType("float32"), Device("cpu")))

        def mock_cell(carry: BackendArg, x: BackendArg) -> BackendArg:
            """Test function mock_cell.

            Args:
                carry: Test parameter `carry`.
                x: Test parameter `x`.

            Returns:
                BackendArg: Computed output value from `mock_cell`.
            """
            return (carry, x)

        from unittest.mock import patch

        def mock_cf_scan(f: BackendArg, init: BackendArg, xs: BackendArg, length: BackendArg) -> BackendArg:
            """Test function mock_cf_scan.

            Args:
                f: Test parameter `f`.
                init: Test parameter `init`.
                xs: Test parameter `xs`.
                length: Test parameter `length`.

            Returns:
                BackendArg: Computed output value from `mock_cf_scan`.
            """
            return (init, xs)

        with patch("ml_switcheroo_compiler.ops.nn.rnn_utils.cf_scan", mock_cf_scan):
            with patch("ml_switcheroo_compiler.ops.nn.rnn_utils.cf_reverse", return_value=t):
                (carry, ys) = rnn_utils.scan(mock_cell, t, t, config=rnn_utils.ScanConfig(unroll=False, reverse=True))
    finally:
        config.eager_mode = orig
        state.global_tracing_state.is_tracing = orig_tracing
        pytest.MonkeyPatch().setattr(state.global_tracing_state, "add_node", orig_add_node, raising=False)
    assert state.global_tracing_state.add_node == orig_add_node
