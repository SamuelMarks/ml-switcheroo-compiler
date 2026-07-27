import ml_switcheroo_compiler.tracing.state as state
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.ops.reductions.frontend_pool import UnpoolOptions, adaptive_avg_pool2d, adaptive_avg_pool3d, adaptive_max_pool2d, adaptive_max_pool3d, fold, fractional_max_pool2d, fractional_max_pool3d, max_unpool1d, max_unpool2d, max_unpool3d, unfold


def test_frontend_pool_tracing():
    orig_eager = config.eager_mode
    orig_tracing = state.global_tracing_state.is_tracing
    orig_add_node = state.global_tracing_state.add_node

    try:
        config.eager_mode = False
        state.global_tracing_state.is_tracing = True
        state.global_tracing_state.add_node = lambda node: None

        class MockData:
            id = "t1"

        class MockTensor:
            def __init__(self, shape):
                self.data = MockData()
                self.shape = shape
                self.dtype = DType("float32")
                self.device = Device("cpu")

            def eval(self):
                return self

        mt3 = MockTensor((1, 1, 1))
        mt4 = MockTensor((1, 1, 1, 1))
        mt5 = MockTensor((1, 1, 1, 1, 1))

        fractional_max_pool2d(mt4, output_size=[1, 1])
        adaptive_avg_pool2d(mt4, output_size=[1, 1])
        adaptive_max_pool2d(mt4, output_size=[1, 1])
        unfold(mt4, kernel_size=(1, 1))
        fold(mt3, output_size=(1, 1), kernel_size=(1, 1))
        fractional_max_pool3d(mt5, output_size=[1, 1, 1])
        adaptive_avg_pool3d(mt5, output_size=[1, 1, 1])
        adaptive_max_pool3d(mt5, output_size=[1, 1, 1], return_indices=True)
        adaptive_max_pool3d(mt5, output_size=[1, 1, 1], return_indices=False)
        max_unpool1d(mt3, mt3, UnpoolOptions(kernel_size=(1,), output_size=(1,)))
        max_unpool2d(mt4, mt4, UnpoolOptions(kernel_size=(1, 1), output_size=(1, 1)))
        max_unpool3d(mt5, mt5, UnpoolOptions(kernel_size=(1, 1, 1), output_size=(1, 1, 1)))

    finally:
        config.eager_mode = orig_eager
        state.global_tracing_state.is_tracing = orig_tracing
        state.global_tracing_state.add_node = orig_add_node
