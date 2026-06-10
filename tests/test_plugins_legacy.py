"""Tests for legacy plugins."""

from ml_switcheroo.plugins_legacy import (
    LegacyPlugin,
    AttentionPacking,
    BatchNormPlugin,
    CheckpointKeys,
    NnxToTorchParams,
    StateContainer,
    StateFlagInjection,
    ContextToFunctionWrap,
    MethodProperty,
    AutoFSDPWrapper,
    OptimizerStep,
    LossWrapper,
    MLXOptimizers,
    Schedulers,
    DataLoader,
    TFDataLoader,
    IOHandler,
    DeviceAllocator,
    DeviceChecks,
    Casting,
    Clipping,
    EinsumPlugin,
    FlattenPlugin,
    GatherPlugin,
    ScatterPlugin,
    InTopKPlugin,
    TopKPlugin,
    PaddingPlugin,
    ReshapePlugin,
    ShapePackingPlugin,
    JaxDecompose,
    InplaceUnroll,
    LoopUnroll,
    StaticUnroll,
    KerasSequential,
    MLXExtras,
    RNGThreading,
    UtilsPlugin,
)


def test_base_plugin() -> None:
    """Test the base plugin class."""
    plugin = LegacyPlugin(config={"a": 1})
    assert plugin.config == {"a": 1}
    state = plugin.apply({})
    assert state["LegacyPlugin"] is True

    plugin2 = LegacyPlugin()
    assert plugin2.config == {}


def test_all_plugins() -> None:
    """Test all plugin stubs."""
    plugins = [
        AttentionPacking(),
        BatchNormPlugin(),
        CheckpointKeys(),
        NnxToTorchParams(),
        StateContainer(),
        StateFlagInjection(),
        ContextToFunctionWrap(),
        MethodProperty(),
        AutoFSDPWrapper(),
        OptimizerStep(),
        LossWrapper(),
        MLXOptimizers(),
        Schedulers(),
        DataLoader(),
        TFDataLoader(),
        IOHandler(),
        DeviceAllocator(),
        DeviceChecks(),
        Casting(),
        Clipping(),
        EinsumPlugin(),
        FlattenPlugin(),
        GatherPlugin(),
        ScatterPlugin(),
        InTopKPlugin(),
        TopKPlugin(),
        PaddingPlugin(),
        ReshapePlugin(),
        ShapePackingPlugin(),
        JaxDecompose(),
        InplaceUnroll(),
        LoopUnroll(),
        StaticUnroll(),
        KerasSequential(),
        MLXExtras(),
        RNGThreading(),
        UtilsPlugin(),
    ]
    for plugin in plugins:
        state = plugin.apply({})
        assert state[plugin.__class__.__name__] is True
