import numpy as np
from ml_switcheroo_compiler import ops
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.tracing.tracer import _tracer
from ml_switcheroo_compiler.backends.registry import BackendRegistry


def setup_module():
    config.eager_mode = True


def test_adjust_brightness():
    x = ops.array(np.random.randn(2, 4, 4, 3).astype(np.float32))
    _ = ops.adjust_brightness(x, delta=0.1)
    # the naive impl just adds delta, let's just make sure it runs
    pass


def test_adjust_brightness_ast():
    config.eager_mode = False
    _tracer.start_tracing("Test")
    x = ops.array(np.array([[[[0.5, 0.5, 0.5]]]]))
    _ = ops.adjust_brightness(x, delta=0.1)
    graph = _tracer.stop_tracing()

    gen_cls = BackendRegistry.get("numpy")
    gen = gen_cls(graph)
    code_str = gen.generate()
    assert "_adjust_brightness" in code_str
    config.eager_mode = True


def test_adjust_contrast():
    x = ops.array(np.random.randn(2, 4, 4, 3).astype(np.float32))
    _ = ops.adjust_contrast(x, contrast_factor=0.5)
    pass


def test_adjust_contrast_ast():
    config.eager_mode = False
    _tracer.start_tracing("Test")
    x = ops.array(np.array([[[[0.5, 0.5, 0.5]]]]))
    _ = ops.adjust_contrast(x, contrast_factor=0.5)
    graph = _tracer.stop_tracing()

    gen_cls = BackendRegistry.get("numpy")
    gen = gen_cls(graph)
    code_str = gen.generate()
    assert "_adjust_contrast" in code_str
    config.eager_mode = True


def test_adjust_hue():
    x = ops.array(np.random.randn(2, 4, 4, 3).astype(np.float32))
    _ = ops.adjust_hue(x, delta=0.5)
    pass


def test_adjust_hue_ast():
    config.eager_mode = False
    _tracer.start_tracing("Test")
    x = ops.array(np.array([[[[0.5, 0.5, 0.5]]]]))
    _ = ops.adjust_hue(x, delta=0.5)
    graph = _tracer.stop_tracing()

    gen_cls = BackendRegistry.get("numpy")
    gen = gen_cls(graph)
    code_str = gen.generate()
    assert "_adjust_hue" in code_str
    config.eager_mode = True


def test_adjust_saturation():
    x = ops.array(np.random.randn(2, 4, 4, 3).astype(np.float32))
    _ = ops.adjust_saturation(x, saturation_factor=0.5)
    pass


def test_adjust_saturation_ast():
    config.eager_mode = False
    _tracer.start_tracing("Test")
    x = ops.array(np.array([[[[0.5, 0.5, 0.5]]]]))
    _ = ops.adjust_saturation(x, saturation_factor=0.5)
    graph = _tracer.stop_tracing()

    gen_cls = BackendRegistry.get("numpy")
    gen = gen_cls(graph)
    code_str = gen.generate()
    assert "_adjust_saturation" in code_str
    config.eager_mode = True


def test_affine_generator():
    angles = ops.array(np.random.randn(2).astype(np.float32))
    shears = ops.array(np.random.randn(2, 2).astype(np.float32))
    zooms = ops.array(np.random.randn(2, 2).astype(np.float32))
    _ = ops.affine_generator(batch_size=2, angles=angles, shears=shears, zooms=zooms)
    # the naive impl returns the first input in numpy fallback, or a specific shape depending on backend


def test_affine_generator_ast():
    config.eager_mode = False
    _tracer.start_tracing("Test")
    angles = ops.array(np.random.randn(2).astype(np.float32))
    shears = ops.array(np.random.randn(2, 2).astype(np.float32))
    zooms = ops.array(np.random.randn(2, 2).astype(np.float32))
    _ = ops.affine_generator(batch_size=2, angles=angles, shears=shears, zooms=zooms)
    graph = _tracer.stop_tracing()

    gen_cls = BackendRegistry.get("numpy")
    gen = gen_cls(graph)
    code_str = gen.generate()
    assert "_affine_generator" in code_str
    config.eager_mode = True


def test_affine_grid():
    theta = ops.array(np.random.randn(2, 2, 3).astype(np.float32))
    _ = ops.affine_grid(theta, size=(2, 3, 4, 4), align_corners=False)


def test_affine_grid_ast():
    config.eager_mode = False
    _tracer.start_tracing("Test")
    theta = ops.array(np.random.randn(2, 2, 3).astype(np.float32))
    _ = ops.affine_grid(theta, size=(2, 3, 4, 4), align_corners=False)
    graph = _tracer.stop_tracing()

    gen_cls = BackendRegistry.get("numpy")
    gen = gen_cls(graph)
    code_str = gen.generate()
    assert "_affine_grid" in code_str
    config.eager_mode = True


def test_affine_transform():
    images = ops.array(np.random.randn(2, 4, 4, 3).astype(np.float32))
    transforms = ops.array(np.random.randn(2, 8).astype(np.float32))
    _ = ops.affine_transform(images, transforms=transforms, interpolation="nearest")
    pass


def test_affine_transform_ast():
    config.eager_mode = False
    _tracer.start_tracing("Test")
    images = ops.array(np.random.randn(2, 4, 4, 3).astype(np.float32))
    transforms = ops.array(np.random.randn(2, 8).astype(np.float32))
    _ = ops.affine_transform(images, transforms=transforms, interpolation="nearest")
    graph = _tracer.stop_tracing()

    gen_cls = BackendRegistry.get("numpy")
    gen = gen_cls(graph)
    code_str = gen.generate()
    assert "_affine_transform" in code_str
    config.eager_mode = True
