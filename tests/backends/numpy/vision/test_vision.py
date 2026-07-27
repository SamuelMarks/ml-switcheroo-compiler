import numpy as np

from ml_switcheroo_compiler.backends.numpy.eager.vision import (
    TransformOptions,
    random_flip_numpy,
    random_rotation_numpy,
    resize_bicubic,
    resize_bilinear,
    resize_lanczos3,
    resize_nearest,
)


def test_resize_bilinear():
    images = np.ones((1, 4, 4, 3), dtype=np.float32)

    # align_corners=False
    res1 = resize_bilinear(np, images, (2, 2))
    assert res1.shape == (1, 2, 2, 3)

    # align_corners=True
    res2 = resize_bilinear(np, images, (2, 2), align_corners=True)
    assert res2.shape == (1, 2, 2, 3)

    # rank 3
    img_rank3 = np.ones((4, 4, 3), dtype=np.float32)
    res3 = resize_bilinear(np, img_rank3, (2, 2))
    assert res3.shape == (2, 2, 3)

    # rank 2
    res_rank2 = resize_bilinear(np, np.ones((4, 4)), (2, 2))
    assert res_rank2.shape == (4, 4)


def test_resize_nearest():
    images = np.ones((1, 4, 4, 3), dtype=np.float32)

    # align_corners=False
    res1 = resize_nearest(np, images, (2, 2))
    assert res1.shape == (1, 2, 2, 3)

    # align_corners=True
    res2 = resize_nearest(np, images, (2, 2), align_corners=True)
    assert res2.shape == (1, 2, 2, 3)

    # rank 3
    img_rank3 = np.ones((4, 4, 3), dtype=np.float32)
    res3 = resize_nearest(np, img_rank3, (2, 2))
    assert res3.shape == (2, 2, 3)

    # rank 2
    res_rank2 = resize_nearest(np, np.ones((4, 4)), (2, 2))
    assert res_rank2.shape == (4, 4)


def test_resize_bicubic():
    images = np.ones((1, 4, 4, 3), dtype=np.float32)
    res = resize_bicubic(np, images, (2, 2))
    assert res.shape == (1, 2, 2, 3)


def test_resize_lanczos3():
    images = np.ones((1, 4, 4, 3), dtype=np.float32)
    res = resize_lanczos3(np, images, (2, 2))
    assert res.shape == (1, 2, 2, 3)


def test_random_flip_numpy():
    images = np.arange(16).reshape(1, 4, 4, 1)

    res1 = random_flip_numpy(np, images, mode="horizontal", seed=42)
    assert res1.shape == (1, 4, 4, 1)

    res2 = random_flip_numpy(np, images, mode="vertical", seed=42)
    assert res2.shape == (1, 4, 4, 1)

    res3 = random_flip_numpy(np, images, mode="horizontal_and_vertical", seed=42)
    assert res3.shape == (1, 4, 4, 1)

    # rank 3
    img_rank3 = np.arange(16).reshape(4, 4, 1)
    res4 = random_flip_numpy(np, img_rank3, seed=42)
    assert res4.shape == (4, 4, 1)

    # rank 2
    res_rank2 = random_flip_numpy(np, np.ones((4, 4)), seed=42)
    assert res_rank2.shape == (4, 4)


def test_random_rotation_numpy():
    images = np.ones((1, 4, 4, 3), dtype=np.float32)

    res1 = random_rotation_numpy(np, images, factor=0.1, options=TransformOptions(seed=42))
    assert res1.shape == (1, 4, 4, 3)

    # factor as tuple
    res2 = random_rotation_numpy(np, images, factor=(-0.1, 0.1), options=TransformOptions(seed=42))
    assert res2.shape == (1, 4, 4, 3)

    # fill mode constant
    res3 = random_rotation_numpy(np, images, factor=0.1, options=TransformOptions(fill_mode="constant", seed=42))
    assert res3.shape == (1, 4, 4, 3)

    # rank 3
    img_rank3 = np.ones((4, 4, 3), dtype=np.float32)
    res4 = random_rotation_numpy(np, img_rank3, factor=0.1, options=TransformOptions(seed=42))
    assert res4.shape == (4, 4, 3)

    # rank 2
    res_rank2 = random_rotation_numpy(np, np.ones((4, 4)), factor=0.1)
    assert res_rank2.shape == (4, 4)
