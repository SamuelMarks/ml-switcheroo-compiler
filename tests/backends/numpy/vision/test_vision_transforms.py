from unittest.mock import patch

import numpy as np

import ml_switcheroo_compiler.backends.numpy.eager.vision_transforms as vt


@patch("ml_switcheroo_compiler.backends.numpy.eager.vision_transforms.random_shear_eager")
@patch("ml_switcheroo_compiler.backends.numpy.eager.vision_transforms.random_perspective_eager")
@patch("ml_switcheroo_compiler.backends.numpy.eager.vision_transforms.random_elastic_transform_eager")
def test_vision_transforms_coverage(mock_elastic, mock_perspective, mock_shear):
    mock_shear.return_value = 1
    res = vt._np_random_shear(np, np.zeros((2, 2, 3)), 1.0)
    assert res == 1

    mock_perspective.return_value = 2
    res = vt._np_random_perspective(np, np.zeros((2, 2, 3)), 1.0)
    assert res == 2

    mock_elastic.return_value = 3
    res = vt._np_random_elastic_transform(np, np.zeros((2, 2, 3)), 1.0, 1.0)
    assert res == 3
