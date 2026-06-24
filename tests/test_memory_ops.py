import numpy as np
from ml_switcheroo_compiler import ops


def test_memory_ops():
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True

    x = ops.array(np.random.randn(2, 4).astype(np.float32))

    # associative_scan
    scan_out = ops.associative_scan(lambda a, b: ops.add(a, b), x)
    assert scan_out is not None

    # convert_to_numpy
    np_x = ops.convert_to_numpy(x)
    assert isinstance(np_x, np.ndarray)

    # numpy
    np_x2 = ops.numpy(x)
    assert isinstance(np_x2, np.ndarray)

    # convert_to_tensor
    tensor_x = ops.convert_to_tensor(np_x)
    assert ops.is_tensor(tensor_x)

    # extract_sequences
    seqs = ops.extract_sequences(x, sequence_length=2, sequence_stride=1)
    assert seqs is not None

    # get_item
    item = ops.get_item(x, 0)
    assert item is not None

    # identity
    idx = ops.identity(4)
    assert idx is not None

    # multi_hot
    indices = ops.array(np.array([[0, 1], [1, 2]]).astype(np.int32))
    mh = ops.multi_hot(indices, num_classes=5)
    assert mh is not None

    # normalize
    n = ops.normalize(x)
    assert n is not None

    # one_hot
    oh = ops.one_hot(ops.array(np.array([1, 2]).astype(np.int32)), num_classes=5)
    assert oh is not None

    # ravel
    r = ops.ravel(x)
    assert r is not None

    # rearrange
    rearr = ops.rearrange(x, "b c -> c b")
    assert rearr is not None

    # saturate_cast
    sat = ops.saturate_cast(x, "float16")
    assert sat is not None

    # scatter_update
    upd = ops.scatter_update(
        x,
        ops.array(np.array([0]).astype(np.int32)),
        ops.array(np.random.randn(1, 4).astype(np.float32)),
    )
    assert upd is not None

    # slice_update
    supd = ops.slice_update(x, [0, 0], x)
    assert supd is not None
