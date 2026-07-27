import numpy as np

import ml_switcheroo_compiler.ops as _ops
from ml_switcheroo_compiler.backends.numpy.eager.math_misc import (
    _np_confusion_matrix,
    _np_decode_csv,
    _np_decode_image,
    _np_descriptive,
    _np_distributions,
    _np_parse_example,
    _np_parse_tensor,
    _np_rawmatmul,
    _np_read_file,
    _np_rem,
    _np_serialize_tensor,
    _np_sparsedensematmul,
    _np_write_file,
)


class MockCallableClass:
    def __call__(self, *args, **kwargs):
        return args


def test_mock_ops_subclass():
    original_rawmatmul = getattr(_ops, "RawMatMul", None)
    original_sparsedensematmul = getattr(_ops, "SparseDenseMatMul", None)
    original_decode_csv = getattr(_ops, "decode_csv", None)
    original_decode_image = getattr(_ops, "decode_image", None)
    original_parse_example = getattr(_ops, "parse_example", None)
    original_parse_tensor = getattr(_ops, "parse_tensor", None)
    original_read_file = getattr(_ops, "read_file", None)
    original_rem = getattr(_ops, "rem", None)
    original_serialize_tensor = getattr(_ops, "serialize_tensor", None)
    original_write_file = getattr(_ops, "write_file", None)
    original_confusion_matrix = getattr(_ops, "confusion_matrix", None)
    original_descriptive = getattr(_ops, "descriptive", None)
    original_distributions = getattr(_ops, "distributions", None)

    try:
        _ops.RawMatMul = MockCallableClass
        _ops.SparseDenseMatMul = MockCallableClass
        _ops.decode_csv = MockCallableClass
        _ops.decode_image = MockCallableClass
        _ops.parse_example = MockCallableClass
        _ops.parse_tensor = MockCallableClass
        _ops.read_file = MockCallableClass
        _ops.rem = MockCallableClass
        _ops.serialize_tensor = MockCallableClass
        _ops.write_file = MockCallableClass
        _ops.confusion_matrix = MockCallableClass
        _ops.descriptive = MockCallableClass
        _ops.distributions = MockCallableClass

        _np_rawmatmul(np, np.ones((2, 2)), np.ones((2, 2)))
        _np_sparsedensematmul(np, np.ones((2, 2)), np.ones((2, 2)))
        _np_decode_csv(np, "1,2,3")
        _np_decode_image(np, b"")
        _np_parse_example(np, b"")
        # For _np_parse_tensor we need a valid arg if using astype fallback
        _np_parse_tensor(np, [1, 2], out_type=np.float32)
        _np_read_file(np, "test.txt")
        print("HASATTR REM:", hasattr(_ops, "rem"))
        _np_rem(np, np.array([5]), np.array([2]))
        _np_serialize_tensor(np, np.ones((2, 2)))
        _np_write_file(np, "test.txt", "hello")
        _np_confusion_matrix(np, np.array([1]), np.array([1]))
        _np_descriptive(np, np.array([1]))
        _np_distributions(np, np.array([1]))

    finally:
        _ops.RawMatMul = original_rawmatmul
        _ops.SparseDenseMatMul = original_sparsedensematmul
        _ops.decode_csv = original_decode_csv
        _ops.decode_image = original_decode_image
        _ops.parse_example = original_parse_example
        _ops.parse_tensor = original_parse_tensor
        _ops.read_file = original_read_file
        _ops.rem = original_rem
        _ops.serialize_tensor = original_serialize_tensor
        _ops.write_file = original_write_file
        _ops.confusion_matrix = original_confusion_matrix
        _ops.descriptive = original_descriptive
        _ops.distributions = original_distributions


def test_mock_ops_missing():
    original_rawmatmul = getattr(_ops, "RawMatMul", None)
    original_sparsedensematmul = getattr(_ops, "SparseDenseMatMul", None)
    original_decode_csv = getattr(_ops, "decode_csv", None)
    original_decode_image = getattr(_ops, "decode_image", None)
    original_parse_example = getattr(_ops, "parse_example", None)
    original_parse_tensor = getattr(_ops, "parse_tensor", None)
    original_read_file = getattr(_ops, "read_file", None)
    original_rem = getattr(_ops, "rem", None)
    original_serialize_tensor = getattr(_ops, "serialize_tensor", None)
    original_write_file = getattr(_ops, "write_file", None)
    original_confusion_matrix = getattr(_ops, "confusion_matrix", None)
    original_descriptive = getattr(_ops, "descriptive", None)
    original_distributions = getattr(_ops, "distributions", None)

    try:
        del _ops.RawMatMul
        del _ops.SparseDenseMatMul
        del _ops.decode_csv
        del _ops.decode_image
        del _ops.parse_example
        del _ops.parse_tensor
        del _ops.read_file
        del _ops.rem
        del _ops.serialize_tensor
        del _ops.write_file
        del _ops.confusion_matrix
        del _ops.descriptive
        del _ops.distributions

        _np_rawmatmul(np, np.ones((2, 2)), np.ones((2, 2)))
        _np_sparsedensematmul(np, np.ones((2, 2)), np.ones((2, 2)))

        try:
            _np_decode_csv(np, "1,2,3")
        except:
            pass
        try:
            _np_decode_image(np, b"")
        except:
            pass
        try:
            _np_parse_example(np, b"")
        except:
            pass
        try:
            _np_parse_tensor(np, b"")
        except:
            pass
        try:
            _np_read_file(np, "test.txt")
        except:
            pass

        print("HASATTR REM:", hasattr(_ops, "rem"))
        _np_rem(np, np.array([5]), np.array([2]))
        _np_serialize_tensor(np, np.ones((2, 2)))

        try:
            _np_write_file(np, "test.txt", "hello")
        except:
            pass

        _np_confusion_matrix(np, np.array([1]), np.array([1]))
        _np_descriptive(np, np.array([1]))
        _np_distributions(np, np.array([1]))

    finally:
        _ops.RawMatMul = original_rawmatmul
        _ops.SparseDenseMatMul = original_sparsedensematmul
        _ops.decode_csv = original_decode_csv
        _ops.decode_image = original_decode_image
        _ops.parse_example = original_parse_example
        _ops.parse_tensor = original_parse_tensor
        _ops.read_file = original_read_file
        _ops.rem = original_rem
        _ops.serialize_tensor = original_serialize_tensor
        _ops.write_file = original_write_file
        _ops.confusion_matrix = original_confusion_matrix
        _ops.descriptive = original_descriptive
        _ops.distributions = original_distributions


def test_confusion_matrix_num_classes():
    _np_confusion_matrix(np, np.array([1]), np.array([1]), num_classes=5)
