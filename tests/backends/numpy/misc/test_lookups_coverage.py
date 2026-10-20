"""Tests for test_lookups_coverage."""

from __future__ import annotations

import typing

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import lookups as lookups_module

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol


def test_lookups_ops() -> None:
    """Test lookup operations directly and through eager registry.

    Returns:
        None
    """
    inputs = np.array([1, 2, 3])
    assert typing.cast("EagerCallableProtocol", np.array_equal)(typing.cast("EagerCallableProtocol", lookups_module._np_hashing)(np, inputs, num_bins=10), inputs)
    reg_hashing = numpy_eager_registry.get("Hashing")
    assert typing.cast("EagerCallableProtocol", np.array_equal)(typing.cast("EagerCallableProtocol", reg_hashing)(np, inputs, num_bins=10), inputs)

    assert typing.cast("EagerCallableProtocol", np.array_equal)(typing.cast("EagerCallableProtocol", lookups_module._np_integer_lookup)(np, inputs), inputs)
    reg_int_lookup = numpy_eager_registry.get("IntegerLookup")
    assert typing.cast("EagerCallableProtocol", np.array_equal)(typing.cast("EagerCallableProtocol", reg_int_lookup)(np, inputs), inputs)

    vocab = np.array([2, 3, 4])
    lookup_res = lookups_module._np_lookup(np, inputs, vocab)
    assert typing.cast("EagerCallableProtocol", np.array_equal)(lookup_res, typing.cast("EagerCallableProtocol", np.array)([0, 0, 1]))

    reg_lookup = numpy_eager_registry.get("Lookup")
    assert reg_lookup is not None

    str_inputs = np.array(["a", "b"])
    assert typing.cast("EagerCallableProtocol", np.array_equal)(typing.cast("EagerCallableProtocol", lookups_module._np_string_lookup)(np, str_inputs), str_inputs)
    reg_str_lookup = numpy_eager_registry.get("StringLookup")
    assert typing.cast("EagerCallableProtocol", np.array_equal)(typing.cast("EagerCallableProtocol", reg_str_lookup)(np, str_inputs), str_inputs)
