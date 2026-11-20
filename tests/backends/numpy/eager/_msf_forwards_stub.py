"""Shared dummy backend stubs and state builders for NumPy special-math test suites."""

from __future__ import annotations

import types
import typing

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import BackendArg
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from tests.testing_utils import MockTestStub, NumpyTestArray


class _MathMiscDummyBackend(MockTestStub):
    """Minimal dummy backend providing `any`, `isinf`, and `isnan` predicates."""

    @staticmethod
    def any(x: BackendArg) -> BackendArg:
        """Evaluate `np.any` on `x`.

        Args:
            x (BackendArg): Input array or scalar.

        Returns:
            BackendArg: Boolean reduction result.
        """
        return bool(np.any(ensure_ndarray(x)))

    @staticmethod
    def isinf(x: BackendArg) -> BackendArg:
        """Evaluate `np.isinf` on `x`.

        Args:
            x (BackendArg): Input array or scalar.

        Returns:
            BackendArg: Elementwise infinity mask.
        """
        return np.isinf(ensure_ndarray(x))

    @staticmethod
    def isnan(x: BackendArg) -> BackendArg:
        """Evaluate `np.isnan` on `x`.

        Args:
            x (BackendArg): Input array or scalar.

        Returns:
            BackendArg: Elementwise NaN mask.
        """
        return np.isnan(ensure_ndarray(x))


class BaseMathMiscState(typing.TypedDict, total=False):
    """Typed shared base state for `_test_math_misc_coverage` split helpers."""

    DummyBk: type[_MathMiscDummyBackend]
    arg1: NumpyTestArray
    arg2: NumpyTestArray
    arg_bool: NumpyTestArray
    arg_int: NumpyTestArray
    bk: types.ModuleType
    dummy_bk: MockTestStub


def _make_base_math_state(part_start: int) -> BaseMathMiscState:
    """Construct initial shared operand state dictionary matching part boundary `part_start`.

    Args:
        part_start (int): 1-based starting part index for the mathematical family slice.

    Returns:
        BaseMathMiscState: Seeded operand and backend dictionary.
    """
    arg1: NumpyTestArray = np.array([0.0, 0.0]) if part_start > 156 else np.array([1.0, 2.0])
    arg2: NumpyTestArray
    if part_start <= 10:
        arg2 = np.array([2.0, 3.0])
    elif part_start <= 81:
        arg2 = np.log(np.array([2.0, 3.0]))
    elif part_start <= 102:
        arg2 = np.log(np.array([2.0, 3.0])) - 1.0
    else:
        arg2 = np.array([1.0, 2.0])
    arg_bool: NumpyTestArray = np.array([True, False])
    arg_int: NumpyTestArray = np.array([1, 2])
    return {
        "DummyBk": _MathMiscDummyBackend,
        "arg1": arg1,
        "arg2": arg2,
        "arg_bool": arg_bool,
        "arg_int": arg_int,
        "bk": np,
        "dummy_bk": _MathMiscDummyBackend(),
    }


class _SimpleForwardsDummyBackend:
    """Test class _SimpleForwardsDummyBackend."""

    def applyoveraxes(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Test function applyoveraxes.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.

        Returns:
            BackendArg: Computed output value from `applyoveraxes`.
        """
        return 0

    def arrayrepr(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Test function arrayrepr.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.

        Returns:
            BackendArg: Computed output value from `arrayrepr`.
        """
        return 1

    def arraystr(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Test function arraystr.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.

        Returns:
            BackendArg: Computed output value from `arraystr`.
        """
        return 2

    def atleast1d(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Test function atleast1d.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.

        Returns:
            BackendArg: Computed output value from `atleast1d`.
        """
        return 3

    def atleast2d(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Test function atleast2d.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.

        Returns:
            BackendArg: Computed output value from `atleast2d`.
        """
        return 4

    def atleast3d(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Test function atleast3d.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.

        Returns:
            BackendArg: Computed output value from `atleast3d`.
        """
        return 5

    def average(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Test function average.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.

        Returns:
            BackendArg: Computed output value from `average`.
        """
        return 6

    def betainc(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Test function betainc.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.

        Returns:
            BackendArg: Computed output value from `betainc`.
        """
        return 7

    def block(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Test function block.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.

        Returns:
            BackendArg: Computed output value from `block`.
        """
        return 8

    def callable(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Test function callable.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.

        Returns:
            BackendArg: Computed output value from `callable`.
        """
        return 9

    def clip(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Test function clip.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.

        Returns:
            BackendArg: Computed output value from `clip`.
        """
        return 10

    def corrcoef(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Test function corrcoef.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.

        Returns:
            BackendArg: Computed output value from `corrcoef`.
        """
        return 11

    def cov(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Test function cov.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.

        Returns:
            BackendArg: Computed output value from `cov`.
        """
        return 12

    def debuginfs(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Test function debuginfs.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.

        Returns:
            BackendArg: Computed output value from `debuginfs`.
        """
        return 13

    def debugnans(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Test function debugnans.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.

        Returns:
            BackendArg: Computed output value from `debugnans`.
        """
        return 14

    def descriptive(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Test function descriptive.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.

        Returns:
            BackendArg: Computed output value from `descriptive`.
        """
        return 15

    def descriptive2(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Test function descriptive2.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.

        Returns:
            BackendArg: Computed output value from `descriptive2`.
        """
        return 16

    def diff(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Test function diff.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.

        Returns:
            BackendArg: Computed output value from `diff`.
        """
        return 17

    def digitize(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Test function digitize.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.

        Returns:
            BackendArg: Computed output value from `digitize`.
        """
        return 18

    def dotgeneral(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Test function dotgeneral.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.

        Returns:
            BackendArg: Computed output value from `dotgeneral`.
        """
        return 19

    def ediff1d(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Test function ediff1d.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.

        Returns:
            BackendArg: Computed output value from `ediff1d`.
        """
        return 20

    def einsumpath(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Test function einsumpath.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.

        Returns:
            BackendArg: Computed output value from `einsumpath`.
        """
        return 21

    def fabs(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Test function fabs.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.

        Returns:
            BackendArg: Computed output value from `fabs`.
        """
        return 22

    def interp(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Test function interp.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.

        Returns:
            BackendArg: Computed output value from `interp`.
        """
        return 23

    def iterable(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Test function iterable.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.

        Returns:
            BackendArg: Computed output value from `iterable`.
        """
        return 24

    def ix(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Test function ix.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.

        Returns:
            BackendArg: Computed output value from `ix`.
        """
        return 25

    def key(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Test function key.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.

        Returns:
            BackendArg: Computed output value from `key`.
        """
        return 26

    def kron(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Test function kron.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.

        Returns:
            BackendArg: Computed output value from `kron`.
        """
        return 27

    def load(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Test function load.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.

        Returns:
            BackendArg: Computed output value from `load`.
        """
        return 28

    def modf(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Test function modf.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.

        Returns:
            BackendArg: Computed output value from `modf`.
        """
        return 29

    def onehot(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Test function onehot.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.

        Returns:
            BackendArg: Computed output value from `onehot`.
        """
        return 30

    def raggeddot(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Test function raggeddot.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.

        Returns:
            BackendArg: Computed output value from `raggeddot`.
        """
        return 31

    def ravelmultiindex(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Test function ravelmultiindex.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.

        Returns:
            BackendArg: Computed output value from `ravelmultiindex`.
        """
        return 32

    def rem(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Test function rem.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.

        Returns:
            BackendArg: Computed output value from `rem`.
        """
        return 33

    def rem2(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Test function rem2.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.

        Returns:
            BackendArg: Computed output value from `rem2`.
        """
        return 34

    def rem3(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Test function rem3.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.

        Returns:
            BackendArg: Computed output value from `rem3`.
        """
        return 35

    def serializetensor(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Test function serializetensor.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.

        Returns:
            BackendArg: Computed output value from `serializetensor`.
        """
        return 36

    def serializetensorcamel(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Test function serializetensorcamel.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.

        Returns:
            BackendArg: Computed output value from `serializetensorcamel`.
        """
        return 37

    def tensor(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Test function tensor.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.

        Returns:
            BackendArg: Computed output value from `tensor`.
        """
        return 38

    def trapz(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Test function trapz.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.

        Returns:
            BackendArg: Computed output value from `trapz`.
        """
        return 39

    def truncatediv(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Test function truncatediv.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.

        Returns:
            BackendArg: Computed output value from `truncatediv`.
        """
        return 41

    def truncatemod(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Test function truncatemod.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.

        Returns:
            BackendArg: Computed output value from `truncatemod`.
        """
        return 42

    def uint(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Test function uint.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.

        Returns:
            BackendArg: Computed output value from `uint`.
        """
        return 43

    def uint8(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Test function uint8.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.

        Returns:
            BackendArg: Computed output value from `uint8`.
        """
        return 44

    def unravelindex(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Test function unravelindex.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.

        Returns:
            BackendArg: Computed output value from `unravelindex`.
        """
        return 45

    def unwrap(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Test function unwrap.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.

        Returns:
            BackendArg: Computed output value from `unwrap`.
        """
        return 46

    def vectorize(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Test function vectorize.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.

        Returns:
            BackendArg: Computed output value from `vectorize`.
        """
        return 47
