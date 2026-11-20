"""Unit tests for Haskell-style backend capability typeclasses and static AST/YAML verifier."""

from __future__ import annotations

import os
import pathlib
from unittest.mock import patch

import numpy as np
import pytest
from ml_switcheroo_ir import IRGraph

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.backends.eager_registry import BackendOperand
from ml_switcheroo_compiler.backends.numpy.generator import NumpyGenerator
from ml_switcheroo_compiler.backends.protocols import StaticEagerBackendAdapter
from ml_switcheroo_compiler.backends.pure_python.generator import PurePythonGenerator
from ml_switcheroo_compiler.backends.pure_python.types import PurePythonTensor
from ml_switcheroo_compiler.backends.sparse.generator import SparseGenerator
from ml_switcheroo_compiler.backends.typeclass_verifier import (
    verify_all_typeclasses,
    verify_backend_ast_capabilities,
    verify_curated_yaml_mappings,
    verify_live_typeclass_instances,
)
from ml_switcheroo_compiler.backends.typeclasses import (
    CORE_EAGER_REQUIRED_OPS,
    AOTCompilerTypeclass,
    CapabilityKind,
    CodeEmitterTypeclass,
    CoreEagerDictionary,
    CoreEagerTypeclass,
    build_core_eager_dictionary_from_ops,
    build_typeclass_op_table,
)


def test_numpy_and_pure_python_typeclass_dictionaries() -> None:
    """Verify CoreEagerDictionary methods, dispatch wrappers, and capability traits on live backends."""
    assert CapabilityKind.CORE_EAGER in NumpyGenerator.declared_capabilities()
    assert CapabilityKind.CORE_EAGER in PurePythonGenerator.declared_capabilities()
    assert CapabilityKind.DOMAIN_EAGER in SparseGenerator.declared_capabilities()
    assert CapabilityKind.CORE_EAGER not in SparseGenerator.declared_capabilities()
    assert CapabilityKind.CORE_EAGER not in BaseGenerator.declared_capabilities()

    np_gen = NumpyGenerator(IRGraph(name="tc_np"))
    assert isinstance(np_gen, CodeEmitterTypeclass)
    assert isinstance(np_gen, AOTCompilerTypeclass)

    np_adapter = NumpyGenerator.get_eager_adapter()
    assert isinstance(np_adapter, StaticEagerBackendAdapter)
    tc = np_adapter.typeclass_dict
    assert isinstance(tc, CoreEagerDictionary)
    assert isinstance(tc, CoreEagerTypeclass)

    z = tc.zeros((2, 2), "float32")
    o = tc.ones((2, 2), "float32")
    arr = tc.array([[1.0, 4.0], [9.0, 16.0]], "float32")
    as_arr = tc.asarray(arr, "float32")
    assert np.allclose(tc.add(z, o), o)
    assert np.allclose(tc.subtract(o, o), z)
    assert np.allclose(tc.multiply(arr, o), arr)
    assert np.allclose(tc.divide(arr, o), arr)
    assert np.allclose(tc.abs(tc.subtract(z, o)), o)
    assert np.allclose(tc.sqrt(arr), np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    assert np.allclose(tc.log(tc.exp(o)), o)
    assert np.allclose(tc.add(tc.multiply(tc.sin(o), tc.sin(o)), tc.multiply(tc.cos(o), tc.cos(o))), o)
    assert tc.tanh(z) is not None
    assert tc.matmul(o, as_arr) is not None
    assert np.asarray(tc.reshape(arr, (4,))).shape == (4,)
    assert np.asarray(tc.transpose(arr, (1, 0))).shape == (2, 2)
    assert float(np.asarray(tc.sum(o, None, False))) == pytest.approx(4.0)
    assert float(np.asarray(tc.mean(o, 0, True))[0, 0]) == pytest.approx(1.0)

    table = build_typeclass_op_table(tc)
    assert set(table.keys()) == set(CORE_EAGER_REQUIRED_OPS)
    assert np.asarray(table["zeros"]((2,), "float32")).shape == (2,)
    assert np.asarray(table["ones"](dtype="float32")).shape == ()
    assert table["array"]([1.0, 2.0], "float32") is not None
    assert table["asarray"](dtype="float32") is not None
    assert table["abs"](np.array(-2.0)) == pytest.approx(2.0)
    assert table["add"](np.array(1.0), np.array(2.0)) == pytest.approx(3.0)
    assert np.asarray(table["reshape"](arr, (4,))).shape == (4,)
    assert np.asarray(table["reshape"](arr, shape=(4,))).shape == (4,)
    assert np.asarray(table["transpose"](arr, (1, 0))).shape == (2, 2)
    assert np.asarray(table["transpose"](arr, axes=(1, 0))).shape == (2, 2)
    assert float(np.asarray(table["sum"](o, 0, False))[0]) == pytest.approx(2.0)
    assert float(np.asarray(table["mean"](o, (0, 1), False))) == pytest.approx(1.0)

    pure_adapter = StaticEagerBackendAdapter(backend_name="tc_only", typeclass_dict=tc)
    assert set(pure_adapter.bound_operations().keys()) == set(CORE_EAGER_REQUIRED_OPS)


def test_auto_synthesized_typeclass_dictionary() -> None:
    """Verify `build_core_eager_dictionary_from_ops` and `StaticEagerBackendAdapter` auto-synthesis."""
    np_adapter = NumpyGenerator.get_eager_adapter()
    assert isinstance(np_adapter, StaticEagerBackendAdapter)
    assert np_adapter.typeclass_dict is not None
    table = build_typeclass_op_table(np_adapter.typeclass_dict)
    arr = np.array([[1.0, 4.0], [9.0, 16.0]], dtype=np.float32)

    synth_adapter = StaticEagerBackendAdapter(backend_name="auto_synth", ops=table)
    synth_tc = synth_adapter.typeclass_dict
    assert synth_tc is not None
    assert np.asarray(synth_tc.zeros((2,), None)).shape == (2,)
    assert np.asarray(synth_tc.zeros((2,), "float32")).shape == (2,)
    assert np.asarray(synth_tc.ones((2,), None)).shape == (2,)
    assert np.asarray(synth_tc.ones((2,), "float32")).shape == (2,)
    assert synth_tc.array([1.0, 2.0], None) is not None
    assert synth_tc.array([1.0, 2.0], "float32") is not None
    assert synth_tc.asarray([1.0, 2.0], None) is not None
    assert synth_tc.asarray([1.0, 2.0], "float32") is not None
    assert np.asarray(synth_tc.transpose(arr, None)).shape == (2, 2)
    assert np.asarray(synth_tc.transpose(arr, (1, 0))).shape == (2, 2)
    assert build_core_eager_dictionary_from_ops(table) is not None


def test_pure_python_typeclass_dictionary_ops() -> None:
    """Verify PurePythonGenerator CoreEagerDictionary methods on both tensor and scalar operands."""
    pp_adapter = PurePythonGenerator.get_eager_adapter()
    assert isinstance(pp_adapter, StaticEagerBackendAdapter)
    pp_tc = pp_adapter.typeclass_dict
    assert pp_tc is not None
    pp_t = pp_tc.array([1.0, 4.0], "float32")
    pp_o = pp_tc.ones((2,), "float32")
    assert isinstance(pp_tc.zeros((2,), "float32"), PurePythonTensor)
    assert isinstance(pp_tc.zeros((), "float32"), PurePythonTensor)
    assert isinstance(pp_tc.array(5.0, "float32"), PurePythonTensor)
    assert isinstance(pp_tc.array(None, "float32"), PurePythonTensor)
    assert isinstance(pp_tc.asarray(pp_t, "float32"), PurePythonTensor)
    assert isinstance(pp_tc.asarray([1.0, 2.0]), PurePythonTensor)
    assert isinstance(pp_tc.abs(pp_t), PurePythonTensor)
    assert isinstance(pp_tc.exp(pp_o), PurePythonTensor)
    assert isinstance(pp_tc.log(pp_o), PurePythonTensor)
    assert isinstance(pp_tc.sqrt(pp_t), PurePythonTensor)
    assert isinstance(pp_tc.sin(pp_o), PurePythonTensor)
    assert isinstance(pp_tc.cos(pp_o), PurePythonTensor)
    assert isinstance(pp_tc.tanh(pp_o), PurePythonTensor)
    assert isinstance(pp_tc.add(pp_t, pp_o), PurePythonTensor)
    assert isinstance(pp_tc.subtract(pp_t, pp_o), PurePythonTensor)
    assert isinstance(pp_tc.multiply(pp_t, pp_o), PurePythonTensor)
    assert isinstance(pp_tc.divide(pp_t, pp_o), PurePythonTensor)
    assert isinstance(pp_tc.matmul(pp_t, pp_o), PurePythonTensor)
    assert isinstance(pp_tc.reshape(pp_t, (2, 1)), PurePythonTensor)
    assert isinstance(pp_tc.reshape(1.0, (1,)), PurePythonTensor)
    assert pp_tc.transpose(pp_t, None) is pp_t
    assert float(str(pp_tc.sum(pp_t, None, False))) == pytest.approx(5.0)
    assert float(str(pp_tc.sum(3.0, None, False))) == pytest.approx(3.0)
    assert float(str(pp_tc.mean(pp_t, None, False))) == pytest.approx(2.5)
    assert float(str(pp_tc.mean(4.0, None, False))) == pytest.approx(4.0)
    assert float(str(pp_tc.abs(-2.0))) == pytest.approx(2.0)
    assert float(str(pp_tc.exp(0.0))) == pytest.approx(1.0)
    assert float(str(pp_tc.log(1.0))) == pytest.approx(0.0)
    assert float(str(pp_tc.sqrt(9.0))) == pytest.approx(3.0)
    assert float(str(pp_tc.sin(0.0))) == pytest.approx(0.0)
    assert float(str(pp_tc.cos(0.0))) == pytest.approx(1.0)
    assert float(str(pp_tc.tanh(0.0))) == pytest.approx(0.0)


def test_typeclass_verifier_ast_and_yaml_branches(tmp_path: pathlib.Path) -> None:
    """Exercise positive and negative branches of AST and curated YAML verifiers.

    Args:
        tmp_path (pathlib.Path): Temporary pytest fixture directory.
    """
    assert verify_all_typeclasses() == []

    missing_dir = str(tmp_path / "does_not_exist")
    assert len(verify_backend_ast_capabilities(missing_dir)) == 1
    assert len(verify_curated_yaml_mappings(missing_dir)) == 1

    ab_dir = tmp_path / "adapter_bindings"
    ab_dir.mkdir()
    (ab_dir / "shard_00.py").write_text("x = 1" + os.linesep, encoding="utf-8")

    b_dir = tmp_path / "dummy_backend"
    b_dir.mkdir()
    (b_dir / "notes.txt").write_text("ignore" + os.linesep, encoding="utf-8")
    bad_src = os.linesep.join(
        [
            "@other_dec",
            "@other_call()",
            "@(factory())('x')",
            "@reg.register_backend('bad1')",
            "class BadNoBase:",
            "    async def helper(self) -> None:",
            "        return None",
            "",
            "@register_backend('ok')",
            "class OkGen(BaseGen):",
            "    def _compile_aot_impl(self) -> None:",
            "        return None",
            "",
        ]
    )
    (b_dir / "generator.py").write_text(bad_src, encoding="utf-8")
    ast_errs = verify_backend_ast_capabilities(str(tmp_path))
    assert any("lacks CodeEmitterTypeclass 'generate'" in e for e in ast_errs)
    assert any("lacks AOTCompilerTypeclass '_compile_aot_impl'" in e for e in ast_errs)

    maps_dir = b_dir / "mappings"
    maps_dir.mkdir()
    (maps_dir / "readme.txt").write_text("skip" + os.linesep, encoding="utf-8")
    (maps_dir / "not_dict.yaml").write_text("- 1" + os.linesep, encoding="utf-8")
    (maps_dir / "unknown_key.yaml").write_text("bogus_field: 123" + os.linesep, encoding="utf-8")
    (maps_dir / "bad_type.yaml").write_text("is_method: [1, 2, 3]" + os.linesep, encoding="utf-8")
    (maps_dir / "blank_op.yaml").write_text("operation: '   '" + os.linesep, encoding="utf-8")
    (maps_dir / "valid_op.yaml").write_text(
        "operation: Add" + os.linesep + "target_api: np.add" + os.linesep,
        encoding="utf-8",
    )

    yaml_errs = verify_curated_yaml_mappings(str(tmp_path))
    assert any("expected YAML mapping dictionary" in e for e in yaml_errs)
    assert any("unknown OpMappingSchema keys" in e for e in yaml_errs)
    assert any("invalid OpMappingSchema" in e for e in yaml_errs)
    assert any("'operation' field must not be blank" in e for e in yaml_errs)


class _FakeBrokenAdapter:
    """Stub eager adapter missing operations and exposing an invalid `typeclass_dict`."""

    typeclass_dict = "invalid_tc"

    def bound_operations(self) -> dict[str, str]:
        """Return empty bound operations dict.

        Returns:
            dict[str, str]: Empty mapping.
        """
        return {}


class _FakeBrokenBackend:
    """Stub backend missing `CODE_EMIT` and `AOT_COMPILE` while declaring `CORE_EAGER`."""

    @classmethod
    def declared_capabilities(cls) -> frozenset[CapabilityKind]:
        """Return broken capability set.

        Returns:
            frozenset[CapabilityKind]: Singleton `CORE_EAGER`.
        """
        return frozenset({CapabilityKind.CORE_EAGER})

    @classmethod
    def get_eager_adapter(cls) -> _FakeBrokenAdapter:
        """Return broken adapter stub.

        Returns:
            _FakeBrokenAdapter: Broken adapter.
        """
        return _FakeBrokenAdapter()


class _FakeNoTcAdapter:
    """Stub eager adapter whose `typeclass_dict` is `None`."""

    def bound_operations(self) -> dict[str, str]:
        """Return empty mapping.

        Returns:
            dict[str, str]: Empty mapping.
        """
        return {}


class _FakeNoTcBackend:
    """Stub eager backend with `typeclass_dict = None`."""

    @classmethod
    def declared_capabilities(cls) -> frozenset[CapabilityKind]:
        """Return valid capability set with `CORE_EAGER`.

        Returns:
            frozenset[CapabilityKind]: Capability set.
        """
        return frozenset({CapabilityKind.CODE_EMIT, CapabilityKind.AOT_COMPILE, CapabilityKind.CORE_EAGER})

    @classmethod
    def get_eager_adapter(cls) -> _FakeNoTcAdapter:
        """Return `_FakeNoTcAdapter`.

        Returns:
            _FakeNoTcAdapter: Adapter without typeclass dictionary.
        """
        return _FakeNoTcAdapter()


class _FakePartialTcAdapter:
    """Stub eager adapter with valid `CoreEagerTypeclass` but empty `bound_operations`."""

    @property
    def typeclass_dict(self) -> CoreEagerTypeclass[BackendOperand] | None:
        """Return valid NumPy `typeclass_dict`.

        Returns:
            CoreEagerTypeclass[BackendOperand] | None: Valid typeclass dictionary.
        """
        np_ad = NumpyGenerator.get_eager_adapter()
        assert isinstance(np_ad, StaticEagerBackendAdapter)
        return np_ad.typeclass_dict

    def bound_operations(self) -> dict[str, str]:
        """Return empty bound ops to trigger missing-op check.

        Returns:
            dict[str, str]: Empty mapping.
        """
        return {}


class _FakePartialTcBackend:
    """Stub backend whose adapter has valid `typeclass_dict` but incomplete `bound_operations`."""

    @classmethod
    def declared_capabilities(cls) -> frozenset[CapabilityKind]:
        """Return valid capability set.

        Returns:
            frozenset[CapabilityKind]: Capability set.
        """
        return frozenset({CapabilityKind.CODE_EMIT, CapabilityKind.AOT_COMPILE, CapabilityKind.CORE_EAGER})

    @classmethod
    def get_eager_adapter(cls) -> _FakePartialTcAdapter:
        """Return `_FakePartialTcAdapter`.

        Returns:
            _FakePartialTcAdapter: Partial adapter.
        """
        return _FakePartialTcAdapter()


class _FakeOptionalMissingBackend:
    """Stub backend whose optional third-party runtime raises `ImportError`."""

    @classmethod
    def declared_capabilities(cls) -> frozenset[CapabilityKind]:
        """Return capability set with `CORE_EAGER`.

        Returns:
            frozenset[CapabilityKind]: Capability set.
        """
        return frozenset({CapabilityKind.CODE_EMIT, CapabilityKind.AOT_COMPILE, CapabilityKind.CORE_EAGER})

    @classmethod
    def get_eager_adapter(cls) -> _FakeNoTcAdapter:
        """Simulate missing optional third-party library.

        Returns:
            _FakeNoTcAdapter: Unreachable due to raised `ImportError`.

        Raises:
            ImportError: Always raised for optional backend test.
        """
        raise ImportError("optional library not installed")


def test_typeclass_verifier_live_instance_branches() -> None:
    """Exercise all error and pass branches of `verify_live_typeclass_instances`."""
    with patch(
        "ml_switcheroo_compiler.backends.typeclass_verifier.BackendRegistry.get", return_value=_FakeBrokenBackend
    ):
        live_errs = verify_live_typeclass_instances(("fake_broken",))
        assert any("missing CapabilityKind.CODE_EMIT" in e for e in live_errs)
        assert any("missing CapabilityKind.AOT_COMPILE" in e for e in live_errs)
        assert any("does not conform to CoreEagerTypeclass" in e for e in live_errs)
        assert any("missing ops" in e for e in live_errs)

    with patch("ml_switcheroo_compiler.backends.typeclass_verifier.BackendRegistry.get", return_value=_FakeNoTcBackend):
        notc_errs = verify_live_typeclass_instances(("fake_notc",))
        assert any("CORE_EAGER backend missing verified typeclass_dict" in e for e in notc_errs)

    with patch(
        "ml_switcheroo_compiler.backends.typeclass_verifier.BackendRegistry.get",
        return_value=_FakeOptionalMissingBackend,
    ):
        assert verify_live_typeclass_instances(("fake_opt_missing",)) == []

    with patch(
        "ml_switcheroo_compiler.backends.typeclass_verifier.BackendRegistry.get",
        side_effect=ImportError("backend module missing"),
    ):
        assert verify_live_typeclass_instances(("fake_reg_import_err",)) == []

    with patch(
        "ml_switcheroo_compiler.backends.typeclass_verifier.BackendRegistry.get", return_value=_FakePartialTcBackend
    ):
        partial_errs = verify_live_typeclass_instances(("fake_partial",))
        assert len(partial_errs) == 1 and "missing ops" in partial_errs[0]

    assert verify_live_typeclass_instances(("metal",)) == []
