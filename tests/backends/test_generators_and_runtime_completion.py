"""Branch-completion tests for backend eager adapters, WGSL validation, JAXpr, C++ runner, and MLX/TF."""

from __future__ import annotations

import pathlib
import sys
import types
from collections.abc import Sequence
from unittest.mock import patch

import pytest
from ml_switcheroo_ir import LogicalGraph

import ml_switcheroo_compiler.backends.tensorflow.profiler as tf_prof_mod
from ml_switcheroo_compiler.backends.awkward.generator import AwkwardGenerator
from ml_switcheroo_compiler.backends.cupy.generator import CupyGenerator
from ml_switcheroo_compiler.backends.dask.generator import DaskGenerator
from ml_switcheroo_compiler.backends.dpnp.eager import _resolve_dpnp_callable
from ml_switcheroo_compiler.backends.dpnp.generator import DPNPGenerator
from ml_switcheroo_compiler.backends.eager_registry import BackendArg
from ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider import _merge_modular_templates
from ml_switcheroo_compiler.backends.edge.wgsl_ast import WGSLNode, WGSLRaw
from ml_switcheroo_compiler.backends.jax.generator import JAXCodeGenerator
from ml_switcheroo_compiler.backends.jax.jaxpr_ingest import _translate_jax_equation
from ml_switcheroo_compiler.backends.keras.generator import KerasCodeGenerator
from ml_switcheroo_compiler.backends.llvm_cpp._cpp_runner import LLVMCPPRunner
from ml_switcheroo_compiler.backends.mlx.eager import _combine_ring_chunks
from ml_switcheroo_compiler.backends.mlx.generator import MLXCodeGenerator
from ml_switcheroo_compiler.backends.numba.generator import NumbaGenerator
from ml_switcheroo_compiler.backends.protocols import EagerBackendAdapter
from ml_switcheroo_compiler.backends.pure_python.generator import PurePythonGenerator
from ml_switcheroo_compiler.backends.pyarrow_compute.generator import PyArrowComputeGenerator
from ml_switcheroo_compiler.backends.pytorch.generator import PyTorchCodeGenerator
from ml_switcheroo_compiler.backends.sparse.generator import SparseGenerator
from ml_switcheroo_compiler.backends.tensorflow.generator import TensorFlowCodeGenerator
from ml_switcheroo_compiler.backends.tensorflow.profiler import TensorFlowProfiler
from ml_switcheroo_compiler.core.shape import OpValue


class _WGSLListAttrNode(WGSLNode):
    """WGSL AST node exposing a mixed list attribute (`values`) for `validate()` loop coverage."""

    def __init__(self, items: list[WGSLNode | str]) -> None:
        """Initialize `.values` with `WGSLNode` and non-`WGSLNode` elements.

        Args:
            items (list[WGSLNode | str]): Mixed sequence of child AST nodes and strings.
        """
        self.values: list[WGSLNode | str] = items

    def emit(self, indent: int = 0) -> str:
        """Emit placeholder WGSL code.

        Args:
            indent (int): Indentation level.

        Returns:
            str: Placeholder WGSL snippet.
        """
        del indent
        return "/* list_node */"


class _NullAvalOutVar:
    """JAXpr output variable satisfying `_JaxVarProtocol` with `.aval` set to `None`."""

    @property
    def aval(self) -> OpValue:
        """Return `None` instead of a shaped abstract value.

        Returns:
            OpValue: `None` abstract value.
        """
        return None


class _DummyJaxEqn:
    """Minimal `JaxprEqnProtocol` equation whose output variable has a `None` `aval`."""

    @property
    def primitive(self) -> str:
        """Return primitive operation name.

        Returns:
            str: `"add"` primitive identifier.
        """
        return "add"

    @property
    def invars(self) -> Sequence[OpValue]:
        """Return empty input variables tuple.

        Returns:
            Sequence[OpValue]: Empty tuple.
        """
        return ()

    @property
    def outvars(self) -> Sequence[OpValue]:
        """Return single output variable with `aval=None`.

        Returns:
            Sequence[OpValue]: Single-element tuple containing `_NullAvalOutVar` wrapped via shape stub.
        """
        return (_NullAvalShapeVar(),)


class _NullAvalShapeVar:
    """Object satisfying both `ShapeProtocol` (`OpValue`) and `_JaxVarProtocol` (`aval=None`)."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Return empty shape tuple so instance satisfies `OpValue`.

        Returns:
            tuple[int, ...]: Empty shape tuple.
        """
        return ()

    @property
    def aval(self) -> OpValue:
        """Return `None` non-`_JaxAvalProtocol` value.

        Returns:
            OpValue: `None`.
        """
        return None


def test_all_backend_generators_get_eager_adapter() -> None:
    """Verify `get_eager_adapter()` across all 13 backend code generators."""
    import importlib
    import importlib.machinery
    from collections.abc import Callable
    from unittest.mock import MagicMock

    def _call_with_optional_stub(
        fn: Callable[[], EagerBackendAdapter],
        pkg_name: str = "",
        submods: tuple[str, ...] = (),
    ) -> EagerBackendAdapter:
        """Invoke `fn()` using real `pkg_name` when importable, or an isolated PEP-451 stub otherwise.

        Args:
            fn (Callable[[], EagerBackendAdapter]): Zero-arg factory returning a backend eager adapter.
            pkg_name (str): Optional root backend package name to probe or stub.
            submods (tuple[str, ...]): Dotted submodule names under `pkg_name` required by the adapter.

        Returns:
            EagerBackendAdapter: Concrete eager backend adapter instance.
        """
        if not pkg_name:
            return fn()
        preexisting = set(sys.modules.keys())
        can_import = True
        try:
            importlib.import_module(pkg_name)
            for sub in submods:
                importlib.import_module(f"{pkg_name}.{sub}")
        except ImportError:
            can_import = False
            for mod_key in list(sys.modules.keys()):
                if mod_key not in preexisting and (mod_key == pkg_name or mod_key.startswith(f"{pkg_name}.")):
                    sys.modules.pop(mod_key, None)
        if can_import:
            return fn()

        stub_mods: dict[str, MagicMock] = {}
        root = MagicMock(name=pkg_name)
        root.__path__ = []
        root.__spec__ = importlib.machinery.ModuleSpec(pkg_name, loader=None, is_package=True)
        stub_mods[pkg_name] = root
        mp = pytest.MonkeyPatch()
        for sub in submods:
            full_sub = f"{pkg_name}.{sub}"
            leaf = MagicMock(name=full_sub)
            leaf.__spec__ = importlib.machinery.ModuleSpec(full_sub, loader=None, is_package=False)
            mp.setattr(root, sub, leaf, raising=False)
            stub_mods[full_sub] = leaf
        with patch.dict(sys.modules, stub_mods):
            return fn()

    expected_pairs: list[tuple[str, EagerBackendAdapter]] = [
        ("awkward", _call_with_optional_stub(AwkwardGenerator.get_eager_adapter, "awkward")),
        ("cupy", _call_with_optional_stub(CupyGenerator.get_eager_adapter, "cupy")),
        ("dask", _call_with_optional_stub(DaskGenerator.get_eager_adapter, "dask", ("array",))),
        ("jax", _call_with_optional_stub(JAXCodeGenerator.get_eager_adapter, "jax", ("numpy", "scipy", "lax"))),
        ("keras", _call_with_optional_stub(KerasCodeGenerator.get_eager_adapter, "keras", ("ops",))),
        ("mlx", _call_with_optional_stub(MLXCodeGenerator.get_eager_adapter, "mlx", ("core",))),
        ("numba", _call_with_optional_stub(NumbaGenerator.get_eager_adapter, "numba")),
        ("pure_python", _call_with_optional_stub(PurePythonGenerator.get_eager_adapter)),
        ("pyarrow", _call_with_optional_stub(PyArrowComputeGenerator.get_eager_adapter, "pyarrow", ("compute",))),
        ("pytorch", _call_with_optional_stub(PyTorchCodeGenerator.get_eager_adapter, "torch")),
        ("sparse", _call_with_optional_stub(SparseGenerator.get_eager_adapter, "sparse")),
        ("tensorflow", _call_with_optional_stub(TensorFlowCodeGenerator.get_eager_adapter, "tensorflow")),
    ]
    for expected_prefix, adapter in expected_pairs:
        assert isinstance(adapter, EagerBackendAdapter)
        assert expected_prefix in adapter.backend_name

    fake_dpnp = types.ModuleType("dpnp")
    fake_dpnp.__spec__ = importlib.machinery.ModuleSpec("dpnp", loader=None, is_package=True)
    with patch.dict(sys.modules, {"dpnp": fake_dpnp}):
        dpnp_adapter = DPNPGenerator.get_eager_adapter()
        assert isinstance(dpnp_adapter, EagerBackendAdapter)
        assert dpnp_adapter.backend_name == "dpnp"


def test_jax_savez_compressed_when_symbol_is_none(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    """Verify `JAXCodeGenerator.savez_compressed` exits cleanly when `jnp.savez_compressed` is `None`.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.
        tmp_path (pathlib.Path): Temporary directory fixture.
    """
    import importlib.util

    if importlib.util.find_spec("jax") is None:
        fake_jax = types.ModuleType("jax")
        fake_jax.__path__ = []
        fake_jnp = types.ModuleType("jax.numpy")
        monkeypatch.setattr(fake_jax, "numpy", fake_jnp, raising=False)
        monkeypatch.setitem(sys.modules, "jax", fake_jax)
        monkeypatch.setitem(sys.modules, "jax.numpy", fake_jnp)

    import jax.numpy as jnp

    out_file = tmp_path / "unused_archive.npz"
    monkeypatch.setattr(jnp, "savez_compressed", None, raising=False)
    JAXCodeGenerator.savez_compressed(str(out_file))
    assert not out_file.exists()


def test_dpnp_resolve_callable_non_callable_root_and_linalg() -> None:
    """Verify `_resolve_dpnp_callable` returns `None` when root and `linalg` attributes are non-callable."""
    linalg_stub = types.ModuleType("dpnp.linalg")
    pytest.MonkeyPatch().setattr(linalg_stub, "slogdet", 123, raising=False)
    dpnp_stub = types.ModuleType("dpnp")
    pytest.MonkeyPatch().setattr(dpnp_stub, "slogdet", 456, raising=False)
    pytest.MonkeyPatch().setattr(dpnp_stub, "linalg", linalg_stub, raising=False)

    assert _resolve_dpnp_callable(dpnp_stub, "slogdet") is None


def test_wgsl_provider_non_dict_templates_and_wgsl_ast_mixed_list() -> None:
    """Verify `_merge_modular_templates` non-dict skips and `WGSLNode.validate` mixed list traversal."""
    loaded: dict[str, dict[str, dict[str, str | list[int] | None]] | dict[str, str] | str | None] = {
        "templates": None,
        "js_orchestration": None,
        "global_bindings": None,
    }
    _merge_modular_templates(loaded)
    assert loaded["templates"] is None
    assert loaded["js_orchestration"] is None

    node = _WGSLListAttrNode([WGSLRaw("let x: f32 = 1.0;"), "plain_non_node_item"])
    node.validate()
    assert node.emit() == "/* list_node */"


def test_jaxpr_ingest_null_aval_and_cpp_runner_known_entry_point() -> None:
    """Verify `_translate_jax_equation` null-`aval` branch and `LLVMCPPRunner` known symbol lookup."""
    graph = LogicalGraph(name="jaxpr_null_aval")
    _translate_jax_equation(_DummyJaxEqn(), graph)
    assert len(graph.nodes) == 1
    created = next(iter(graph.nodes.values()))
    assert created.op_type == "Add"
    assert created.shape_metadata == ()

    cpp_src = """
    extern "C" int evaluate(int x) {
        return x + 10;
    }
    """
    runner = LLVMCPPRunner()
    fn = runner.compile_and_load(cpp_src, entry_point="evaluate")
    assert callable(fn)


def test_mlx_combine_ring_chunks_host_fallbacks_and_tf_profiler_empty_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `_combine_ring_chunks(None, ...)` reductions and `TensorFlowProfiler` `tf=None` step.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.
    """
    assert _combine_ring_chunks(None, 3.0, 4.0, "PROD") == 12.0
    assert _combine_ring_chunks(None, 3.0, 7.0, "MAX") == 7.0
    assert _combine_ring_chunks(None, 9.0, 2.0, "MAX") == 9.0
    assert _combine_ring_chunks(None, 3.0, 7.0, "MIN") == 3.0
    assert _combine_ring_chunks(None, 9.0, 2.0, "MIN") == 2.0
    assert _combine_ring_chunks(None, 3.0, 4.0, "SUM") == 7.0

    monkeypatch.setattr(tf_prof_mod, "tf", None)
    profiler = TensorFlowProfiler()
    empty_inputs: list[BackendArg] = []
    step_fn = profiler._build_step_fn(False, None, empty_inputs, None)
    assert step_fn() == 0.0
