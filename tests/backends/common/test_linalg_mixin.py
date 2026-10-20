"""Test module."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.backends.common.mixins.linalg import LinearAlgebraASTVisitor

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue


class DummyGenerator:
    """Test class DummyGenerator."""

    def __init__(self) -> None:
        """Test function __init__."""
        self.lines = typing.cast("EagerTensorProtocol", [])

    def get_fallback_prefix(self) -> EagerValue:
        """Test function get_fallback_prefix."""
        return "bk"

    def add_line(self, line: EagerValue) -> None:
        """Test function add_line.

        Args:
            line: Test parameter `line`.
        """
        self.lines.append(line)


class DummyGeneratorDotted:
    """Test class DummyGeneratorDotted."""

    def __init__(self) -> None:
        """Test function __init__."""
        self.lines = typing.cast("EagerTensorProtocol", [])

    def get_fallback_prefix(self) -> EagerValue:
        """Test function get_fallback_prefix."""
        return "np."

    def add_line(self, line: EagerValue) -> None:
        """Test function add_line.

        Args:
            line: Test parameter `line`.
        """
        self.lines.append(line)


class DummyGeneratorMlx:
    """Test class DummyGeneratorMlx."""

    def __init__(self) -> None:
        """Test function __init__."""
        self.lines = typing.cast("EagerTensorProtocol", [])

    def get_fallback_prefix(self) -> EagerValue:
        """Test function get_fallback_prefix."""
        return "mlx"

    def add_line(self, line: EagerValue) -> None:
        """Test function add_line.

        Args:
            line: Test parameter `line`.
        """
        self.lines.append(line)


class DummyVisitor(LinearAlgebraASTVisitor):
    """Test class DummyVisitor."""

    def __init__(self, gen: EagerValue = None) -> None:
        """Test function __init__.

        Args:
            gen: Test parameter `gen`.
        """
        setattr(self, "_generator", gen or DummyGenerator())


class DummyNode:
    """Test class DummyNode."""

    def __init__(self, attrs: EagerValue = None) -> None:
        """Test function __init__.

        Args:
            attrs: Test parameter `attrs`.
        """
        self.attributes = attrs or {}


def test_linalg_mixin() -> None:
    """Test function test_linalg_mixin."""
    vis = DummyVisitor()
    node = DummyNode()

    assert typing.cast("EagerCallableProtocol", vis.visit_CholeskyVjp)(node, ["a", "b"]) == "_cholesky_vjp_eager(bk, a, b)"
    assert typing.cast("EagerTensorProtocol", getattr(vis._generator, "lines")) == ["    from ml_switcheroo_compiler.backends.eager.linalg import _cholesky_vjp_eager"]
    assert typing.cast("EagerCallableProtocol", vis.visit_CholeskyVjp)(node, ["a", "b"]) == "_cholesky_vjp_eager(bk, a, b)"

    assert typing.cast("EagerCallableProtocol", vis.visit_LuVjp)(node, ["a", "b", "c", "d"]) == "_lu_vjp_eager(bk, a, b, c, d)"
    assert "    from ml_switcheroo_compiler.backends.eager.linalg import _lu_vjp_eager" in typing.cast("EagerTensorProtocol", getattr(vis._generator, "lines"))

    assert typing.cast("EagerCallableProtocol", vis.visit_BandPart)(node, ["a"], num_lower=1, num_upper=2) == "bk_band_part(a, 1, 2)"
    assert typing.cast("EagerCallableProtocol", vis.visit_BandPart)(node, ["a"]) == "bk_band_part(a, -1, -1)"

    assert typing.cast("EagerCallableProtocol", vis.visit_BandedTriangularSolve)(node, ["a", "b"], lower=True, adjoint=True) == "bk_banded_triangular_solve(a, b, lower=True, adjoint=True)"
    assert typing.cast("EagerCallableProtocol", vis.visit_BandedTriangularSolve)(node, ["a", "b"]) == "bk_banded_triangular_solve(a, b, lower=False, adjoint=False)"

    node_gather = DummyNode({"lhs_indices": 2, "rhs_indices": 3})
    assert typing.cast("EagerCallableProtocol", vis.visit_GatherMm)(node_gather, ["a", "b", "li", "ri"]) == "bk.gather_mm(a, b, lhs_indices=li, rhs_indices=ri)"
    assert typing.cast("EagerCallableProtocol", vis.visit_GatherMm)(typing.cast("EagerCallableProtocol", DummyNode)(), ["a", "b"]) == "bk.gather_mm(a, b)"

    node_seg = DummyNode({"segments": 2})
    assert typing.cast("EagerCallableProtocol", vis.visit_SegmentedMm)(node_seg, ["a", "b", "seg"]) == "bk.segmented_mm(a, b, seg)"

    node_block = DummyNode()
    assert typing.cast("EagerCallableProtocol", vis.visit_BlockMaskedMm)(node_block, ["a", "b"]) == "bk.matmul(a, b)"

    node_q = DummyNode({"group_size": 32, "bits": 8, "return_idx": 1})
    assert typing.cast("EagerCallableProtocol", vis.visit_Quantize)(node_q, ["a"]) == "a"
    assert typing.cast("EagerCallableProtocol", vis.visit_QuantizedMatmul)(node_q, ["x", "w", "s", "b"]) == "bk.matmul(x, w.T if True else w)"
    assert typing.cast("EagerCallableProtocol", vis.visit_GatherQMM)(node_q, ["x", "w", "s", "b", "i"]) == "bk.matmul(x, w[i].T if True else w[i])"

    vis_mlx = typing.cast("EagerCallableProtocol", DummyVisitor)(typing.cast("EagerCallableProtocol", DummyGeneratorMlx)())
    assert typing.cast("EagerCallableProtocol", vis_mlx.visit_Quantize)(node_q, ["a"]) == "mx.quantize(a, group_size=32, bits=8)[1]"
    assert typing.cast("EagerCallableProtocol", vis_mlx.visit_QuantizedMatmul)(node_q, ["x", "w", "s", "b"]) == "mx.quantized_matmul(x, w, s, b, transpose=True, group_size=32, bits=8)"
    assert typing.cast("EagerCallableProtocol", vis_mlx.visit_GatherQMM)(node_q, ["x", "w", "s", "b", "i"]) == "mx.gather_qmm(x, w, s, b, i, transpose=True, group_size=32, bits=8)"

    assert typing.cast("EagerCallableProtocol", vis.visit_QrVjp)(node, ["a", "b", "c"]) == "_qr_vjp_eager(bk, a, b, c)"
    assert typing.cast("EagerCallableProtocol", vis.visit_QrVjp)(node, ["a", "b", "c"]) == "_qr_vjp_eager(bk, a, b, c)"
    assert typing.cast("EagerCallableProtocol", vis.visit_SvdVjp)(node, ["a", "b", "c", "d"], compute_uv=True) == "_svd_vjp_eager(bk, a, b, c, d, compute_uv=True)"
    assert typing.cast("EagerCallableProtocol", vis.visit_SvdVjp)(node, ["a", "b"], compute_uv=False) == "_svd_vjp_eager(bk, a, b, compute_uv=False)"


def test_linalg_mixin_dotted_prefix() -> None:
    """Test function test_linalg_mixin_dotted_prefix."""
    vis = typing.cast("EagerCallableProtocol", DummyVisitor)(typing.cast("EagerCallableProtocol", DummyGeneratorDotted)())
    node = DummyNode()
    assert typing.cast("EagerCallableProtocol", vis.visit_CholeskyVjp)(node, ["a", "b"]) == "_cholesky_vjp_eager(np, a, b)"
    assert typing.cast("EagerCallableProtocol", vis.visit_LuVjp)(node, ["a", "b", "c", "d"]) == "_lu_vjp_eager(np, a, b, c, d)"


def test_linalg_mixin_re_import() -> None:
    """Test function test_linalg_mixin_re_import."""
    vis = DummyVisitor()
    node = DummyNode()
    typing.cast("EagerCallableProtocol", vis.visit_LuVjp)(node, ["a", "b", "c", "d"])
    typing.cast("EagerCallableProtocol", vis.visit_LuVjp)(node, ["a", "b", "c", "d"])
    assert node is not None
