"""Test module."""

from ml_switcheroo_compiler.backends.common.mixins.linalg import LinearAlgebraASTVisitor


class DummyGenerator:
    def __init__(self):
        self.lines = []

    def get_fallback_prefix(self):
        return "bk"

    def add_line(self, line):
        self.lines.append(line)


class DummyGeneratorDotted:
    def __init__(self):
        self.lines = []

    def get_fallback_prefix(self):
        return "np."

    def add_line(self, line):
        self.lines.append(line)


class DummyGeneratorMlx:
    def __init__(self):
        self.lines = []

    def get_fallback_prefix(self):
        return "mlx"

    def add_line(self, line):
        self.lines.append(line)


class DummyVisitor(LinearAlgebraASTVisitor):
    def __init__(self, gen=None):
        self._generator = gen or DummyGenerator()


class DummyNode:
    def __init__(self, attrs=None):
        self.attributes = attrs or {}


def test_linalg_mixin():
    vis = DummyVisitor()
    node = DummyNode()

    assert vis.visit_CholeskyVjp(node, ["a", "b"]) == "_cholesky_vjp_eager(bk, a, b)"
    assert vis._generator.lines == ["    from ml_switcheroo_compiler.backends.eager.linalg import _cholesky_vjp_eager"]
    assert vis.visit_CholeskyVjp(node, ["a", "b"]) == "_cholesky_vjp_eager(bk, a, b)"

    assert vis.visit_LuVjp(node, ["a", "b", "c", "d"]) == "_lu_vjp_eager(bk, a, b, c, d)"
    assert "    from ml_switcheroo_compiler.backends.eager.linalg import _lu_vjp_eager" in vis._generator.lines

    assert vis.visit_BandPart(node, ["a"], num_lower=1, num_upper=2) == "bk_band_part(a, 1, 2)"
    assert vis.visit_BandPart(node, ["a"]) == "bk_band_part(a, -1, -1)"

    assert vis.visit_BandedTriangularSolve(node, ["a", "b"], lower=True, adjoint=True) == "bk_banded_triangular_solve(a, b, lower=True, adjoint=True)"
    assert vis.visit_BandedTriangularSolve(node, ["a", "b"]) == "bk_banded_triangular_solve(a, b, lower=False, adjoint=False)"

    node_gather = DummyNode({"lhs_indices": 2, "rhs_indices": 3})
    assert vis.visit_GatherMm(node_gather, ["a", "b", "li", "ri"]) == "bk.gather_mm(a, b, lhs_indices=li, rhs_indices=ri)"
    assert vis.visit_GatherMm(DummyNode(), ["a", "b"]) == "bk.gather_mm(a, b)"

    node_seg = DummyNode({"segments": 2})
    assert vis.visit_SegmentedMm(node_seg, ["a", "b", "seg"]) == "bk.segmented_mm(a, b, seg)"

    node_block = DummyNode()
    assert vis.visit_BlockMaskedMm(node_block, ["a", "b"]) == "bk.matmul(a, b)"

    node_q = DummyNode({"group_size": 32, "bits": 8, "return_idx": 1})
    assert vis.visit_Quantize(node_q, ["a"]) == "a"
    assert vis.visit_QuantizedMatmul(node_q, ["x", "w", "s", "b"]) == "bk.matmul(x, w.T if True else w)"
    assert vis.visit_GatherQMM(node_q, ["x", "w", "s", "b", "i"]) == "bk.matmul(x, w[i].T if True else w[i])"

    vis_mlx = DummyVisitor(DummyGeneratorMlx())
    assert vis_mlx.visit_Quantize(node_q, ["a"]) == "mx.quantize(a, group_size=32, bits=8)[1]"
    assert vis_mlx.visit_QuantizedMatmul(node_q, ["x", "w", "s", "b"]) == "mx.quantized_matmul(x, w, s, b, transpose=True, group_size=32, bits=8)"
    assert vis_mlx.visit_GatherQMM(node_q, ["x", "w", "s", "b", "i"]) == "mx.gather_qmm(x, w, s, b, i, transpose=True, group_size=32, bits=8)"

    assert vis.visit_QrVjp(node, ["a", "b", "c"]) == "_qr_vjp_eager(bk, a, b, c)"
    assert vis.visit_QrVjp(node, ["a", "b", "c"]) == "_qr_vjp_eager(bk, a, b, c)"
    assert vis.visit_SvdVjp(node, ["a", "b", "c", "d"], compute_uv=True) == "_svd_vjp_eager(bk, a, b, c, d, compute_uv=True)"
    assert vis.visit_SvdVjp(node, ["a", "b"], compute_uv=False) == "_svd_vjp_eager(bk, a, b, compute_uv=False)"


def test_linalg_mixin_dotted_prefix():
    vis = DummyVisitor(DummyGeneratorDotted())
    node = DummyNode()
    assert vis.visit_CholeskyVjp(node, ["a", "b"]) == "_cholesky_vjp_eager(np, a, b)"
    assert vis.visit_LuVjp(node, ["a", "b", "c", "d"]) == "_lu_vjp_eager(np, a, b, c, d)"


def test_linalg_mixin_re_import():
    vis = DummyVisitor()
    node = DummyNode()
    vis.visit_LuVjp(node, ["a", "b", "c", "d"])
    vis.visit_LuVjp(node, ["a", "b", "c", "d"])
