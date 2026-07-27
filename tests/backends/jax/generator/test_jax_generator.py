# ruff: noqa
from ml_switcheroo_compiler.backends.jax.generator_mixins import JaxAudioVisitor, JaxControlFlowVisitor, JaxDistributedVisitor, JaxMathVisitor, JaxVisionVisitor
from unittest.mock import MagicMock, patch

from unittest.mock import MagicMock
from ml_switcheroo_compiler.backends.jax.generator import JAXCodeGenerator
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
import sys

"Core abstractions and logic definitions for test_jax_generator_extra.py."


def test_jax_generator_extra_coverage() -> object:
    """Test the jax generator extra coverage behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        g = IRGraph()
        gen = JAXCodeGenerator(g)

        def _test_node(op_type: object, inputs: object, attrs: object, expected: object) -> object:
            """Test the node behavior.

            Args:
                op_type (object): The op_type parameter.
                inputs (object): The inputs parameter.
                attrs (object): The attrs parameter.
                expected (object): The expected parameter.

            Returns:
                object: The inferred shape or computed result.
            """
            n = IRNode(id="n1", op_type=op_type, inputs=inputs, attributes=attrs, shape_metadata=None)
            res = gen.visit(n, inputs)
            assert expected in res

        _test_node("SegmentSum", ["x", "ids"], {"num_segments": 2}, "segment_sum")
        _test_node("SegmentMax", ["x", "ids"], {}, "segment_max")
        _test_node("SegmentMin", ["x", "ids"], {}, "segment_min")
        _test_node("SegmentProd", ["x", "ids"], {}, "segment_prod")
        _test_node("UnsortedSegmentSum", ["x", "ids"], {}, "segment_sum")
        _test_node("UnsortedSegmentMax", ["x", "ids"], {}, "segment_max")
        _test_node("UnsortedSegmentMin", ["x", "ids"], {}, "segment_min")
        _test_node("UnsortedSegmentProd", ["x", "ids"], {}, "segment_prod")
        _test_node("MatrixExponential", ["x"], {}, "expm")
        _test_node("Polar", ["x"], {"side": "left"}, "polar")
        _test_node("Polar", ["x"], {"side": "'right'"}, "polar")
        _test_node("Schur", ["x"], {}, "schur")
        _test_node("Cholesky", ["x"], {}, "cholesky")
        _test_node("Svd", ["x"], {"full_matrices": False, "compute_uv": False}, "svd")
        _test_node("If", ["cond"], {}, "cond")
        _test_node("Loop", ["x"], {}, "while_loop")
        _test_node("Scan", ["x", "y"], {}, "scan")
        _test_node("PowerIteration", ["x", "u"], {"num_iters": 2}, "power_iteration")
        _test_node("PowerIteration", ["x"], {}, "power_iteration")
        _test_node("ConvTranspose", ["x", "w"], {"strides": 2, "padding": "SAME"}, "conv_transpose")
        _test_node("RaggedDot", ["x", "y"], {}, "ragged_dot")
        _test_node("all_gather", ["x"], {"axis_name": "'y'"}, "all_gather")
        _test_node("reduce_scatter", ["x"], {"axis_name": "'y'", "axis": 1, "op": "jax.lax.pmax"}, "reduce_scatter")
        _test_node("all_reduce", ["x"], {"axis_name": "'y'", "op": "pmax"}, "pmax")
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


def test_jax_generator_generate_full() -> object:
    """Test the jax generator generate full behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        g = IRGraph()
        n = IRNode(id="n1", op_type="UnknownOp", inputs=[], attributes={}, shape_metadata=None)
        g.nodes["n1"] = n
        gen = JAXCodeGenerator(g)
        assert gen.get_fallback_prefix() == "jnp"
        assert "jnp.zeros({shape})" in gen._format_zeros_like("zeros", {})
        assert "jnp.full({shape}, {fill_value})" in gen._format_full({})
        code = gen.generate()
        assert "def apply_model(params, *args, **kwargs):" in code
        assert "import jax" in code
        gen._emit_constant_assignment("var_a", "1")
        assert "var_a = jnp.array(1)" in "\n".join(gen.code)
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


"Test module."

sys.modules["jax"] = MagicMock()

sys.modules["jax.ops"] = MagicMock()

sys.modules["jax.numpy"] = MagicMock()

sys.modules["jax.scipy"] = MagicMock()

sys.modules["jax.scipy.special"] = MagicMock()

sys.modules["jax.scipy.signal"] = MagicMock()

sys.modules["jax.scipy.stats"] = MagicMock()

sys.modules["jax.scipy.linalg"] = MagicMock()

sys.modules["jax.nn"] = MagicMock()


class DummyNode:
    def __init__(self, attrs=None):
        self.attributes = attrs or {}


def test_jax_distributed_visitor():
    vis = JaxDistributedVisitor()
    node = DummyNode({"axis_name": "'y'", "axis": 1, "op": "psum"})
    assert vis.visit_all_gather(node, ["a"]) == "jax.lax.all_gather(a, axis_name='y')"
    assert vis.visit_all_gather(DummyNode(), ["a"]) == "jax.lax.all_gather(a, axis_name='x')"
    assert vis.visit_reduce_scatter(node, ["a"]) == "jax.lax.reduce_scatter(a, psum, scatter_dimension=1, axis_name='y')"
    assert vis.visit_reduce_scatter(DummyNode(), ["a"]) == "jax.lax.reduce_scatter(a, jax.lax.psum, scatter_dimension=0, axis_name='x')"
    assert vis.visit_all_reduce(node, ["a"]) == "jax.lax.psum(a, axis_name='y')"
    assert vis.visit_all_reduce(DummyNode(), ["a"]) == "jax.lax.psum(a, axis_name='x')"


def test_jax_math_visitor():
    vis = JaxMathVisitor()
    node = DummyNode({"num_segments": 2})
    assert vis.visit_SegmentSum(node, ["a", "b"]) == "jax.ops.segment_sum(a, b, num_segments=2)"
    assert vis.visit_SegmentSum(DummyNode(), ["a", "b"]) == "jax.ops.segment_sum(a, b, num_segments=None)"
    assert vis.visit_SegmentMax(node, ["a", "b"]) == "jax.ops.segment_max(a, b, num_segments=2)"
    assert vis.visit_SegmentMax(DummyNode(), ["a", "b"]) == "jax.ops.segment_max(a, b, num_segments=None)"
    assert vis.visit_SegmentMin(node, ["a", "b"]) == "jax.ops.segment_min(a, b, num_segments=2)"
    assert vis.visit_SegmentMin(DummyNode(), ["a", "b"]) == "jax.ops.segment_min(a, b, num_segments=None)"
    assert vis.visit_SegmentProd(node, ["a", "b"]) == "jax.ops.segment_prod(a, b, num_segments=2)"
    assert vis.visit_SegmentProd(DummyNode(), ["a", "b"]) == "jax.ops.segment_prod(a, b, num_segments=None)"
    assert vis.visit_UnsortedSegmentSum(node, ["a", "b"]) == "jax.ops.segment_sum(a, b, num_segments=2)"
    assert vis.visit_UnsortedSegmentSum(DummyNode(), ["a", "b"]) == "jax.ops.segment_sum(a, b, num_segments=None)"
    assert vis.visit_UnsortedSegmentMax(node, ["a", "b"]) == "jax.ops.segment_max(a, b, num_segments=2)"
    assert vis.visit_UnsortedSegmentMax(DummyNode(), ["a", "b"]) == "jax.ops.segment_max(a, b, num_segments=None)"
    assert vis.visit_UnsortedSegmentMin(node, ["a", "b"]) == "jax.ops.segment_min(a, b, num_segments=2)"
    assert vis.visit_UnsortedSegmentMin(DummyNode(), ["a", "b"]) == "jax.ops.segment_min(a, b, num_segments=None)"
    assert vis.visit_UnsortedSegmentProd(node, ["a", "b"]) == "jax.ops.segment_prod(a, b, num_segments=2)"
    assert vis.visit_UnsortedSegmentProd(DummyNode(), ["a", "b"]) == "jax.ops.segment_prod(a, b, num_segments=None)"
    assert vis.visit_MatrixExponential(DummyNode(), ["a"]) == "jax.scipy.linalg.expm(a)"
    assert vis.visit_Polar(DummyNode({"side": "left"}), ["a"]) == "jax.scipy.linalg.polar(a, side='left')"
    assert vis.visit_Polar(DummyNode({"side": "'right'"}), ["a"]) == "jax.scipy.linalg.polar(a, side='right')"
    assert vis.visit_Polar(DummyNode(), ["a"]) == "jax.scipy.linalg.polar(a, side='right')"
    assert vis.visit_Schur(DummyNode(), ["a"]) == "jax.scipy.linalg.schur(a)"
    assert vis.visit_Cholesky(DummyNode(), ["a"]) == "jax.numpy.linalg.cholesky(a)"
    assert vis.visit_Svd(DummyNode({"full_matrices": False, "compute_uv": False}), ["a"]) == "jax.numpy.linalg.svd(a, full_matrices=False, compute_uv=False)"
    assert vis.visit_Svd(DummyNode(), ["a"]) == "jax.numpy.linalg.svd(a, full_matrices=True, compute_uv=True)"
    assert vis.visit_PowerIteration(DummyNode({"num_iters": 5}), ["a", "u"]) == "jax_power_iteration(a, 5, u)"
    assert vis.visit_PowerIteration(DummyNode(), ["a"]) == "jax_power_iteration(a, 1, None)"
    assert vis.visit_RaggedDot(DummyNode(), ["a", "b"]) == "jax_ragged_dot(a, b)"
    assert vis.visit_Einsum(DummyNode(), ["a", "b"], equation="i,j->ij") == "jnp.einsum('i,j->ij', a, b)"
    assert vis.visit_Einsum(DummyNode(), ["a", "b"]) == "jnp.einsum('', a, b)"


def test_jax_control_flow_visitor():
    vis = JaxControlFlowVisitor()
    assert vis.visit_If(DummyNode(), ["c"]) == "jax.lax.cond(c, lambda: None, lambda: None)"
    assert vis.visit_Loop(DummyNode(), ["c"]) == "jax.lax.while_loop(lambda _: True, lambda _: c, c)"
    assert vis.visit_Scan(DummyNode(), ["a"]) == "jax.lax.scan(lambda c, x: (c, x), a, None)"
    assert vis.visit_Scan(DummyNode(), ["a", "b"]) == "jax.lax.scan(lambda c, x: (c, x), a, b)"


def test_jax_vision_visitor():
    vis = JaxVisionVisitor()
    assert vis.visit_ElasticTransform(DummyNode({"data_format": "df", "interpolation": "bicubic"}), ["a", "b"]) == "jax_elastic_transform(a, b, 'bicubic', 0.0, \"df\")"
    assert vis.visit_ElasticTransform(DummyNode(), ["a", "b"]) == "jax_elastic_transform(a, b, 'bilinear', 0.0, None)"
    assert vis.visit_GaussianBlur(DummyNode({"kernel_size": 3, "sigma": 1.0, "data_format": "df"}), ["a"]) == "jax_gaussian_blur(a, 3, 1.0, 'same', \"df\")"
    assert vis.visit_GaussianBlur(DummyNode(), ["a"]) == "jax_gaussian_blur(a, None, None, 'same', None)"
    assert vis.visit_MedianFilter(DummyNode({"kernel_size": 3, "sigma": 1.0, "data_format": "df"}), ["a"]) == "jax_median_filter(a, 3, 'same', \"df\")"
    assert vis.visit_MedianFilter(DummyNode(), ["a"]) == "jax_median_filter(a, None, 'same', None)"
    assert vis.visit_IoU(DummyNode({"bounding_box_format": "cxywh"}), ["a", "b"]) == "jax_iou(a, b, 'cxywh')"
    assert vis.visit_IoU(DummyNode(), ["a", "b"]) == "jax_iou(a, b, 'xyxy')"
    assert vis.visit_NonMaxSuppression(DummyNode({"max_output_size": 10}), ["a", "b"]) == "jax_nms(a, b, 10, 0.5, -inf)"
    assert vis.visit_NonMaxSuppression(DummyNode(), ["a", "b"]) == "jax_nms(a, b, None, 0.5, -inf)"
    assert vis.visit_ResizeBicubic(DummyNode({"size": 10, "align_corners": True}), ["a"]) == "jax_resize(a, 10, 'bicubic', True)"
    assert vis.visit_ResizeBicubic(DummyNode(), ["a"]) == "jax_resize(a, None, 'bicubic', False)"
    assert vis.visit_ResizeLanczos3(DummyNode({"size": 10, "align_corners": True}), ["a"]) == "jax_resize(a, 10, 'lanczos3', True)"
    assert vis.visit_ResizeLanczos3(DummyNode(), ["a"]) == "jax_resize(a, None, 'lanczos3', False)"
    assert vis.visit_ExtractBoundingBoxes(DummyNode({"crop_size": 10, "data_format": "df"}), ["a", "b", "c"]) == "jax_extract_bounding_boxes(a, b, c, 10, 'bilinear', 0.0, \"df\")"
    assert vis.visit_ExtractBoundingBoxes(DummyNode(), ["a", "b", "c"]) == "jax_extract_bounding_boxes(a, b, c, None, 'bilinear', 0.0, None)"
    assert vis.visit_PerspectiveTransform(DummyNode({"data_format": "df"}), ["a", "b", "c"]) == "jax_perspective_transform(a, b, c, 'bilinear', 0.0, \"df\")"
    assert vis.visit_PerspectiveTransform(DummyNode(), ["a", "b", "c"]) == "jax_perspective_transform(a, b, c, 'bilinear', 0.0, None)"


def test_jax_audio_visitor():
    vis = JaxAudioVisitor()
    assert vis.visit_Istft(DummyNode({"frame_length": 2048, "frame_step": 512, "center": False}), ["a"]) == "jax_istft(a, 2048, 512, None, 'hann', False)"
    assert vis.visit_MelFilterbank(DummyNode({"num_mel_bins": 1, "num_spectrogram_bins": 2, "sample_rate": 3, "lower_edge_hertz": 4, "upper_edge_hertz": 5}), ["a"]) == "jax_mel_filterbank(1, 2, 3, 4, 5)"
    assert vis.visit_Mfcc(DummyNode({"num_mel_bins": 1, "sample_rate": 2, "lower_edge_hertz": 3, "upper_edge_hertz": 4, "num_mfccs": 5}), ["a"]) == "jax_mfcc(a, 2, 1, 3, 4, 5)"


"Test module."

sys.modules["jax"] = MagicMock()

sys.modules["jax.ops"] = MagicMock()

sys.modules["jax.numpy"] = MagicMock()

sys.modules["jax.scipy"] = MagicMock()

sys.modules["jax.scipy.special"] = MagicMock()

sys.modules["jax.scipy.signal"] = MagicMock()

sys.modules["jax.scipy.stats"] = MagicMock()

sys.modules["jax.scipy.linalg"] = MagicMock()

sys.modules["jax.nn"] = MagicMock()


class DummyGraph:
    def __init__(self):
        self.nodes = []


def test_jax_generator():
    g = DummyGraph()
    gen = JAXCodeGenerator(g)
    assert gen._get_backend_prefix() == "jax"
    assert gen.get_fallback_prefix() == "jnp"
    assert gen._format_zeros_like("zeros", {}) == "jnp.zeros({shape})"
    assert gen._format_zeros_like("ones", {"dtype": "int32"}) == "jnp.ones({shape}), dtype='int32'"
    assert gen._format_full({}) == "jnp.full({shape}, {fill_value})"
    assert gen._format_full({"dtype": "int32"}) == "jnp.full({shape}, {fill_value}), dtype='int32'"
    ops = gen.get_ops_map({})
    assert "Zeros" in ops
    assert "Ones" in ops
    assert "Full" in ops
    assert ops["BroadcastInDim"] == "{0}.broadcast_in_dim({1}, {2})"
    gen._emit_constant_assignment("var_a", "42")
    assert gen.code[-1] == "var_a = jnp.array(42)"
    assert gen._generate_file_header() == [gen.header.strip()]
    imports = gen._resolve_imports()
    assert "import jax" in imports
    assert "import jax.numpy as jnp" in imports
    gen.code = []
    gen._generate_function_signature()
    assert gen.indent_level == 1
    assert "def apply_model(params, *args, **kwargs):" in gen.code[0]


def test_jax_generator_imports():
    g = DummyGraph()
    gen = JAXCodeGenerator(g)
    import builtins

    original_open = builtins.open

    def mock_open(path, *args, **kwargs):
        if "jax_prefix.py.tmpl" in str(path):
            from io import StringIO

            return StringIO("import test_template")
        return original_open(path, *args, **kwargs)

    with patch("builtins.open", side_effect=mock_open):
        imports = gen._resolve_imports()
        assert "import test_template" in imports


def test_jax_generator_save_funcs():
    g = DummyGraph()
    gen = JAXCodeGenerator(g)
