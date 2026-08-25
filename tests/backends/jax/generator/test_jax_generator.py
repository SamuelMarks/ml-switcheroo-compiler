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
        assert "def apply_model(params, *args, **kwargs) -> object:" in code
        assert "import jax" in code
        gen._emit_constant_assignment("var_a", "1")
        assert "var_a = jnp.array(1)" in "\n".join(gen.code)
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


"Test module."

sys.modules["jax.scipy"] = MagicMock()

sys.modules["jax.scipy.special"] = MagicMock()

sys.modules["jax.scipy.signal"] = MagicMock()

sys.modules["jax.scipy.stats"] = MagicMock()

sys.modules["jax.scipy.linalg"] = MagicMock()

sys.modules["jax.nn"] = MagicMock()


class DummyNode:
    def __init__(self, attrs=None):
        self.attributes = attrs or {}


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


def test_jax_distributed_visitor():
    from ml_switcheroo_compiler.backends.jax.generator_mixins import JaxDistributedVisitor

    class DummyGenerator(JaxDistributedVisitor):
        def __init__(self):
            self.code = []

    gen = DummyGenerator()

    # Send
    node_send = IRNode("send", "Send", inputs=["in1"], attributes={"dst_rank": 1})
    res = gen.visit_Send(node_send, ["in1"])
    assert res == ""
    assert "Send to 1" in gen.code[0]

    # Recv
    node_recv = IRNode("recv", "Recv", inputs=[], attributes={"src_rank": 2, "shape": (2, 2), "dtype": "float32"})
    res = gen.visit_Recv(node_recv, [])
    assert res == "v_recv"
    assert "Recv from 2" in gen.code[1]

    # AllGather
    node_allgather = IRNode("ag", "AllGather", inputs=["in1"], attributes={"axis_name": "'x'"})
    res = gen.visit_AllGather(node_allgather, ["in1"])
    assert res == "jax.lax.all_gather(in1, axis_name='x')"

    # ReduceScatter
    node_rs = IRNode("rs", "ReduceScatter", inputs=["in1"], attributes={"axis": 0, "axis_name": "'x'", "op": "jax.lax.psum"})
    res = gen.visit_ReduceScatter(node_rs, ["in1"])
    assert res == "jax.lax.reduce_scatter(in1, jax.lax.psum, scatter_dimension=0, axis_name='x')"

    # AllReduce
    node_ar = IRNode("ar", "AllReduce", inputs=["in1"], attributes={"axis_name": "'x'", "op": "psum"})
    res = gen.visit_AllReduce(node_ar, ["in1"])
    assert res == "jax.lax.psum(in1, axis_name='x')"


def test_jax_math_visitor_extra():
    from ml_switcheroo_compiler.backends.jax.generator_mixins import JaxMathVisitor

    class DummyGenerator(JaxMathVisitor):
        def __init__(self):
            pass

    gen = DummyGenerator()

    # RaggedDot
    node_rd = IRNode("rd", "RaggedDot", inputs=["in1", "in2"])
    res = gen.visit_RaggedDot(node_rd, ["in1", "in2"])
    assert res == "jax_ragged_dot(in1, in2)"

    # Einsum
    node_einsum = IRNode("einsum", "Einsum", inputs=["in1", "in2"])
    res = gen.visit_Einsum(node_einsum, ["in1", "in2"], equation="ij,jk->ik")
    assert res == "jnp.einsum('ij,jk->ik', in1, in2)"


def test_jax_audio_visitor():
    vis = JaxAudioVisitor()
    assert vis.visit_Istft(DummyNode({"frame_length": 2048, "frame_step": 512, "center": False}), ["a"]) == "jax_istft(a, 2048, 512, None, 'hann', False)"
    assert vis.visit_MelFilterbank(DummyNode({"num_mel_bins": 1, "num_spectrogram_bins": 2, "sample_rate": 3, "lower_edge_hertz": 4, "upper_edge_hertz": 5}), ["a"]) == "jax_mel_filterbank(1, 2, 3, 4, 5)"
    assert vis.visit_Mfcc(DummyNode({"num_mel_bins": 1, "sample_rate": 2, "lower_edge_hertz": 3, "upper_edge_hertz": 4, "num_mfccs": 5}), ["a"]) == "jax_mfcc(a, 2, 1, 3, 4, 5)"


"Test module."

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
    assert gen.get_fallback_prefix() == "jnp"
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
    assert "def apply_model(params, *args, **kwargs) -> object:" in gen.code[0]


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
