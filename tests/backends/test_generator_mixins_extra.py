"""Module docstring."""

from ml_switcheroo_compiler.backends.common.generator_mixins import SharedASTGeneratorVisitor
from ml_switcheroo_compiler.backends.common.mixins.nn import GroupNormConfig


class DummyGenerator(SharedASTGeneratorVisitor):
    """Class docstring."""

    def _get_backend_prefix(self) -> object:
        """Function docstring."""
        return "dummy"

    def visit(self, node: object, input_vars: object, **kwargs: object) -> object:
        """Function docstring."""
        return getattr(self, f"visit_{node.op_type}")(node, input_vars, **kwargs)


def test_shared_ast_generator_mixins() -> object:
    """Function docstring."""
    gen = DummyGenerator()

    class DummyNode:
        """Class docstring."""

        op_type = "Dummy"

        def __init__(self, kwargs: object = None) -> object:
            """Function docstring."""
            self.attributes = kwargs or {}

    input_vars = ["v1", "v2", "v3", "v4", "v5", "v6"]

    gen.visit_AdaptiveAvgPool2D(DummyNode({"output_size": (2, 2)}), input_vars[:1])
    gen.visit_AdaptiveMaxPool2D(DummyNode({"output_size": (2, 2)}), input_vars[:1])
    gen.visit_AddN(DummyNode(), input_vars[:3])
    gen.visit_AccumulateN(DummyNode(), input_vars[:3])
    gen.visit_Scan(DummyNode(), input_vars[:2])
    gen.visit_Switch(DummyNode(), input_vars[:3])
    gen.visit_TimeDistributed(DummyNode({"wrapped_op_name": "dummy"}), input_vars[:1])
    gen.visit_ActivityRegularization(DummyNode({"l1": 0.1, "l2": 0.2}), input_vars[:1])
    gen.visit_ActivityRegularization(DummyNode({"l1": 0.0, "l2": 0.2}), input_vars[:1])
    gen.visit_ActivityRegularization(DummyNode({"l1": 0.1, "l2": 0.0}), input_vars[:1])
    gen.visit_ActivityRegularization(DummyNode({"l1": 0.0, "l2": 0.0}), input_vars[:1])
    gen.visit_AdjustBrightness(DummyNode({"delta": 0.1}), input_vars[:1])
    gen.visit_AdjustContrast(DummyNode({"contrast_factor": 1.5}), input_vars[:1])
    gen.visit_AdjustHue(DummyNode({"delta": 0.1}), input_vars[:1])
    gen.visit_AdjustSaturation(DummyNode({"saturation_factor": 1.5}), input_vars[:1])
    gen.visit_AffineGenerator(DummyNode(), input_vars[:4])
    gen.visit_AffineGrid(DummyNode(), input_vars[:1])
    gen.visit_AffineTransform(DummyNode(), input_vars[:2])
    gen.visit_AllGather(DummyNode({"axis": 0}), input_vars[:1])
    gen.visit_AllReduce(DummyNode({"op": "sum"}), input_vars[:1])
    gen.visit_AllToAll(DummyNode({"split_axis": 0, "concat_axis": 1}), input_vars[:1])
    gen.visit_GroupNorm(DummyNode({"num_groups": 2, "epsilon": 1e-5}), input_vars[:3])
    gen.visit_GroupMean(DummyNode({"num_groups": 2}), input_vars[:1])
    gen.visit_GroupVariance(DummyNode({"num_groups": 2}), input_vars[:1])
    config = GroupNormConfig(
        prefix="dummy",
        module="dummy_mod",
        reshape="reshape",
        mean="mean",
        var="var",
        sqrt="sqrt",
        dim_arg="axis=",
        keepdim_arg="keepdims=",
    )
    gen._get_group_norm_code(config)
    gen.visit_AlphaDropout(DummyNode({"rate": 0.5}), input_vars[:1])
    gen.visit_Angle(DummyNode(), input_vars[:1])
    gen.visit_ApproxMaxK(DummyNode({"k": 2}), input_vars[:1])
    gen.visit_ApproxMaxKIndices(DummyNode({"k": 2}), input_vars[:1])
    gen.visit_ApproxMinK(DummyNode({"k": 2}), input_vars[:1])
    gen.visit_ApproxMinKIndices(DummyNode({"k": 2}), input_vars[:1])
    gen.visit_ArgSort(DummyNode({"axis": -1}), input_vars[:1])
    gen.visit_Argwhere(DummyNode(), input_vars[:1])
    gen.visit_Argpartition(DummyNode({"kth": 1}), input_vars[:1])
    gen.visit_AsString(DummyNode(), input_vars[:1])
    gen.visit_Assert(DummyNode({"msg": "error"}), input_vars[:1])
    gen.visit_Assign(DummyNode(), input_vars[:2])
    gen.visit_AssignAdd(DummyNode(), input_vars[:2])
    gen.visit_AssignSub(DummyNode(), input_vars[:2])
    gen.visit_AssociativeScan(DummyNode(), input_vars[:2])
    gen.visit_AugMix(DummyNode(), input_vars[:1])
    gen.visit_AutoContrast(DummyNode(), input_vars[:1])
    gen.visit_AxisIndex(DummyNode({"axis_name": "batch"}), input_vars[:1])
    gen.visit_Ball(DummyNode({"p": 2.0, "d": 3}), input_vars[:1])
    gen.visit_BandPart(DummyNode({"num_lower": 1, "num_upper": 1}), input_vars[:1])
    gen.visit_BandedTriangularSolve(DummyNode({"lower": True}), input_vars[:2])
    gen.visit_Bartlett(DummyNode(), input_vars[:1])
    gen.visit_BesselI0(DummyNode(), input_vars[:1])
    gen.visit_BesselI0e(DummyNode(), input_vars[:1])
    gen.visit_BesselI1(DummyNode(), input_vars[:1])
    gen.visit_BesselI1e(DummyNode(), input_vars[:1])
    gen.visit_BesselJ0(DummyNode(), input_vars[:1])
    gen.visit_BesselJ1(DummyNode(), input_vars[:1])
    gen.visit_BesselJn(DummyNode(), input_vars[:2])
    gen.visit_BesselK0(DummyNode(), input_vars[:1])
    gen.visit_BesselK0e(DummyNode(), input_vars[:1])
    gen.visit_BesselK1(DummyNode(), input_vars[:1])
    gen.visit_BesselK1e(DummyNode(), input_vars[:1])
    gen.visit_BesselY0(DummyNode(), input_vars[:1])
    gen.visit_BesselY1(DummyNode(), input_vars[:1])
    gen.visit_Beta(DummyNode(), input_vars[:3])
    gen.visit_Betainc(DummyNode(), input_vars[:3])


def test_shared_ast_generator_mixins2() -> object:
    """Function docstring."""
    gen = DummyGenerator()

    class DummyNode:
        """Class docstring."""

        op_type = "Dummy"

        def __init__(self, kwargs: object = None) -> object:
            """Function docstring."""
            self.attributes = kwargs or {}

    input_vars = ["v1", "v2"]
    gen.visit_AddN(DummyNode(), [])

    class DummyConfig:
        """Class docstring."""

        training = True
        noise_shape = None
        seed = 42

    # kwargs to AlphaDropout usually goes through **kwargs of visit_AlphaDropout, wait
    # The signature is visit_AlphaDropout(self, node: object, input_vars: list[str], **kwargs: object) -> str:
    # Actually, rate and config come from node.attributes typically, but let's look at the implementation
    # It says `rate = kwargs.get("rate", 0.5)`
    # Ah, it gets them from `kwargs` directly, which means the transpiler might pass node attributes as kwargs.
    gen.visit_AlphaDropout(DummyNode(), input_vars[:1], rate=0.5, config=DummyConfig())
