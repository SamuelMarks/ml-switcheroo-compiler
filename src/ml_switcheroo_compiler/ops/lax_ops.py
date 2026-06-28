"""LAX ops mocks."""
# ruff: noqa: E402, ANN001, ANN002, ANN003, ANN201, ANN202, D100, D103

from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.backends.registry import get_active_backend

ops_to_mock = [
    "ApproxMaxK",
    "ApproxMinK",
    "TopK",
    "SortKeyVal",
    "Clz",
    "PopulationCount",
    "BitcastConvertType",
    "ReducePrecision",
    "Conv",
    "ConvTranspose",
    "ConvGeneralDilatedLocal",
    "ConvGeneralDilatedPatches",
    "ConvWithGeneralPadding",
    "CustomLinearSolve",
    "CustomRoot",
    "ForiLoop",
    "WhileLoop",
    "AssociativeScan",
    "ScanBind",
    "Betainc",
    "ErfInv",
    "Igamma",
    "Igammac",
    "Polygamma",
    "Zeta",
    "BesselI0e",
    "BesselI1e",
    "DynamicSliceInDim",
    "DynamicUpdateSliceInDim",
    "DynamicIndexInDim",
    "DynamicUpdateIndexInDim",
    "SliceInDim",
    "ScatterApply",
    "ScatterMax",
    "ScatterMin",
    "ScatterMul",
    "Cummax",
    "Cummin",
    "Cumprod",
    "Cumlogsumexp",
    "AllGather",
    "Pdot",
    "Pmax",
    "Pmin",
    "PsumScatter",
    "Ppermute",
    "Pshuffle",
    "Pswapaxes",
    "AllToAll",
    "Pbroadcast",
    "IgammaGradA",
    "RandomGammaGrad",
    "RaggedDot",
]

import re

for op_name in ops_to_mock:

    class _MockOpDef(OpDef):
        def infer_shape(self, *args, **kwargs):
            return args[0].shape if args and hasattr(args[0], "shape") else ()

    _MockOpDef.op_name = op_name
    _MockOpDef.__name__ = op_name
    try:
        register_op(op_name)(_MockOpDef)
    except ValueError:
        pass

    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", op_name)
    snake = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()

    def make_func(op):
        def func(*args, **kwargs):
            if config.eager_mode:
                return get_active_backend().execute_op(op, *args, **kwargs)
            else:
                from ml_switcheroo_compiler.tracing import _tracer

                if getattr(_tracer, "is_tracing", False) and _tracer.active_graph is not None:
                    node = _tracer.active_graph.create_node(op_type=op)
                    for a in args:
                        if hasattr(a, "_node"):
                            node.add_input(a._node)
                    _tracer.active_graph.add_node(node)
                    import ml_switcheroo_compiler.core.tensor as tensor_mod

                    # Mock output proxy
                    out_t = tensor_mod.Tensor(None, tensor_mod.TensorConfig((), None, None))
                    out_t._node = node
                    return out_t
                return args[0] if args else None

        return func

    globals()[snake] = make_func(op_name)
