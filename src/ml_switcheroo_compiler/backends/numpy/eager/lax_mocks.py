"""Mocks for LAX ops."""
# ruff: noqa: PLR2004, ANN001, ANN002, ANN003, ANN201, ANN202, D100, D103

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry

ops_to_mock = [
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
    "ScanBind",
    "Betainc",
    "ErfInv",
    "Igamma",
    "Igammac",
    "Polygamma",
    "Zeta",
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
    "AllReduce",
    "AllToAll",
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

for op in ops_to_mock:

    def make_mock(op_name):
        def _mock(np_mod, *args, **kwargs):
            if op_name in ["ForiLoop", "WhileLoop"]:
                return args[2] if len(args) > 2 else np_mod.array(0)
            if len(args) > 0:
                return args[0]
            return np_mod.array(0)

        return _mock

    numpy_eager_registry.register(op)(make_mock(op))
