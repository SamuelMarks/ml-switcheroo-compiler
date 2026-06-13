"""Tests for module."""

import numpy as np
import pytest

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.pax import (
    ELU,
    GELU,
    CubedReLU,
    LeakyReLU,
    PaxModule,
    ReLU,
    ReLU6,
    Sigmoid,
    SigmoidCrossEntropy,
    SiLU,
    SquaredReLU,
    Swish,
    Tanh,
)


def test_pax_module_base() -> None:
    """Test pax module base."""
    mod = PaxModule(my_arg=5)
    assert mod.my_arg == 5
    mod.init_weights()  # Should not error
    with pytest.raises(NotImplementedError):
        mod()


def test_pax_activations() -> None:
    """Test pax activations."""
    config.eager_mode = True
    x_val = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    x = Tensor(x_val, x_val.shape, DType.Float32, Device("cpu"))

    # CubedReLU
    act = CubedReLU()
    out = act(x)
    assert out.shape == x.shape

    # ELU
    act = ELU(alpha=1.0)
    out = act(x)
    assert out.shape == x.shape

    # GELU
    act = GELU()
    out = act(x)
    assert out.shape == x.shape

    # LeakyReLU
    act = LeakyReLU()
    out = act(x)
    assert out.shape == x.shape

    # ReLU
    act = ReLU()
    out = act(x)
    assert out.shape == x.shape

    # ReLU6
    act = ReLU6()
    out = act(x)
    assert out.shape == x.shape

    # SiLU / Swish
    act = SiLU()
    out = act(x)
    assert out.shape == x.shape

    act = Swish()
    out = act(x)
    assert out.shape == x.shape

    # Sigmoid
    act = Sigmoid()
    out = act(x)
    assert out.shape == x.shape

    # SigmoidCrossEntropy
    act = SigmoidCrossEntropy()
    y_val = np.array([0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    y = Tensor(y_val, y_val.shape, DType.Float32, Device("cpu"))
    out = act(x, y)
    assert out.shape == x.shape

    # SquaredReLU
    act = SquaredReLU()
    out = act(x)
    assert out.shape == x.shape

    # Tanh
    act = Tanh()
    out = act(x)
    assert out.shape == x.shape


def test_pax_attention_and_conv() -> None:
    """Test pax attention and conv."""
    config.eager_mode = True
    x_val = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    x = Tensor(x_val, x_val.shape, DType.Float32, Device("cpu"))

    from ml_switcheroo_compiler import pax

    # Attention
    assert pax.AttentionProjection()(x).shape == x.shape
    assert pax.DotProductAttention()(x, x, x).shape == x.shape
    assert pax.DotProductAttentionWithContext()(x, x, x, x).shape == x.shape
    assert pax.DotProductAttentionWithContextXL()(x, x, x, x).shape == x.shape
    assert pax.DotProductAttentionXL()(x, x, x).shape == x.shape
    assert pax.GroupedQueryAttention()(x, x, x).shape == x.shape
    assert pax.LocalSelfAttention()(x).shape == x.shape
    assert pax.LocalSelfAttentionAlibi()(x).shape == x.shape
    assert pax.LocalSelfAttentionRelativeBias()(x).shape == x.shape
    assert pax.LocalSelfAttentionXL()(x).shape == x.shape
    assert pax.PerDimScale()(x).shape == x.shape
    assert pax.RelativeBias()(x).shape == x.shape

    # Convolutions
    assert pax.CausalDepthwiseConv1D()(x).shape == x.shape
    assert pax.Conv2D()(x).shape == x.shape
    assert pax.ConvBNAct()(x).shape == x.shape
    assert pax.ConvBNActWithPadding()(x).shape == x.shape
    assert pax.DepthwiseConv1D()(x).shape == x.shape
    assert pax.GlobalPooling()(x).shape == x.shape
    assert pax.LightConv1D()(x).shape == x.shape
    assert pax.Pooling()(x).shape == x.shape
    assert pax.Pooling1D()(x).shape == x.shape


def test_pax_all_other_modules() -> None:
    """Test pax all other modules."""
    config.eager_mode = True
    x_val = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    x = Tensor(x_val, x_val.shape, DType.Float32, Device("cpu"))

    from ml_switcheroo_compiler import pax

    # Core & Base Modules
    assert pax.AutodiffCheckpointType()(x).shape == x.shape
    assert pax.Bias()(x).shape == x.shape
    assert pax.Dropout()(x).shape == x.shape
    assert pax.Einsum()(x).shape == x.shape
    assert pax.EinsumOp()(x).shape == x.shape
    assert pax.Identity()(x).shape == x.shape
    assert pax.LayerwiseShardablePipelined()(x).shape == x.shape
    assert pax.Linear()(x).shape == x.shape
    assert pax.MLPBlock()(x).shape == x.shape
    assert pax.MaskedLmDataAugmenter()(x).shape == x.shape
    assert pax.MultitaskResidualAdapter()(x).shape == x.shape
    assert pax.Repeat()(x).shape == x.shape
    assert pax.Sequential()(x).shape == x.shape
    assert pax.SpectrumAugmenter()(x).shape == x.shape
    assert pax.StackingOverTime()(x).shape == x.shape
    assert pax.StochasticResidual()(x).shape == x.shape
    assert pax.VanillaBlock()(x).shape == x.shape
    assert pax.VitEntryLayers()(x).shape == x.shape
    assert pax.VitExitLayers()(x).shape == x.shape

    # Embeddings & Softmax
    assert pax.Embedding()(x).shape == x.shape
    assert pax.FullSoftmax()(x).shape == x.shape
    assert pax.GShardSharedEmbeddingSoftmax()(x).shape == x.shape
    assert pax.Ngrammer()(x).shape == x.shape
    assert pax.PositionalEmbedding()(x).shape == x.shape
    assert pax.PositionalEmbedding2D()(x).shape == x.shape
    assert pax.RandomVectorQuantizer()(x).shape == x.shape
    assert pax.SharedEmbeddingSoftmax()(x).shape == x.shape
    assert pax.TrainablePositionalEmbedding()(x).shape == x.shape
    assert pax.VQNgrammer()(x).shape == x.shape
    assert pax.VectorQuantization()(x).shape == x.shape
    assert pax.VectorQuantizer()(x).shape == x.shape

    # Models & Architectures
    assert pax.BertModel()(x).shape == x.shape
    assert pax.BiTemperedLoss()(x).shape == x.shape
    assert pax.BregmanPCA()(x).shape == x.shape
    assert pax.ClassificationMLPModel()(x).shape == x.shape
    assert pax.ClassificationModel()(x).shape == x.shape
    assert pax.Conformer()(x).shape == x.shape
    assert pax.LanguageModel()(x).shape == x.shape
    assert pax.LanguageModelContinuousBatching()(x).shape == x.shape
    assert pax.LanguageModelDPO()(x).shape == x.shape
    assert pax.LanguageModelType()(x).shape == x.shape
    assert pax.ResNet()(x).shape == x.shape
    assert pax.ResNetBlock()(x).shape == x.shape
    assert pax.SequenceModel()(x).shape == x.shape
    assert pax.VanillaNet()(x).shape == x.shape

    # Normalizations
    assert pax.BatchNorm()(x).shape == x.shape
    assert pax.GroupNorm()(x).shape == x.shape
    assert pax.IdentityNorm()(x).shape == x.shape
    assert pax.LayerNorm()(x).shape == x.shape
    assert pax.LayerNormalizedLstmCellSimple()(x).shape == x.shape
    assert pax.RmsNorm()(x).shape == x.shape
    assert pax.RmsNormNoScale()(x).shape == x.shape
    assert pax.SelfAttentionWithNormAndResidual()(x).shape == x.shape

    # RNNs & SSMs
    assert pax.CifgLstmCellSimple()(x).shape == x.shape
    assert pax.FRnn()(x).shape == x.shape
    assert pax.LstmCellSimple()(x).shape == x.shape
    assert pax.LstmFrnn()(x).shape == x.shape
    assert pax.SSM()(x).shape == x.shape
    assert pax.SSMGated()(x).shape == x.shape
    assert pax.StackFrnn()(x).shape == x.shape
    assert pax.TemporalShifting()(x).shape == x.shape

    # Transformers
    assert pax.AdaptedTransformerFeedForward()(x).shape == x.shape
    assert pax.FeedForward()(x).shape == x.shape
    assert pax.PipelinedTransformer()(x).shape == x.shape
    assert pax.SSMTransformer()(x).shape == x.shape
    assert pax.StackedTransformer()(x).shape == x.shape
    assert pax.StackedTransformerRepeated()(x).shape == x.shape
    assert pax.Transformer()(x).shape == x.shape
    assert pax.TransformerEncoderDecoder()(x).shape == x.shape
    assert pax.TransformerFeedForward()(x).shape == x.shape
    assert pax.TransformerFeedForwardMoe()(x).shape == x.shape
    assert pax.TransformerLm()(x).shape == x.shape
    assert pax.VisionTransformer()(x).shape == x.shape
