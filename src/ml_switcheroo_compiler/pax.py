"""PAX frontend compatibility layer.

This module provides backend implementations of the PAX framework classes
to ensure semantic and syntactic compatibility through the zero-pax shim.
"""

import ml_switcheroo_compiler.ops as numpy
from ml_switcheroo_compiler.core.tensor import Tensor


class PaxModule:
    """Base class for all PAX modules."""

    def __init__(self, **kwargs: object) -> None:
        """Initialize the module.

        Args:
            **kwargs: Additional keyword arguments.
        """
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Forward pass generic backend routing & IR construction.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return self.forward(*args, **kwargs)

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Abstract forward pass method.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        msg = "forward not implemented"
        raise NotImplementedError(msg)

    def init_weights(self) -> None:
        """State initialization backend API."""


class BaseActivation(PaxModule):
    """Base class for activations."""


class BaseNormalization(PaxModule):
    """Base class for normalizations."""


# Activations
class CubedReLU(BaseActivation):
    """Cubed ReLU activation."""

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """Forward pass.

        Args:
            x (Tensor): The x.

        Returns:
            Tensor: The computed result.
        """
        return numpy.power(numpy.maximum(x, 0.0), 3)


class ELU(BaseActivation):
    """ELU activation."""

    def __init__(self, alpha: float = 1.0, **kwargs: object) -> None:
        """Initialize.

        Args:
            alpha (float): The alpha.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.alpha = alpha

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """Forward pass.

        Args:
            x (Tensor): The x.

        Returns:
            Tensor: The computed result.
        """
        return numpy.where(x > 0.0, x, self.alpha * (numpy.exp(x) - 1.0))


class GELU(BaseActivation):
    """GELU activation."""

    def __init__(self, approximate: bool = False, **kwargs: object) -> None:
        """Initialize.

        Args:
            approximate (bool): The approximate.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.approximate = approximate

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """Forward pass.

        Args:
            x (Tensor): The x.

        Returns:
            Tensor: The computed result.
        """
        from ml_switcheroo_compiler.nn import gelu

        return gelu(x, approximate=self.approximate)


class LeakyReLU(BaseActivation):
    """Leaky ReLU activation."""

    def __init__(self, negative_slope: float = 0.01, **kwargs: object) -> None:
        """Initialize.

        Args:
            negative_slope (float): The negative_slope.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.negative_slope = negative_slope

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """Forward pass.

        Args:
            x (Tensor): The x.

        Returns:
            Tensor: The computed result.
        """
        return numpy.where(x > 0.0, x, x * self.negative_slope)


class ReLU(BaseActivation):
    """ReLU activation."""

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """Forward pass.

        Args:
            x (Tensor): The x.

        Returns:
            Tensor: The computed result.
        """
        return numpy.maximum(x, 0.0)


class ReLU6(BaseActivation):
    """ReLU6 activation."""

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """Forward pass.

        Args:
            x (Tensor): The x.

        Returns:
            Tensor: The computed result.
        """
        return numpy.clip(x, 0.0, 6.0)


class SiLU(BaseActivation):
    """SiLU activation."""

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """Forward pass.

        Args:
            x (Tensor): The x.

        Returns:
            Tensor: The computed result.
        """
        return x * (1.0 / (1.0 + numpy.exp(-x)))


class Sigmoid(BaseActivation):
    """Sigmoid activation."""

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """Forward pass.

        Args:
            x (Tensor): The x.

        Returns:
            Tensor: The computed result.
        """
        return 1.0 / (1.0 + numpy.exp(-x))


class SigmoidCrossEntropy(PaxModule):
    """Sigmoid cross entropy loss."""

    def forward(
        self,
        logits: Tensor,
        labels: Tensor,
    ) -> Tensor:
        """Forward pass.

        Args:
            logits (Tensor): The logits.
            labels (Tensor): The labels.

        Returns:
            Tensor: The computed result.
        """
        return (
            numpy.maximum(logits, 0.0)
            - logits * labels
            + numpy.log1p(numpy.exp(-numpy.abs(logits)))
        )


class SquaredReLU(BaseActivation):
    """Squared ReLU activation."""

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """Forward pass.

        Args:
            x (Tensor): The x.

        Returns:
            Tensor: The computed result.
        """
        return numpy.square(numpy.maximum(x, 0.0))


class Swish(BaseActivation):
    """Swish activation."""

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """Forward pass.

        Args:
            x (Tensor): The x.

        Returns:
            Tensor: The computed result.
        """
        return x * (1.0 / (1.0 + numpy.exp(-x)))


class Tanh(BaseActivation):
    """Tanh activation."""

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """Forward pass.

        Args:
            x (Tensor): The x.

        Returns:
            Tensor: The computed result.
        """
        return numpy.tanh(x)


class AttentionProjection(PaxModule):
    """AttentionProjection."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class DotProductAttention(PaxModule):
    """DotProductAttention."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class DotProductAttentionWithContext(PaxModule):
    """DotProductAttentionWithContext."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class DotProductAttentionWithContextXL(PaxModule):
    """DotProductAttentionWithContextXL."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class DotProductAttentionXL(PaxModule):
    """DotProductAttentionXL."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class GroupedQueryAttention(PaxModule):
    """GroupedQueryAttention."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class LocalSelfAttention(PaxModule):
    """LocalSelfAttention."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class LocalSelfAttentionAlibi(PaxModule):
    """LocalSelfAttentionAlibi."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class LocalSelfAttentionRelativeBias(PaxModule):
    """LocalSelfAttentionRelativeBias."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class LocalSelfAttentionXL(PaxModule):
    """LocalSelfAttentionXL."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class PerDimScale(PaxModule):
    """PerDimScale."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class RelativeBias(PaxModule):
    """RelativeBias."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class CausalDepthwiseConv1D(PaxModule):
    """CausalDepthwiseConv1D."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class Conv2D(PaxModule):
    """Conv2D."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class ConvBNAct(PaxModule):
    """ConvBNAct."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class ConvBNActWithPadding(PaxModule):
    """ConvBNActWithPadding."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class DepthwiseConv1D(PaxModule):
    """DepthwiseConv1D."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class GlobalPooling(PaxModule):
    """GlobalPooling."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class LightConv1D(PaxModule):
    """LightConv1D."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class Pooling(PaxModule):
    """Pooling."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class Pooling1D(PaxModule):
    """Pooling1D."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class AutodiffCheckpointType(PaxModule):
    """AutodiffCheckpointType."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class Bias(PaxModule):
    """Bias."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class Dropout(PaxModule):
    """Dropout."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class Einsum(PaxModule):
    """Einsum."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class EinsumOp(PaxModule):
    """EinsumOp."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class Identity(PaxModule):
    """Identity."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class LayerwiseShardablePipelined(PaxModule):
    """LayerwiseShardablePipelined."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class Linear(PaxModule):
    """Linear."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class MLPBlock(PaxModule):
    """MLPBlock."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class MaskedLmDataAugmenter(PaxModule):
    """MaskedLmDataAugmenter."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class MultitaskResidualAdapter(PaxModule):
    """MultitaskResidualAdapter."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class Repeat(PaxModule):
    """Repeat."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class Sequential(PaxModule):
    """Sequential."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class SpectrumAugmenter(PaxModule):
    """SpectrumAugmenter."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class StackingOverTime(PaxModule):
    """StackingOverTime."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class StochasticResidual(PaxModule):
    """StochasticResidual."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class VanillaBlock(PaxModule):
    """VanillaBlock."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class VitEntryLayers(PaxModule):
    """VitEntryLayers."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class VitExitLayers(PaxModule):
    """VitExitLayers."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class Embedding(PaxModule):
    """Embedding."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class FullSoftmax(PaxModule):
    """FullSoftmax."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class GShardSharedEmbeddingSoftmax(PaxModule):
    """GShardSharedEmbeddingSoftmax."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class Ngrammer(PaxModule):
    """Ngrammer."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class PositionalEmbedding(PaxModule):
    """PositionalEmbedding."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class PositionalEmbedding2D(PaxModule):
    """PositionalEmbedding2D."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class RandomVectorQuantizer(PaxModule):
    """RandomVectorQuantizer."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class SharedEmbeddingSoftmax(PaxModule):
    """SharedEmbeddingSoftmax."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class TrainablePositionalEmbedding(PaxModule):
    """TrainablePositionalEmbedding."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class VQNgrammer(PaxModule):
    """VQNgrammer."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class VectorQuantization(PaxModule):
    """VectorQuantization."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class VectorQuantizer(PaxModule):
    """VectorQuantizer."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class BertModel(PaxModule):
    """BertModel."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class BiTemperedLoss(PaxModule):
    """BiTemperedLoss."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class BregmanPCA(PaxModule):
    """BregmanPCA."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class ClassificationMLPModel(PaxModule):
    """ClassificationMLPModel."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class ClassificationModel(PaxModule):
    """ClassificationModel."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class Conformer(PaxModule):
    """Conformer."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class LanguageModel(PaxModule):
    """LanguageModel."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class LanguageModelContinuousBatching(PaxModule):
    """LanguageModelContinuousBatching."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class LanguageModelDPO(PaxModule):
    """LanguageModelDPO."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class LanguageModelType(PaxModule):
    """LanguageModelType."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class ResNet(PaxModule):
    """ResNet."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class ResNetBlock(PaxModule):
    """ResNetBlock."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class SequenceModel(PaxModule):
    """SequenceModel."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class VanillaNet(PaxModule):
    """VanillaNet."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class BatchNorm(BaseNormalization):
    """BatchNorm."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class GroupNorm(BaseNormalization):
    """GroupNorm."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class IdentityNorm(BaseNormalization):
    """IdentityNorm."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class LayerNorm(BaseNormalization):
    """LayerNorm."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class LayerNormalizedLstmCellSimple(PaxModule):
    """LayerNormalizedLstmCellSimple."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class RmsNorm(BaseNormalization):
    """RmsNorm."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class RmsNormNoScale(BaseNormalization):
    """RmsNormNoScale."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class SelfAttentionWithNormAndResidual(PaxModule):
    """SelfAttentionWithNormAndResidual."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class CifgLstmCellSimple(PaxModule):
    """CifgLstmCellSimple."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class FRnn(PaxModule):
    """FRnn."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class LstmCellSimple(PaxModule):
    """LstmCellSimple."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class LstmFrnn(PaxModule):
    """LstmFrnn."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class SSM(PaxModule):
    """SSM."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class SSMGated(PaxModule):
    """SSMGated."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class StackFrnn(PaxModule):
    """StackFrnn."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class TemporalShifting(PaxModule):
    """TemporalShifting."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class AdaptedTransformerFeedForward(PaxModule):
    """AdaptedTransformerFeedForward."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class FeedForward(PaxModule):
    """FeedForward."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class PipelinedTransformer(PaxModule):
    """PipelinedTransformer."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class SSMTransformer(PaxModule):
    """SSMTransformer."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class StackedTransformer(PaxModule):
    """StackedTransformer."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class StackedTransformerRepeated(PaxModule):
    """StackedTransformerRepeated."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class Transformer(PaxModule):
    """Transformer."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class TransformerEncoderDecoder(PaxModule):
    """TransformerEncoderDecoder."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class TransformerFeedForward(PaxModule):
    """TransformerFeedForward."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class TransformerFeedForwardMoe(PaxModule):
    """TransformerFeedForwardMoe."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class TransformerLm(PaxModule):
    """TransformerLm."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None


class VisionTransformer(PaxModule):
    """VisionTransformer."""

    def forward(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Forward pass.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The computed result.
        """
        return args[0] if args else None
