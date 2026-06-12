# PAX Frontend Compatibility Plan

This document tracks the backend implementation requirements in `ml-switcheroo-compiler` to ensure 100% semantic and syntactic compatibility with the `pax` framework through the `zero-pax` frontend shim.

## Core Operations & Bindings
- [ ] `numpy` backend operations binding
- [ ] `__call__` / `forward` generic backend routing & IR construction
- [ ] `init_weights` state initialization backend API

## Activations
- [ ] `BaseActivation`
- [ ] `CubedReLU`
- [ ] `ELU`
- [ ] `GELU`
- [ ] `LeakyReLU`
- [ ] `ReLU`
- [ ] `ReLU6`
- [ ] `SiLU`
- [ ] `Sigmoid`
- [ ] `SigmoidCrossEntropy`
- [ ] `SquaredReLU`
- [ ] `Swish`
- [ ] `Tanh`

## Attention
- [ ] `AttentionProjection`
- [ ] `DotProductAttention`
- [ ] `DotProductAttentionWithContext`
- [ ] `DotProductAttentionWithContextXL`
- [ ] `DotProductAttentionXL`
- [ ] `GroupedQueryAttention`
- [ ] `LocalSelfAttention`
- [ ] `LocalSelfAttentionAlibi`
- [ ] `LocalSelfAttentionRelativeBias`
- [ ] `LocalSelfAttentionXL`
- [ ] `PerDimScale`
- [ ] `RelativeBias`

## Convolutions
- [ ] `CausalDepthwiseConv1D`
- [ ] `Conv2D`
- [ ] `ConvBNAct`
- [ ] `ConvBNActWithPadding`
- [ ] `DepthwiseConv1D`
- [ ] `GlobalPooling`
- [ ] `LightConv1D`
- [ ] `Pooling`
- [ ] `Pooling1D`

## Core & Base Modules
- [ ] `AutodiffCheckpointType`
- [ ] `Bias`
- [ ] `Dropout`
- [ ] `Einsum`
- [ ] `EinsumOp`
- [ ] `Identity`
- [ ] `LayerwiseShardablePipelined`
- [ ] `Linear`
- [ ] `MLPBlock`
- [ ] `MaskedLmDataAugmenter`
- [ ] `MultitaskResidualAdapter`
- [ ] `Repeat`
- [ ] `Sequential`
- [ ] `SpectrumAugmenter`
- [ ] `StackingOverTime`
- [ ] `StochasticResidual`
- [ ] `VanillaBlock`
- [ ] `VitEntryLayers`
- [ ] `VitExitLayers`

## Embeddings & Softmax
- [ ] `Embedding`
- [ ] `FullSoftmax`
- [ ] `GShardSharedEmbeddingSoftmax`
- [ ] `Ngrammer`
- [ ] `PositionalEmbedding`
- [ ] `PositionalEmbedding2D`
- [ ] `RandomVectorQuantizer`
- [ ] `SharedEmbeddingSoftmax`
- [ ] `TrainablePositionalEmbedding`
- [ ] `VQNgrammer`
- [ ] `VectorQuantization`
- [ ] `VectorQuantizer`

## Models & Architectures
- [ ] `BertModel`
- [ ] `BiTemperedLoss`
- [ ] `BregmanPCA`
- [ ] `ClassificationMLPModel`
- [ ] `ClassificationModel`
- [ ] `Conformer`
- [ ] `LanguageModel`
- [ ] `LanguageModelContinuousBatching`
- [ ] `LanguageModelDPO`
- [ ] `LanguageModelType`
- [ ] `ResNet`
- [ ] `ResNetBlock`
- [ ] `SequenceModel`
- [ ] `VanillaNet`

## Normalizations
- [ ] `BaseNormalization`
- [ ] `BatchNorm`
- [ ] `GroupNorm`
- [ ] `IdentityNorm`
- [ ] `LayerNorm`
- [ ] `LayerNormalizedLstmCellSimple`
- [ ] `RmsNorm`
- [ ] `RmsNormNoScale`
- [ ] `SelfAttentionWithNormAndResidual`

## RNNs & SSMs
- [ ] `CifgLstmCellSimple`
- [ ] `FRnn`
- [ ] `LstmCellSimple`
- [ ] `LstmFrnn`
- [ ] `SSM`
- [ ] `SSMGated`
- [ ] `StackFrnn`
- [ ] `TemporalShifting`

## Transformers
- [ ] `AdaptedTransformerFeedForward`
- [ ] `FeedForward`
- [ ] `PipelinedTransformer`
- [ ] `SSMTransformer`
- [ ] `StackedTransformer`
- [ ] `StackedTransformerRepeated`
- [ ] `Transformer`
- [ ] `TransformerEncoderDecoder`
- [ ] `TransformerFeedForward`
- [ ] `TransformerFeedForwardMoe`
- [ ] `TransformerLm`
- [ ] `VisionTransformer`
