# ml-switcheroo-compiler PAX Compliance Plan

This document outlines the exhaustive checklist for `ml-switcheroo-compiler` to implement in order to fully support the `pax` (and `zero-pax` shim) API surface, enabling 100% semantic and syntactic test passing.

## Core Framework

| Checkbox | Name | Function/Class Signature | Docstring | Notes |
|---|---|---|---|---|
| [x] | `numpy` | N/A | Shim/Module/Hook required. | |
| [ ] | `__call__` | N/A | Shim/Module/Hook required. | |
| [ ] | `init_weights` | N/A | Shim/Module/Hook required. | |

## Activations

| Checkbox | Name | Function/Class Signature | Docstring | Notes |
|---|---|---|---|---|
| [ ] | `BaseActivation` | N/A | Represents the BaseActivation configuration and behavior.  This class encapsulates the functionality for BaseActivation. | |
| [ ] | `CubedReLU` | `__call__(self, x, *args, **kwargs)` | Represents the CubedReLU configuration and behavior.  This class encapsulates the functionality for CubedReLU. | |
| [ ] | `ELU` | `__call__(self, x, *args, **kwargs)` | Represents the ELU configuration and behavior.  This class encapsulates the functionality for ELU. | |
| [ ] | `GELU` | `__call__(self, x, *args, **kwargs)` | Represents the GELU configuration and behavior.  This class encapsulates the functionality for GELU. | |
| [ ] | `LeakyReLU` | `__call__(self, x, *args, **kwargs)` | Represents the LeakyReLU configuration and behavior.  This class encapsulates the functionality for LeakyReLU. | |
| [ ] | `ReLU` | `__call__(self, x, *args, **kwargs)` | Represents the ReLU configuration and behavior.  This class encapsulates the functionality for ReLU. | |
| [ ] | `ReLU6` | `__call__(self, x, *args, **kwargs)` | Represents the ReLU6 configuration and behavior.  This class encapsulates the functionality for ReLU6. | |
| [ ] | `SiLU` | `__call__(self, x, *args, **kwargs)` | Represents the SiLU configuration and behavior.  This class encapsulates the functionality for SiLU. | |
| [x] | `Sigmoid` | `__call__(self, x, *args, **kwargs)` | Represents the Sigmoid configuration and behavior.  This class encapsulates the functionality for Sigmoid. | |
| [ ] | `SigmoidCrossEntropy` | `__call__(self, logits, labels, *args, **kwargs)` | Represents the SigmoidCrossEntropy configuration and behavior.  This class encapsulates the functionality for SigmoidCrossEntropy. | |
| [ ] | `SquaredReLU` | `__call__(self, x, *args, **kwargs)` | Represents the SquaredReLU configuration and behavior.  This class encapsulates the functionality for SquaredReLU. | |
| [x] | `Swish` | `__call__(self, x, *args, **kwargs)` | Represents the Swish configuration and behavior.  This class encapsulates the functionality for Swish. | |
| [x] | `Tanh` | `__call__(self, x, *args, **kwargs)` | Represents the Tanh configuration and behavior.  This class encapsulates the functionality for Tanh. | |

## Attention

| Checkbox | Name | Function/Class Signature | Docstring | Notes |
|---|---|---|---|---|
| [ ] | `AttentionProjection` | `__call__(self, inputs, w, bias, *args, **kwargs)` | Represents the AttentionProjection configuration and behavior.  This class encapsulates the functionality for AttentionProjection. | |
| [ ] | `DotProductAttention` | `__call__(self, query, key, value, query_w, key_w, value_w, atten_mask, *args, **kwargs)` | Represents the DotProductAttention configuration and behavior.  This class encapsulates the functionality for DotProductAttention. | |
| [ ] | `DotProductAttentionWithContext` | `__call__(self, query, key, value, *args, **kwargs)` | Represents the DotProductAttentionWithContext configuration and behavior.  This class encapsulates the functionality for DotProductAttentionWithContext. | |
| [ ] | `DotProductAttentionWithContextXL` | `__call__(self, query, key, value, *args, **kwargs)` | Represents the DotProductAttentionWithContextXL configuration and behavior.  This class encapsulates the functionality for DotProductAttentionWithContextXL. | |
| [ ] | `DotProductAttentionXL` | `__call__(self, query, key, value, *args, **kwargs)` | Represents the DotProductAttentionXL configuration and behavior.  This class encapsulates the functionality for DotProductAttentionXL. | |
| [ ] | `GroupedQueryAttention` | `__call__(self, query, key, value, query_w, key_w, value_w, atten_mask, *args, **kwargs)` | Represents the GroupedQueryAttention configuration and behavior.  This class encapsulates the functionality for GroupedQueryAttention. | |
| [ ] | `LocalSelfAttention` | `__call__(self, query, key, value, *args, **kwargs)` | Represents the LocalSelfAttention configuration and behavior.  This class encapsulates the functionality for LocalSelfAttention. | |
| [ ] | `LocalSelfAttentionAlibi` | `__call__(self, query, key, value, *args, **kwargs)` | Represents the LocalSelfAttentionAlibi configuration and behavior.  This class encapsulates the functionality for LocalSelfAttentionAlibi. | |
| [ ] | `LocalSelfAttentionRelativeBias` | `__call__(self, query, key, value, *args, **kwargs)` | Represents the LocalSelfAttentionRelativeBias configuration and behavior.  This class encapsulates the functionality for LocalSelfAttentionRelativeBias. | |
| [ ] | `LocalSelfAttentionXL` | `__call__(self, query, key, value, *args, **kwargs)` | Represents the LocalSelfAttentionXL configuration and behavior.  This class encapsulates the functionality for LocalSelfAttentionXL. | |
| [ ] | `PerDimScale` | `__call__(self, inputs, scale, *args, **kwargs)` | Represents the PerDimScale configuration and behavior.  This class encapsulates the functionality for PerDimScale. | |
| [ ] | `RelativeBias` | N/A | Represents the RelativeBias configuration and behavior.  This class encapsulates the functionality for RelativeBias. | |

## Convolutions

| Checkbox | Name | Function/Class Signature | Docstring | Notes |
|---|---|---|---|---|
| [ ] | `CausalDepthwiseConv1D` | `__call__(self, inputs, w, *args, **kwargs)` | Represents the CausalDepthwiseConv1D configuration and behavior.  This class encapsulates the functionality for CausalDepthwiseConv1D. | |
| [x] | `Conv2D` | `__call__(self, inputs, w, bias, *args, **kwargs)` | Represents the Conv2D configuration and behavior.  This class encapsulates the functionality for Conv2D. | |
| [ ] | `ConvBNAct` | `__call__(self, inputs, w, bias, bn_gamma, bn_beta, *args, **kwargs)` | Represents the ConvBNAct configuration and behavior.  This class encapsulates the functionality for ConvBNAct. | |
| [ ] | `ConvBNActWithPadding` | `__call__(self, inputs, paddings, w, bias, bn_gamma, bn_beta, *args, **kwargs)` | Represents the ConvBNActWithPadding configuration and behavior.  This class encapsulates the functionality for ConvBNActWithPadding. | |
| [ ] | `DepthwiseConv1D` | `__call__(self, inputs, w, *args, **kwargs)` | Represents the DepthwiseConv1D configuration and behavior.  This class encapsulates the functionality for DepthwiseConv1D. | |
| [ ] | `GlobalPooling` | `__call__(self, inputs, epsilon, compatible_paddings, *args, **kwargs)` | Represents the GlobalPooling configuration and behavior.  This class encapsulates the functionality for GlobalPooling. | |
| [ ] | `LightConv1D` | `__call__(self, inputs, paddings, w, *args, **kwargs)` | Represents the LightConv1D configuration and behavior.  This class encapsulates the functionality for LightConv1D. | |
| [ ] | `Pooling` | `__call__(self, inputs, paddings, *args, **kwargs)` | Represents the Pooling configuration and behavior.  This class encapsulates the functionality for Pooling. | |
| [ ] | `Pooling1D` | `__call__(self, inputs, paddings, *args, **kwargs)` | Represents the Pooling1D configuration and behavior.  This class encapsulates the functionality for Pooling1D. | |

## Core & Base Layers

| Checkbox | Name | Function/Class Signature | Docstring | Notes |
|---|---|---|---|---|
| [ ] | `AutodiffCheckpointType` | `__call__(self, inputs, *args, **kwargs)` | Represents the AutodiffCheckpointType configuration and behavior.  This class encapsulates the functionality for AutodiffCheckpointType. | |
| [ ] | `Bias` | `__call__(self, inputs, b, *args, **kwargs)` | Represents the Bias configuration and behavior.  This class encapsulates the functionality for Bias. | |
| [x] | `Dropout` | `__call__(self, inputs, *args, **kwargs)` | Represents the Dropout configuration and behavior.  This class encapsulates the functionality for Dropout. | |
| [x] | `Einsum` | `__call__(self, *args, **kwargs)` | Represents the Einsum configuration and behavior.  This class encapsulates the functionality for Einsum. | |
| [ ] | `EinsumOp` | `__call__(self, *args, **kwargs)` | Represents the EinsumOp configuration and behavior.  This class encapsulates the functionality for EinsumOp. | |
| [ ] | `Identity` | `__call__(self, inputs, *args, **kwargs)` | Represents the Identity configuration and behavior.  This class encapsulates the functionality for Identity. | |
| [ ] | `LayerwiseShardablePipelined` | `__call__(self, inputs, *args, **kwargs)` | Represents the LayerwiseShardablePipelined configuration and behavior.  This class encapsulates the functionality for LayerwiseShardablePipelined. | |
| [x] | `Linear` | `__call__(self, inputs, w, *args, **kwargs)` | Represents the Linear configuration and behavior.  This class encapsulates the functionality for Linear. | |
| [ ] | `MLPBlock` | `__call__(self, inputs, *args, **kwargs)` | Represents the MLPBlock configuration and behavior.  This class encapsulates the functionality for MLPBlock. | |
| [ ] | `MaskedLmDataAugmenter` | `__call__(self, inputs, *args, **kwargs)` | Represents the MaskedLmDataAugmenter configuration and behavior.  This class encapsulates the functionality for MaskedLmDataAugmenter. | |
| [ ] | `MultitaskResidualAdapter` | `__call__(self, inputs, *args, **kwargs)` | Represents the MultitaskResidualAdapter configuration and behavior.  This class encapsulates the functionality for MultitaskResidualAdapter. | |
| [x] | `Repeat` | `__call__(self, inputs, *args, **kwargs)` | Represents the Repeat configuration and behavior.  This class encapsulates the functionality for Repeat. | |
| [ ] | `Sequential` | `__call__(self, inputs, *args, **kwargs)` | Represents the Sequential configuration and behavior.  This class encapsulates the functionality for Sequential. | |
| [ ] | `SpectrumAugmenter` | `__call__(self, inputs, *args, **kwargs)` | Represents the SpectrumAugmenter configuration and behavior.  This class encapsulates the functionality for SpectrumAugmenter. | |
| [ ] | `StackingOverTime` | `__call__(self, inputs, *args, **kwargs)` | Represents the StackingOverTime configuration and behavior.  This class encapsulates the functionality for StackingOverTime. | |
| [ ] | `StochasticResidual` | `__call__(self, inputs, residual, *args, **kwargs)` | Represents the StochasticResidual configuration and behavior.  This class encapsulates the functionality for StochasticResidual. | |
| [ ] | `VanillaBlock` | `__call__(self, inputs, *args, **kwargs)` | Represents the VanillaBlock configuration and behavior.  This class encapsulates the functionality for VanillaBlock. | |
| [ ] | `VitEntryLayers` | `__call__(self, inputs, *args, **kwargs)` | Represents the VitEntryLayers configuration and behavior.  This class encapsulates the functionality for VitEntryLayers. | |
| [ ] | `VitExitLayers` | `__call__(self, inputs, *args, **kwargs)` | Represents the VitExitLayers configuration and behavior.  This class encapsulates the functionality for VitExitLayers. | |

## Embeddings & Softmax

| Checkbox | Name | Function/Class Signature | Docstring | Notes |
|---|---|---|---|---|
| [x] | `Embedding` | `__call__(self, ids, w, *args, **kwargs)` | Represents the Embedding configuration and behavior.  This class encapsulates the functionality for Embedding. | |
| [ ] | `FullSoftmax` | `__call__(self, logits, *args, **kwargs)` | Represents the FullSoftmax configuration and behavior.  This class encapsulates the functionality for FullSoftmax. | |
| [ ] | `GShardSharedEmbeddingSoftmax` | `__call__(self, inputs, w, *args, **kwargs)` | Represents the GShardSharedEmbeddingSoftmax configuration and behavior.  This class encapsulates the functionality for GShardSharedEmbeddingSoftmax. | |
| [ ] | `Ngrammer` | `__call__(self, inputs, *args, **kwargs)` | Represents the Ngrammer configuration and behavior.  This class encapsulates the functionality for Ngrammer. | |
| [ ] | `PositionalEmbedding` | `__call__(self, seq_length, position, *args, **kwargs)` | Represents the PositionalEmbedding configuration and behavior.  This class encapsulates the functionality for PositionalEmbedding. | |
| [ ] | `PositionalEmbedding2D` | `__call__(self, *args, **kwargs)` | Represents the PositionalEmbedding2D configuration and behavior.  This class encapsulates the functionality for PositionalEmbedding2D. | |
| [ ] | `RandomVectorQuantizer` | `__call__(self, inputs, *args, **kwargs)` | Represents the RandomVectorQuantizer configuration and behavior.  This class encapsulates the functionality for RandomVectorQuantizer. | |
| [ ] | `SharedEmbeddingSoftmax` | `__call__(self, inputs, w, *args, **kwargs)` | Represents the SharedEmbeddingSoftmax configuration and behavior.  This class encapsulates the functionality for SharedEmbeddingSoftmax. | |
| [ ] | `TrainablePositionalEmbedding` | `__call__(self, seq_length, position, w, *args, **kwargs)` | Represents the TrainablePositionalEmbedding configuration and behavior.  This class encapsulates the functionality for TrainablePositionalEmbedding. | |
| [ ] | `VQNgrammer` | `__call__(self, inputs, *args, **kwargs)` | Represents the VQNgrammer configuration and behavior.  This class encapsulates the functionality for VQNgrammer. | |
| [ ] | `VectorQuantization` | `__call__(self, inputs, w, *args, **kwargs)` | Represents the VectorQuantization configuration and behavior.  This class encapsulates the functionality for VectorQuantization. | |
| [ ] | `VectorQuantizer` | `__call__(self, inputs, w, *args, **kwargs)` | Represents the VectorQuantizer configuration and behavior.  This class encapsulates the functionality for VectorQuantizer. | |

## Models & Architectures

| Checkbox | Name | Function/Class Signature | Docstring | Notes |
|---|---|---|---|---|
| [ ] | `BertModel` | `__call__(self, inputs, *args, **kwargs)` | Represents the BertModel configuration and behavior.  This class encapsulates the functionality for BertModel. | |
| [ ] | `BiTemperedLoss` | `__call__(self, logits, labels, *args, **kwargs)` | Represents the BiTemperedLoss configuration and behavior.  This class encapsulates the functionality for BiTemperedLoss. | |
| [ ] | `BregmanPCA` | `__call__(self, inputs, *args, **kwargs)` | Represents the BregmanPCA configuration and behavior.  This class encapsulates the functionality for BregmanPCA. | |
| [ ] | `ClassificationMLPModel` | N/A | Represents the ClassificationMLPModel configuration and behavior.  This class encapsulates the functionality for ClassificationMLPModel. | |
| [ ] | `ClassificationModel` | N/A | Represents the ClassificationModel configuration and behavior.  This class encapsulates the functionality for ClassificationModel. | |
| [ ] | `Conformer` | `__call__(self, inputs, *args, **kwargs)` | Represents the Conformer configuration and behavior.  This class encapsulates the functionality for Conformer. | |
| [ ] | `LanguageModel` | `__call__(self, inputs, *args, **kwargs)` | Represents the LanguageModel configuration and behavior.  This class encapsulates the functionality for LanguageModel. | |
| [ ] | `LanguageModelContinuousBatching` | `__call__(self, inputs, *args, **kwargs)` | Represents the LanguageModelContinuousBatching configuration and behavior.  This class encapsulates the functionality for LanguageModelContinuousBatching. | |
| [ ] | `LanguageModelDPO` | `__call__(self, inputs, *args, **kwargs)` | Represents the LanguageModelDPO configuration and behavior.  This class encapsulates the functionality for LanguageModelDPO. | |
| [ ] | `LanguageModelType` | N/A | Shim/Module/Hook required. | |
| [ ] | `ResNet` | `__call__(self, inputs, *args, **kwargs)` | Represents the ResNet configuration and behavior.  This class encapsulates the functionality for ResNet. | |
| [ ] | `ResNetBlock` | `__call__(self, inputs, *args, **kwargs)` | Represents the ResNetBlock configuration and behavior.  This class encapsulates the functionality for ResNetBlock. | |
| [ ] | `SequenceModel` | `__call__(self, inputs, *args, **kwargs)` | Represents the SequenceModel configuration and behavior.  This class encapsulates the functionality for SequenceModel. | |
| [ ] | `VanillaNet` | `__call__(self, inputs, *args, **kwargs)` | Represents the VanillaNet configuration and behavior.  This class encapsulates the functionality for VanillaNet. | |

## Normalizations

| Checkbox | Name | Function/Class Signature | Docstring | Notes |
|---|---|---|---|---|
| [ ] | `BaseNormalization` | `__call__(self, inputs, paddings, *args, **kwargs)` | Represents the BaseNormalization configuration and behavior.  This class encapsulates the functionality for BaseNormalization. | |
| [x] | `BatchNorm` | `__call__(self, inputs, paddings, beta, gamma, *args, **kwargs)` | Represents the BatchNorm configuration and behavior.  This class encapsulates the functionality for BatchNorm. | |
| [x] | `GroupNorm` | `__call__(self, inputs, paddings, gamma, beta, *args, **kwargs)` | Represents the GroupNorm configuration and behavior.  This class encapsulates the functionality for GroupNorm. | |
| [ ] | `IdentityNorm` | `__call__(self, inputs, paddings, *args, **kwargs)` | Represents the IdentityNorm configuration and behavior.  This class encapsulates the functionality for IdentityNorm. | |
| [x] | `LayerNorm` | `__call__(self, inputs, paddings, scale, bias, *args, **kwargs)` | Represents the LayerNorm configuration and behavior.  This class encapsulates the functionality for LayerNorm. | |
| [ ] | `LayerNormalizedLstmCellSimple` | `__call__(self, state0, act, padding, reset_mask, wm, b, ln_scale, *args, **kwargs)` | Represents the LayerNormalizedLstmCellSimple configuration and behavior.  This class encapsulates the functionality for LayerNormalizedLstmCellSimple. | |
| [x] | `RmsNorm` | `__call__(self, inputs, paddings, scale, *args, **kwargs)` | Represents the RmsNorm configuration and behavior.  This class encapsulates the functionality for RmsNorm. | |
| [ ] | `RmsNormNoScale` | `__call__(self, inputs, paddings, *args, **kwargs)` | Represents the RmsNormNoScale configuration and behavior.  This class encapsulates the functionality for RmsNormNoScale. | |
| [ ] | `SelfAttentionWithNormAndResidual` | `__call__(self, inputs, *args, **kwargs)` | Represents the SelfAttentionWithNormAndResidual configuration and behavior.  This class encapsulates the functionality for SelfAttentionWithNormAndResidual. | |

## RNNs & SSMs

| Checkbox | Name | Function/Class Signature | Docstring | Notes |
|---|---|---|---|---|
| [ ] | `CifgLstmCellSimple` | `__call__(self, state0, act, padding, reset_mask, wm, b, *args, **kwargs)` | Represents the CifgLstmCellSimple configuration and behavior.  This class encapsulates the functionality for CifgLstmCellSimple. | |
| [ ] | `FRnn` | `__call__(self, inputs, state0, w, *args, **kwargs)` | Represents the FRnn configuration and behavior.  This class encapsulates the functionality for FRnn. | |
| [ ] | `LstmCellSimple` | `__call__(self, state0, act, padding, reset_mask, wm, b, *args, **kwargs)` | Represents the LstmCellSimple configuration and behavior.  This class encapsulates the functionality for LstmCellSimple. | |
| [ ] | `LstmFrnn` | `__call__(self, inputs, state0, w, *args, **kwargs)` | Represents the LstmFrnn configuration and behavior.  This class encapsulates the functionality for LstmFrnn. | |
| [ ] | `SSM` | `__call__(self, inputs, *args, **kwargs)` | Represents the SSM configuration and behavior.  This class encapsulates the functionality for SSM. | |
| [ ] | `SSMGated` | `__call__(self, inputs, *args, **kwargs)` | Represents the SSMGated configuration and behavior.  This class encapsulates the functionality for SSMGated. | |
| [ ] | `StackFrnn` | `__call__(self, inputs, *args, **kwargs)` | Represents the StackFrnn configuration and behavior.  This class encapsulates the functionality for StackFrnn. | |
| [ ] | `TemporalShifting` | `__call__(self, inputs, *args, **kwargs)` | Represents the TemporalShifting configuration and behavior.  This class encapsulates the functionality for TemporalShifting. | |

## Transformers

| Checkbox | Name | Function/Class Signature | Docstring | Notes |
|---|---|---|---|---|
| [ ] | `AdaptedTransformerFeedForward` | `__call__(self, inputs, w1, w2, *args, **kwargs)` | Represents the AdaptedTransformerFeedForward configuration and behavior.  This class encapsulates the functionality for AdaptedTransformerFeedForward. | |
| [ ] | `FeedForward` | N/A | Represents the FeedForward configuration and behavior.  This class encapsulates the functionality for FeedForward. | |
| [ ] | `PipelinedTransformer` | `__call__(self, inputs, *args, **kwargs)` | Represents the PipelinedTransformer configuration and behavior.  This class encapsulates the functionality for PipelinedTransformer. | |
| [ ] | `SSMTransformer` | `__call__(self, inputs, *args, **kwargs)` | Represents the SSMTransformer configuration and behavior.  This class encapsulates the functionality for SSMTransformer. | |
| [ ] | `StackedTransformer` | `__call__(self, inputs, *args, **kwargs)` | Represents the StackedTransformer configuration and behavior.  This class encapsulates the functionality for StackedTransformer. | |
| [ ] | `StackedTransformerRepeated` | `__call__(self, inputs, *args, **kwargs)` | Represents the StackedTransformerRepeated configuration and behavior.  This class encapsulates the functionality for StackedTransformerRepeated. | |
| [ ] | `Transformer` | `__call__(self, inputs, *args, **kwargs)` | Represents the Transformer configuration and behavior.  This class encapsulates the functionality for Transformer. | |
| [ ] | `TransformerEncoderDecoder` | `__call__(self, inputs, *args, **kwargs)` | Represents the TransformerEncoderDecoder configuration and behavior.  This class encapsulates the functionality for TransformerEncoderDecoder. | |
| [ ] | `TransformerFeedForward` | `__call__(self, inputs, w1, w2, *args, **kwargs)` | Represents the TransformerFeedForward configuration and behavior.  This class encapsulates the functionality for TransformerFeedForward. | |
| [ ] | `TransformerFeedForwardMoe` | `__call__(self, inputs, w1, w2, *args, **kwargs)` | Represents the TransformerFeedForwardMoe configuration and behavior.  This class encapsulates the functionality for TransformerFeedForwardMoe. | |
| [ ] | `TransformerLm` | `__call__(self, inputs, *args, **kwargs)` | Represents the TransformerLm configuration and behavior.  This class encapsulates the functionality for TransformerLm. | |
| [ ] | `VisionTransformer` | `__call__(self, inputs, *args, **kwargs)` | Represents the VisionTransformer configuration and behavior.  This class encapsulates the functionality for VisionTransformer. | |
