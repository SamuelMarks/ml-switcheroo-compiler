from ml_switcheroo_compiler.ops import get_op


def test_a_0():
    op = get_op("A")
    assert op is not None


def test_alibi_1():
    op = get_op("ALiBi")
    assert op is not None


def test_arbitrary_2():
    op = get_op("ARBITRARY")
    assert op is not None


def test_abs_3():
    op = get_op("Abs")
    assert op is not None


def test_abs__4():
    op = get_op("Abs_")
    assert op is not None


def test_abstracttoken_5():
    op = get_op("AbstractToken")
    assert op is not None


def test_abstractvariable_6():
    op = get_op("AbstractVariable")
    assert op is not None


def test_acceleratorerror_7():
    op = get_op("AcceleratorError")
    assert op is not None


def test_accumulaten_8():
    op = get_op("AccumulateN")
    assert op is not None


def test_accuracy_9():
    op = get_op("Accuracy")
    assert op is not None


def test_accuracymode_10():
    op = get_op("AccuracyMode")
    assert op is not None


def test_acos_11():
    op = get_op("Acos")
    assert op is not None


def test_acos__12():
    op = get_op("Acos_")
    assert op is not None


def test_acosh_13():
    op = get_op("Acosh")
    assert op is not None


def test_acosh__14():
    op = get_op("Acosh_")
    assert op is not None


def test_activationlayer_15():
    op = get_op("ActivationLayer")
    assert op is not None


def test_activityregularization_16():
    op = get_op("ActivityRegularization")
    assert op is not None


def test_adadelta_17():
    op = get_op("Adadelta")
    assert op is not None


def test_adafactor_18():
    op = get_op("Adafactor")
    assert op is not None


def test_adagrad_19():
    op = get_op("Adagrad")
    assert op is not None


def test_adam_20():
    op = get_op("Adam")
    assert op is not None


def test_adamax_21():
    op = get_op("Adamax")
    assert op is not None


def test_adamw_22():
    op = get_op("Adamw")
    assert op is not None


def test_adaptedtransformerfeedforward_23():
    op = get_op("AdaptedTransformerFeedForward")
    assert op is not None


def test_adaptiveaveragepooling1d_24():
    op = get_op("AdaptiveAveragePooling1D")
    assert op is not None


def test_adaptiveaveragepooling2d_25():
    op = get_op("AdaptiveAveragePooling2D")
    assert op is not None


def test_adaptiveaveragepooling3d_26():
    op = get_op("AdaptiveAveragePooling3D")
    assert op is not None


def test_adaptiveavgpool1d_27():
    op = get_op("AdaptiveAvgPool1d")
    assert op is not None


def test_adaptiveavgpool2d_28():
    op = get_op("AdaptiveAvgPool2d")
    assert op is not None


def test_adaptiveavgpool3d_29():
    op = get_op("AdaptiveAvgPool3d")
    assert op is not None


def test_adaptivegradclipstate_30():
    op = get_op("AdaptiveGradClipState")
    assert op is not None


def test_adaptivelogsoftmaxwithloss_31():
    op = get_op("AdaptiveLogSoftmaxWithLoss")
    assert op is not None


def test_adaptivemaxpool1d_32():
    op = get_op("AdaptiveMaxPool1d")
    assert op is not None


def test_adaptivemaxpool2d_33():
    op = get_op("AdaptiveMaxPool2d")
    assert op is not None


def test_adaptivemaxpool3d_34():
    op = get_op("AdaptiveMaxPool3d")
    assert op is not None


def test_adaptivemaxpooling1d_35():
    op = get_op("AdaptiveMaxPooling1D")
    assert op is not None


def test_adaptivemaxpooling2d_36():
    op = get_op("AdaptiveMaxPooling2D")
    assert op is not None


def test_adaptivemaxpooling3d_37():
    op = get_op("AdaptiveMaxPooling3D")
    assert op is not None


def test_add_38():
    op = get_op("Add")
    assert op is not None


def test_adddecayedweightsstate_39():
    op = get_op("AddDecayedWeightsState")
    assert op is not None


def test_addlayer_40():
    op = get_op("AddLayer")
    assert op is not None


def test_addn_41():
    op = get_op("AddN")
    assert op is not None


def test_addnoisestate_42():
    op = get_op("AddNoiseState")
    assert op is not None


def test_addbmm_43():
    op = get_op("Addbmm")
    assert op is not None


def test_addcdiv_44():
    op = get_op("Addcdiv")
    assert op is not None


def test_addcmul_45():
    op = get_op("Addcmul")
    assert op is not None


def test_additiveattention_46():
    op = get_op("AdditiveAttention")
    assert op is not None


def test_addmm_47():
    op = get_op("Addmm")
    assert op is not None


def test_addmv_48():
    op = get_op("Addmv")
    assert op is not None


def test_addmv__49():
    op = get_op("Addmv_")
    assert op is not None


def test_addr_50():
    op = get_op("Addr")
    assert op is not None


def test_adjoint_51():
    op = get_op("Adjoint")
    assert op is not None


def test_affinegrid_52():
    op = get_op("AffineGrid")
    assert op is not None


def test_affinegridgenerator_53():
    op = get_op("AffineGridGenerator")
    assert op is not None


def test_aggregationtype_54():
    op = get_op("AggregationType")
    assert op is not None


def test_aliascopy_55():
    op = get_op("AliasCopy")
    assert op is not None


def test_aliasdb_56():
    op = get_op("AliasDb")
    assert op is not None


def test_aligntensors_57():
    op = get_op("AlignTensors")
    assert op is not None


def test_allgather_58():
    op = get_op("AllGather")
    assert op is not None


def test_allreduce_59():
    op = get_op("AllReduce")
    assert op is not None


def test_alltoall_60():
    op = get_op("AllToAll")
    assert op is not None


def test_alltoshardedlinear_61():
    op = get_op("AllToShardedLinear")
    assert op is not None


def test_allclose_62():
    op = get_op("Allclose")
    assert op is not None


def test_alphadropout_63():
    op = get_op("AlphaDropout")
    assert op is not None


def test_alphadropout__64():
    op = get_op("AlphaDropout_")
    assert op is not None


def test_aminmax_65():
    op = get_op("Aminmax")
    assert op is not None


def test_anyop_66():
    op = get_op("AnyOp")
    assert op is not None


def test_anytype_67():
    op = get_op("AnyType")
    assert op is not None


def test_applyalongaxis_68():
    op = get_op("ApplyAlongAxis")
    assert op is not None


def test_applycaller_69():
    op = get_op("ApplyCaller")
    assert op is not None


def test_applyiffinitestate_70():
    op = get_op("ApplyIfFiniteState")
    assert op is not None


def test_applyoveraxes_71():
    op = get_op("ApplyOverAxes")
    assert op is not None


def test_approxmaxk_72():
    op = get_op("ApproxMaxK")
    assert op is not None


def test_approxmink_73():
    op = get_op("ApproxMinK")
    assert op is not None


def test_arange_74():
    op = get_op("Arange")
    assert op is not None


def test_arccos__75():
    op = get_op("Arccos_")
    assert op is not None


def test_arccosh__76():
    op = get_op("Arccosh_")
    assert op is not None


def test_arch_77():
    op = get_op("Arch")
    assert op is not None


def test_arcsin__78():
    op = get_op("Arcsin_")
    assert op is not None


def test_arcsinh__79():
    op = get_op("Arcsinh_")
    assert op is not None


def test_arctan__80():
    op = get_op("Arctan_")
    assert op is not None


def test_arctanh__81():
    op = get_op("Arctanh_")
    assert op is not None


def test_aredeterministicalgorithmsenabled_82():
    op = get_op("AreDeterministicAlgorithmsEnabled")
    assert op is not None


def test_argsort_83():
    op = get_op("ArgSort")
    assert op is not None


def test_argument_84():
    op = get_op("Argument")
    assert op is not None


def test_argumentspec_85():
    op = get_op("ArgumentSpec")
    assert op is not None


def test_arrayat_86():
    op = get_op("ArrayAt")
    assert op is not None


def test_arrayequal_87():
    op = get_op("ArrayEqual")
    assert op is not None


def test_arrayequiv_88():
    op = get_op("ArrayEquiv")
    assert op is not None


def test_arrayiterator_89():
    op = get_op("ArrayIterator")
    assert op is not None


def test_arraylike_90():
    op = get_op("ArrayLike")
    assert op is not None


def test_arraymapping_91():
    op = get_op("ArrayMapping")
    assert op is not None


def test_arraynamespaceinfo_92():
    op = get_op("ArrayNamespaceInfo")
    assert op is not None


def test_arrayorsparse_93():
    op = get_op("ArrayOrSparse")
    assert op is not None


def test_arrayrefdef_94():
    op = get_op("ArrayRefDef")
    assert op is not None


def test_arrayrefoutput_95():
    op = get_op("ArrayRefOutput")
    assert op is not None


def test_arrayrepr_96():
    op = get_op("ArrayRepr")
    assert op is not None


def test_arraysplit_97():
    op = get_op("ArraySplit")
    assert op is not None


def test_arraystr_98():
    op = get_op("ArrayStr")
    assert op is not None


def test_asanyarray_99():
    op = get_op("AsAnyArray")
    assert op is not None


def test_asarraychkfinite_100():
    op = get_op("AsArrayChkFinite")
    assert op is not None


def test_asarrayvars_101():
    op = get_op("AsArrayVars")
    assert op is not None


def test_ascontiguousarray_102():
    op = get_op("AsContiguousArray")
    assert op is not None


def test_asfortranarray_103():
    op = get_op("AsFortranArray")
    assert op is not None


def test_ashijaxvars_104():
    op = get_op("AsHijaxVars")
    assert op is not None


def test_asimmutablevars_105():
    op = get_op("AsImmutableVars")
    assert op is not None


def test_asmatrix_106():
    op = get_op("AsMatrix")
    assert op is not None


def test_asmutablevars_107():
    op = get_op("AsMutableVars")
    assert op is not None


def test_aspytreevars_108():
    op = get_op("AsPytreeVars")
    assert op is not None


def test_asrefvars_109():
    op = get_op("AsRefVars")
    assert op is not None


def test_asstrided_110():
    op = get_op("AsStrided")
    assert op is not None


def test_asstridedcopy_111():
    op = get_op("AsStridedCopy")
    assert op is not None


def test_asstridedscatter_112():
    op = get_op("AsStridedScatter")
    assert op is not None


def test_asstrided__113():
    op = get_op("AsStrided_")
    assert op is not None


def test_asstring_114():
    op = get_op("AsString")
    assert op is not None


def test_astensor_115():
    op = get_op("AsTensor")
    assert op is not None


def test_asin_116():
    op = get_op("Asin")
    assert op is not None


def test_asin__117():
    op = get_op("Asin_")
    assert op is not None


def test_asinh_118():
    op = get_op("Asinh")
    assert op is not None


def test_asinh__119():
    op = get_op("Asinh_")
    assert op is not None


def test_assertclose_120():
    op = get_op("AssertClose")
    assert op is not None


def test_assignvariable_121():
    op = get_op("AssignVariable")
    assert op is not None


def test_associativescan_122():
    op = get_op("AssociativeScan")
    assert op is not None


def test_asynccopyimplementation_123():
    op = get_op("AsyncCopyImplementation")
    assert op is not None


def test_asynceval_124():
    op = get_op("AsyncEval")
    assert op is not None


def test_asyncmanager_125():
    op = get_op("AsyncManager")
    assert op is not None


def test_atan_126():
    op = get_op("Atan")
    assert op is not None


def test_atan2_127():
    op = get_op("Atan2")
    assert op is not None


def test_atan__128():
    op = get_op("Atan_")
    assert op is not None


def test_atanh_129():
    op = get_op("Atanh")
    assert op is not None


def test_atanh__130():
    op = get_op("Atanh_")
    assert op is not None


def test_atleast1d_131():
    op = get_op("Atleast1d")
    assert op is not None


def test_atleast2d_132():
    op = get_op("Atleast2d")
    assert op is not None


def test_atleast3d_133():
    op = get_op("Atleast3d")
    assert op is not None


def test_atom_134():
    op = get_op("Atom")
    assert op is not None


def test_attention_135():
    op = get_op("Attention")
    assert op is not None


def test_attentionlayer_136():
    op = get_op("AttentionLayer")
    assert op is not None


def test_attentionprojection_137():
    op = get_op("AttentionProjection")
    assert op is not None


def test_attrpriority_138():
    op = get_op("AttrPriority")
    assert op is not None


def test_attribute_139():
    op = get_op("Attribute")
    assert op is not None


def test_attributestatus_140():
    op = get_op("AttributeStatus")
    assert op is not None


def test_augmix_141():
    op = get_op("AugMix")
    assert op is not None


def test_autocontrast_142():
    op = get_op("AutoContrast")
    assert op is not None


def test_autocast_143():
    op = get_op("Autocast")
    assert op is not None


def test_autocastdecrementnesting_144():
    op = get_op("AutocastDecrementNesting")
    assert op is not None


def test_autocastincrementnesting_145():
    op = get_op("AutocastIncrementNesting")
    assert op is not None


def test_autodiffcheckpointtype_146():
    op = get_op("AutodiffCheckpointType")
    assert op is not None


def test_auxoutput_147():
    op = get_op("AuxOutput")
    assert op is not None


def test_auxrequest_148():
    op = get_op("AuxRequest")
    assert op is not None


def test_averagegradients_149():
    op = get_op("AverageGradients")
    assert op is not None


def test_averagelayer_150():
    op = get_op("AverageLayer")
    assert op is not None


def test_averagepooling1d_151():
    op = get_op("AveragePooling1D")
    assert op is not None


def test_averagepooling2d_152():
    op = get_op("AveragePooling2D")
    assert op is not None


def test_averagepooling3d_153():
    op = get_op("AveragePooling3D")
    assert op is not None


def test_avg_154():
    op = get_op("Avg")
    assert op is not None


def test_avgpool1d_155():
    op = get_op("AvgPool1D")
    assert op is not None


def test_avgpool2d_156():
    op = get_op("AvgPool2D")
    assert op is not None


def test_avgpool3d_157():
    op = get_op("AvgPool3D")
    assert op is not None


def test_avgpool_158():
    op = get_op("Avgpool")
    assert op is not None


def test_awaittype_159():
    op = get_op("AwaitType")
    assert op is not None


def test_axisindex_160():
    op = get_op("AxisIndex")
    assert op is not None


def test_b_161():
    op = get_op("B")
    assert op is not None


def test_bceloss_162():
    op = get_op("BCELoss")
    assert op is not None


def test_bcewithlogitsloss_163():
    op = get_op("BCEWithLogitsLoss")
    assert op is not None


def test_bcoo_164():
    op = get_op("BCOO")
    assert op is not None


def test_bcooproperties_165():
    op = get_op("BCOOProperties")
    assert op is not None


def test_bcsr_166():
    op = get_op("BCSR")
    assert op is not None


def test_bcsrproperties_167():
    op = get_op("BCSRProperties")
    assert op is not None


def test_bfloat16storage_168():
    op = get_op("BFloat16Storage")
    assert op is not None


def test_bfloat16tensor_169():
    op = get_op("BFloat16Tensor")
    assert op is not None


def test_blocks_are_contiguous_170():
    op = get_op("BLOCKS_ARE_CONTIGUOUS")
    assert op is not None


def test_block_m_171():
    op = get_op("BLOCK_M")
    assert op is not None


def test_block_m1_172():
    op = get_op("BLOCK_M1")
    assert op is not None


def test_block_m2_173():
    op = get_op("BLOCK_M2")
    assert op is not None


def test_block_n_174():
    op = get_op("BLOCK_N")
    assert op is not None


def test_block_n1_175():
    op = get_op("BLOCK_N1")
    assert op is not None


def test_block_n2_176():
    op = get_op("BLOCK_N2")
    assert op is not None


def test_block_size_177():
    op = get_op("BLOCK_SIZE")
    assert op is not None


def test_backend_178():
    op = get_op("Backend")
    assert op is not None


def test_backwardhookfunction_179():
    op = get_op("BackwardHookFunction")
    assert op is not None


def test_baddbmm_180():
    op = get_op("Baddbmm")
    assert op is not None


def test_bandpart_181():
    op = get_op("BandPart")
    assert op is not None


def test_bandedtriangularsolve_182():
    op = get_op("BandedTriangularSolve")
    assert op is not None


def test_barrierref_183():
    op = get_op("BarrierRef")
    assert op is not None


def test_barriertimeouterror_184():
    op = get_op("BarrierTimeoutError")
    assert op is not None


def test_bartlettwindow_185():
    op = get_op("BartlettWindow")
    assert op is not None


def test_baseactivation_186():
    op = get_op("BaseActivation")
    assert op is not None


def test_basenormalization_187():
    op = get_op("BaseNormalization")
    assert op is not None


def test_basepruningmethod_188():
    op = get_op("BasePruningMethod")
    assert op is not None


def test_baserepr_189():
    op = get_op("BaseRepr")
    assert op is not None


def test_batchnorm_190():
    op = get_op("BatchNorm")
    assert op is not None


def test_batchnorm1d_191():
    op = get_op("BatchNorm1d")
    assert op is not None


def test_batchnorm2d_192():
    op = get_op("BatchNorm2d")
    assert op is not None


def test_batchnorm3d_193():
    op = get_op("BatchNorm3d")
    assert op is not None


def test_batchnormbackwardelemt_194():
    op = get_op("BatchNormBackwardElemt")
    assert op is not None


def test_batchnormbackwardreduce_195():
    op = get_op("BatchNormBackwardReduce")
    assert op is not None


def test_batchnormelemt_196():
    op = get_op("BatchNormElemt")
    assert op is not None


def test_batchnormgatherstats_197():
    op = get_op("BatchNormGatherStats")
    assert op is not None


def test_batchnormgatherstatswithcounts_198():
    op = get_op("BatchNormGatherStatsWithCounts")
    assert op is not None


def test_batchnormstats_199():
    op = get_op("BatchNormStats")
    assert op is not None


def test_batchnormupdatestats_200():
    op = get_op("BatchNormUpdateStats")
    assert op is not None


def test_batchnormalization_201():
    op = get_op("BatchNormalization")
    assert op is not None


def test_batchstat_202():
    op = get_op("BatchStat")
    assert op is not None


def test_batcheddotgeneralproperties_203():
    op = get_op("BatchedDotGeneralProperties")
    assert op is not None


def test_benchmarkconfig_204():
    op = get_op("BenchmarkConfig")
    assert op is not None


def test_benchmarkexecutionstats_205():
    op = get_op("BenchmarkExecutionStats")
    assert op is not None


def test_bernoulli_206():
    op = get_op("Bernoulli")
    assert op is not None


def test_bertmodel_207():
    op = get_op("BertModel")
    assert op is not None


def test_besseli0_208():
    op = get_op("BesselI0")
    assert op is not None


def test_besseli0e_209():
    op = get_op("BesselI0e")
    assert op is not None


def test_besseli1_210():
    op = get_op("BesselI1")
    assert op is not None


def test_besseli1e_211():
    op = get_op("BesselI1e")
    assert op is not None


def test_besseljn_212():
    op = get_op("BesselJn")
    assert op is not None


def test_beta_213():
    op = get_op("Beta")
    assert op is not None


def test_betacdf_214():
    op = get_op("BetaCdf")
    assert op is not None


def test_betapdf_215():
    op = get_op("BetaPdf")
    assert op is not None


def test_betainc_216():
    op = get_op("Betainc")
    assert op is not None


def test_bitemperedloss_217():
    op = get_op("BiTemperedLoss")
    assert op is not None


def test_binarycrossentropywithlogits_218():
    op = get_op("BinaryCrossEntropyWithLogits")
    assert op is not None


def test_binarycrossentropy_219():
    op = get_op("BinaryCrossentropy")
    assert op is not None


def test_binaryfocalcrossentropy_220():
    op = get_op("BinaryFocalCrossentropy")
    assert op is not None


def test_binaryrepr_221():
    op = get_op("BinaryRepr")
    assert op is not None


def test_bincount_222():
    op = get_op("Bincount")
    assert op is not None


def test_binomcdf_223():
    op = get_op("BinomCdf")
    assert op is not None


def test_binompmf_224():
    op = get_op("BinomPmf")
    assert op is not None


def test_binomial_225():
    op = get_op("Binomial")
    assert op is not None


def test_bit_226():
    op = get_op("Bit")
    assert op is not None


def test_bitcast_227():
    op = get_op("BitCast")
    assert op is not None


def test_bits_228():
    op = get_op("Bits")
    assert op is not None


def test_bits16_229():
    op = get_op("Bits16")
    assert op is not None


def test_bits1x8_230():
    op = get_op("Bits1x8")
    assert op is not None


def test_bits2x4_231():
    op = get_op("Bits2x4")
    assert op is not None


def test_bits4x2_232():
    op = get_op("Bits4x2")
    assert op is not None


def test_bits8_233():
    op = get_op("Bits8")
    assert op is not None


def test_bitwiseand_234():
    op = get_op("BitwiseAnd")
    assert op is not None


def test_bitwisecount_235():
    op = get_op("BitwiseCount")
    assert op is not None


def test_bitwiseinvert_236():
    op = get_op("BitwiseInvert")
    assert op is not None


def test_bitwiseleftshift_237():
    op = get_op("BitwiseLeftShift")
    assert op is not None


def test_bitwisenot_238():
    op = get_op("BitwiseNot")
    assert op is not None


def test_bitwiseor_239():
    op = get_op("BitwiseOr")
    assert op is not None


def test_bitwiserightshift_240():
    op = get_op("BitwiseRightShift")
    assert op is not None


def test_bitwisexor_241():
    op = get_op("BitwiseXor")
    assert op is not None


def test_blackmanwindow_242():
    op = get_op("BlackmanWindow")
    assert op is not None


def test_blockdiag_243():
    op = get_op("BlockDiag")
    assert op is not None


def test_blockmask_244():
    op = get_op("BlockMask")
    assert op is not None


def test_blockmaskedmm_245():
    op = get_op("BlockMaskedMM")
    assert op is not None


def test_blocksizes_246():
    op = get_op("BlockSizes")
    assert op is not None


def test_blockwise128x128_247():
    op = get_op("BlockWise128x128")
    assert op is not None


def test_blockwise1x128_248():
    op = get_op("BlockWise1x128")
    assert op is not None


def test_blockwise1x16_249():
    op = get_op("BlockWise1x16")
    assert op is not None


def test_blockwise1x32_250():
    op = get_op("BlockWise1x32")
    assert op is not None


def test_bmm_251():
    op = get_op("Bmm")
    assert op is not None


def test_bool_252():
    op = get_op("Bool")
    assert op is not None


def test_boolformat_253():
    op = get_op("BoolFormat")
    assert op is not None


def test_boolscalar_254():
    op = get_op("BoolScalar")
    assert op is not None


def test_boolstorage_255():
    op = get_op("BoolStorage")
    assert op is not None


def test_booltensor_256():
    op = get_op("BoolTensor")
    assert op is not None


def test_booltype_257():
    op = get_op("BoolType")
    assert op is not None


def test_boolvaluesapi_258():
    op = get_op("BoolValuesApi")
    assert op is not None


def test_booleanmask_259():
    op = get_op("BooleanMask")
    assert op is not None


def test_bregmanpca_260():
    op = get_op("BregmanPCA")
    assert op is not None


def test_broadcast_261():
    op = get_op("Broadcast")
    assert op is not None


def test_broadcastarrays_262():
    op = get_op("BroadcastArrays")
    assert op is not None


def test_broadcastindim_263():
    op = get_op("BroadcastInDim")
    assert op is not None


def test_broadcastlike_264():
    op = get_op("BroadcastLike")
    assert op is not None


def test_broadcastshapes_265():
    op = get_op("BroadcastShapes")
    assert op is not None


def test_broadcasttensors_266():
    op = get_op("BroadcastTensors")
    assert op is not None


def test_broadcastto_267():
    op = get_op("BroadcastTo")
    assert op is not None


def test_broadcasttorank_268():
    op = get_op("BroadcastToRank")
    assert op is not None


def test_broadcasted_269():
    op = get_op("Broadcasted")
    assert op is not None


def test_broadcastediota_270():
    op = get_op("BroadcastedIota")
    assert op is not None


def test_bucketize_271():
    op = get_op("Bucketize")
    assert op is not None


def test_buffer_272():
    op = get_op("Buffer")
    assert op is not None


def test_bufferdict_273():
    op = get_op("BufferDict")
    assert op is not None


def test_buildmatrix_274():
    op = get_op("BuildMatrix")
    assert op is not None


def test_busdaycalendar_275():
    op = get_op("BusDayCalendar")
    assert op is not None


def test_busdaycount_276():
    op = get_op("BusDayCount")
    assert op is not None


def test_busdayoffset_277():
    op = get_op("BusDayOffset")
    assert op is not None


def test_bwdfn_278():
    op = get_op("BwdFn")
    assert op is not None


def test_bytestorage_279():
    op = get_op("ByteStorage")
    assert op is not None


def test_bytetensor_280():
    op = get_op("ByteTensor")
    assert op is not None


def test_bytetype_281():
    op = get_op("ByteType")
    assert op is not None


def test_bytestype_282():
    op = get_op("BytesType")
    assert op is not None


def test_c_283():
    op = get_op("C")
    assert op is not None


def test_callsite_regex_284():
    op = get_op("CALLSITE_REGEX")
    assert op is not None


def test_cclass_285():
    op = get_op("CClass")
    assert op is not None


def test_ccolindicescopy_286():
    op = get_op("CColIndicesCopy")
    assert op is not None


def test_cmem_287():
    op = get_op("CMEM")
    assert op is not None


def test_cmpf_impls_288():
    op = get_op("CMPF_IMPLS")
    assert op is not None


def test_cmpi_impls_289():
    op = get_op("CMPI_IMPLS")
    assert op is not None


def test_collective_attr_290():
    op = get_op("COLLECTIVE_ATTR")
    assert op is not None


def test_collective_metadata_size_291():
    op = get_op("COLLECTIVE_METADATA_SIZE")
    assert op is not None


def test_col_layout_292():
    op = get_op("COL_LAYOUT")
    assert op is not None


def test_coo_293():
    op = get_op("COO")
    assert op is not None


def test_cooinfo_294():
    op = get_op("COOInfo")
    assert op is not None


def test_core_parallel_295():
    op = get_op("CORE_PARALLEL")
    assert op is not None


def test_cp_async_296():
    op = get_op("CP_ASYNC")
    assert op is not None


def test_csc_297():
    op = get_op("CSC")
    assert op is not None


def test_csr_298():
    op = get_op("CSR")
    assert op is not None


def test_cusparse_data_dtypes_299():
    op = get_op("CUSPARSE_DATA_DTYPES")
    assert op is not None


def test_cusparse_index_dtypes_300():
    op = get_op("CUSPARSE_INDEX_DTYPES")
    assert op is not None


def test_c__301():
    op = get_op("C_")
    assert op is not None


def test_cache_302():
    op = get_op("Cache")
    assert op is not None


def test_cachedpartial_303():
    op = get_op("CachedPartial")
    assert op is not None


def test_callinfo_304():
    op = get_op("CallInfo")
    assert op is not None


def test_callstack_305():
    op = get_op("CallStack")
    assert op is not None


def test_calltfeffect_306():
    op = get_op("CallTfEffect")
    assert op is not None


def test_calltforderedeffect_307():
    op = get_op("CallTfOrderedEffect")
    assert op is not None


def test_callableproxy_308():
    op = get_op("CallableProxy")
    assert op is not None


def test_cancast_309():
    op = get_op("CanCast")
    assert op is not None


def test_capsule_310():
    op = get_op("Capsule")
    assert op is not None


def test_carry_311():
    op = get_op("Carry")
    assert op is not None


def test_cartesianprod_312():
    op = get_op("CartesianProd")
    assert op is not None


def test_cast_313():
    op = get_op("Cast")
    assert op is not None


def test_castbfloat16_314():
    op = get_op("CastBFloat16")
    assert op is not None


def test_castbool_315():
    op = get_op("CastBool")
    assert op is not None


def test_castbyte_316():
    op = get_op("CastByte")
    assert op is not None


def test_castchar_317():
    op = get_op("CastChar")
    assert op is not None


def test_castdouble_318():
    op = get_op("CastDouble")
    assert op is not None


def test_castfloat_319():
    op = get_op("CastFloat")
    assert op is not None


def test_casthalf_320():
    op = get_op("CastHalf")
    assert op is not None


def test_castint_321():
    op = get_op("CastInt")
    assert op is not None


def test_castint64_322():
    op = get_op("CastInt64")
    assert op is not None


def test_castlong_323():
    op = get_op("CastLong")
    assert op is not None


def test_castshort_324():
    op = get_op("CastShort")
    assert op is not None


def test_categoricalcrossentropy_325():
    op = get_op("CategoricalCrossentropy")
    assert op is not None


def test_categoricalfocalcrossentropy_326():
    op = get_op("CategoricalFocalCrossentropy")
    assert op is not None


def test_categoricalgeneralizedcrossentropy_327():
    op = get_op("CategoricalGeneralizedCrossEntropy")
    assert op is not None


def test_categoricalhinge_328():
    op = get_op("CategoricalHinge")
    assert op is not None


def test_categoryencoding_329():
    op = get_op("CategoryEncoding")
    assert op is not None


def test_cauchy_330():
    op = get_op("Cauchy")
    assert op is not None


def test_causalbias_331():
    op = get_op("CausalBias")
    assert op is not None


def test_causaldepthwiseconv1d_332():
    op = get_op("CausalDepthwiseConv1D")
    assert op is not None


def test_causalmask_333():
    op = get_op("CausalMask")
    assert op is not None


def test_causalsegmentmask_334():
    op = get_op("CausalSegmentMask")
    assert op is not None


def test_causalvariant_335():
    op = get_op("CausalVariant")
    assert op is not None


def test_cdist_336():
    op = get_op("Cdist")
    assert op is not None


def test_ceilinplace_337():
    op = get_op("CeilInplace")
    assert op is not None


def test_celuinplace_338():
    op = get_op("CeluInplace")
    assert op is not None


def test_centercrop_339():
    op = get_op("CenterCrop")
    assert op is not None


def test_chainmatmul_340():
    op = get_op("ChainMatmul")
    assert op is not None


def test_channelshuffle_341():
    op = get_op("ChannelShuffle")
    assert op is not None


def test_channelslast_342():
    op = get_op("ChannelsLast")
    assert op is not None


def test_channelslast3d_343():
    op = get_op("ChannelsLast3d")
    assert op is not None


def test_charstorage_344():
    op = get_op("CharStorage")
    assert op is not None


def test_chartensor_345():
    op = get_op("CharTensor")
    assert op is not None


def test_charactertype_346():
    op = get_op("CharacterType")
    assert op is not None


def test_checkpytree_347():
    op = get_op("CheckPytree")
    assert op is not None


def test_checkifyfn_348():
    op = get_op("CheckifyFn")
    assert op is not None


def test_checkpoint_349():
    op = get_op("Checkpoint")
    assert op is not None


def test_chisquare_350():
    op = get_op("Chisquare")
    assert op is not None


def test_cholesky_351():
    op = get_op("Cholesky")
    assert op is not None


def test_choleskyinverse_352():
    op = get_op("CholeskyInverse")
    assert op is not None


def test_choleskysolve_353():
    op = get_op("CholeskySolve")
    assert op is not None


def test_chooseqparamsoptimized_354():
    op = get_op("ChooseQParamsOptimized")
    assert op is not None


def test_chunk_355():
    op = get_op("Chunk")
    assert op is not None


def test_chunked_356():
    op = get_op("Chunked")
    assert op is not None


def test_chunkedcausalmask_357():
    op = get_op("ChunkedCausalMask")
    assert op is not None


def test_cifglstmcellsimple_358():
    op = get_op("CifgLstmCellSimple")
    assert op is not None


def test_circleloss_359():
    op = get_op("CircleLoss")
    assert op is not None


def test_circularpad1d_360():
    op = get_op("CircularPad1d")
    assert op is not None


def test_circularpad2d_361():
    op = get_op("CircularPad2d")
    assert op is not None


def test_circularpad3d_362():
    op = get_op("CircularPad3d")
    assert op is not None


def test_clampinplace_363():
    op = get_op("ClampInplace")
    assert op is not None


def test_clampmax_364():
    op = get_op("ClampMax")
    assert op is not None


def test_clampmaxinplace_365():
    op = get_op("ClampMaxInplace")
    assert op is not None


def test_clampmin_366():
    op = get_op("ClampMin")
    assert op is not None


def test_clampmin__367():
    op = get_op("ClampMin_")
    assert op is not None


def test_classproperty_368():
    op = get_op("ClassProperty")
    assert op is not None


def test_classtype_369():
    op = get_op("ClassType")
    assert op is not None


def test_classificationmlpmodel_370():
    op = get_op("ClassificationMLPModel")
    assert op is not None


def test_classificationmodel_371():
    op = get_op("ClassificationModel")
    assert op is not None


def test_clearautocastcache_372():
    op = get_op("ClearAutocastCache")
    assert op is not None


def test_clearcache_373():
    op = get_op("ClearCache")
    assert op is not None


def test_clip_374():
    op = get_op("Clip")
    assert op is not None


def test_clipbyglobalnormstate_375():
    op = get_op("ClipByGlobalNormState")
    assert op is not None


def test_clipgradnorm_376():
    op = get_op("ClipGradNorm")
    assert op is not None


def test_clipstate_377():
    op = get_op("ClipState")
    assert op is not None


def test_clip__378():
    op = get_op("Clip_")
    assert op is not None


def test_clone_379():
    op = get_op("Clone")
    assert op is not None


def test_clusterbarrier_380():
    op = get_op("ClusterBarrier")
    assert op is not None


def test_code_381():
    op = get_op("Code")
    assert op is not None


def test_colindicescopy_382():
    op = get_op("ColIndicesCopy")
    assert op is not None


def test_collapse_383():
    op = get_op("Collapse")
    assert op is not None


def test_collapseleadingindicestransform_384():
    op = get_op("CollapseLeadingIndicesTransform")
    assert op is not None


def test_collectivebarrierref_385():
    op = get_op("CollectiveBarrierRef")
    assert op is not None


def test_color_386():
    op = get_op("Color")
    assert op is not None


def test_columnconcatenation_387():
    op = get_op("ColumnConcatenation")
    assert op is not None


def test_columnstack_388():
    op = get_op("ColumnStack")
    assert op is not None


def test_combinations_389():
    op = get_op("Combinations")
    assert op is not None


def test_combinemasks_390():
    op = get_op("CombineMasks")
    assert op is not None


def test_commadecimalpointlocale_391():
    op = get_op("CommaDecimalPointLocale")
    assert op is not None


def test_commontype_392():
    op = get_op("CommonType")
    assert op is not None


def test_compilationunit_393():
    op = get_op("CompilationUnit")
    assert op is not None


def test_compilefn_394():
    op = get_op("CompileFn")
    assert op is not None


def test_compiled_395():
    op = get_op("Compiled")
    assert op is not None


def test_compiledwithcxx11abi_396():
    op = get_op("CompiledWithCxx11Abi")
    assert op is not None


def test_completeargumentspec_397():
    op = get_op("CompleteArgumentSpec")
    assert op is not None


def test_complex32_398():
    op = get_op("Complex32")
    assert op is not None


def test_complexdouble_399():
    op = get_op("ComplexDouble")
    assert op is not None


def test_complexdoublestorage_400():
    op = get_op("ComplexDoubleStorage")
    assert op is not None


def test_complexfloatstorage_401():
    op = get_op("ComplexFloatStorage")
    assert op is not None


def test_complexfloatingformat_402():
    op = get_op("ComplexFloatingFormat")
    assert op is not None


def test_complexhalf_403():
    op = get_op("ComplexHalf")
    assert op is not None


def test_complexlongdouble_404():
    op = get_op("ComplexLongDouble")
    assert op is not None


def test_complexop_405():
    op = get_op("ComplexOp")
    assert op is not None


def test_complextype_406():
    op = get_op("ComplexType")
    assert op is not None


def test_complexwarning_407():
    op = get_op("ComplexWarning")
    assert op is not None


def test_computeattentionmasksforextendstep_408():
    op = get_op("ComputeAttentionMasksForExtendStep")
    assert op is not None


def test_computeattentionmasksforfprop_409():
    op = get_op("ComputeAttentionMasksForFprop")
    assert op is not None


def test_computecv_410():
    op = get_op("ComputeCv")
    assert op is not None


def test_computemoments_411():
    op = get_op("ComputeMoments")
    assert op is not None


def test_concatenatelayer_412():
    op = get_op("ConcatenateLayer")
    assert op is not None


def test_concretemoduletype_413():
    op = get_op("ConcreteModuleType")
    assert op is not None


def test_concretemoduletypebuilder_414():
    op = get_op("ConcreteModuleTypeBuilder")
    assert op is not None


def test_cond_415():
    op = get_op("Cond")
    assert op is not None


def test_conditionallymaskstate_416():
    op = get_op("ConditionallyMaskState")
    assert op is not None


def test_conditionallytransformstate_417():
    op = get_op("ConditionallyTransformState")
    assert op is not None


def test_config_418():
    op = get_op("Config")
    assert op is not None


def test_conformer_419():
    op = get_op("Conformer")
    assert op is not None


def test_confusionmatrix_420():
    op = get_op("ConfusionMatrix")
    assert op is not None


def test_conjphysical_421():
    op = get_op("ConjPhysical")
    assert op is not None


def test_conjphysical__422():
    op = get_op("ConjPhysical_")
    assert op is not None


def test_constant_423():
    op = get_op("Constant")
    assert op is not None


def test_constantpad1d_424():
    op = get_op("ConstantPad1d")
    assert op is not None


def test_constantpad2d_425():
    op = get_op("ConstantPad2d")
    assert op is not None


def test_constantpad3d_426():
    op = get_op("ConstantPad3d")
    assert op is not None


def test_constantpadnd_427():
    op = get_op("ConstantPadNd")
    assert op is not None


def test_constraint_428():
    op = get_op("Constraint")
    assert op is not None


def test_constraintsystem_429():
    op = get_op("ConstraintSystem")
    assert op is not None


def test_constraintsystemderivationrule_430():
    op = get_op("ConstraintSystemDerivationRule")
    assert op is not None


def test_constraintsystemderivationruleresult_431():
    op = get_op("ConstraintSystemDerivationRuleResult")
    assert op is not None


def test_constructfloat16_432():
    op = get_op("ConstructFloat16")
    assert op is not None


def test_constructfloat32_433():
    op = get_op("ConstructFloat32")
    assert op is not None


def test_constructfloat64_434():
    op = get_op("ConstructFloat64")
    assert op is not None


def test_container_435():
    op = get_op("Container")
    assert op is not None


def test_contiguous_436():
    op = get_op("Contiguous")
    assert op is not None


def test_contiguousformat_437():
    op = get_op("ContiguousFormat")
    assert op is not None


def test_controldependency_438():
    op = get_op("ControlDependency")
    assert op is not None


def test_controlvariate_439():
    op = get_op("ControlVariate")
    assert op is not None


def test_conv1d_440():
    op = get_op("Conv1D")
    assert op is not None


def test_conv1dtranspose_441():
    op = get_op("Conv1DTranspose")
    assert op is not None


def test_conv2d_442():
    op = get_op("Conv2D")
    assert op is not None


def test_conv2dtranspose_443():
    op = get_op("Conv2DTranspose")
    assert op is not None


def test_conv3d_444():
    op = get_op("Conv3D")
    assert op is not None


def test_conv3dtranspose_445():
    op = get_op("Conv3DTranspose")
    assert op is not None


def test_convbnact_446():
    op = get_op("ConvBNAct")
    assert op is not None


def test_convbnactwithpadding_447():
    op = get_op("ConvBNActWithPadding")
    assert op is not None


def test_convdimensionnumbers_448():
    op = get_op("ConvDimensionNumbers")
    assert op is not None


def test_convgeneraldilated_449():
    op = get_op("ConvGeneralDilated")
    assert op is not None


def test_convlstm1d_450():
    op = get_op("ConvLSTM1d")
    assert op is not None


def test_convlstm2d_451():
    op = get_op("ConvLSTM2d")
    assert op is not None


def test_convlstm3d_452():
    op = get_op("ConvLSTM3d")
    assert op is not None


def test_convnet_453():
    op = get_op("ConvNet")
    assert op is not None


def test_convt_454():
    op = get_op("ConvT")
    assert op is not None


def test_convtranspose_455():
    op = get_op("ConvTranspose")
    assert op is not None


def test_convtranspose1d_456():
    op = get_op("ConvTranspose1D")
    assert op is not None


def test_convtranspose2d_457():
    op = get_op("ConvTranspose2D")
    assert op is not None


def test_convtranspose3d_458():
    op = get_op("ConvTranspose3D")
    assert op is not None


def test_convtransposeshapetuple_459():
    op = get_op("ConvTransposeShapeTuple")
    assert op is not None


def test_convertpaddingstomask_460():
    op = get_op("ConvertPaddingsToMask")
    assert op is not None


def test_convolution_461():
    op = get_op("Convolution")
    assert op is not None


def test_convolution1d_462():
    op = get_op("Convolution1D")
    assert op is not None


def test_convolution1dtranspose_463():
    op = get_op("Convolution1DTranspose")
    assert op is not None


def test_convolution2d_464():
    op = get_op("Convolution2D")
    assert op is not None


def test_convolution2dtranspose_465():
    op = get_op("Convolution2DTranspose")
    assert op is not None


def test_convolution3d_466():
    op = get_op("Convolution3d")
    assert op is not None


def test_convolution3dtranspose_467():
    op = get_op("Convolution3dTranspose")
    assert op is not None


def test_convolve_468():
    op = get_op("Convolve")
    assert op is not None


def test_convolve2d_469():
    op = get_op("Convolve2d")
    assert op is not None


def test_cos_470():
    op = get_op("Cos")
    assert op is not None


def test_cos__471():
    op = get_op("Cos_")
    assert op is not None


def test_cosh__472():
    op = get_op("Cosh_")
    assert op is not None


def test_cosineannealinglr_473():
    op = get_op("CosineAnnealingLR")
    assert op is not None


def test_cosineembeddingloss_474():
    op = get_op("CosineEmbeddingLoss")
    assert op is not None


def test_cosinesimilarity_475():
    op = get_op("CosineSimilarity")
    assert op is not None


def test_countnonzero_476():
    op = get_op("CountNonzero")
    assert op is not None


def test_createtoken_477():
    op = get_op("CreateToken")
    assert op is not None


def test_cropping1d_478():
    op = get_op("Cropping1d")
    assert op is not None


def test_cropping2d_479():
    op = get_op("Cropping2d")
    assert op is not None


def test_cropping3d_480():
    op = get_op("Cropping3d")
    assert op is not None


def test_cross_481():
    op = get_op("Cross")
    assert op is not None


def test_crossentropyloss_482():
    op = get_op("CrossEntropyLoss")
    assert op is not None


def test_crossmaplrn2d_483():
    op = get_op("CrossMapLRN2d")
    assert op is not None


def test_crowindicescopy_484():
    op = get_op("CrowIndicesCopy")
    assert op is not None


def test_cusparseefficiencywarning_485():
    op = get_op("CuSparseEfficiencyWarning")
    assert op is not None


def test_cuberoot_486():
    op = get_op("CubeRoot")
    assert op is not None


def test_cubedrelu_487():
    op = get_op("CubedReLU")
    assert op is not None


def test_cudaavailable_488():
    op = get_op("CudaAvailable")
    assert op is not None


def test_cudnnaffinegridgenerator_489():
    op = get_op("CudnnAffineGridGenerator")
    assert op is not None


def test_cudnnbatchnorm_490():
    op = get_op("CudnnBatchNorm")
    assert op is not None


def test_cudnnconvolution_491():
    op = get_op("CudnnConvolution")
    assert op is not None


def test_cudnnconvolutionaddrelu_492():
    op = get_op("CudnnConvolutionAddRelu")
    assert op is not None


def test_cudnnconvolutionrelu_493():
    op = get_op("CudnnConvolutionRelu")
    assert op is not None


def test_cudnnconvolutiontranspose_494():
    op = get_op("CudnnConvolutionTranspose")
    assert op is not None


def test_cudnngridsampler_495():
    op = get_op("CudnnGridSampler")
    assert op is not None


def test_cudnnisacceptable_496():
    op = get_op("CudnnIsAcceptable")
    assert op is not None


def test_cumlogsumexp_497():
    op = get_op("Cumlogsumexp")
    assert op is not None


def test_cummax_498():
    op = get_op("Cummax")
    assert op is not None


def test_cummin_499():
    op = get_op("Cummin")
    assert op is not None


def test_cumprod_500():
    op = get_op("Cumprod")
    assert op is not None


def test_cumsum_501():
    op = get_op("Cumsum")
    assert op is not None


def test_cumulativelogsumexp_502():
    op = get_op("CumulativeLogsumexp")
    assert op is not None


def test_cumulativeprod_503():
    op = get_op("CumulativeProd")
    assert op is not None


def test_cumulativereduction_504():
    op = get_op("CumulativeReduction")
    assert op is not None


def test_cumulativesum_505():
    op = get_op("CumulativeSum")
    assert op is not None


def test_cumulativetrapezoid_506():
    op = get_op("CumulativeTrapezoid")
    assert op is not None


def test_cupti_507():
    op = get_op("Cupti")
    assert op is not None


def test_currentupdatecontext_508():
    op = get_op("CurrentUpdateContext")
    assert op is not None


def test_customautograd_509():
    op = get_op("CustomAutograd")
    assert op is not None


def test_customfrommask_510():
    op = get_op("CustomFromMask")
    assert op is not None


def test_customlinearsolve_511():
    op = get_op("CustomLinearSolve")
    assert op is not None


def test_customvjp_512():
    op = get_op("CustomVjp")
    assert op is not None


def test_customvjpfnwrapper_513():
    op = get_op("CustomVjpFnWrapper")
    assert op is not None


def test_cutmix_514():
    op = get_op("CutMix")
    assert op is not None


def test_cvexpectedvalue_515():
    op = get_op("CvExpectedValue")
    assert op is not None


def test_cvstate_516():
    op = get_op("CvState")
    assert op is not None


def test_default_mask_value_517():
    op = get_op("DEFAULT_MASK_VALUE")
    assert op is not None


def test_default_native_serialization_518():
    op = get_op("DEFAULT_NATIVE_SERIALIZATION")
    assert op is not None


def test_default_num_tracing_attempts_519():
    op = get_op("DEFAULT_NUM_TRACING_ATTEMPTS")
    assert op is not None


def test_device_id_attr_520():
    op = get_op("DEVICE_ID_ATTR")
    assert op is not None


def test_down_521():
    op = get_op("DOWN")
    assert op is not None


def test_dynamic_522():
    op = get_op("DYNAMIC")
    assert op is not None


def test_dynamic32_523():
    op = get_op("DYNAMIC32")
    assert op is not None


def test_dataelem_524():
    op = get_op("DataElem")
    assert op is not None


def test_dataloader_525():
    op = get_op("DataLoader")
    assert op is not None


def test_dataparallel_526():
    op = get_op("DataParallel")
    assert op is not None


def test_datetime64_527():
    op = get_op("Datetime64")
    assert op is not None


def test_datetimeasstring_528():
    op = get_op("DatetimeAsString")
    assert op is not None


def test_datetimedata_529():
    op = get_op("DatetimeData")
    assert op is not None


def test_datetimeformat_530():
    op = get_op("DatetimeFormat")
    assert op is not None


def test_dct_531():
    op = get_op("Dct")
    assert op is not None


def test_debuginfs_532():
    op = get_op("DebugInfs")
    assert op is not None


def test_debugnans_533():
    op = get_op("DebugNans")
    assert op is not None


def test_deepcopymemotable_534():
    op = get_op("DeepCopyMemoTable")
    assert op is not None


def test_defmodel_535():
    op = get_op("DefModel")
    assert op is not None


def test_defaultgenerator_536():
    op = get_op("DefaultGenerator")
    assert op is not None


def test_deg2rad__537():
    op = get_op("Deg2Rad_")
    assert op is not None


def test_delayedaccessor_538():
    op = get_op("DelayedAccessor")
    assert op is not None


def test_dense_539():
    op = get_op("Dense")
    assert op is not None


def test_depthwiseconv1d_540():
    op = get_op("DepthwiseConv1d")
    assert op is not None


def test_depthwiseconv2d_541():
    op = get_op("DepthwiseConv2D")
    assert op is not None


def test_dequantize_542():
    op = get_op("Dequantize")
    assert op is not None


def test_derivationcontext_543():
    op = get_op("DerivationContext")
    assert op is not None


def test_deserializationstoragecontext_544():
    op = get_op("DeserializationStorageContext")
    assert op is not None


def test_deserializelayer_545():
    op = get_op("DeserializeLayer")
    assert op is not None


def test_deserializeoptimizer_546():
    op = get_op("DeserializeOptimizer")
    assert op is not None


def test_det_547():
    op = get_op("Det")
    assert op is not None


def test_detach_548():
    op = get_op("Detach")
    assert op is not None


def test_detachcopy_549():
    op = get_op("DetachCopy")
    assert op is not None


def test_detach__550():
    op = get_op("Detach_")
    assert op is not None


def test_devicecpu_551():
    op = get_op("DeviceCpu")
    assert op is not None


def test_devicelist_552():
    op = get_op("DeviceList")
    assert op is not None


def test_deviceobjtype_553():
    op = get_op("DeviceObjType")
    assert op is not None


def test_deviceputreplicated_554():
    op = get_op("DevicePutReplicated")
    assert op is not None


def test_deviceputsharded_555():
    op = get_op("DevicePutSharded")
    assert op is not None


def test_devicetype_556():
    op = get_op("DeviceType")
    assert op is not None


def test_diagembed_557():
    op = get_op("DiagEmbed")
    assert op is not None


def test_diagindices_558():
    op = get_op("DiagIndices")
    assert op is not None


def test_diagindicesfrom_559():
    op = get_op("DiagIndicesFrom")
    assert op is not None


def test_diagonal_560():
    op = get_op("Diagonal")
    assert op is not None


def test_diagonalcopy_561():
    op = get_op("DiagonalCopy")
    assert op is not None


def test_diagonalscatter_562():
    op = get_op("DiagonalScatter")
    assert op is not None


def test_dialectbarrierref_563():
    op = get_op("DialectBarrierRef")
    assert op is not None


def test_dict_564():
    op = get_op("Dict")
    assert op is not None


def test_dicttype_565():
    op = get_op("DictType")
    assert op is not None


def test_diffstate_566():
    op = get_op("DiffState")
    assert op is not None


def test_digamma_567():
    op = get_op("Digamma")
    assert op is not None


def test_dirichlet_568():
    op = get_op("Dirichlet")
    assert op is not None


def test_disablecompile_569():
    op = get_op("DisableCompile")
    assert op is not None


def test_disabletorchfunction_570():
    op = get_op("DisableTorchFunction")
    assert op is not None


def test_disabletorchfunctionsubclass_571():
    op = get_op("DisableTorchFunctionSubclass")
    assert op is not None


def test_disabledsafetycheck_572():
    op = get_op("DisabledSafetyCheck")
    assert op is not None


def test_discretization_573():
    op = get_op("Discretization")
    assert op is not None


def test_dispatchkey_574():
    op = get_op("DispatchKey")
    assert op is not None


def test_dispatchkeyset_575():
    op = get_op("DispatchKeySet")
    assert op is not None


def test_dist_576():
    op = get_op("Dist")
    assert op is not None


def test_distributeddataparallel_577():
    op = get_op("DistributedDataParallel")
    assert op is not None


def test_distributeddataparallelcpu_578():
    op = get_op("DistributedDataParallelCPU")
    assert op is not None


def test_distributedsum_579():
    op = get_op("DistributedSum")
    assert op is not None


def test_div_580():
    op = get_op("Div")
    assert op is not None


def test_dividenonan_581():
    op = get_op("DivideNoNan")
    assert op is not None


def test_divides_582():
    op = get_op("Divides")
    assert op is not None


def test_dot_583():
    op = get_op("Dot")
    assert op is not None


def test_dotalgorithm_584():
    op = get_op("DotAlgorithm")
    assert op is not None


def test_dotalgorithmpreset_585():
    op = get_op("DotAlgorithmPreset")
    assert op is not None


def test_dotgeneral_586():
    op = get_op("DotGeneral")
    assert op is not None


def test_dotlist_587():
    op = get_op("DotList")
    assert op is not None


def test_dotproductattention_588():
    op = get_op("DotProductAttention")
    assert op is not None


def test_dotproductattentionwithcontext_589():
    op = get_op("DotProductAttentionWithContext")
    assert op is not None


def test_dotproductattentionwithcontextxl_590():
    op = get_op("DotProductAttentionWithContextXL")
    assert op is not None


def test_dotproductattentionxl_591():
    op = get_op("DotProductAttentionXL")
    assert op is not None


def test_doublesidedmaxwell_592():
    op = get_op("DoubleSidedMaxwell")
    assert op is not None


def test_doublestorage_593():
    op = get_op("DoubleStorage")
    assert op is not None


def test_doubletensor_594():
    op = get_op("DoubleTensor")
    assert op is not None


def test_drawboundingboxes_595():
    op = get_op("DrawBoundingBoxes")
    assert op is not None


def test_dropout_596():
    op = get_op("Dropout")
    assert op is not None


def test_dropout__597():
    op = get_op("Dropout_")
    assert op is not None


def test_dsmm_598():
    op = get_op("Dsmm")
    assert op is not None


def test_dtypecategory_599():
    op = get_op("DtypeCategory")
    assert op is not None


def test_dtypeobject_600():
    op = get_op("DtypeObject")
    assert op is not None


def test_dynamicpartition_601():
    op = get_op("DynamicPartition")
    assert op is not None


def test_dynamicslice_602():
    op = get_op("DynamicSlice")
    assert op is not None


def test_dynamicstitch_603():
    op = get_op("DynamicStitch")
    assert op is not None


def test_dynamicupdateslice_604():
    op = get_op("DynamicUpdateSlice")
    assert op is not None


def test_elu_605():
    op = get_op("ELU")
    assert op is not None


def test_enter_606():
    op = get_op("ENTER")
    assert op is not None


def test_editdistance_607():
    op = get_op("EditDistance")
    assert op is not None


def test_eig_608():
    op = get_op("Eig")
    assert op is not None


def test_eigimplementation_609():
    op = get_op("EigImplementation")
    assert op is not None


def test_eigresult_610():
    op = get_op("EigResult")
    assert op is not None


def test_eigh_611():
    op = get_op("Eigh")
    assert op is not None


def test_eighimplementation_612():
    op = get_op("EighImplementation")
    assert op is not None


def test_eighresult_613():
    op = get_op("EighResult")
    assert op is not None


def test_eightridiagonal_614():
    op = get_op("EighTridiagonal")
    assert op is not None


def test_eigvalsh_615():
    op = get_op("Eigvalsh")
    assert op is not None


def test_einsum_616():
    op = get_op("Einsum")
    assert op is not None


def test_einsumdense_617():
    op = get_op("EinsumDense")
    assert op is not None


def test_einsumlayer_618():
    op = get_op("EinsumLayer")
    assert op is not None


def test_einsumop_619():
    op = get_op("EinsumOp")
    assert op is not None


def test_einsumpath_620():
    op = get_op("EinsumPath")
    assert op is not None


def test_emastate_621():
    op = get_op("EmaState")
    assert op is not None


def test_embed_622():
    op = get_op("Embed")
    assert op is not None


def test_embedding_623():
    op = get_op("Embedding")
    assert op is not None


def test_embeddingbag_624():
    op = get_op("EmbeddingBag")
    assert op is not None


def test_embeddingrenorm__625():
    op = get_op("EmbeddingRenorm_")
    assert op is not None


def test_emptylike_626():
    op = get_op("EmptyLike")
    assert op is not None


def test_emptypermuted_627():
    op = get_op("EmptyPermuted")
    assert op is not None


def test_emptyquantized_628():
    op = get_op("EmptyQuantized")
    assert op is not None


def test_emptystrided_629():
    op = get_op("EmptyStrided")
    assert op is not None


def test_enablecompile_630():
    op = get_op("EnableCompile")
    assert op is not None


def test_enablegrad_631():
    op = get_op("EnableGrad")
    assert op is not None


def test_enumtype_632():
    op = get_op("EnumType")
    assert op is not None


def test_equal_633():
    op = get_op("Equal")
    assert op is not None


def test_equalaggregate_634():
    op = get_op("EqualAggregate")
    assert op is not None


def test_equalelementwise_635():
    op = get_op("EqualElementwise")
    assert op is not None


def test_equalization_636():
    op = get_op("Equalization")
    assert op is not None


def test_equals_637():
    op = get_op("Equals")
    assert op is not None


def test_erf_638():
    op = get_op("Erf")
    assert op is not None


def test_erf__639():
    op = get_op("Erf_")
    assert op is not None


def test_erfcinv_640():
    op = get_op("ErfcInv")
    assert op is not None


def test_erfc__641():
    op = get_op("Erfc_")
    assert op is not None


def test_erfinv_642():
    op = get_op("Erfinv")
    assert op is not None


def test_errorreport_643():
    op = get_op("ErrorReport")
    assert op is not None


def test_euclideannorm_644():
    op = get_op("EuclideanNorm")
    assert op is not None


def test_eulergamma_645():
    op = get_op("EulerGamma")
    assert op is not None


def test_evalmode_646():
    op = get_op("EvalMode")
    assert op is not None


def test_evalshape_647():
    op = get_op("EvalShape")
    assert op is not None


def test_event_648():
    op = get_op("Event")
    assert op is not None


def test_everything_649():
    op = get_op("Everything")
    assert op is not None


def test_excludedispatchkeyguard_650():
    op = get_op("ExcludeDispatchKeyGuard")
    assert op is not None


def test_executionplan_651():
    op = get_op("ExecutionPlan")
    assert op is not None


def test_exit_652():
    op = get_op("Exit")
    assert op is not None


def test_exp_653():
    op = get_op("Exp")
    assert op is not None


def test_exp2__654():
    op = get_op("Exp2_")
    assert op is not None


def test_exp__655():
    op = get_op("Exp_")
    assert op is not None


def test_expandcopy_656():
    op = get_op("ExpandCopy")
    assert op is not None


def test_expanddims_657():
    op = get_op("ExpandDims")
    assert op is not None


def test_experimentalenablenumpybehavior_658():
    op = get_op("ExperimentalEnableNumpyBehavior")
    assert op is not None


def test_expm1__659():
    op = get_op("Expm1_")
    assert op is not None


def test_exponential_660():
    op = get_op("Exponential")
    assert op is not None


def test_exportfunction_661():
    op = get_op("ExportFunction")
    assert op is not None


def test_exporttodot_662():
    op = get_op("ExportToDot")
    assert op is not None


def test_exporter_663():
    op = get_op("Exporter")
    assert op is not None


def test_expr_664():
    op = get_op("Expr")
    assert op is not None


def test_extractvolumepatches_665():
    op = get_op("ExtractVolumePatches")
    assert op is not None


def test_f_666():
    op = get_op("F")
    assert op is not None


def test_force_use_flex_attention_667():
    op = get_op("FORCE_USE_FLEX_ATTENTION")
    assert op is not None


def test_frnn_668():
    op = get_op("FRnn")
    assert op is not None


def test_fwd_compat_ir_version_669():
    op = get_op("FWD_COMPAT_IR_VERSION")
    assert op is not None


def test_fakequantizeperchannelaffine_670():
    op = get_op("FakeQuantizePerChannelAffine")
    assert op is not None


def test_fakequantizepertensoraffine_671():
    op = get_op("FakeQuantizePerTensorAffine")
    assert op is not None


def test_false__672():
    op = get_op("False_")
    assert op is not None


def test_faninconcat_673():
    op = get_op("FanInConcat")
    assert op is not None


def test_faninsum_674():
    op = get_op("FanInSum")
    assert op is not None


def test_fanout_675():
    op = get_op("FanOut")
    assert op is not None


def test_fatalerror_676():
    op = get_op("FatalError")
    assert op is not None


def test_fbgemmlinearfp16weight_677():
    op = get_op("FbgemmLinearFp16Weight")
    assert op is not None


def test_fbgemmlinearfp16weightfp32activation_678():
    op = get_op("FbgemmLinearFp16WeightFp32Activation")
    assert op is not None


def test_fbgemmlinearint8weight_679():
    op = get_op("FbgemmLinearInt8Weight")
    assert op is not None


def test_fbgemmlinearint8weightfp32activation_680():
    op = get_op("FbgemmLinearInt8WeightFp32Activation")
    assert op is not None


def test_fbgemmlinearquantizeweight_681():
    op = get_op("FbgemmLinearQuantizeWeight")
    assert op is not None


def test_fbgemmpackgemmmatrixfp16_682():
    op = get_op("FbgemmPackGemmMatrixFp16")
    assert op is not None


def test_fbgemmpackquantizedmatrix_683():
    op = get_op("FbgemmPackQuantizedMatrix")
    assert op is not None


def test_featurealphadropout_684():
    op = get_op("FeatureAlphaDropout")
    assert op is not None


def test_featurealphadropout__685():
    op = get_op("FeatureAlphaDropout_")
    assert op is not None


def test_featuredropout_686():
    op = get_op("FeatureDropout")
    assert op is not None


def test_featuredropout__687():
    op = get_op("FeatureDropout_")
    assert op is not None


def test_feedforward_688():
    op = get_op("FeedForward")
    assert op is not None


def test_fft_689():
    op = get_op("Fft")
    assert op is not None


def test_fft2d_690():
    op = get_op("Fft2d")
    assert op is not None


def test_fft3d_691():
    op = get_op("Fft3d")
    assert op is not None


def test_ffttype_692():
    op = get_op("FftType")
    assert op is not None


def test_fftconvolve_693():
    op = get_op("Fftconvolve")
    assert op is not None


def test_fftn_694():
    op = get_op("Fftn")
    assert op is not None


def test_fftnd_695():
    op = get_op("Fftnd")
    assert op is not None


def test_fftshift_696():
    op = get_op("Fftshift")
    assert op is not None


def test_filecheck_697():
    op = get_op("FileCheck")
    assert op is not None


def test_fill_698():
    op = get_op("Fill")
    assert op is not None


def test_filldiagonal_699():
    op = get_op("FillDiagonal")
    assert op is not None


def test_fill__700():
    op = get_op("Fill_")
    assert op is not None


def test_filterall_701():
    op = get_op("FilterAll")
    assert op is not None


def test_filterany_702():
    op = get_op("FilterAny")
    assert op is not None


def test_filtereverything_703():
    op = get_op("FilterEverything")
    assert op is not None


def test_filterstate_704():
    op = get_op("FilterState")
    assert op is not None


def test_findduplicates_705():
    op = get_op("FindDuplicates")
    assert op is not None


def test_fix__706():
    op = get_op("Fix_")
    assert op is not None


def test_flatstate_707():
    op = get_op("FlatState")
    assert op is not None


def test_flatnonsense_708():
    op = get_op("Flatnonsense")
    assert op is not None


def test_flaxlayer_709():
    op = get_op("FlaxLayer")
    assert op is not None


def test_flexkerneloptions_710():
    op = get_op("FlexKernelOptions")
    assert op is not None


def test_float16_711():
    op = get_op("Float16")
    assert op is not None


def test_float32_712():
    op = get_op("Float32")
    assert op is not None


def test_float4e2m1fnx2_713():
    op = get_op("Float4E2m1fnX2")
    assert op is not None


def test_float64_714():
    op = get_op("Float64")
    assert op is not None


def test_float8e4m3fn_715():
    op = get_op("Float8E4m3fn")
    assert op is not None


def test_float8e4m3fnuz_716():
    op = get_op("Float8E4m3fnuz")
    assert op is not None


def test_float8e5m2_717():
    op = get_op("Float8E5m2")
    assert op is not None


def test_float8e5m2fnuz_718():
    op = get_op("Float8E5m2fnuz")
    assert op is not None


def test_float8e8m0fnu_719():
    op = get_op("Float8E8m0fnu")
    assert op is not None


def test_floatpower_720():
    op = get_op("FloatPower")
    assert op is not None


def test_floatstorage_721():
    op = get_op("FloatStorage")
    assert op is not None


def test_floattensor_722():
    op = get_op("FloatTensor")
    assert op is not None


def test_floattype_723():
    op = get_op("FloatType")
    assert op is not None


def test_floatingformat_724():
    op = get_op("FloatingFormat")
    assert op is not None


def test_floordivide_725():
    op = get_op("FloorDivide")
    assert op is not None


def test_floormod_726():
    op = get_op("FloorMod")
    assert op is not None


def test_floor__727():
    op = get_op("Floor_")
    assert op is not None


def test_fold_728():
    op = get_op("Fold")
    assert op is not None


def test_forresult_729():
    op = get_op("ForResult")
    assert op is not None


def test_foriloop_730():
    op = get_op("ForiLoop")
    assert op is not None


def test_foriloopbodyfn_731():
    op = get_op("ForiLoopBodyFn")
    assert op is not None


def test_fork_732():
    op = get_op("Fork")
    assert op is not None


def test_forkrngs_733():
    op = get_op("ForkRngs")
    assert op is not None


def test_formatfloatpositional_734():
    op = get_op("FormatFloatPositional")
    assert op is not None


def test_formatfloatscientific_735():
    op = get_op("FormatFloatScientific")
    assert op is not None


def test_frac_736():
    op = get_op("Frac")
    assert op is not None


def test_frac__737():
    op = get_op("Frac_")
    assert op is not None


def test_fractionalavgpool_738():
    op = get_op("FractionalAvgPool")
    assert op is not None


def test_fractionalmaxpool_739():
    op = get_op("FractionalMaxPool")
    assert op is not None


def test_fractionalmaxpool2d_740():
    op = get_op("FractionalMaxPool2d")
    assert op is not None


def test_fractionalmaxpool3d_741():
    op = get_op("FractionalMaxPool3d")
    assert op is not None


def test_fragmentedarray_742():
    op = get_op("FragmentedArray")
    assert op is not None


def test_fragmentedlayout_743():
    op = get_op("FragmentedLayout")
    assert op is not None


def test_frame_744():
    op = get_op("Frame")
    assert op is not None


def test_frobeniusnorm_745():
    op = get_op("FrobeniusNorm")
    assert op is not None


def test_fromdlpack_746():
    op = get_op("FromDlpack")
    assert op is not None


def test_fromflatstate_747():
    op = get_op("FromFlatState")
    assert op is not None


def test_fromnumpy_748():
    op = get_op("FromNumpy")
    assert op is not None


def test_fromregex_749():
    op = get_op("FromRegex")
    assert op is not None


def test_fromtensorslices_750():
    op = get_op("FromTensorSlices")
    assert op is not None


def test_fromtree_751():
    op = get_op("FromTree")
    assert op is not None


def test_frombuffer_752():
    op = get_op("Frombuffer")
    assert op is not None


def test_fromfile_753():
    op = get_op("Fromfile")
    assert op is not None


def test_fromfunction_754():
    op = get_op("Fromfunction")
    assert op is not None


def test_fromiter_755():
    op = get_op("Fromiter")
    assert op is not None


def test_frompyfunc_756():
    op = get_op("Frompyfunc")
    assert op is not None


def test_fromstring_757():
    op = get_op("Fromstring")
    assert op is not None


def test_ftrl_758():
    op = get_op("Ftrl")
    assert op is not None


def test_full_759():
    op = get_op("Full")
    assert op is not None


def test_fulllike_760():
    op = get_op("FullLike")
    assert op is not None


def test_fullmask_761():
    op = get_op("FullMask")
    assert op is not None


def test_fullsoftmax_762():
    op = get_op("FullSoftmax")
    assert op is not None


def test_fulltypedescr_763():
    op = get_op("FullTypeDescr")
    assert op is not None


def test_funcnamesuffix_764():
    op = get_op("FuncNameSuffix")
    assert op is not None


def test_function_765():
    op = get_op("Function")
    assert op is not None


def test_functionapi_766():
    op = get_op("FunctionApi")
    assert op is not None


def test_functionexporter_767():
    op = get_op("FunctionExporter")
    assert op is not None


def test_functioninfo_768():
    op = get_op("FunctionInfo")
    assert op is not None


def test_functionschema_769():
    op = get_op("FunctionSchema")
    assert op is not None


def test_fusedaddrelu_770():
    op = get_op("FusedAddRelu")
    assert op is not None


def test_fusedlogexp_771():
    op = get_op("FusedLogExp")
    assert op is not None


def test_fusedmovingavgobsfakequant_772():
    op = get_op("FusedMovingAvgObsFakeQuant")
    assert op is not None


def test_fusedmultiplyadd_773():
    op = get_op("FusedMultiplyAdd")
    assert op is not None


def test_future_774():
    op = get_op("Future")
    assert op is not None


def test_futuretype_775():
    op = get_op("FutureType")
    assert op is not None


def test_fwdfn_776():
    op = get_op("FwdFn")
    assert op is not None


def test_gelu_777():
    op = get_op("GELU")
    assert op is not None


def test_global_broadcast_778():
    op = get_op("GLOBAL_BROADCAST")
    assert op is not None


def test_gmem_779():
    op = get_op("GMEM")
    assert op is not None


def test_grid_sample_interpolation_modes_780():
    op = get_op("GRID_SAMPLE_INTERPOLATION_MODES")
    assert op is not None


def test_grid_sample_padding_modes_781():
    op = get_op("GRID_SAMPLE_PADDING_MODES")
    assert op is not None


def test_gru_782():
    op = get_op("GRU")
    assert op is not None


def test_grucell_783():
    op = get_op("GRUCell")
    assert op is not None


def test_gshardsharedembeddingsoftmax_784():
    op = get_op("GShardSharedEmbeddingSoftmax")
    assert op is not None


def test_gamma_785():
    op = get_op("Gamma")
    assert op is not None


def test_gammacdf_786():
    op = get_op("GammaCdf")
    assert op is not None


def test_gammapdf_787():
    op = get_op("GammaPdf")
    assert op is not None


def test_gather_788():
    op = get_op("Gather")
    assert op is not None


def test_gatherdimensionnumbers_789():
    op = get_op("GatherDimensionNumbers")
    assert op is not None


def test_gathermm_790():
    op = get_op("GatherMM")
    assert op is not None


def test_gathernd_791():
    op = get_op("GatherNd")
    assert op is not None


def test_gatherqmm_792():
    op = get_op("GatherQMM")
    assert op is not None


def test_gatherscattermode_793():
    op = get_op("GatherScatterMode")
    assert op is not None


def test_gaussiandropout_794():
    op = get_op("GaussianDropout")
    assert op is not None


def test_gaussiannllloss_795():
    op = get_op("GaussianNLLLoss")
    assert op is not None


def test_gaussiannoise_796():
    op = get_op("GaussianNoise")
    assert op is not None


def test_gaussiannll_797():
    op = get_op("Gaussiannll")
    assert op is not None


def test_gcd__798():
    op = get_op("Gcd_")
    assert op is not None


def test_geluapprox_799():
    op = get_op("GeluApprox")
    assert op is not None


def test_gelufastapprox_800():
    op = get_op("GeluFastApprox")
    assert op is not None


def test_genfromtxt_801():
    op = get_op("GenFromTxt")
    assert op is not None


def test_generalconv_802():
    op = get_op("GeneralConv")
    assert op is not None


def test_generalconvtranspose_803():
    op = get_op("GeneralConvTranspose")
    assert op is not None


def test_generalizednormal_804():
    op = get_op("GeneralizedNormal")
    assert op is not None


def test_generatedumpfn_805():
    op = get_op("GenerateDumpFn")
    assert op is not None


def test_generator_806():
    op = get_op("Generator")
    assert op is not None


def test_genericpytree_807():
    op = get_op("GenericPytree")
    assert op is not None


def test_geomspace_808():
    op = get_op("Geomspace")
    assert op is not None


def test_ger_809():
    op = get_op("Ger")
    assert op is not None


def test_getabstractmodel_810():
    op = get_op("GetAbstractModel")
    assert op is not None


def test_getactivememory_811():
    op = get_op("GetActiveMemory")
    assert op is not None


def test_getattr_812():
    op = get_op("GetAttr")
    assert op is not None


def test_getautocastcpudtype_813():
    op = get_op("GetAutocastCpuDtype")
    assert op is not None


def test_getautocastdtype_814():
    op = get_op("GetAutocastDtype")
    assert op is not None


def test_getautocastgpudtype_815():
    op = get_op("GetAutocastGpuDtype")
    assert op is not None


def test_getautocastipudtype_816():
    op = get_op("GetAutocastIpuDtype")
    assert op is not None


def test_getautocastxladtype_817():
    op = get_op("GetAutocastXlaDtype")
    assert op is not None


def test_getbigramids_818():
    op = get_op("GetBigramIds")
    assert op is not None


def test_getcachememory_819():
    op = get_op("GetCacheMemory")
    assert op is not None


def test_getdefaultdevice_820():
    op = get_op("GetDefaultDevice")
    assert op is not None


def test_getdefaultdtype_821():
    op = get_op("GetDefaultDtype")
    assert op is not None


def test_getdefaultstream_822():
    op = get_op("GetDefaultStream")
    assert op is not None


def test_getdeterministicdebugmode_823():
    op = get_op("GetDeterministicDebugMode")
    assert op is not None


def test_getdevice_824():
    op = get_op("GetDevice")
    assert op is not None


def test_getdevicemodule_825():
    op = get_op("GetDeviceModule")
    assert op is not None


def test_getfilepath_826():
    op = get_op("GetFilePath")
    assert op is not None


def test_getfloat32matmulprecision_827():
    op = get_op("GetFloat32MatmulPrecision")
    assert op is not None


def test_getinclude_828():
    op = get_op("GetInclude")
    assert op is not None


def test_getitem_829():
    op = get_op("GetItem")
    assert op is not None


def test_getnamedsharding_830():
    op = get_op("GetNamedSharding")
    assert op is not None


def test_getnuminteropthreads_831():
    op = get_op("GetNumInteropThreads")
    assert op is not None


def test_getnumthreads_832():
    op = get_op("GetNumThreads")
    assert op is not None


def test_getoptimizer_833():
    op = get_op("GetOptimizer")
    assert op is not None


def test_getpartitionspec_834():
    op = get_op("GetPartitionSpec")
    assert op is not None


def test_getpeakmemory_835():
    op = get_op("GetPeakMemory")
    assert op is not None


def test_getprintoptions_836():
    op = get_op("GetPrintoptions")
    assert op is not None


def test_getrngstate_837():
    op = get_op("GetRngState")
    assert op is not None


def test_getstate_838():
    op = get_op("GetState")
    assert op is not None


def test_getvariable_839():
    op = get_op("GetVariable")
    assert op is not None


def test_globalasynccheckpointmanager_840():
    op = get_op("GlobalAsyncCheckpointManager")
    assert op is not None


def test_globalasynccheckpointmanagerbase_841():
    op = get_op("GlobalAsyncCheckpointManagerBase")
    assert op is not None


def test_globalaveragepooling1d_842():
    op = get_op("GlobalAveragePooling1D")
    assert op is not None


def test_globalaveragepooling2d_843():
    op = get_op("GlobalAveragePooling2D")
    assert op is not None


def test_globalaveragepooling3d_844():
    op = get_op("GlobalAveragePooling3D")
    assert op is not None


def test_globalavgpool1d_845():
    op = get_op("GlobalAvgPool1D")
    assert op is not None


def test_globalavgpool2d_846():
    op = get_op("GlobalAvgPool2D")
    assert op is not None


def test_globalavgpool3d_847():
    op = get_op("GlobalAvgPool3D")
    assert op is not None


def test_globalbroadcast_848():
    op = get_op("GlobalBroadcast")
    assert op is not None


def test_globalmaxpool1d_849():
    op = get_op("GlobalMaxPool1D")
    assert op is not None


def test_globalmaxpool2d_850():
    op = get_op("GlobalMaxPool2D")
    assert op is not None


def test_globalmaxpool3d_851():
    op = get_op("GlobalMaxPool3D")
    assert op is not None


def test_globalmaxpooling1d_852():
    op = get_op("GlobalMaxPooling1D")
    assert op is not None


def test_globalmaxpooling2d_853():
    op = get_op("GlobalMaxPooling2D")
    assert op is not None


def test_globalmaxpooling3d_854():
    op = get_op("GlobalMaxPooling3D")
    assert op is not None


def test_globalpooling_855():
    op = get_op("GlobalPooling")
    assert op is not None


def test_globalvarapi_856():
    op = get_op("GlobalVarApi")
    assert op is not None


def test_gpu_857():
    op = get_op("Gpu")
    assert op is not None


def test_gradfn_858():
    op = get_op("GradFn")
    assert op is not None


def test_gradscaler_859():
    op = get_op("GradScaler")
    assert op is not None


def test_gradient_860():
    op = get_op("Gradient")
    assert op is not None


def test_graph_861():
    op = get_op("Graph")
    assert op is not None


def test_graphcontext_862():
    op = get_op("GraphContext")
    assert op is not None


def test_graphdefstate_863():
    op = get_op("GraphDefState")
    assert op is not None


def test_graphexecutorstate_864():
    op = get_op("GraphExecutorState")
    assert op is not None


def test_graphnodeimpl_865():
    op = get_op("GraphNodeImpl")
    assert op is not None


def test_graphstate_866():
    op = get_op("GraphState")
    assert op is not None


def test_grayscale_867():
    op = get_op("Grayscale")
    assert op is not None


def test_greater_868():
    op = get_op("Greater")
    assert op is not None


def test_greaterequal_869():
    op = get_op("GreaterEqual")
    assert op is not None


def test_gridsample_870():
    op = get_op("GridSample")
    assert op is not None


def test_gridsampler_871():
    op = get_op("GridSampler")
    assert op is not None


def test_gridsampler2d_872():
    op = get_op("GridSampler2d")
    assert op is not None


def test_gridsampler3d_873():
    op = get_op("GridSampler3d")
    assert op is not None


def test_groupinfo_874():
    op = get_op("GroupInfo")
    assert op is not None


def test_groupmetadata_875():
    op = get_op("GroupMetadata")
    assert op is not None


def test_groupnorm_876():
    op = get_op("GroupNorm")
    assert op is not None


def test_groupnormalization_877():
    op = get_op("GroupNormalization")
    assert op is not None


def test_groupqueryattention_878():
    op = get_op("GroupQueryAttention")
    assert op is not None


def test_groupedqueryattention_879():
    op = get_op("GroupedQueryAttention")
    assert op is not None


def test_gumbel_880():
    op = get_op("Gumbel")
    assert op is not None


def test_hbm_881():
    op = get_op("HBM")
    assert op is not None


def test_head_dim_minor_882():
    op = get_op("HEAD_DIM_MINOR")
    assert op is not None


def test_host_883():
    op = get_op("HOST")
    assert op is not None


def test_hadamardtransform_884():
    op = get_op("HadamardTransform")
    assert op is not None


def test_halfstorage_885():
    op = get_op("HalfStorage")
    assert op is not None


def test_halftensor_886():
    op = get_op("HalfTensor")
    assert op is not None


def test_hammingwindow_887():
    op = get_op("HammingWindow")
    assert op is not None


def test_hannwindow_888():
    op = get_op("HannWindow")
    assert op is not None


def test_hardsilu_889():
    op = get_op("HardSilu")
    assert op is not None


def test_hardswish_890():
    op = get_op("Hardswish")
    assert op is not None


def test_haslapack_891():
    op = get_op("HasLapack")
    assert op is not None


def test_hasmkl_892():
    op = get_op("HasMkl")
    assert op is not None


def test_hasopenmp_893():
    op = get_op("HasOpenmp")
    assert op is not None


def test_hasspectral_894():
    op = get_op("HasSpectral")
    assert op is not None


def test_hastag_895():
    op = get_op("HasTag")
    assert op is not None


def test_hashtensor_896():
    op = get_op("HashTensor")
    assert op is not None


def test_hashedcrossing_897():
    op = get_op("HashedCrossing")
    assert op is not None


def test_hashing_898():
    op = get_op("Hashing")
    assert op is not None


def test_hfft_899():
    op = get_op("Hfft")
    assert op is not None


def test_hijaxvariable_900():
    op = get_op("HijaxVariable")
    assert op is not None


def test_hijaxvariablemeta_901():
    op = get_op("HijaxVariableMeta")
    assert op is not None


def test_hingeembeddingloss_902():
    op = get_op("HingeEmbeddingLoss")
    assert op is not None


def test_histc_903():
    op = get_op("Histc")
    assert op is not None


def test_histogram_904():
    op = get_op("Histogram")
    assert op is not None


def test_histogram2d_905():
    op = get_op("Histogram2d")
    assert op is not None


def test_histogrambinedges_906():
    op = get_op("HistogramBinEdges")
    assert op is not None


def test_histogramdd_907():
    op = get_op("Histogramdd")
    assert op is not None


def test_hlopass_908():
    op = get_op("HloPass")
    assert op is not None


def test_hsmm_909():
    op = get_op("Hsmm")
    assert op is not None


def test_hspmm_910():
    op = get_op("Hspmm")
    assert op is not None


def test_huberloss_911():
    op = get_op("HuberLoss")
    assert op is not None


def test_i0_912():
    op = get_op("I0")
    assert op is not None


def test_i0__913():
    op = get_op("I0_")
    assert op is not None


def test_iodescriptor_914():
    op = get_op("IODescriptor")
    assert op is not None


def test_idct_915():
    op = get_op("Idct")
    assert op is not None


def test_identitynorm_916():
    op = get_op("IdentityNorm")
    assert op is not None


def test_if_917():
    op = get_op("If")
    assert op is not None


def test_ifft_918():
    op = get_op("Ifft")
    assert op is not None


def test_ifft2_919():
    op = get_op("Ifft2")
    assert op is not None


def test_ifft2d_920():
    op = get_op("Ifft2d")
    assert op is not None


def test_ifft3d_921():
    op = get_op("Ifft3d")
    assert op is not None


def test_ifftn_922():
    op = get_op("Ifftn")
    assert op is not None


def test_ifftnd_923():
    op = get_op("Ifftnd")
    assert op is not None


def test_ifftshift_924():
    op = get_op("Ifftshift")
    assert op is not None


def test_igamma_925():
    op = get_op("Igamma")
    assert op is not None


def test_igammagrada_926():
    op = get_op("IgammaGradA")
    assert op is not None


def test_igammamode_927():
    op = get_op("IgammaMode")
    assert op is not None


def test_igammac_928():
    op = get_op("Igammac")
    assert op is not None


def test_importfunction_929():
    op = get_op("ImportFunction")
    assert op is not None


def test_importirmodule_930():
    op = get_op("ImportIrModule")
    assert op is not None


def test_intopk_931():
    op = get_op("InTopK")
    assert op is not None


def test_indexadd_932():
    op = get_op("IndexAdd")
    assert op is not None


def test_indexcopy_933():
    op = get_op("IndexCopy")
    assert op is not None


def test_indexexp_934():
    op = get_op("IndexExp")
    assert op is not None


def test_indexexpression_935():
    op = get_op("IndexExpression")
    assert op is not None


def test_indexfill_936():
    op = get_op("IndexFill")
    assert op is not None


def test_indexindim_937():
    op = get_op("IndexInDim")
    assert op is not None


def test_indexmap_938():
    op = get_op("IndexMap")
    assert op is not None


def test_indexput_939():
    op = get_op("IndexPut")
    assert op is not None


def test_indexput__940():
    op = get_op("IndexPut_")
    assert op is not None


def test_indexreduce_941():
    op = get_op("IndexReduce")
    assert op is not None


def test_indexselect_942():
    op = get_op("IndexSelect")
    assert op is not None


def test_indextransform_943():
    op = get_op("IndexTransform")
    assert op is not None


def test_indextype_944():
    op = get_op("IndexType")
    assert op is not None


def test_indexespytreedef_945():
    op = get_op("IndexesPytreeDef")
    assert op is not None


def test_indexingstrategy_946():
    op = get_op("IndexingStrategy")
    assert op is not None


def test_infeed_947():
    op = get_op("Infeed")
    assert op is not None


def test_inferencemode_948():
    op = get_op("InferenceMode")
    assert op is not None


def test_inferredop_949():
    op = get_op("InferredOp")
    assert op is not None


def test_inferredtype_950():
    op = get_op("InferredType")
    assert op is not None


def test_infinite_951():
    op = get_op("Infinite")
    assert op is not None


def test_info_952():
    op = get_op("Info")
    assert op is not None


def test_initfn_953():
    op = get_op("InitFn")
    assert op is not None


def test_initnumthreads_954():
    op = get_op("InitNumThreads")
    assert op is not None


def test_initialseed_955():
    op = get_op("InitialSeed")
    assert op is not None


def test_initializer_956():
    op = get_op("Initializer")
    assert op is not None


def test_injecthyperparamsstate_957():
    op = get_op("InjectHyperparamsState")
    assert op is not None


def test_injectstatefulhyperparamsstate_958():
    op = get_op("InjectStatefulHyperparamsState")
    assert op is not None


def test_input_959():
    op = get_op("Input")
    assert op is not None


def test_inputlayer_960():
    op = get_op("InputLayer")
    assert op is not None


def test_inputspec_961():
    op = get_op("InputSpec")
    assert op is not None


def test_instancenorm_962():
    op = get_op("InstanceNorm")
    assert op is not None


def test_instancenorm1d_963():
    op = get_op("InstanceNorm1d")
    assert op is not None


def test_instancenorm2d_964():
    op = get_op("InstanceNorm2d")
    assert op is not None


def test_instancenorm3d_965():
    op = get_op("InstanceNorm3d")
    assert op is not None


def test_int16_966():
    op = get_op("Int16")
    assert op is not None


def test_int16tensor_967():
    op = get_op("Int16Tensor")
    assert op is not None


def test_int3_968():
    op = get_op("Int3")
    assert op is not None


def test_int32_969():
    op = get_op("Int32")
    assert op is not None


def test_int5_970():
    op = get_op("Int5")
    assert op is not None


def test_int6_971():
    op = get_op("Int6")
    assert op is not None


def test_int64_972():
    op = get_op("Int64")
    assert op is not None


def test_int7_973():
    op = get_op("Int7")
    assert op is not None


def test_intrepr_974():
    op = get_op("IntRepr")
    assert op is not None


def test_intstorage_975():
    op = get_op("IntStorage")
    assert op is not None


def test_inttensor_976():
    op = get_op("IntTensor")
    assert op is not None


def test_inttype_977():
    op = get_op("IntType")
    assert op is not None


def test_intc_978():
    op = get_op("Intc")
    assert op is not None


def test_integerformat_979():
    op = get_op("IntegerFormat")
    assert op is not None


def test_integerlookup_980():
    op = get_op("IntegerLookup")
    assert op is not None


def test_interfacetype_981():
    op = get_op("InterfaceType")
    assert op is not None


def test_intermediate_982():
    op = get_op("Intermediate")
    assert op is not None


def test_intp_983():
    op = get_op("Intp")
    assert op is not None


def test_inv_984():
    op = get_op("Inv")
    assert op is not None


def test_inverse_985():
    op = get_op("Inverse")
    assert op is not None


def test_inversemdct_986():
    op = get_op("InverseMdct")
    assert op is not None


def test_inverseshorttimefouriertransform_987():
    op = get_op("InverseShortTimeFourierTransform")
    assert op is not None


def test_invertpermutation_988():
    op = get_op("InvertPermutation")
    assert op is not None


def test_irfft_989():
    op = get_op("Irfft")
    assert op is not None


def test_irfft2_990():
    op = get_op("Irfft2")
    assert op is not None


def test_irfft2d_991():
    op = get_op("Irfft2d")
    assert op is not None


def test_irfft3d_992():
    op = get_op("Irfft3d")
    assert op is not None


def test_irfftn_993():
    op = get_op("Irfftn")
    assert op is not None


def test_irfftnd_994():
    op = get_op("Irfftnd")
    assert op is not None


def test_isanomalychecknanenabled_995():
    op = get_op("IsAnomalyCheckNanEnabled")
    assert op is not None


def test_isanomalyenabled_996():
    op = get_op("IsAnomalyEnabled")
    assert op is not None


def test_isautocastcacheenabled_997():
    op = get_op("IsAutocastCacheEnabled")
    assert op is not None


def test_isautocastcpuenabled_998():
    op = get_op("IsAutocastCpuEnabled")
    assert op is not None


def test_isautocastenabled_999():
    op = get_op("IsAutocastEnabled")
    assert op is not None


def test_isautocastipuenabled_1000():
    op = get_op("IsAutocastIpuEnabled")
    assert op is not None


def test_isautocastxlaenabled_1001():
    op = get_op("IsAutocastXlaEnabled")
    assert op is not None


def test_isavailable_1002():
    op = get_op("IsAvailable")
    assert op is not None


def test_isbusday_1003():
    op = get_op("IsBusDay")
    assert op is not None


def test_isconj_1004():
    op = get_op("IsConj")
    assert op is not None


def test_isdata_1005():
    op = get_op("IsData")
    assert op is not None


def test_isdeterministicalgorithmswarnonlyenabled_1006():
    op = get_op("IsDeterministicAlgorithmsWarnOnlyEnabled")
    assert op is not None


def test_isdistributed_1007():
    op = get_op("IsDistributed")
    assert op is not None


def test_isfloatingpoint_1008():
    op = get_op("IsFloatingPoint")
    assert op is not None


def test_isgradenabled_1009():
    op = get_op("IsGradEnabled")
    assert op is not None


def test_isinference_1010():
    op = get_op("IsInference")
    assert op is not None


def test_isinferencemodeenabled_1011():
    op = get_op("IsInferenceModeEnabled")
    assert op is not None


def test_isnat_1012():
    op = get_op("IsNaT")
    assert op is not None


def test_isneg_1013():
    op = get_op("IsNeg")
    assert op is not None


def test_isnondecreasing_1014():
    op = get_op("IsNonDecreasing")
    assert op is not None


def test_isnonzero_1015():
    op = get_op("IsNonZero")
    assert op is not None


def test_issamesize_1016():
    op = get_op("IsSameSize")
    assert op is not None


def test_issigned_1017():
    op = get_op("IsSigned")
    assert op is not None


def test_isstorage_1018():
    op = get_op("IsStorage")
    assert op is not None


def test_isstrictlyincreasing_1019():
    op = get_op("IsStrictlyIncreasing")
    assert op is not None


def test_istensor_1020():
    op = get_op("IsTensor")
    assert op is not None


def test_istransferable_1021():
    op = get_op("IsTransferable")
    assert op is not None


def test_isvalidmmatiling_1022():
    op = get_op("IsValidMmaTiling")
    assert op is not None


def test_isvulkanavailable_1023():
    op = get_op("IsVulkanAvailable")
    assert op is not None


def test_iswarnalwaysenabled_1024():
    op = get_op("IsWarnAlwaysEnabled")
    assert op is not None


def test_istft_1025():
    op = get_op("Istft")
    assert op is not None


def test_iterchildren_1026():
    op = get_op("IterChildren")
    assert op is not None


def test_itergraph_1027():
    op = get_op("IterGraph")
    assert op is not None


def test_itermodules_1028():
    op = get_op("IterModules")
    assert op is not None


def test_iterable_1029():
    op = get_op("Iterable")
    assert op is not None


def test_jitexception_1030():
    op = get_op("JITException")
    assert op is not None


def test_jagged_1031():
    op = get_op("Jagged")
    assert op is not None


def test_jaxlayer_1032():
    op = get_op("JaxLayer")
    assert op is not None


def test_jaxruntimeerror_1033():
    op = get_op("JaxRuntimeError")
    assert op is not None


def test_jettrace_1034():
    op = get_op("JetTrace")
    assert op is not None


def test_jettracer_1035():
    op = get_op("JetTracer")
    assert op is not None


def test_jitfn_1036():
    op = get_op("JitFn")
    assert op is not None


def test_jitwrapped_1037():
    op = get_op("JitWrapped")
    assert op is not None


def test_joinpoint_1038():
    op = get_op("JoinPoint")
    assert op is not None


def test_kernel_arg_id_attr_1039():
    op = get_op("KERNEL_ARG_ID_ATTR")
    assert op is not None


def test_kldivloss_1040():
    op = get_op("KLDivLoss")
    assert op is not None


def test_kldivergenceloss_1041():
    op = get_op("KLDivergenceLoss")
    assert op is not None


def test_known_kernels_1042():
    op = get_op("KNOWN_KERNELS")
    assert op is not None


def test_k_hi_32_1043():
    op = get_op("K_HI_32")
    assert op is not None


def test_k_lo_32_1044():
    op = get_op("K_LO_32")
    assert op is not None


def test_kaiserwindow_1045():
    op = get_op("KaiserWindow")
    assert op is not None


def test_kerneltype_1046():
    op = get_op("KernelType")
    assert op is not None


def test_keydata_1047():
    op = get_op("KeyData")
    assert op is not None


def test_keyentry_1048():
    op = get_op("KeyEntry")
    assert op is not None


def test_keyimpl_1049():
    op = get_op("KeyImpl")
    assert op is not None


def test_keylessinitializer_1050():
    op = get_op("KeylessInitializer")
    assert op is not None


def test_kldiv_1051():
    op = get_op("Kldiv")
    assert op is not None


def test_kthvalue_1052():
    op = get_op("KthValue")
    assert op is not None


def test_l1_1053():
    op = get_op("L1")
    assert op is not None


def test_l1loss_1054():
    op = get_op("L1Loss")
    assert op is not None


def test_l1unstructured_1055():
    op = get_op("L1Unstructured")
    assert op is not None


def test_l2normalize_1056():
    op = get_op("L2Normalize")
    assert op is not None


def test_lbeta_1057():
    op = get_op("LBeta")
    assert op is not None


def test_loc_regex_1058():
    op = get_op("LOC_REGEX")
    assert op is not None


def test_lower_right_1059():
    op = get_op("LOWER_RIGHT")
    assert op is not None


def test_lppool1d_1060():
    op = get_op("LPPool1d")
    assert op is not None


def test_lppool2d_1061():
    op = get_op("LPPool2d")
    assert op is not None


def test_lppool3d_1062():
    op = get_op("LPPool3d")
    assert op is not None


def test_lstm_1063():
    op = get_op("LSTM")
    assert op is not None


def test_lstmcell_1064():
    op = get_op("LSTMCell")
    assert op is not None


def test_lusolve_1065():
    op = get_op("LUSolve")
    assert op is not None


def test_lamb_1066():
    op = get_op("Lamb")
    assert op is not None


def test_lambda_1067():
    op = get_op("Lambda")
    assert op is not None


def test_lane_1068():
    op = get_op("Lane")
    assert op is not None


def test_languagemodel_1069():
    op = get_op("LanguageModel")
    assert op is not None


def test_languagemodelcontinuousbatching_1070():
    op = get_op("LanguageModelContinuousBatching")
    assert op is not None


def test_languagemodeldpo_1071():
    op = get_op("LanguageModelDPO")
    assert op is not None


def test_languagemodeltype_1072():
    op = get_op("LanguageModelType")
    assert op is not None


def test_laplace_1073():
    op = get_op("Laplace")
    assert op is not None


def test_launchcontext_1074():
    op = get_op("LaunchContext")
    assert op is not None


def test_layer_1075():
    op = get_op("Layer")
    assert op is not None


def test_layernorm_1076():
    op = get_op("LayerNorm")
    assert op is not None


def test_layernormalization_1077():
    op = get_op("LayerNormalization")
    assert op is not None


def test_layernormalizedlstmcellsimple_1078():
    op = get_op("LayerNormalizedLstmCellSimple")
    assert op is not None


def test_layerwrapper_1079():
    op = get_op("LayerWrapper")
    assert op is not None


def test_layerwiseshardablepipelined_1080():
    op = get_op("LayerwiseShardablePipelined")
    assert op is not None


def test_layout_1081():
    op = get_op("Layout")
    assert op is not None


def test_lazybatchnorm1d_1082():
    op = get_op("LazyBatchNorm1d")
    assert op is not None


def test_lazybatchnorm2d_1083():
    op = get_op("LazyBatchNorm2d")
    assert op is not None


def test_lazybatchnorm3d_1084():
    op = get_op("LazyBatchNorm3d")
    assert op is not None


def test_lazyconv1d_1085():
    op = get_op("LazyConv1d")
    assert op is not None


def test_lazyconv2d_1086():
    op = get_op("LazyConv2d")
    assert op is not None


def test_lazyconv3d_1087():
    op = get_op("LazyConv3d")
    assert op is not None


def test_lazyconvtranspose1d_1088():
    op = get_op("LazyConvTranspose1d")
    assert op is not None


def test_lazyconvtranspose2d_1089():
    op = get_op("LazyConvTranspose2d")
    assert op is not None


def test_lazyconvtranspose3d_1090():
    op = get_op("LazyConvTranspose3d")
    assert op is not None


def test_lazyinstancenorm1d_1091():
    op = get_op("LazyInstanceNorm1d")
    assert op is not None


def test_lazyinstancenorm2d_1092():
    op = get_op("LazyInstanceNorm2d")
    assert op is not None


def test_lazyinstancenorm3d_1093():
    op = get_op("LazyInstanceNorm3d")
    assert op is not None


def test_lazylinear_1094():
    op = get_op("LazyLinear")
    assert op is not None


def test_lazymodulemixin_1095():
    op = get_op("LazyModuleMixin")
    assert op is not None


def test_lcm__1096():
    op = get_op("Lcm_")
    assert op is not None


def test_ldexp__1097():
    op = get_op("Ldexp_")
    assert op is not None


def test_leafattr_1098():
    op = get_op("LeafAttr")
    assert op is not None


def test_leakyrelu_1099():
    op = get_op("LeakyReLU")
    assert op is not None


def test_leftshift_1100():
    op = get_op("LeftShift")
    assert op is not None


def test_legacycontiguousformat_1101():
    op = get_op("LegacyContiguousFormat")
    assert op is not None


def test_lerp_1102():
    op = get_op("Lerp")
    assert op is not None


def test_less_1103():
    op = get_op("Less")
    assert op is not None


def test_lessequal_1104():
    op = get_op("LessEqual")
    assert op is not None


def test_liftedmodule_1105():
    op = get_op("LiftedModule")
    assert op is not None


def test_lightconv1d_1106():
    op = get_op("LightConv1D")
    assert op is not None


def test_linalgerror_1107():
    op = get_op("LinAlgError")
    assert op is not None


def test_linear_1108():
    op = get_op("Linear")
    assert op is not None


def test_lineargeneral_1109():
    op = get_op("LinearGeneral")
    assert op is not None


def test_lineart_1110():
    op = get_op("LinearT")
    assert op is not None


def test_lion_1111():
    op = get_op("Lion")
    assert op is not None


def test_list_1112():
    op = get_op("List")
    assert op is not None


def test_listadd_1113():
    op = get_op("ListAdd")
    assert op is not None


def test_listaverage_1114():
    op = get_op("ListAverage")
    assert op is not None


def test_listconcatenate_1115():
    op = get_op("ListConcatenate")
    assert op is not None


def test_listmaximum_1116():
    op = get_op("ListMaximum")
    assert op is not None


def test_listminimum_1117():
    op = get_op("ListMinimum")
    assert op is not None


def test_listmultiply_1118():
    op = get_op("ListMultiply")
    assert op is not None


def test_listsubtract_1119():
    op = get_op("ListSubtract")
    assert op is not None


def test_listtype_1120():
    op = get_op("ListType")
    assert op is not None


def test_litescriptmodule_1121():
    op = get_op("LiteScriptModule")
    assert op is not None


def test_littleendian_1122():
    op = get_op("LittleEndian")
    assert op is not None


def test_lnstructured_1123():
    op = get_op("LnStructured")
    assert op is not None


def test_lora_1124():
    op = get_op("LoRA")
    assert op is not None


def test_loralinear_1125():
    op = get_op("LoRALinear")
    assert op is not None


def test_loraparam_1126():
    op = get_op("LoRAParam")
    assert op is not None


def test_loadtxt_1127():
    op = get_op("LoadTxt")
    assert op is not None


def test_lobpcg_1128():
    op = get_op("LobPCG")
    assert op is not None


def test_localmask_1129():
    op = get_op("LocalMask")
    assert op is not None


def test_localresponsenorm_1130():
    op = get_op("LocalResponseNorm")
    assert op is not None


def test_localselfattention_1131():
    op = get_op("LocalSelfAttention")
    assert op is not None


def test_localselfattentionalibi_1132():
    op = get_op("LocalSelfAttentionAlibi")
    assert op is not None


def test_localselfattentionrelativebias_1133():
    op = get_op("LocalSelfAttentionRelativeBias")
    assert op is not None


def test_localselfattentionxl_1134():
    op = get_op("LocalSelfAttentionXL")
    assert op is not None


def test_location_1135():
    op = get_op("Location")
    assert op is not None


def test_lockinglogger_1136():
    op = get_op("LockingLogger")
    assert op is not None


def test_log_1137():
    op = get_op("Log")
    assert op is not None


def test_log10__1138():
    op = get_op("Log10_")
    assert op is not None


def test_log1p__1139():
    op = get_op("Log1p_")
    assert op is not None


def test_log2__1140():
    op = get_op("Log2_")
    assert op is not None


def test_logcumsumexp_1141():
    op = get_op("LogCumSumExp")
    assert op is not None


def test_logsoftmax_1142():
    op = get_op("LogSoftmax")
    assert op is not None


def test_log__1143():
    op = get_op("Log_")
    assert op is not None


def test_logcosh_1144():
    op = get_op("Logcosh")
    assert op is not None


def test_logcumsumexp_1145():
    op = get_op("Logcumsumexp")
    assert op is not None


def test_loggamma_1146():
    op = get_op("Loggamma")
    assert op is not None


def test_loggerbase_1147():
    op = get_op("LoggerBase")
    assert op is not None


def test_logicaland_1148():
    op = get_op("LogicalAnd")
    assert op is not None


def test_logicalaxisrules_1149():
    op = get_op("LogicalAxisRules")
    assert op is not None


def test_logicalnot_1150():
    op = get_op("LogicalNot")
    assert op is not None


def test_logicalor_1151():
    op = get_op("LogicalOr")
    assert op is not None


def test_logicalxor_1152():
    op = get_op("LogicalXor")
    assert op is not None


def test_logistic_1153():
    op = get_op("Logistic")
    assert op is not None


def test_logit__1154():
    op = get_op("Logit_")
    assert op is not None


def test_lognormal_1155():
    op = get_op("Lognormal")
    assert op is not None


def test_longlong_1156():
    op = get_op("LongLong")
    assert op is not None


def test_longstorage_1157():
    op = get_op("LongStorage")
    assert op is not None


def test_longtensor_1158():
    op = get_op("LongTensor")
    assert op is not None


def test_lookup_1159():
    op = get_op("Lookup")
    assert op is not None


def test_loop_1160():
    op = get_op("Loop")
    assert op is not None


def test_loss_1161():
    op = get_op("Loss")
    assert op is not None


def test_lossscaleoptimizer_1162():
    op = get_op("LossScaleOptimizer")
    assert op is not None


def test_lowered_1163():
    op = get_op("Lowered")
    assert op is not None


def test_loweringcontext_1164():
    op = get_op("LoweringContext")
    assert op is not None


def test_loweringsemantics_1165():
    op = get_op("LoweringSemantics")
    assert op is not None


def test_lstmcellsimple_1166():
    op = get_op("LstmCellSimple")
    assert op is not None


def test_lstmfrnn_1167():
    op = get_op("LstmFrnn")
    assert op is not None


def test_lstsq_1168():
    op = get_op("Lstsq")
    assert op is not None


def test_lu_1169():
    op = get_op("Lu")
    assert op is not None


def test_lufactor_1170():
    op = get_op("LuFactor")
    assert op is not None


def test_lumatrixinverse_1171():
    op = get_op("LuMatrixInverse")
    assert op is not None


def test_lureconstruct_1172():
    op = get_op("LuReconstruct")
    assert op is not None


def test_lusolve_1173():
    op = get_op("LuSolve")
    assert op is not None


def test_luunpack_1174():
    op = get_op("LuUnpack")
    assert op is not None


def test_lutfn_1175():
    op = get_op("LutFn")
    assert op is not None


def test_m_1176():
    op = get_op("M")
    assert op is not None


def test_ma_1177():
    op = get_op("MA")
    assert op is not None


def test_mask_value_1178():
    op = get_op("MASK_VALUE")
    assert op is not None


def test_matmul_tol_1179():
    op = get_op("MATMUL_TOL")
    assert op is not None


def test_max_int8_1180():
    op = get_op("MAX_INT8")
    assert op is not None


def test_max_pages_per_seq_1181():
    op = get_op("MAX_PAGES_PER_SEQ")
    assert op is not None


def test_mbarrier_bytes_1182():
    op = get_op("MBARRIER_BYTES")
    assert op is not None


def test_metadata_regex_1183():
    op = get_op("METADATA_REGEX")
    assert op is not None


def test_min_block_size_1184():
    op = get_op("MIN_BLOCK_SIZE")
    assert op is not None


def test_mlpblock_1185():
    op = get_op("MLPBlock")
    assert op is not None


def test_mmalayouts_1186():
    op = get_op("MMALayouts")
    assert op is not None


def test_mn_1187():
    op = get_op("MN")
    assert op is not None


def test_mosaic_gpu_smem_alloc_attr_1188():
    op = get_op("MOSAIC_GPU_SMEM_ALLOC_ATTR")
    assert op is not None


def test_mseloss_1189():
    op = get_op("MSELoss")
    assert op is not None


def test_mul_a_1190():
    op = get_op("MUL_A")
    assert op is not None


def test_mul_b_1191():
    op = get_op("MUL_B")
    assert op is not None


def test_makeattentionmask_1192():
    op = get_op("MakeAttentionMask")
    assert op is not None


def test_makecausalmask_1193():
    op = get_op("MakeCausalMask")
    assert op is not None


def test_manualseed_1194():
    op = get_op("ManualSeed")
    assert op is not None


def test_map_1195():
    op = get_op("Map")
    assert op is not None


def test_mapstate_1196():
    op = get_op("MapState")
    assert op is not None


def test_maptracer_1197():
    op = get_op("MapTracer")
    assert op is not None


def test_mappingreprmixin_1198():
    op = get_op("MappingReprMixin")
    assert op is not None


def test_marginrankingloss_1199():
    op = get_op("MarginRankingLoss")
    assert op is not None


def test_marginranking_1200():
    op = get_op("Marginranking")
    assert op is not None


def test_maskfunctiontype_1201():
    op = get_op("MaskFunctionType")
    assert op is not None


def test_maskindices_1202():
    op = get_op("MaskIndices")
    assert op is not None


def test_maskinfo_1203():
    op = get_op("MaskInfo")
    assert op is not None


def test_maskedfill_1204():
    op = get_op("MaskedFill")
    assert op is not None


def test_maskedlmdataaugmenter_1205():
    op = get_op("MaskedLmDataAugmenter")
    assert op is not None


def test_maskednode_1206():
    op = get_op("MaskedNode")
    assert op is not None


def test_maskedscatter_1207():
    op = get_op("MaskedScatter")
    assert op is not None


def test_maskedselect_1208():
    op = get_op("MaskedSelect")
    assert op is not None


def test_maskedstate_1209():
    op = get_op("MaskedState")
    assert op is not None


def test_masking_1210():
    op = get_op("Masking")
    assert op is not None


def test_matmul_1211():
    op = get_op("MatMul")
    assert op is not None


def test_matmul_1212():
    op = get_op("Matmul")
    assert op is not None


def test_matmuldimension_1213():
    op = get_op("MatmulDimension")
    assert op is not None


def test_matrix_1214():
    op = get_op("Matrix")
    assert op is not None


def test_matrixexp_1215():
    op = get_op("MatrixExp")
    assert op is not None


def test_matrixexponential_1216():
    op = get_op("MatrixExponential")
    assert op is not None


def test_matrixnorm_1217():
    op = get_op("MatrixNorm")
    assert op is not None


def test_matrixpower_1218():
    op = get_op("MatrixPower")
    assert op is not None


def test_matrixrank_1219():
    op = get_op("MatrixRank")
    assert op is not None


def test_matrixtranspose_1220():
    op = get_op("MatrixTranspose")
    assert op is not None


def test_max_1221():
    op = get_op("Max")
    assert op is not None


def test_maxnumboundingboxes_1222():
    op = get_op("MaxNumBoundingBoxes")
    assert op is not None


def test_maxpool1d_1223():
    op = get_op("MaxPool1D")
    assert op is not None


def test_maxpool1dwithindices_1224():
    op = get_op("MaxPool1dWithIndices")
    assert op is not None


def test_maxpool2d_1225():
    op = get_op("MaxPool2D")
    assert op is not None


def test_maxpool3d_1226():
    op = get_op("MaxPool3D")
    assert op is not None


def test_maxpooling1d_1227():
    op = get_op("MaxPooling1d")
    assert op is not None


def test_maxpooling2d_1228():
    op = get_op("MaxPooling2d")
    assert op is not None


def test_maxpooling3d_1229():
    op = get_op("MaxPooling3d")
    assert op is not None


def test_maxunpool1d_1230():
    op = get_op("MaxUnpool1d")
    assert op is not None


def test_maxunpool2d_1231():
    op = get_op("MaxUnpool2d")
    assert op is not None


def test_maxunpool3d_1232():
    op = get_op("MaxUnpool3d")
    assert op is not None


def test_maximum_1233():
    op = get_op("Maximum")
    assert op is not None


def test_maxpool_1234():
    op = get_op("Maxpool")
    assert op is not None


def test_maxwell_1235():
    op = get_op("Maxwell")
    assert op is not None


def test_maysharememory_1236():
    op = get_op("MayShareMemory")
    assert op is not None


def test_mdct_1237():
    op = get_op("Mdct")
    assert op is not None


def test_mean_1238():
    op = get_op("Mean")
    assert op is not None


def test_melspectrogram_1239():
    op = get_op("MelSpectrogram")
    assert op is not None


def test_memreftransform_1240():
    op = get_op("MemRefTransform")
    assert op is not None


def test_memoryformat_1241():
    op = get_op("MemoryFormat")
    assert op is not None


def test_memoryspace_1242():
    op = get_op("MemorySpace")
    assert op is not None


def test_mergecontext_1243():
    op = get_op("MergeContext")
    assert op is not None


def test_mergestate_1244():
    op = get_op("MergeState")
    assert op is not None


def test_mergetypefromtypecomment_1245():
    op = get_op("MergeTypeFromTypeComment")
    assert op is not None


def test_meshaxisname_1246():
    op = get_op("MeshAxisName")
    assert op is not None


def test_meshcomputation_1247():
    op = get_op("MeshComputation")
    assert op is not None


def test_meshexecutable_1248():
    op = get_op("MeshExecutable")
    assert op is not None


def test_metric_1249():
    op = get_op("Metric")
    assert op is not None


def test_metricstate_1250():
    op = get_op("MetricState")
    assert op is not None


def test_mgrid_1251():
    op = get_op("Mgrid")
    assert op is not None


def test_min_1252():
    op = get_op("Min")
    assert op is not None


def test_minpool_1253():
    op = get_op("MinPool")
    assert op is not None


def test_minscalartype_1254():
    op = get_op("MinScalarType")
    assert op is not None


def test_mintypecode_1255():
    op = get_op("MinTypeCode")
    assert op is not None


def test_minversion_1256():
    op = get_op("MinVersion")
    assert op is not None


def test_minimum_1257():
    op = get_op("Minimum")
    assert op is not None


def test_miopenbatchnorm_1258():
    op = get_op("MiopenBatchNorm")
    assert op is not None


def test_miopenconvolution_1259():
    op = get_op("MiopenConvolution")
    assert op is not None


def test_miopenconvolutionaddrelu_1260():
    op = get_op("MiopenConvolutionAddRelu")
    assert op is not None


def test_miopenconvolutionrelu_1261():
    op = get_op("MiopenConvolutionRelu")
    assert op is not None


def test_miopenconvolutiontranspose_1262():
    op = get_op("MiopenConvolutionTranspose")
    assert op is not None


def test_miopendepthwiseconvolution_1263():
    op = get_op("MiopenDepthwiseConvolution")
    assert op is not None


def test_miopenrnn_1264():
    op = get_op("MiopenRnn")
    assert op is not None


def test_mismatchcapierror_1265():
    op = get_op("MismatchCAPIError")
    assert op is not None


def test_mixup_1266():
    op = get_op("MixUp")
    assert op is not None


def test_mkldnnadaptiveavgpool2d_1267():
    op = get_op("MkldnnAdaptiveAvgPool2d")
    assert op is not None


def test_mkldnnconvolution_1268():
    op = get_op("MkldnnConvolution")
    assert op is not None


def test_mkldnnlinearbackwardweights_1269():
    op = get_op("MkldnnLinearBackwardWeights")
    assert op is not None


def test_mkldnnmaxpool2d_1270():
    op = get_op("MkldnnMaxPool2d")
    assert op is not None


def test_mkldnnmaxpool3d_1271():
    op = get_op("MkldnnMaxPool3d")
    assert op is not None


def test_mkldnnrnnlayer_1272():
    op = get_op("MkldnnRnnLayer")
    assert op is not None


def test_mlirloweringrule_1273():
    op = get_op("MlirLoweringRule")
    assert op is not None


def test_mlirloweringruleresult_1274():
    op = get_op("MlirLoweringRuleResult")
    assert op is not None


def test_mliroperation_1275():
    op = get_op("MlirOperation")
    assert op is not None


def test_mm_1276():
    op = get_op("Mm")
    assert op is not None


def test_model_1277():
    op = get_op("Model")
    assert op is not None


def test_modelandoptimizer_1278():
    op = get_op("ModelAndOptimizer")
    assert op is not None


def test_modulebase_1279():
    op = get_op("ModuleBase")
    assert op is not None


def test_modulecontext_1280():
    op = get_op("ModuleContext")
    assert op is not None


def test_moduledefapply_1281():
    op = get_op("ModuleDefApply")
    assert op is not None


def test_moduledict_1282():
    op = get_op("ModuleDict")
    assert op is not None


def test_modulelist_1283():
    op = get_op("ModuleList")
    assert op is not None


def test_modulemeta_1284():
    op = get_op("ModuleMeta")
    assert op is not None


def test_modulestackentry_1285():
    op = get_op("ModuleStackEntry")
    assert op is not None


def test_modulestate_1286():
    op = get_op("ModuleState")
    assert op is not None


def test_movedim_1287():
    op = get_op("Movedim")
    assert op is not None


def test_mse_1288():
    op = get_op("Mse")
    assert op is not None


def test_msort_1289():
    op = get_op("Msort")
    assert op is not None


def test_multidot_1290():
    op = get_op("MultiDot")
    assert op is not None


def test_multiheadmask_1291():
    op = get_op("MultiHeadMask")
    assert op is not None


def test_multilabelmarginloss_1292():
    op = get_op("MultiLabelMarginLoss")
    assert op is not None


def test_multilabelsoftmarginloss_1293():
    op = get_op("MultiLabelSoftMarginLoss")
    assert op is not None


def test_multimarginloss_1294():
    op = get_op("MultiMarginLoss")
    assert op is not None


def test_multimetric_1295():
    op = get_op("MultiMetric")
    assert op is not None


def test_multipageasynccopydescriptor_1296():
    op = get_op("MultiPageAsyncCopyDescriptor")
    assert op is not None


def test_multisteps_1297():
    op = get_op("MultiSteps")
    assert op is not None


def test_multistepsstate_1298():
    op = get_op("MultiStepsState")
    assert op is not None


def test_multitransformstate_1299():
    op = get_op("MultiTransformState")
    assert op is not None


def test_multiheadattention_1300():
    op = get_op("MultiheadAttention")
    assert op is not None


def test_multimemreductionop_1301():
    op = get_op("MultimemReductionOp")
    assert op is not None


def test_multimemref_1302():
    op = get_op("MultimemRef")
    assert op is not None


def test_multinomial_1303():
    op = get_op("Multinomial")
    assert op is not None


def test_multiply_1304():
    op = get_op("Multiply")
    assert op is not None


def test_multiplynonan_1305():
    op = get_op("MultiplyNoNan")
    assert op is not None


def test_multitaskresidualadapter_1306():
    op = get_op("MultitaskResidualAdapter")
    assert op is not None


def test_multivariatenormal_1307():
    op = get_op("MultivariateNormal")
    assert op is not None


def test_muon_1308():
    op = get_op("Muon")
    assert op is not None


def test_mutablearrayrepr_1309():
    op = get_op("MutableArrayRepr")
    assert op is not None


def test_mv_1310():
    op = get_op("Mv")
    assert op is not None


def test_mvlgamma_1311():
    op = get_op("Mvlgamma")
    assert op is not None


def test_natype_1312():
    op = get_op("NAType")
    assert op is not None


def test_ndindexer_1313():
    op = get_op("NDIndexer")
    assert op is not None


def test_nllloss_1314():
    op = get_op("NLLLoss")
    assert op is not None


def test_nllloss2d_1315():
    op = get_op("NLLLoss2d")
    assert op is not None


def test_nnxmeta_1316():
    op = get_op("NNXMeta")
    assert op is not None


def test_nn_dim_numbers_1317():
    op = get_op("NN_DIM_NUMBERS")
    assert op is not None


def test_none_1318():
    op = get_op("NONE")
    assert op is not None


def test_no_swizzle_1319():
    op = get_op("NO_SWIZZLE")
    assert op is not None


def test_nt_dim_numbers_1320():
    op = get_op("NT_DIM_NUMBERS")
    assert op is not None


def test_num_lanes_1321():
    op = get_op("NUM_LANES")
    assert op is not None


def test_num_sublanes_1322():
    op = get_op("NUM_SUBLANES")
    assert op is not None


def test_nadam_1323():
    op = get_op("Nadam")
    assert op is not None


def test_namestack_1324():
    op = get_op("NameStack")
    assert op is not None


def test_namedshape_1325():
    op = get_op("NamedShape")
    assert op is not None


def test_names_1326():
    op = get_op("Names")
    assert op is not None


def test_nantonum_1327():
    op = get_op("NanToNum")
    assert op is not None


def test_nantonum__1328():
    op = get_op("NanToNum_")
    assert op is not None


def test_nanosleep_1329():
    op = get_op("Nanosleep")
    assert op is not None


def test_narrow_1330():
    op = get_op("Narrow")
    assert op is not None


def test_narrowcopy_1331():
    op = get_op("NarrowCopy")
    assert op is not None


def test_nativebatchnorm_1332():
    op = get_op("NativeBatchNorm")
    assert op is not None


def test_nativechannelshuffle_1333():
    op = get_op("NativeChannelShuffle")
    assert op is not None


def test_nativedropout_1334():
    op = get_op("NativeDropout")
    assert op is not None


def test_nativegroupnorm_1335():
    op = get_op("NativeGroupNorm")
    assert op is not None


def test_nativelayernorm_1336():
    op = get_op("NativeLayerNorm")
    assert op is not None


def test_nativenorm_1337():
    op = get_op("NativeNorm")
    assert op is not None


def test_nativeserializationimpl_1338():
    op = get_op("NativeSerializationImpl")
    assert op is not None


def test_ndenumerate_1339():
    op = get_op("NdEnumerate")
    assert op is not None


def test_ndindex_1340():
    op = get_op("NdIndex")
    assert op is not None


def test_nditer_1341():
    op = get_op("NdIter")
    assert op is not None


def test_negative_1342():
    op = get_op("Negative")
    assert op is not None


def test_negativealias_1343():
    op = get_op("NegativeAlias")
    assert op is not None


def test_negativealias__1344():
    op = get_op("NegativeAlias_")
    assert op is not None


def test_negative__1345():
    op = get_op("Negative_")
    assert op is not None


def test_nestediters_1346():
    op = get_op("NestedIters")
    assert op is not None


def test_nestedstaterepr_1347():
    op = get_op("NestedStateRepr")
    assert op is not None


def test_newaxis_1348():
    op = get_op("NewAxis")
    assert op is not None


def test_newop_1349():
    op = get_op("NewOp")
    assert op is not None


def test_newvariable_1350():
    op = get_op("NewVariable")
    assert op is not None


def test_ngrammer_1351():
    op = get_op("Ngrammer")
    assert op is not None


def test_nll_1352():
    op = get_op("Nll")
    assert op is not None


def test_nodceeffect_1353():
    op = get_op("NoDCEEffect")
    assert op is not None


def test_nograd_1354():
    op = get_op("NoGrad")
    assert op is not None


def test_nosharding_1355():
    op = get_op("NoSharding")
    assert op is not None


def test_noupdate_1356():
    op = get_op("NoUpdate")
    assert op is not None


def test_node_1357():
    op = get_op("Node")
    assert op is not None


def test_nodeattr_1358():
    op = get_op("NodeAttr")
    assert op is not None


def test_nodedef_1359():
    op = get_op("NodeDef")
    assert op is not None


def test_nodeimplbase_1360():
    op = get_op("NodeImplBase")
    assert op is not None


def test_noderef_1361():
    op = get_op("NodeRef")
    assert op is not None


def test_nodestates_1362():
    op = get_op("NodeStates")
    assert op is not None


def test_nondynamicallyquantizablelinear_1363():
    op = get_op("NonDynamicallyQuantizableLinear")
    assert op is not None


def test_nonnegativeparamsstate_1364():
    op = get_op("NonNegativeParamsState")
    assert op is not None


def test_nonzerostatic_1365():
    op = get_op("NonZeroStatic")
    assert op is not None


def test_nonedumper_1366():
    op = get_op("NoneDumper")
    assert op is not None


def test_nonetype_1367():
    op = get_op("NoneType")
    assert op is not None


def test_nooplogger_1368():
    op = get_op("NoopLogger")
    assert op is not None


def test_nop_1369():
    op = get_op("Nop")
    assert op is not None


def test_norm_1370():
    op = get_op("Norm")
    assert op is not None


def test_normcdf_1371():
    op = get_op("NormCdf")
    assert op is not None


def test_normexceptdim_1372():
    op = get_op("NormExceptDim")
    assert op is not None


def test_normpdf_1373():
    op = get_op("NormPdf")
    assert op is not None


def test_normal_1374():
    op = get_op("Normal")
    assert op is not None


def test_normalization_1375():
    op = get_op("Normalization")
    assert op is not None


def test_not_1376():
    op = get_op("Not")
    assert op is not None


def test_notequal_1377():
    op = get_op("NotEqual")
    assert op is not None


def test_notequalalias_1378():
    op = get_op("NotEqualAlias")
    assert op is not None


def test_notmapped_1379():
    op = get_op("NotMapped")
    assert op is not None


def test_notoftype_1380():
    op = get_op("NotOfType")
    assert op is not None


def test_nothing_1381():
    op = get_op("Nothing")
    assert op is not None


def test_nuclearnorm_1382():
    op = get_op("NuclearNorm")
    assert op is not None


def test_numbertype_1383():
    op = get_op("NumberType")
    assert op is not None


def test_numpymask_1384():
    op = get_op("NumpyMask")
    assert op is not None


def test_ogrid_1385():
    op = get_op("OGrid")
    assert op is not None


def test_operand_1386():
    op = get_op("OPERAND")
    assert op is not None


def test_optimized_1387():
    op = get_op("OPTIMIZED")
    assert op is not None


def test_original_kernel_arg_attr_1388():
    op = get_op("ORIGINAL_KERNEL_ARG_ATTR")
    assert op is not None


def test_object_1389():
    op = get_op("Object")
    assert op is not None


def test_objectcontext_1390():
    op = get_op("ObjectContext")
    assert op is not None


def test_objectinfo_1391():
    op = get_op("ObjectInfo")
    assert op is not None


def test_oftype_1392():
    op = get_op("OfType")
    assert op is not None


def test_ogrid_1393():
    op = get_op("Ogrid")
    assert op is not None


def test_ondeviceprofiler_1394():
    op = get_op("OnDeviceProfiler")
    assert op is not None


def test_onehot_1395():
    op = get_op("OneHot")
    assert op is not None


def test_ones_1396():
    op = get_op("Ones")
    assert op is not None


def test_oneslike_1397():
    op = get_op("OnesLike")
    assert op is not None


def test_op_1398():
    op = get_op("Op")
    assert op is not None


def test_operatorinfo_1399():
    op = get_op("OperatorInfo")
    assert op is not None


def test_optarray_1400():
    op = get_op("OptArray")
    assert op is not None


def test_optstate_1401():
    op = get_op("OptState")
    assert op is not None


def test_optvariable_1402():
    op = get_op("OptVariable")
    assert op is not None


def test_optaxtest_1403():
    op = get_op("OptaxTest")
    assert op is not None


def test_optimizedlstmcell_1404():
    op = get_op("OptimizedLSTMCell")
    assert op is not None


def test_optimizer_1405():
    op = get_op("Optimizer")
    assert op is not None


def test_optimizerstate_1406():
    op = get_op("OptimizerState")
    assert op is not None


def test_optional_1407():
    op = get_op("Optional")
    assert op is not None


def test_optionaltype_1408():
    op = get_op("OptionalType")
    assert op is not None


def test_ordereddictwrapper_1409():
    op = get_op("OrderedDictWrapper")
    assert op is not None


def test_orgqr_1410():
    op = get_op("OrgQr")
    assert op is not None


def test_ormqr_1411():
    op = get_op("OrmQr")
    assert op is not None


def test_orthogonal_1412():
    op = get_op("Orthogonal")
    assert op is not None


def test_outofmemoryerror_1413():
    op = get_op("OutOfMemoryError")
    assert op is not None


def test_outer_1414():
    op = get_op("Outer")
    assert op is not None


def test_outfeed_1415():
    op = get_op("Outfeed")
    assert op is not None


def test_overlapandadd_1416():
    op = get_op("OverlapAndAdd")
    assert op is not None


def test_prescale_qk_1417():
    op = get_op("PRESCALE_QK")
    assert op is not None


def test_prngkey_1418():
    op = get_op("PRNGKey")
    assert op is not None


def test_pruning_type_1419():
    op = get_op("PRUNING_TYPE")
    assert op is not None


def test_python_runfiles_1420():
    op = get_op("PYTHON_RUNFILES")
    assert op is not None


def test_packedsequence_1421():
    op = get_op("PackedSequence")
    assert op is not None


def test_packedsequence__1422():
    op = get_op("PackedSequence_")
    assert op is not None


def test_padstatfunc_1423():
    op = get_op("PadStatFunc")
    assert op is not None


def test_paddingtype_1424():
    op = get_op("PaddingType")
    assert op is not None


def test_pairwisedistance_1425():
    op = get_op("PairwiseDistance")
    assert op is not None


def test_param_1426():
    op = get_op("Param")
    assert op is not None


def test_parameter_1427():
    op = get_op("Parameter")
    assert op is not None


def test_parameterdict_1428():
    op = get_op("ParameterDict")
    assert op is not None


def test_parameterlist_1429():
    op = get_op("ParameterList")
    assert op is not None


def test_paramsfn_1430():
    op = get_op("ParamsFn")
    assert op is not None


def test_pareto_1431():
    op = get_op("Pareto")
    assert op is not None


def test_parseerror_1432():
    op = get_op("ParseError")
    assert op is not None


def test_parseir_1433():
    op = get_op("ParseIr")
    assert op is not None


def test_parseschema_1434():
    op = get_op("ParseSchema")
    assert op is not None


def test_parsetypecomment_1435():
    op = get_op("ParseTypeComment")
    assert op is not None


def test_parsedindex_1436():
    op = get_op("ParsedIndex")
    assert op is not None


def test_partialstate_1437():
    op = get_op("PartialState")
    assert op is not None


def test_partition1d_1438():
    op = get_op("Partition1D")
    assert op is not None


def test_partitionname_1439():
    op = get_op("PartitionName")
    assert op is not None


def test_partitionstate_1440():
    op = get_op("PartitionState")
    assert op is not None


def test_partitionsorreplicated_1441():
    op = get_op("PartitionsOrReplicated")
    assert op is not None


def test_pass_1442():
    op = get_op("Pass")
    assert op is not None


def test_pathcontains_1443():
    op = get_op("PathContains")
    assert op is not None


def test_pathin_1444():
    op = get_op("PathIn")
    assert op is not None


def test_pcalowrank_1445():
    op = get_op("PcaLowRank")
    assert op is not None


def test_pdot_1446():
    op = get_op("Pdot")
    assert op is not None


def test_perchannelaffine_1447():
    op = get_op("PerChannelAffine")
    assert op is not None


def test_perchannelaffinefloatqparams_1448():
    op = get_op("PerChannelAffineFloatQParams")
    assert op is not None


def test_perchannelsymmetric_1449():
    op = get_op("PerChannelSymmetric")
    assert op is not None


def test_perdimscale_1450():
    op = get_op("PerDimScale")
    assert op is not None


def test_pertensoraffine_1451():
    op = get_op("PerTensorAffine")
    assert op is not None


def test_pertensorsymmetric_1452():
    op = get_op("PerTensorSymmetric")
    assert op is not None


def test_permute_1453():
    op = get_op("Permute")
    assert op is not None


def test_permutecopy_1454():
    op = get_op("PermuteCopy")
    assert op is not None


def test_permutedims_1455():
    op = get_op("PermuteDims")
    assert op is not None


def test_perturbation_1456():
    op = get_op("Perturbation")
    assert op is not None


def test_picklemodule_1457():
    op = get_op("PickleModule")
    assert op is not None


def test_pinv_1458():
    op = get_op("Pinv")
    assert op is not None


def test_pinverse_1459():
    op = get_op("Pinverse")
    assert op is not None


def test_pipeline_1460():
    op = get_op("Pipeline")
    assert op is not None


def test_pipelinedtransformer_1461():
    op = get_op("PipelinedTransformer")
    assert op is not None


def test_pixelshuffle_1462():
    op = get_op("PixelShuffle")
    assert op is not None


def test_pixelunshuffle_1463():
    op = get_op("PixelUnshuffle")
    assert op is not None


def test_pmap_1464():
    op = get_op("Pmap")
    assert op is not None


def test_pmapexecutable_1465():
    op = get_op("PmapExecutable")
    assert op is not None


def test_pmapfn_1466():
    op = get_op("PmapFn")
    assert op is not None


def test_pmapsharding_1467():
    op = get_op("PmapSharding")
    assert op is not None


def test_pmax_1468():
    op = get_op("Pmax")
    assert op is not None


def test_pmean_1469():
    op = get_op("Pmean")
    assert op is not None


def test_pmin_1470():
    op = get_op("Pmin")
    assert op is not None


def test_poisson_1471():
    op = get_op("Poisson")
    assert op is not None


def test_poissoncdf_1472():
    op = get_op("PoissonCdf")
    assert op is not None


def test_poissonnllloss_1473():
    op = get_op("PoissonNLLLoss")
    assert op is not None


def test_poissonpmf_1474():
    op = get_op("PoissonPmf")
    assert op is not None


def test_poly_1475():
    op = get_op("Poly")
    assert op is not None


def test_poly1d_1476():
    op = get_op("Poly1d")
    assert op is not None


def test_polyshape_1477():
    op = get_op("PolyShape")
    assert op is not None


def test_polyadd_1478():
    op = get_op("Polyadd")
    assert op is not None


def test_polyder_1479():
    op = get_op("Polyder")
    assert op is not None


def test_polydiv_1480():
    op = get_op("Polydiv")
    assert op is not None


def test_polyfit_1481():
    op = get_op("Polyfit")
    assert op is not None


def test_polygamma_1482():
    op = get_op("Polygamma")
    assert op is not None


def test_polyint_1483():
    op = get_op("Polyint")
    assert op is not None


def test_polymul_1484():
    op = get_op("Polymul")
    assert op is not None


def test_polysub_1485():
    op = get_op("Polysub")
    assert op is not None


def test_polyval_1486():
    op = get_op("Polyval")
    assert op is not None


def test_pool_1487():
    op = get_op("Pool")
    assert op is not None


def test_pooling_1488():
    op = get_op("Pooling")
    assert op is not None


def test_pooling1d_1489():
    op = get_op("Pooling1D")
    assert op is not None


def test_positionalembedding_1490():
    op = get_op("PositionalEmbedding")
    assert op is not None


def test_positionalembedding2d_1491():
    op = get_op("PositionalEmbedding2D")
    assert op is not None


def test_power_1492():
    op = get_op("Power")
    assert op is not None


def test_ppermute_1493():
    op = get_op("Ppermute")
    assert op is not None


def test_precision_1494():
    op = get_op("Precision")
    assert op is not None


def test_precisiontype_1495():
    op = get_op("PrecisionType")
    assert op is not None


def test_prefixmapping_1496():
    op = get_op("PrefixMapping")
    assert op is not None


def test_preparemultiprocessingenvironment_1497():
    op = get_op("PrepareMultiprocessingEnvironment")
    assert op is not None


def test_preserveformat_1498():
    op = get_op("PreserveFormat")
    assert op is not None


def test_prettymapping_1499():
    op = get_op("PrettyMapping")
    assert op is not None


def test_prettysequence_1500():
    op = get_op("PrettySequence")
    assert op is not None


def test_prioritystr_1501():
    op = get_op("PriorityStr")
    assert op is not None


def test_privateops_1502():
    op = get_op("PrivateOps")
    assert op is not None


def test_prmt_1503():
    op = get_op("Prmt")
    assert op is not None


def test_profilerallowcudagraphcuptilazyreinitcuda12_1504():
    op = get_op("ProfilerAllowCudagraphCuptiLazyReinitCuda12")
    assert op is not None


def test_profilerspec_1505():
    op = get_op("ProfilerSpec")
    assert op is not None


def test_projectlastdim_1506():
    op = get_op("ProjectLastDim")
    assert op is not None


def test_promotetypes_1507():
    op = get_op("PromoteTypes")
    assert op is not None


def test_pruningcontainer_1508():
    op = get_op("PruningContainer")
    assert op is not None


def test_pshuffle_1509():
    op = get_op("Pshuffle")
    assert op is not None


def test_psum_1510():
    op = get_op("Psum")
    assert op is not None


def test_psumscatter_1511():
    op = get_op("PsumScatter")
    assert op is not None


def test_pswapaxes_1512():
    op = get_op("Pswapaxes")
    assert op is not None


def test_purestate_1513():
    op = get_op("PureState")
    assert op is not None


def test_putalongaxis_1514():
    op = get_op("PutAlongAxis")
    assert op is not None


def test_pyobjecttype_1515():
    op = get_op("PyObjectType")
    assert op is not None


def test_pytorchfilereader_1516():
    op = get_op("PyTorchFileReader")
    assert op is not None


def test_pytorchfilewriter_1517():
    op = get_op("PyTorchFileWriter")
    assert op is not None


def test_pytreefuture_1518():
    op = get_op("PyTreeFuture")
    assert op is not None


def test_pytreet_1519():
    op = get_op("PyTreeT")
    assert op is not None


def test_pytree_1520():
    op = get_op("Pytree")
    assert op is not None


def test_pytreemeta_1521():
    op = get_op("PytreeMeta")
    assert op is not None


def test_pytreenodeimpl_1522():
    op = get_op("PytreeNodeImpl")
    assert op is not None


def test_pytreestate_1523():
    op = get_op("PytreeState")
    assert op is not None


def test_qfunctional_1524():
    op = get_op("QFunctional")
    assert op is not None


def test_qint32_1525():
    op = get_op("QInt32")
    assert op is not None


def test_qint32storage_1526():
    op = get_op("QInt32Storage")
    assert op is not None


def test_qint8_1527():
    op = get_op("QInt8")
    assert op is not None


def test_qint8storage_1528():
    op = get_op("QInt8Storage")
    assert op is not None


def test_qkvlayout_1529():
    op = get_op("QKVLayout")
    assert op is not None


def test_qperchannelaxis_1530():
    op = get_op("QPerChannelAxis")
    assert op is not None


def test_qperchannelscales_1531():
    op = get_op("QPerChannelScales")
    assert op is not None


def test_qperchannelzeropoints_1532():
    op = get_op("QPerChannelZeroPoints")
    assert op is not None


def test_qqlinear_1533():
    op = get_op("QQLinear")
    assert op is not None


def test_qr_1534():
    op = get_op("QR")
    assert op is not None


def test_qrresult_1535():
    op = get_op("QRResult")
    assert op is not None


def test_qscale_1536():
    op = get_op("QScale")
    assert op is not None


def test_qscheme_1537():
    op = get_op("QScheme")
    assert op is not None


def test_quint2x4storage_1538():
    op = get_op("QUInt2x4Storage")
    assert op is not None


def test_quint4x2storage_1539():
    op = get_op("QUInt4x2Storage")
    assert op is not None


def test_quint8storage_1540():
    op = get_op("QUInt8Storage")
    assert op is not None


def test_qzeropoint_1541():
    op = get_op("QZeroPoint")
    assert op is not None


def test_qqmm_1542():
    op = get_op("Qqmm")
    assert op is not None


def test_qr_1543():
    op = get_op("Qr")
    assert op is not None


def test_quantizeperchannel_1544():
    op = get_op("QuantizePerChannel")
    assert op is not None


def test_quantizepertensor_1545():
    op = get_op("QuantizePerTensor")
    assert op is not None


def test_quantizepertensordynamic_1546():
    op = get_op("QuantizePerTensorDynamic")
    assert op is not None


def test_quantizevector_1547():
    op = get_op("QuantizeVector")
    assert op is not None


def test_quantizedalltoshardedlinear_1548():
    op = get_op("QuantizedAllToShardedLinear")
    assert op is not None


def test_quantizedbatchnorm_1549():
    op = get_op("QuantizedBatchNorm")
    assert op is not None


def test_quantizedembedding_1550():
    op = get_op("QuantizedEmbedding")
    assert op is not None


def test_quantizedgru_1551():
    op = get_op("QuantizedGru")
    assert op is not None


def test_quantizedgrucell_1552():
    op = get_op("QuantizedGruCell")
    assert op is not None


def test_quantizedlinear_1553():
    op = get_op("QuantizedLinear")
    assert op is not None


def test_quantizedlstm_1554():
    op = get_op("QuantizedLstm")
    assert op is not None


def test_quantizedlstmcell_1555():
    op = get_op("QuantizedLstmCell")
    assert op is not None


def test_quantizedmatmul_1556():
    op = get_op("QuantizedMatmul")
    assert op is not None


def test_quantizedmaxpool1d_1557():
    op = get_op("QuantizedMaxPool1d")
    assert op is not None


def test_quantizedmaxpool2d_1558():
    op = get_op("QuantizedMaxPool2d")
    assert op is not None


def test_quantizedmaxpool3d_1559():
    op = get_op("QuantizedMaxPool3d")
    assert op is not None


def test_quantizedrnnrelucell_1560():
    op = get_op("QuantizedRnnReluCell")
    assert op is not None


def test_quantizedrnntanhcell_1561():
    op = get_op("QuantizedRnnTanhCell")
    assert op is not None


def test_quantizedshardedtoalllinear_1562():
    op = get_op("QuantizedShardedToAllLinear")
    assert op is not None


def test_quantizedtensor_1563():
    op = get_op("QuantizedTensor")
    assert op is not None


def test_quint2x4_1564():
    op = get_op("Quint2x4")
    assert op is not None


def test_quint4x2_1565():
    op = get_op("Quint4x2")
    assert op is not None


def test_quint8_1566():
    op = get_op("Quint8")
    assert op is not None


def test_r_1567():
    op = get_op("R")
    assert op is not None


def test_rclass_1568():
    op = get_op("RClass")
    assert op is not None


def test_reg_1569():
    op = get_op("REG")
    assert op is not None


def test_regs_1570():
    op = get_op("REGS")
    assert op is not None


def test_rmsnorm_1571():
    op = get_op("RMSNorm")
    assert op is not None


def test_rmsnormalization_1572():
    op = get_op("RMSNormalization")
    assert op is not None


def test_rmsprop_1573():
    op = get_op("RMSprop")
    assert op is not None


def test_rnn_1574():
    op = get_op("RNN")
    assert op is not None


def test_rnnbase_1575():
    op = get_op("RNNBase")
    assert op is not None


def test_rnncell_1576():
    op = get_op("RNNCell")
    assert op is not None


def test_rnncellbase_1577():
    op = get_op("RNNCellBase")
    assert op is not None


def test_rnncelldevicewrapper_1578():
    op = get_op("RNNCellDeviceWrapper")
    assert op is not None


def test_rnncelldropoutwrapper_1579():
    op = get_op("RNNCellDropoutWrapper")
    assert op is not None


def test_rnncellresidualwrapper_1580():
    op = get_op("RNNCellResidualWrapper")
    assert op is not None


def test_rows_guaranteed_safe_1581():
    op = get_op("ROWS_GUARANTEED_SAFE")
    assert op is not None


def test_row_layout_1582():
    op = get_op("ROW_LAYOUT")
    assert op is not None


def test_rpc_available_1583():
    op = get_op("RPC_AVAILABLE")
    assert op is not None


def test_rreftype_1584():
    op = get_op("RRefType")
    assert op is not None


def test_runtime_path_1585():
    op = get_op("RUNTIME_PATH")
    assert op is not None


def test_r__1586():
    op = get_op("R_")
    assert op is not None


def test_rad2deg__1587():
    op = get_op("Rad2Deg_")
    assert op is not None


def test_raggedconstant_1588():
    op = get_op("RaggedConstant")
    assert op is not None


def test_raggedcrosshashed_1589():
    op = get_op("RaggedCrossHashed")
    assert op is not None


def test_raggeddotdimensionnumbers_1590():
    op = get_op("RaggedDotDimensionNumbers")
    assert op is not None


def test_raggeddotmode_1591():
    op = get_op("RaggedDotMode")
    assert op is not None


def test_raggedrange_1592():
    op = get_op("RaggedRange")
    assert op is not None


def test_raggedrowsplitstosegmentids_1593():
    op = get_op("RaggedRowSplitsToSegmentIds")
    assert op is not None


def test_raggedsegmentidstorowsplits_1594():
    op = get_op("RaggedSegmentIdsToRowSplits")
    assert op is not None


def test_raggedstack_1595():
    op = get_op("RaggedStack")
    assert op is not None


def test_raggedstackdynamicpartitions_1596():
    op = get_op("RaggedStackDynamicPartitions")
    assert op is not None


def test_rand_1597():
    op = get_op("Rand")
    assert op is not None


def test_randaugment_1598():
    op = get_op("RandAugment")
    assert op is not None


def test_randint_1599():
    op = get_op("RandInt")
    assert op is not None


def test_randintlike_1600():
    op = get_op("RandIntLike")
    assert op is not None


def test_randlike_1601():
    op = get_op("RandLike")
    assert op is not None


def test_randperm_1602():
    op = get_op("RandPerm")
    assert op is not None


def test_randn_1603():
    op = get_op("Randn")
    assert op is not None


def test_randnlike_1604():
    op = get_op("RandnLike")
    assert op is not None


def test_randomalgorithm_1605():
    op = get_op("RandomAlgorithm")
    assert op is not None


def test_randombernoulli_1606():
    op = get_op("RandomBernoulli")
    assert op is not None


def test_randombrightness_1607():
    op = get_op("RandomBrightness")
    assert op is not None


def test_randomcategorical_1608():
    op = get_op("RandomCategorical")
    assert op is not None


def test_randomcolordegeneration_1609():
    op = get_op("RandomColorDegeneration")
    assert op is not None


def test_randomcolorjitter_1610():
    op = get_op("RandomColorJitter")
    assert op is not None


def test_randomcontrast_1611():
    op = get_op("RandomContrast")
    assert op is not None


def test_randomcrop_1612():
    op = get_op("RandomCrop")
    assert op is not None


def test_randomelastictransform_1613():
    op = get_op("RandomElasticTransform")
    assert op is not None


def test_randomerasing_1614():
    op = get_op("RandomErasing")
    assert op is not None


def test_randomflip_1615():
    op = get_op("RandomFlip")
    assert op is not None


def test_randomgammagrad_1616():
    op = get_op("RandomGammaGrad")
    assert op is not None


def test_randomgammap_1617():
    op = get_op("RandomGammaP")
    assert op is not None


def test_randomgaussianblur_1618():
    op = get_op("RandomGaussianBlur")
    assert op is not None


def test_randomgrayscale_1619():
    op = get_op("RandomGrayscale")
    assert op is not None


def test_randomhorizontalflip_1620():
    op = get_op("RandomHorizontalFlip")
    assert op is not None


def test_randomhue_1621():
    op = get_op("RandomHue")
    assert op is not None


def test_randominvert_1622():
    op = get_op("RandomInvert")
    assert op is not None


def test_randomnormal_1623():
    op = get_op("RandomNormal")
    assert op is not None


def test_randomperspective_1624():
    op = get_op("RandomPerspective")
    assert op is not None


def test_randomposterization_1625():
    op = get_op("RandomPosterization")
    assert op is not None


def test_randomrandint_1626():
    op = get_op("RandomRandint")
    assert op is not None


def test_randomrotation_1627():
    op = get_op("RandomRotation")
    assert op is not None


def test_randomsaturation_1628():
    op = get_op("RandomSaturation")
    assert op is not None


def test_randomsharpness_1629():
    op = get_op("RandomSharpness")
    assert op is not None


def test_randomshear_1630():
    op = get_op("RandomShear")
    assert op is not None


def test_randomsplit_1631():
    op = get_op("RandomSplit")
    assert op is not None


def test_randomstructured_1632():
    op = get_op("RandomStructured")
    assert op is not None


def test_randomtranslation_1633():
    op = get_op("RandomTranslation")
    assert op is not None


def test_randomuniform_1634():
    op = get_op("RandomUniform")
    assert op is not None


def test_randomunstructured_1635():
    op = get_op("RandomUnstructured")
    assert op is not None


def test_randomvectorquantizer_1636():
    op = get_op("RandomVectorQuantizer")
    assert op is not None


def test_randomverticalflip_1637():
    op = get_op("RandomVerticalFlip")
    assert op is not None


def test_randomzoom_1638():
    op = get_op("RandomZoom")
    assert op is not None


def test_range_1639():
    op = get_op("Range")
    assert op is not None


def test_ravelmultiindex_1640():
    op = get_op("RavelMultiIndex")
    assert op is not None


def test_rayleigh_1641():
    op = get_op("Rayleigh")
    assert op is not None


def test_readvariable_1642():
    op = get_op("ReadVariable")
    assert op is not None


def test_readvitals_1643():
    op = get_op("ReadVitals")
    assert op is not None


def test_realifclose_1644():
    op = get_op("RealIfClose")
    assert op is not None


def test_reciprocalnonan_1645():
    op = get_op("ReciprocalNoNan")
    assert op is not None


def test_reciprocal__1646():
    op = get_op("Reciprocal_")
    assert op is not None


def test_recursed_1647():
    op = get_op("Recursed")
    assert op is not None


def test_recursivemap_1648():
    op = get_op("RecursiveMap")
    assert op is not None


def test_redirect_1649():
    op = get_op("Redirect")
    assert op is not None


def test_reduceeuclideannorm_1650():
    op = get_op("ReduceEuclideanNorm")
    assert op is not None


def test_reducemax_1651():
    op = get_op("ReduceMax")
    assert op is not None


def test_reducemean_1652():
    op = get_op("ReduceMean")
    assert op is not None


def test_reducemin_1653():
    op = get_op("ReduceMin")
    assert op is not None


def test_reduceprod_1654():
    op = get_op("ReduceProd")
    assert op is not None


def test_reducescatter_1655():
    op = get_op("ReduceScatter")
    assert op is not None


def test_reducestd_1656():
    op = get_op("ReduceStd")
    assert op is not None


def test_reducesum_1657():
    op = get_op("ReduceSum")
    assert op is not None


def test_reducevariance_1658():
    op = get_op("ReduceVariance")
    assert op is not None


def test_reductionkind_1659():
    op = get_op("ReductionKind")
    assert op is not None


def test_redux_1660():
    op = get_op("Redux")
    assert op is not None


def test_refmap_1661():
    op = get_op("RefMap")
    assert op is not None


def test_reftree_1662():
    op = get_op("RefTree")
    assert op is not None


def test_reflectionpad1d_1663():
    op = get_op("ReflectionPad1d")
    assert op is not None


def test_reflectionpad2d_1664():
    op = get_op("ReflectionPad2d")
    assert op is not None


def test_reflectionpad3d_1665():
    op = get_op("ReflectionPad3d")
    assert op is not None


def test_registerdatatype_1666():
    op = get_op("RegisterDataType")
    assert op is not None


def test_registerlayout_1667():
    op = get_op("RegisterLayout")
    assert op is not None


def test_registervariablename_1668():
    op = get_op("RegisterVariableName")
    assert op is not None


def test_relativebias_1669():
    op = get_op("RelativeBias")
    assert op is not None


def test_relayout_1670():
    op = get_op("Relayout")
    assert op is not None


def test_relu_1671():
    op = get_op("Relu")
    assert op is not None


def test_renorm_1672():
    op = get_op("Renorm")
    assert op is not None


def test_repeatinterleave_1673():
    op = get_op("RepeatInterleave")
    assert op is not None


def test_repeatvector_1674():
    op = get_op("RepeatVector")
    assert op is not None


def test_repeated_1675():
    op = get_op("Repeated")
    assert op is not None


def test_replacebypuredict_1676():
    op = get_op("ReplaceByPureDict")
    assert op is not None


def test_replicated_1677():
    op = get_op("Replicated")
    assert op is not None


def test_replicationerror_1678():
    op = get_op("ReplicationError")
    assert op is not None


def test_replicationpad1d_1679():
    op = get_op("ReplicationPad1d")
    assert op is not None


def test_replicationpad2d_1680():
    op = get_op("ReplicationPad2d")
    assert op is not None


def test_replicationpad3d_1681():
    op = get_op("ReplicationPad3d")
    assert op is not None


def test_reprcontext_1682():
    op = get_op("ReprContext")
    assert op is not None


def test_representable_1683():
    op = get_op("Representable")
    assert op is not None


def test_resnet_1684():
    op = get_op("ResNet")
    assert op is not None


def test_resnetblock_1685():
    op = get_op("ResNetBlock")
    assert op is not None


def test_rescaling_1686():
    op = get_op("Rescaling")
    assert op is not None


def test_resetpeakmemory_1687():
    op = get_op("ResetPeakMemory")
    assert op is not None


def test_reshape_1688():
    op = get_op("Reshape")
    assert op is not None


def test_resize_1689():
    op = get_op("Resize")
    assert op is not None


def test_resizeassparse__1690():
    op = get_op("ResizeAsSparse_")
    assert op is not None


def test_resizeas__1691():
    op = get_op("ResizeAs_")
    assert op is not None


def test_resizing_1692():
    op = get_op("Resizing")
    assert op is not None


def test_resolveconj_1693():
    op = get_op("ResolveConj")
    assert op is not None


def test_resolveneg_1694():
    op = get_op("ResolveNeg")
    assert op is not None


def test_restoreintpaths_1695():
    op = get_op("RestoreIntPaths")
    assert op is not None


def test_restorerngs_1696():
    op = get_op("RestoreRngs")
    assert op is not None


def test_resulttype_1697():
    op = get_op("ResultType")
    assert op is not None


def test_reverse_1698():
    op = get_op("Reverse")
    assert op is not None


def test_reversibleembedding_1699():
    op = get_op("ReversibleEmbedding")
    assert op is not None


def test_rfft_1700():
    op = get_op("Rfft")
    assert op is not None


def test_rfft2_1701():
    op = get_op("Rfft2")
    assert op is not None


def test_rfft2d_1702():
    op = get_op("Rfft2d")
    assert op is not None


def test_rfft3d_1703():
    op = get_op("Rfft3d")
    assert op is not None


def test_rfftfreq_1704():
    op = get_op("Rfftfreq")
    assert op is not None


def test_rfftn_1705():
    op = get_op("Rfftn")
    assert op is not None


def test_rfftnd_1706():
    op = get_op("Rfftnd")
    assert op is not None


def test_rgbtoyiq_1707():
    op = get_op("RgbToYiq")
    assert op is not None


def test_rgbtoyuv_1708():
    op = get_op("RgbToYuv")
    assert op is not None


def test_rightshift_1709():
    op = get_op("RightShift")
    assert op is not None


def test_rmsnormnoscale_1710():
    op = get_op("RmsNormNoScale")
    assert op is not None


def test_rngbitgenerator_1711():
    op = get_op("RngBitGenerator")
    assert op is not None


def test_rngcount_1712():
    op = get_op("RngCount")
    assert op is not None


def test_rngkey_1713():
    op = get_op("RngKey")
    assert op is not None


def test_rngstate_1714():
    op = get_op("RngState")
    assert op is not None


def test_rngstream_1715():
    op = get_op("RngStream")
    assert op is not None


def test_rnguniform_1716():
    op = get_op("RngUniform")
    assert op is not None


def test_rngs_1717():
    op = get_op("Rngs")
    assert op is not None


def test_rnnrelu_1718():
    op = get_op("RnnRelu")
    assert op is not None


def test_rnnrelucell_1719():
    op = get_op("RnnReluCell")
    assert op is not None


def test_rnntanh_1720():
    op = get_op("RnnTanh")
    assert op is not None


def test_rnntanhcell_1721():
    op = get_op("RnnTanhCell")
    assert op is not None


def test_rope_1722():
    op = get_op("RoPE")
    assert op is not None


def test_rooflineresult_1723():
    op = get_op("RooflineResult")
    assert op is not None


def test_rooflinerulecontext_1724():
    op = get_op("RooflineRuleContext")
    assert op is not None


def test_rooflineshape_1725():
    op = get_op("RooflineShape")
    assert op is not None


def test_roots_1726():
    op = get_op("Roots")
    assert op is not None


def test_round__1727():
    op = get_op("Round_")
    assert op is not None


def test_roundingmethod_1728():
    op = get_op("RoundingMethod")
    assert op is not None


def test_rowindicescopy_1729():
    op = get_op("RowIndicesCopy")
    assert op is not None


def test_rowstack_1730():
    op = get_op("RowStack")
    assert op is not None


def test_rowwise_1731():
    op = get_op("RowWise")
    assert op is not None


def test_rsqrt__1732():
    op = get_op("Rsqrt_")
    assert op is not None


def test_rsub_1733():
    op = get_op("Rsub")
    assert op is not None


def test_scoped_regex_1734():
    op = get_op("SCOPED_REGEX")
    assert op is not None


def test_selu_1735():
    op = get_op("SELU")
    assert op is not None


def test_semaphore_1736():
    op = get_op("SEMAPHORE")
    assert op is not None


def test_seq_minor_1737():
    op = get_op("SEQ_MINOR")
    assert op is not None


def test_sgd_1738():
    op = get_op("SGD")
    assert op is not None


def test_singleton_instance_registry_1739():
    op = get_op("SINGLETON_INSTANCE_REGISTRY")
    assert op is not None


def test_singleton_object_store_1740():
    op = get_op("SINGLETON_OBJECT_STORE")
    assert op is not None


def test_singleton_result_store_1741():
    op = get_op("SINGLETON_RESULT_STORE")
    assert op is not None


def test_smemtiling_1742():
    op = get_op("SMEMTiling")
    assert op is not None


def test_smem_banks_1743():
    op = get_op("SMEM_BANKS")
    assert op is not None


def test_smem_bank_bytes_1744():
    op = get_op("SMEM_BANK_BYTES")
    assert op is not None


def test_src_regex_1745():
    op = get_op("SRC_REGEX")
    assert op is not None


def test_ssm_1746():
    op = get_op("SSM")
    assert op is not None


def test_ssmgated_1747():
    op = get_op("SSMGated")
    assert op is not None


def test_ssmtransformer_1748():
    op = get_op("SSMTransformer")
    assert op is not None


def test_stable_hlo_1749():
    op = get_op("STABLE_HLO")
    assert op is not None


def test_stftspectrogram_1750():
    op = get_op("STFTSpectrogram")
    assert op is not None


def test_subcore_parallel_1751():
    op = get_op("SUBCORE_PARALLEL")
    assert op is not None


def test_supported_f8_types_1752():
    op = get_op("SUPPORTED_F8_TYPES")
    assert op is not None


def test_svd_1753():
    op = get_op("SVD")
    assert op is not None


def test_svdresult_1754():
    op = get_op("SVDResult")
    assert op is not None


def test_swizzle_32_4_4_1755():
    op = get_op("SWIZZLE_32_4_4")
    assert op is not None


def test_saddmm_1756():
    op = get_op("Saddmm")
    assert op is not None


def test_save_1757():
    op = get_op("Save")
    assert op is not None


def test_savegguf_1758():
    op = get_op("SaveGguf")
    assert op is not None


def test_savesafetensors_1759():
    op = get_op("SaveSafetensors")
    assert op is not None


def test_savetxt_1760():
    op = get_op("SaveTxt")
    assert op is not None


def test_savezcompressed_1761():
    op = get_op("SavezCompressed")
    assert op is not None


def test_scalarmul_1762():
    op = get_op("ScalarMul")
    assert op is not None


def test_scalartensor_1763():
    op = get_op("ScalarTensor")
    assert op is not None


def test_scalartype_1764():
    op = get_op("ScalarType")
    assert op is not None


def test_scalebytrustratiostate_1765():
    op = get_op("ScaleByTrustRatioState")
    assert op is not None


def test_scalestate_1766():
    op = get_op("ScaleState")
    assert op is not None


def test_scalingtype_1767():
    op = get_op("ScalingType")
    assert op is not None


def test_scan_1768():
    op = get_op("Scan")
    assert op is not None


def test_scanfn_1769():
    op = get_op("ScanFn")
    assert op is not None


def test_scatter_1770():
    op = get_op("Scatter")
    assert op is not None


def test_scatteradd_1771():
    op = get_op("ScatterAdd")
    assert op is not None


def test_scatterdimensionnumbers_1772():
    op = get_op("ScatterDimensionNumbers")
    assert op is not None


def test_scatternd_1773():
    op = get_op("ScatterND")
    assert op is not None


def test_scatternd_1774():
    op = get_op("ScatterNd")
    assert op is not None


def test_scatterreduce_1775():
    op = get_op("ScatterReduce")
    assert op is not None


def test_scope_1776():
    op = get_op("Scope")
    assert op is not None


def test_scriptclass_1777():
    op = get_op("ScriptClass")
    assert op is not None


def test_scriptclassfunction_1778():
    op = get_op("ScriptClassFunction")
    assert op is not None


def test_scriptdict_1779():
    op = get_op("ScriptDict")
    assert op is not None


def test_scriptdictiterator_1780():
    op = get_op("ScriptDictIterator")
    assert op is not None


def test_scriptdictkeyiterator_1781():
    op = get_op("ScriptDictKeyIterator")
    assert op is not None


def test_scriptfunction_1782():
    op = get_op("ScriptFunction")
    assert op is not None


def test_scriptlist_1783():
    op = get_op("ScriptList")
    assert op is not None


def test_scriptlistiterator_1784():
    op = get_op("ScriptListIterator")
    assert op is not None


def test_scriptmethod_1785():
    op = get_op("ScriptMethod")
    assert op is not None


def test_scriptmodule_1786():
    op = get_op("ScriptModule")
    assert op is not None


def test_scriptmoduleserializer_1787():
    op = get_op("ScriptModuleSerializer")
    assert op is not None


def test_scriptobject_1788():
    op = get_op("ScriptObject")
    assert op is not None


def test_scriptobjectproperty_1789():
    op = get_op("ScriptObjectProperty")
    assert op is not None


def test_sctypedict_1790():
    op = get_op("SctypeDict")
    assert op is not None


def test_seedgenerator_1791():
    op = get_op("SeedGenerator")
    assert op is not None


def test_segmentids_1792():
    op = get_op("SegmentIds")
    assert op is not None


def test_segmentmask_1793():
    op = get_op("SegmentMask")
    assert op is not None


def test_segmentmax_1794():
    op = get_op("SegmentMax")
    assert op is not None


def test_segmentmean_1795():
    op = get_op("SegmentMean")
    assert op is not None


def test_segmentmin_1796():
    op = get_op("SegmentMin")
    assert op is not None


def test_segmentprod_1797():
    op = get_op("SegmentProd")
    assert op is not None


def test_segmentreduce_1798():
    op = get_op("SegmentReduce")
    assert op is not None


def test_segmentsum_1799():
    op = get_op("SegmentSum")
    assert op is not None


def test_segmentedmm_1800():
    op = get_op("SegmentedMm")
    assert op is not None


def test_selectcopy_1801():
    op = get_op("SelectCopy")
    assert op is not None


def test_selectscatter_1802():
    op = get_op("SelectScatter")
    assert op is not None


def test_selfattentionwithnormandresidual_1803():
    op = get_op("SelfAttentionWithNormAndResidual")
    assert op is not None


def test_semaphoreref_1804():
    op = get_op("SemaphoreRef")
    assert op is not None


def test_separableconv1d_1805():
    op = get_op("SeparableConv1D")
    assert op is not None


def test_separableconv2d_1806():
    op = get_op("SeparableConv2D")
    assert op is not None


def test_separableconvolution1d_1807():
    op = get_op("SeparableConvolution1D")
    assert op is not None


def test_separableconvolution2d_1808():
    op = get_op("SeparableConvolution2D")
    assert op is not None


def test_sequencemodel_1809():
    op = get_op("SequenceModel")
    assert op is not None


def test_sequencereprmixin_1810():
    op = get_op("SequenceReprMixin")
    assert op is not None


def test_sequential_1811():
    op = get_op("Sequential")
    assert op is not None


def test_serializationstoragecontext_1812():
    op = get_op("SerializationStorageContext")
    assert op is not None


def test_serializelayer_1813():
    op = get_op("SerializeLayer")
    assert op is not None


def test_serializeoptimizer_1814():
    op = get_op("SerializeOptimizer")
    assert op is not None


def test_setanomalyenabled_1815():
    op = get_op("SetAnomalyEnabled")
    assert op is not None


def test_setautocastcacheenabled_1816():
    op = get_op("SetAutocastCacheEnabled")
    assert op is not None


def test_setautocastcpudtype_1817():
    op = get_op("SetAutocastCpuDtype")
    assert op is not None


def test_setautocastcpuenabled_1818():
    op = get_op("SetAutocastCpuEnabled")
    assert op is not None


def test_setautocastdtype_1819():
    op = get_op("SetAutocastDtype")
    assert op is not None


def test_setautocastenabled_1820():
    op = get_op("SetAutocastEnabled")
    assert op is not None


def test_setautocastgpudtype_1821():
    op = get_op("SetAutocastGpuDtype")
    assert op is not None


def test_setautocastipudtype_1822():
    op = get_op("SetAutocastIpUDtype")
    assert op is not None


def test_setautocastipuenabled_1823():
    op = get_op("SetAutocastIpUEnabled")
    assert op is not None


def test_setautocastxladtype_1824():
    op = get_op("SetAutocastXlaDtype")
    assert op is not None


def test_setautocastxlaenabled_1825():
    op = get_op("SetAutocastXlaEnabled")
    assert op is not None


def test_setcachelimit_1826():
    op = get_op("SetCacheLimit")
    assert op is not None


def test_setdefaultdevice_1827():
    op = get_op("SetDefaultDevice")
    assert op is not None


def test_setdefaultdtype_1828():
    op = get_op("SetDefaultDtype")
    assert op is not None


def test_setdefaultstream_1829():
    op = get_op("SetDefaultStream")
    assert op is not None


def test_setdefaulttensortype_1830():
    op = get_op("SetDefaultTensorType")
    assert op is not None


def test_setdeterministicdebugmode_1831():
    op = get_op("SetDeterministicDebugMode")
    assert op is not None


def test_setfloat32matmulprecision_1832():
    op = get_op("SetFloat32MatmulPrecision")
    assert op is not None


def test_setflushdenormal_1833():
    op = get_op("SetFlushDenormal")
    assert op is not None


def test_setgradenabled_1834():
    op = get_op("SetGradEnabled")
    assert op is not None


def test_setmemorylimit_1835():
    op = get_op("SetMemoryLimit")
    assert op is not None


def test_setmetadata_1836():
    op = get_op("SetMetadata")
    assert op is not None


def test_setmode_1837():
    op = get_op("SetMode")
    assert op is not None


def test_setmodeinfo_1838():
    op = get_op("SetModeInfo")
    assert op is not None


def test_setnuminteropthreads_1839():
    op = get_op("SetNumInteropThreads")
    assert op is not None


def test_setnumthreads_1840():
    op = get_op("SetNumThreads")
    assert op is not None


def test_setprintoptions_1841():
    op = get_op("SetPrintoptions")
    assert op is not None


def test_setrngstate_1842():
    op = get_op("SetRngState")
    assert op is not None


def test_setvariable_1843():
    op = get_op("SetVariable")
    assert op is not None


def test_setvital_1844():
    op = get_op("SetVital")
    assert op is not None


def test_setwarnalways_1845():
    op = get_op("SetWarnAlways")
    assert op is not None


def test_setwiredlimit_1846():
    op = get_op("SetWiredLimit")
    assert op is not None


def test_sgn_1847():
    op = get_op("Sgn")
    assert op is not None


def test_shapedtypestructtree_1848():
    op = get_op("ShapeDtypeStructTree")
    assert op is not None


def test_shapetree_1849():
    op = get_op("ShapeTree")
    assert op is not None


def test_shardmap_1850():
    op = get_op("ShardMap")
    assert op is not None


def test_shardmapfn_1851():
    op = get_op("ShardMapFn")
    assert op is not None


def test_shardedaxis_1852():
    op = get_op("ShardedAxis")
    assert op is not None


def test_shardedtoalllinear_1853():
    op = get_op("ShardedToAllLinear")
    assert op is not None


def test_shardingspec_1854():
    op = get_op("ShardingSpec")
    assert op is not None


def test_sharedembeddingsoftmax_1855():
    op = get_op("SharedEmbeddingSoftmax")
    assert op is not None


def test_sharesmemory_1856():
    op = get_op("SharesMemory")
    assert op is not None


def test_shortstorage_1857():
    op = get_op("ShortStorage")
    assert op is not None


def test_shouldskipupdatefunction_1858():
    op = get_op("ShouldSkipUpdateFunction")
    assert op is not None


def test_showconfig_1859():
    op = get_op("ShowConfig")
    assert op is not None


def test_showruntime_1860():
    op = get_op("ShowRuntime")
    assert op is not None


def test_silu_1861():
    op = get_op("SiLU")
    assert op is not None


def test_sigmoid_1862():
    op = get_op("Sigmoid")
    assert op is not None


def test_sigmoidcrossentropy_1863():
    op = get_op("SigmoidCrossEntropy")
    assert op is not None


def test_sigmoid__1864():
    op = get_op("Sigmoid_")
    assert op is not None


def test_simplecell_1865():
    op = get_op("SimpleCell")
    assert op is not None


def test_simpleobjectrepr_1866():
    op = get_op("SimpleObjectRepr")
    assert op is not None


def test_simplernn_1867():
    op = get_op("SimpleRNN")
    assert op is not None


def test_simplernncell_1868():
    op = get_op("SimpleRNNCell")
    assert op is not None


def test_sin_1869():
    op = get_op("Sin")
    assert op is not None


def test_sin__1870():
    op = get_op("Sin_")
    assert op is not None


def test_sinc__1871():
    op = get_op("Sinc_")
    assert op is not None


def test_singlesidecollectiveeffect_1872():
    op = get_op("SingleSideCollectiveEffect")
    assert op is not None


def test_sinh__1873():
    op = get_op("Sinh_")
    assert op is not None


def test_sinusoidalpositionalencoding_1874():
    op = get_op("SinusoidalPositionalEncoding")
    assert op is not None


def test_sizebytes_1875():
    op = get_op("SizeBytes")
    assert op is not None


def test_slicecopy_1876():
    op = get_op("SliceCopy")
    assert op is not None


def test_sliceinverse_1877():
    op = get_op("SliceInverse")
    assert op is not None


def test_slicescatter_1878():
    op = get_op("SliceScatter")
    assert op is not None


def test_sliceupdate_1879():
    op = get_op("SliceUpdate")
    assert op is not None


def test_slogdet_1880():
    op = get_op("Slogdet")
    assert op is not None


def test_slogdetresult_1881():
    op = get_op("SlogdetResult")
    assert op is not None


def test_smm_1882():
    op = get_op("Smm")
    assert op is not None


def test_smoothl1loss_1883():
    op = get_op("SmoothL1Loss")
    assert op is not None


def test_smoothl1_1884():
    op = get_op("Smoothl1")
    assert op is not None


def test_snapshotstate_1885():
    op = get_op("SnapshotState")
    assert op is not None


def test_sobolsample_1886():
    op = get_op("SobolSample")
    assert op is not None


def test_softmarginloss_1887():
    op = get_op("SoftMarginLoss")
    assert op is not None


def test_softmax_1888():
    op = get_op("Softmax")
    assert op is not None


def test_softmax2d_1889():
    op = get_op("Softmax2d")
    assert op is not None


def test_softplus_1890():
    op = get_op("Softplus")
    assert op is not None


def test_solarization_1891():
    op = get_op("Solarization")
    assert op is not None


def test_solve_1892():
    op = get_op("Solve")
    assert op is not None


def test_sort_1893():
    op = get_op("Sort")
    assert op is not None


def test_sortcomplex_1894():
    op = get_op("SortComplex")
    assert op is not None


def test_sourcemapdump_1895():
    op = get_op("SourceMapDump")
    assert op is not None


def test_sourcemapgeneratorfn_1896():
    op = get_op("SourceMapGeneratorFn")
    assert op is not None


def test_spacetobatch_1897():
    op = get_op("SpaceToBatch")
    assert op is not None


def test_spacetobatchnd_1898():
    op = get_op("SpaceToBatchND")
    assert op is not None


def test_sparsebincount_1899():
    op = get_op("SparseBincount")
    assert op is not None


def test_sparsebsc_1900():
    op = get_op("SparseBsc")
    assert op is not None


def test_sparsebsctensor_1901():
    op = get_op("SparseBscTensor")
    assert op is not None


def test_sparsebsr_1902():
    op = get_op("SparseBsr")
    assert op is not None


def test_sparsebsrtensor_1903():
    op = get_op("SparseBsrTensor")
    assert op is not None


def test_sparsecompressed_1904():
    op = get_op("SparseCompressed")
    assert op is not None


def test_sparsecompressedtensor_1905():
    op = get_op("SparseCompressedTensor")
    assert op is not None


def test_sparsecoo_1906():
    op = get_op("SparseCoo")
    assert op is not None


def test_sparsecootensor_1907():
    op = get_op("SparseCooTensor")
    assert op is not None


def test_sparsecrosshashed_1908():
    op = get_op("SparseCrossHashed")
    assert op is not None


def test_sparsecsc_1909():
    op = get_op("SparseCsc")
    assert op is not None


def test_sparsecsctensor_1910():
    op = get_op("SparseCscTensor")
    assert op is not None


def test_sparsecsr_1911():
    op = get_op("SparseCsr")
    assert op is not None


def test_sparsecsrtensor_1912():
    op = get_op("SparseCsrTensor")
    assert op is not None


def test_sparseefficiencyerror_1913():
    op = get_op("SparseEfficiencyError")
    assert op is not None


def test_sparseefficiencywarning_1914():
    op = get_op("SparseEfficiencyWarning")
    assert op is not None


def test_sparseexpanddims_1915():
    op = get_op("SparseExpandDims")
    assert op is not None


def test_sparseeye_1916():
    op = get_op("SparseEye")
    assert op is not None


def test_sparsefillemptyrows_1917():
    op = get_op("SparseFillEmptyRows")
    assert op is not None


def test_sparseinfo_1918():
    op = get_op("SparseInfo")
    assert op is not None


def test_sparselayout_1919():
    op = get_op("SparseLayout")
    assert op is not None


def test_sparsemapvalues_1920():
    op = get_op("SparseMapValues")
    assert op is not None


def test_sparsemask_1921():
    op = get_op("SparseMask")
    assert op is not None


def test_sparsemaximum_1922():
    op = get_op("SparseMaximum")
    assert op is not None


def test_sparseminimum_1923():
    op = get_op("SparseMinimum")
    assert op is not None


def test_sparseplus_1924():
    op = get_op("SparsePlus")
    assert op is not None


def test_sparsereducemax_1925():
    op = get_op("SparseReduceMax")
    assert op is not None


def test_sparsereducesum_1926():
    op = get_op("SparseReduceSum")
    assert op is not None


def test_sparsereorder_1927():
    op = get_op("SparseReorder")
    assert op is not None


def test_sparseresetshape_1928():
    op = get_op("SparseResetShape")
    assert op is not None


def test_sparsereshape_1929():
    op = get_op("SparseReshape")
    assert op is not None


def test_sparseretain_1930():
    op = get_op("SparseRetain")
    assert op is not None


def test_sparsesegmentmean_1931():
    op = get_op("SparseSegmentMean")
    assert op is not None


def test_sparsesegmentsqrtn_1932():
    op = get_op("SparseSegmentSqrtN")
    assert op is not None


def test_sparsesegmentsum_1933():
    op = get_op("SparseSegmentSum")
    assert op is not None


def test_sparsesigmoid_1934():
    op = get_op("SparseSigmoid")
    assert op is not None


def test_sparseslice_1935():
    op = get_op("SparseSlice")
    assert op is not None


def test_sparsesoftmax_1936():
    op = get_op("SparseSoftmax")
    assert op is not None


def test_sparsetestcase_1937():
    op = get_op("SparseTestCase")
    assert op is not None


def test_sparsetoindicator_1938():
    op = get_op("SparseToIndicator")
    assert op is not None


def test_sparsetrace_1939():
    op = get_op("SparseTrace")
    assert op is not None


def test_sparsetracer_1940():
    op = get_op("SparseTracer")
    assert op is not None


def test_sparsetranspose_1941():
    op = get_op("SparseTranspose")
    assert op is not None


def test_sparsemax_1942():
    op = get_op("Sparsemax")
    assert op is not None


def test_sparsifyenv_1943():
    op = get_op("SparsifyEnv")
    assert op is not None


def test_sparsifyvalue_1944():
    op = get_op("SparsifyValue")
    assert op is not None


def test_spatialdropout1d_1945():
    op = get_op("SpatialDropout1D")
    assert op is not None


def test_spatialdropout2d_1946():
    op = get_op("SpatialDropout2D")
    assert op is not None


def test_spatialdropout3d_1947():
    op = get_op("SpatialDropout3D")
    assert op is not None


def test_specialgamma_1948():
    op = get_op("SpecialGamma")
    assert op is not None


def test_specialization_1949():
    op = get_op("Specialization")
    assert op is not None


def test_specs_1950():
    op = get_op("Specs")
    assert op is not None


def test_spectralnorm_1951():
    op = get_op("SpectralNorm")
    assert op is not None


def test_spectralnormloadstatedictprehook_1952():
    op = get_op("SpectralNormLoadStateDictPreHook")
    assert op is not None


def test_spectralnormstatedicthook_1953():
    op = get_op("SpectralNormStateDictHook")
    assert op is not None


def test_spectralnormalization_1954():
    op = get_op("SpectralNormalization")
    assert op is not None


def test_spectrumaugmenter_1955():
    op = get_op("SpectrumAugmenter")
    assert op is not None


def test_splashattentionkernel_1956():
    op = get_op("SplashAttentionKernel")
    assert op is not None


def test_splashcustomreturntype_1957():
    op = get_op("SplashCustomReturnType")
    assert op is not None


def test_splashresidualstype_1958():
    op = get_op("SplashResidualsType")
    assert op is not None


def test_splitbackups_1959():
    op = get_op("SplitBackups")
    assert op is not None


def test_splitcontext_1960():
    op = get_op("SplitContext")
    assert op is not None


def test_splitcopy_1961():
    op = get_op("SplitCopy")
    assert op is not None


def test_splitgraph_1962():
    op = get_op("SplitGraph")
    assert op is not None


def test_splitrngs_1963():
    op = get_op("SplitRngs")
    assert op is not None


def test_splitstate_1964():
    op = get_op("SplitState")
    assert op is not None


def test_splitwithsizes_1965():
    op = get_op("SplitWithSizes")
    assert op is not None


def test_spmm_1966():
    op = get_op("Spmm")
    assert op is not None


def test_sqrt_1967():
    op = get_op("Sqrt")
    assert op is not None


def test_sqrt__1968():
    op = get_op("Sqrt_")
    assert op is not None


def test_sqrtm_1969():
    op = get_op("Sqrtm")
    assert op is not None


def test_square__1970():
    op = get_op("Square_")
    assert op is not None


def test_squareddifference_1971():
    op = get_op("SquaredDifference")
    assert op is not None


def test_squaredhinge_1972():
    op = get_op("SquaredHinge")
    assert op is not None


def test_squaredrelu_1973():
    op = get_op("SquaredReLU")
    assert op is not None


def test_squeeze_1974():
    op = get_op("Squeeze")
    assert op is not None


def test_squeezecopy_1975():
    op = get_op("SqueezeCopy")
    assert op is not None


def test_sspaddmm_1976():
    op = get_op("Sspaddmm")
    assert op is not None


def test_stackfrnn_1977():
    op = get_op("StackFrnn")
    assert op is not None


def test_stackedrnncells_1978():
    op = get_op("StackedRNNCells")
    assert op is not None


def test_stackedtransformer_1979():
    op = get_op("StackedTransformer")
    assert op is not None


def test_stackedtransformerrepeated_1980():
    op = get_op("StackedTransformerRepeated")
    assert op is not None


def test_stackingovertime_1981():
    op = get_op("StackingOverTime")
    assert op is not None


def test_stage_1982():
    op = get_op("Stage")
    assert op is not None


def test_staggeredtransferplan_1983():
    op = get_op("StaggeredTransferPlan")
    assert op is not None


def test_staggeredtransferplanimpl_1984():
    op = get_op("StaggeredTransferPlanImpl")
    assert op is not None


def test_standardize_1985():
    op = get_op("Standardize")
    assert op is not None


def test_stateaxes_1986():
    op = get_op("StateAxes")
    assert op is not None


def test_statesharding_1987():
    op = get_op("StateSharding")
    assert op is not None


def test_staticcache_1988():
    op = get_op("StaticCache")
    assert op is not None


def test_staticelem_1989():
    op = get_op("StaticElem")
    assert op is not None


def test_staticmodule_1990():
    op = get_op("StaticModule")
    assert op is not None


def test_statistics_1991():
    op = get_op("Statistics")
    assert op is not None


def test_stdmean_1992():
    op = get_op("StdMean")
    assert op is not None


def test_stealref_1993():
    op = get_op("StealRef")
    assert op is not None


def test_stepactivation_1994():
    op = get_op("StepActivation")
    assert op is not None


def test_steplr_1995():
    op = get_op("StepLR")
    assert op is not None


def test_stft_1996():
    op = get_op("Stft")
    assert op is not None


def test_stochasticresidual_1997():
    op = get_op("StochasticResidual")
    assert op is not None


def test_stopgradient_1998():
    op = get_op("StopGradient")
    assert op is not None


def test_storage_1999():
    op = get_op("Storage")
    assert op is not None


def test_storagebase_2000():
    op = get_op("StorageBase")
    assert op is not None


def test_str__2001():
    op = get_op("Str_")
    assert op is not None


def test_stream_2002():
    op = get_op("Stream")
    assert op is not None


def test_streamcontext_2003():
    op = get_op("StreamContext")
    assert op is not None


def test_streamobjtype_2004():
    op = get_op("StreamObjType")
    assert op is not None


def test_strided_2005():
    op = get_op("Strided")
    assert op is not None


def test_stridedslice_2006():
    op = get_op("StridedSlice")
    assert op is not None


def test_string_2007():
    op = get_op("String")
    assert op is not None


def test_stringlookup_2008():
    op = get_op("StringLookup")
    assert op is not None


def test_stringtype_2009():
    op = get_op("StringType")
    assert op is not None


def test_structuredvoidformat_2010():
    op = get_op("StructuredVoidFormat")
    assert op is not None


def test_subarrayformat_2011():
    op = get_op("SubArrayFormat")
    assert op is not None


def test_subtract_2012():
    op = get_op("Subtract")
    assert op is not None


def test_subtractlayer_2013():
    op = get_op("SubtractLayer")
    assert op is not None


def test_sum_2014():
    op = get_op("Sum")
    assert op is not None


def test_sumpool_2015():
    op = get_op("SumPool")
    assert op is not None


def test_svd_2016():
    op = get_op("Svd")
    assert op is not None


def test_svdalgorithm_2017():
    op = get_op("SvdAlgorithm")
    assert op is not None


def test_svdlowrank_2018():
    op = get_op("SvdLowrank")
    assert op is not None


def test_svdvals_2019():
    op = get_op("Svdvals")
    assert op is not None


def test_swapdims_2020():
    op = get_op("Swapdims")
    assert op is not None


def test_swish_2021():
    op = get_op("Swish")
    assert op is not None


def test_switch_2022():
    op = get_op("Switch")
    assert op is not None


def test_swizzletransform_2023():
    op = get_op("SwizzleTransform")
    assert op is not None


def test_swizzletype_2024():
    op = get_op("SwizzleType")
    assert op is not None


def test_symbool_2025():
    op = get_op("SymBool")
    assert op is not None


def test_symbooltype_2026():
    op = get_op("SymBoolType")
    assert op is not None


def test_symconstrainrange_2027():
    op = get_op("SymConstrainRange")
    assert op is not None


def test_symconstrainrangeforsize_2028():
    op = get_op("SymConstrainRangeForSize")
    assert op is not None


def test_symfloat_2029():
    op = get_op("SymFloat")
    assert op is not None


def test_symfreshsize_2030():
    op = get_op("SymFreshSize")
    assert op is not None


def test_symint_2031():
    op = get_op("SymInt")
    assert op is not None


def test_syminttype_2032():
    op = get_op("SymIntType")
    assert op is not None


def test_symite_2033():
    op = get_op("SymIte")
    assert op is not None


def test_symmax_2034():
    op = get_op("SymMax")
    assert op is not None


def test_symmin_2035():
    op = get_op("SymMin")
    assert op is not None


def test_symnot_2036():
    op = get_op("SymNot")
    assert op is not None


def test_symsqrt_2037():
    op = get_op("SymSqrt")
    assert op is not None


def test_symsum_2038():
    op = get_op("SymSum")
    assert op is not None


def test_symeig_2039():
    op = get_op("Symeig")
    assert op is not None


def test_syncbatchnorm_2040():
    op = get_op("SyncBatchNorm")
    assert op is not None


def test_synchronize_2041():
    op = get_op("Synchronize")
    assert op is not None


def test_t_2042():
    op = get_op("T")
    assert op is not None


def test_tcgen05_col_layout_2043():
    op = get_op("TCGEN05_COL_LAYOUT")
    assert op is not None


def test_tcgen05_layout_2044():
    op = get_op("TCGEN05_LAYOUT")
    assert op is not None


def test_tcgen05_row_layout_2045():
    op = get_op("TCGEN05_ROW_LAYOUT")
    assert op is not None


def test_tcgen05_smem_descriptor_bit_2046():
    op = get_op("TCGEN05_SMEM_DESCRIPTOR_BIT")
    assert op is not None


def test_tcgen05_transposed_layout_2047():
    op = get_op("TCGEN05_TRANSPOSED_LAYOUT")
    assert op is not None


def test_tcopy_2048():
    op = get_op("TCopy")
    assert op is not None


def test_td_2049():
    op = get_op("TD")
    assert op is not None


def test_tfsmlayer_2050():
    op = get_op("TFSMLayer")
    assert op is not None


def test_tma_2051():
    op = get_op("TMA")
    assert op is not None


def test_tmabarrier_2052():
    op = get_op("TMABarrier")
    assert op is not None


def test_tmareductionop_2053():
    op = get_op("TMAReductionOp")
    assert op is not None


def test_tma_descriptor_alignment_2054():
    op = get_op("TMA_DESCRIPTOR_ALIGNMENT")
    assert op is not None


def test_tma_descriptor_bytes_2055():
    op = get_op("TMA_DESCRIPTOR_BYTES")
    assert op is not None


def test_tma_gather_indices_layout_2056():
    op = get_op("TMA_GATHER_INDICES_LAYOUT")
    assert op is not None


def test_tmemlayout_2057():
    op = get_op("TMEMLayout")
    assert op is not None


def test_tmemref_2058():
    op = get_op("TMEMRef")
    assert op is not None


def test_tmem_max_cols_2059():
    op = get_op("TMEM_MAX_COLS")
    assert op is not None


def test_tmem_rows_2060():
    op = get_op("TMEM_ROWS")
    assert op is not None


def test_transposed_layout_2061():
    op = get_op("TRANSPOSED_LAYOUT")
    assert op is not None


def test_trans_b_dim_numbers_2062():
    op = get_op("TRANS_B_DIM_NUMBERS")
    assert op is not None


def test_tuned_block_sizes_2063():
    op = get_op("TUNED_BLOCK_SIZES")
    assert op is not None


def test_tuple_2064():
    op = get_op("TUPLE")
    assert op is not None


def test_type_checking_2065():
    op = get_op("TYPE_CHECKING")
    assert op is not None


def test_t_destination_2066():
    op = get_op("T_destination")
    assert op is not None


def test_t_module_2067():
    op = get_op("T_module")
    assert op is not None


def test_tag_2068():
    op = get_op("Tag")
    assert op is not None


def test_takealongaxis_2069():
    op = get_op("TakeAlongAxis")
    assert op is not None


def test_takealongdim_2070():
    op = get_op("TakeAlongDim")
    assert op is not None


def test_tan__2071():
    op = get_op("Tan_")
    assert op is not None


def test_tanh_2072():
    op = get_op("Tanh")
    assert op is not None


def test_tanh__2073():
    op = get_op("Tanh_")
    assert op is not None


def test_temporalshifting_2074():
    op = get_op("TemporalShifting")
    assert op is not None


def test_tensordataset_2075():
    op = get_op("TensorDataset")
    assert op is not None


def test_tensorscatteradd_2076():
    op = get_op("TensorScatterAdd")
    assert op is not None


def test_tensorscattermax_2077():
    op = get_op("TensorScatterMax")
    assert op is not None


def test_tensorscattermin_2078():
    op = get_op("TensorScatterMin")
    assert op is not None


def test_tensorscatterupdate_2079():
    op = get_op("TensorScatterUpdate")
    assert op is not None


def test_tensorsplit_2080():
    op = get_op("TensorSplit")
    assert op is not None


def test_tensortype_2081():
    op = get_op("TensorType")
    assert op is not None


def test_tensorwise_2082():
    op = get_op("TensorWise")
    assert op is not None


def test_tensordot_2083():
    op = get_op("Tensordot")
    assert op is not None


def test_tensorinv_2084():
    op = get_op("Tensorinv")
    assert op is not None


def test_tensorsolve_2085():
    op = get_op("Tensorsolve")
    assert op is not None


def test_test_2086():
    op = get_op("Test")
    assert op is not None


def test_textvectorization_2087():
    op = get_op("TextVectorization")
    assert op is not None


def test_tfconcretefunction_2088():
    op = get_op("TfConcreteFunction")
    assert op is not None


def test_tfval_2089():
    op = get_op("TfVal")
    assert op is not None


def test_threadsubset_2090():
    op = get_op("ThreadSubset")
    assert op is not None


def test_throughputbenchmark_2091():
    op = get_op("ThroughputBenchmark")
    assert op is not None


def test_tiletransform_2092():
    op = get_op("TileTransform")
    assert op is not None


def test_tiledlayout_2093():
    op = get_op("TiledLayout")
    assert op is not None


def test_tiledlayoutimpl_2094():
    op = get_op("TiledLayoutImpl")
    assert op is not None


def test_timedistributed_2095():
    op = get_op("TimeDistributed")
    assert op is not None


def test_timedelta64_2096():
    op = get_op("Timedelta64")
    assert op is not None


def test_timedeltaformat_2097():
    op = get_op("TimedeltaFormat")
    assert op is not None


def test_todlpack_2098():
    op = get_op("ToDlpack")
    assert op is not None


def test_toflatstate_2099():
    op = get_op("ToFlatState")
    assert op is not None


def test_tolinen_2100():
    op = get_op("ToLinen")
    assert op is not None


def test_tolinenpartial_2101():
    op = get_op("ToLinenPartial")
    assert op is not None


def test_tonnx_2102():
    op = get_op("ToNNX")
    assert op is not None


def test_topuredict_2103():
    op = get_op("ToPureDict")
    assert op is not None


def test_totensor_2104():
    op = get_op("ToTensor")
    assert op is not None


def test_totree_2105():
    op = get_op("ToTree")
    assert op is not None


def test_tolerance_2106():
    op = get_op("Tolerance")
    assert op is not None


def test_topk_2107():
    op = get_op("TopK")
    assert op is not None


def test_topkvalues_2108():
    op = get_op("TopKValues")
    assert op is not None


def test_topologydescription_2109():
    op = get_op("TopologyDescription")
    assert op is not None


def test_torchfunctional_2110():
    op = get_op("TorchFunctional")
    assert op is not None


def test_torchload_2111():
    op = get_op("TorchLoad")
    assert op is not None


def test_torchmodulewrapper_2112():
    op = get_op("TorchModuleWrapper")
    assert op is not None


def test_torchsave_2113():
    op = get_op("TorchSave")
    assert op is not None


def test_trace_2114():
    op = get_op("Trace")
    assert op is not None


def test_tracestate_2115():
    op = get_op("TraceState")
    assert op is not None


def test_tracetag_2116():
    op = get_op("TraceTag")
    assert op is not None


def test_traced_2117():
    op = get_op("Traced")
    assert op is not None


def test_tracingstate_2118():
    op = get_op("TracingState")
    assert op is not None


def test_trainmode_2119():
    op = get_op("TrainMode")
    assert op is not None


def test_trainstate_2120():
    op = get_op("TrainState")
    assert op is not None


def test_trainablepositionalembedding_2121():
    op = get_op("TrainablePositionalEmbedding")
    assert op is not None


def test_transferconnection_2122():
    op = get_op("TransferConnection")
    assert op is not None


def test_transferplan_2123():
    op = get_op("TransferPlan")
    assert op is not None


def test_transferserver_2124():
    op = get_op("TransferServer")
    assert op is not None


def test_transformer_2125():
    op = get_op("Transformer")
    assert op is not None


def test_transformerdecoder_2126():
    op = get_op("TransformerDecoder")
    assert op is not None


def test_transformerdecoderlayer_2127():
    op = get_op("TransformerDecoderLayer")
    assert op is not None


def test_transformerencoder_2128():
    op = get_op("TransformerEncoder")
    assert op is not None


def test_transformerencoderdecoder_2129():
    op = get_op("TransformerEncoderDecoder")
    assert op is not None


def test_transformerencoderlayer_2130():
    op = get_op("TransformerEncoderLayer")
    assert op is not None


def test_transformerfeedforward_2131():
    op = get_op("TransformerFeedForward")
    assert op is not None


def test_transformerfeedforwardmoe_2132():
    op = get_op("TransformerFeedForwardMoe")
    assert op is not None


def test_transformerlm_2133():
    op = get_op("TransformerLm")
    assert op is not None


def test_transpose_2134():
    op = get_op("Transpose")
    assert op is not None


def test_transpose2d_2135():
    op = get_op("Transpose2D")
    assert op is not None


def test_transposecopy_2136():
    op = get_op("TransposeCopy")
    assert op is not None


def test_transposetransform_2137():
    op = get_op("TransposeTransform")
    assert op is not None


def test_trapz_2138():
    op = get_op("Trapz")
    assert op is not None


def test_treebwdfn_2139():
    op = get_op("TreeBwdFn")
    assert op is not None


def test_treecheckifyfn_2140():
    op = get_op("TreeCheckifyFn")
    assert op is not None


def test_treecompiled_2141():
    op = get_op("TreeCompiled")
    assert op is not None


def test_treecondfn_2142():
    op = get_op("TreeCondFn")
    assert op is not None


def test_treecustomvjp_2143():
    op = get_op("TreeCustomVjp")
    assert op is not None


def test_treecustomvjpfn_2144():
    op = get_op("TreeCustomVjpFn")
    assert op is not None


def test_treeevalshapefn_2145():
    op = get_op("TreeEvalShapeFn")
    assert op is not None


def test_treeforiloopbodyfn_2146():
    op = get_op("TreeForiLoopBodyFn")
    assert op is not None


def test_treefwdfn_2147():
    op = get_op("TreeFwdFn")
    assert op is not None


def test_treegradfn_2148():
    op = get_op("TreeGradFn")
    assert op is not None


def test_treejitfn_2149():
    op = get_op("TreeJitFn")
    assert op is not None


def test_treejitwrapped_2150():
    op = get_op("TreeJitWrapped")
    assert op is not None


def test_treejvpfn_2151():
    op = get_op("TreeJvpFn")
    assert op is not None


def test_treelowered_2152():
    op = get_op("TreeLowered")
    assert op is not None


def test_treenodedef_2153():
    op = get_op("TreeNodeDef")
    assert op is not None


def test_treepmapfn_2154():
    op = get_op("TreePmapFn")
    assert op is not None


def test_treerematfn_2155():
    op = get_op("TreeRematFn")
    assert op is not None


def test_treescanfn_2156():
    op = get_op("TreeScanFn")
    assert op is not None


def test_treeshardmapfn_2157():
    op = get_op("TreeShardMapFn")
    assert op is not None


def test_treetraced_2158():
    op = get_op("TreeTraced")
    assert op is not None


def test_treevjpfn_2159():
    op = get_op("TreeVjpFn")
    assert op is not None


def test_treevmapfn_2160():
    op = get_op("TreeVmapFn")
    assert op is not None


def test_treewhileloopbodyfn_2161():
    op = get_op("TreeWhileLoopBodyFn")
    assert op is not None


def test_triinv_2162():
    op = get_op("TriInv")
    assert op is not None


def test_triangular_2163():
    op = get_op("Triangular")
    assert op is not None


def test_triangularsolve_2164():
    op = get_op("TriangularSolve")
    assert op is not None


def test_tridiagonalmatmul_2165():
    op = get_op("TridiagonalMatmul")
    assert op is not None


def test_tridiagonalsolve_2166():
    op = get_op("TridiagonalSolve")
    assert op is not None


def test_trilindices_2167():
    op = get_op("TrilIndices")
    assert op is not None


def test_trilindicesfrom_2168():
    op = get_op("TrilIndicesFrom")
    assert op is not None


def test_trimzeros_2169():
    op = get_op("TrimZeros")
    assert op is not None


def test_tripletmarginloss_2170():
    op = get_op("TripletMarginLoss")
    assert op is not None


def test_tripletmarginwithdistanceloss_2171():
    op = get_op("TripletMarginWithDistanceLoss")
    assert op is not None


def test_triuindices_2172():
    op = get_op("TriuIndices")
    assert op is not None


def test_triuindicesfrom_2173():
    op = get_op("TriuIndicesFrom")
    assert op is not None


def test_trivialtransferplan_2174():
    op = get_op("TrivialTransferPlan")
    assert op is not None


def test_trivialtransferplanimpl_2175():
    op = get_op("TrivialTransferPlanImpl")
    assert op is not None


def test_truediv_2176():
    op = get_op("TrueDiv")
    assert op is not None


def test_truedivide_2177():
    op = get_op("TrueDivide")
    assert op is not None


def test_true__2178():
    op = get_op("True_")
    assert op is not None


def test_truncinplace_2179():
    op = get_op("TruncInplace")
    assert op is not None


def test_truncatediv_2180():
    op = get_op("TruncateDiv")
    assert op is not None


def test_truncatemod_2181():
    op = get_op("TruncateMod")
    assert op is not None


def test_tuningconfig_2182():
    op = get_op("TuningConfig")
    assert op is not None


def test_tupletype_2183():
    op = get_op("TupleType")
    assert op is not None


def test_typeapi_2184():
    op = get_op("TypeApi")
    assert op is not None


def test_typedescription_2185():
    op = get_op("TypeDescription")
    assert op is not None


def test_typevara_2186():
    op = get_op("TypeVarA")
    assert op is not None


def test_typevarm_2187():
    op = get_op("TypeVarM")
    assert op is not None


def test_typecodes_2188():
    op = get_op("Typecodes")
    assert op is not None


def test_typedstorage_2189():
    op = get_op("TypedStorage")
    assert op is not None


def test_typename_2190():
    op = get_op("Typename")
    assert op is not None


def test_ubyte_2191():
    op = get_op("UByte")
    assert op is not None


def test_ufunctypeerror_2192():
    op = get_op("UFuncTypeError")
    assert op is not None


def test_uintc_2193():
    op = get_op("UIntC")
    assert op is not None


def test_uintp_2194():
    op = get_op("UIntP")
    assert op is not None


def test_ulong_2195():
    op = get_op("ULong")
    assert op is not None


def test_ulonglong_2196():
    op = get_op("ULongLong")
    assert op is not None


def test_up_2197():
    op = get_op("UP")
    assert op is not None


def test_upper_left_2198():
    op = get_op("UPPER_LEFT")
    assert op is not None


def test_use_tma_2199():
    op = get_op("USE_TMA")
    assert op is not None


def test_ushort_2200():
    op = get_op("UShort")
    assert op is not None


def test_uuid_2201():
    op = get_op("UUID")
    assert op is not None


def test_uuidmanager_2202():
    op = get_op("UUIDManager")
    assert op is not None


def test_uint1_2203():
    op = get_op("Uint1")
    assert op is not None


def test_uint3_2204():
    op = get_op("Uint3")
    assert op is not None


def test_uint5_2205():
    op = get_op("Uint5")
    assert op is not None


def test_uint6_2206():
    op = get_op("Uint6")
    assert op is not None


def test_uint7_2207():
    op = get_op("Uint7")
    assert op is not None


def test_uint8_2208():
    op = get_op("Uint8")
    assert op is not None


def test_unbind_2209():
    op = get_op("Unbind")
    assert op is not None


def test_unbindcopy_2210():
    op = get_op("UnbindCopy")
    assert op is not None


def test_unfoldcopy_2211():
    op = get_op("UnfoldCopy")
    assert op is not None


def test_unicode_2212():
    op = get_op("Unicode")
    assert op is not None


def test_uniform_2213():
    op = get_op("Uniform")
    assert op is not None


def test_unifytypelist_2214():
    op = get_op("UnifyTypeList")
    assert op is not None


def test_uninitializedbuffer_2215():
    op = get_op("UninitializedBuffer")
    assert op is not None


def test_uninitializedparameter_2216():
    op = get_op("UninitializedParameter")
    assert op is not None


def test_uninitializedtensormixin_2217():
    op = get_op("UninitializedTensorMixin")
    assert op is not None


def test_union_2218():
    op = get_op("Union")
    assert op is not None


def test_uniontype_2219():
    op = get_op("UnionType")
    assert op is not None


def test_uniqueall_2220():
    op = get_op("UniqueAll")
    assert op is not None


def test_uniqueconsecutive_2221():
    op = get_op("UniqueConsecutive")
    assert op is not None


def test_uniquecounts_2222():
    op = get_op("UniqueCounts")
    assert op is not None


def test_uniqueinverse_2223():
    op = get_op("UniqueInverse")
    assert op is not None


def test_uniquevalues_2224():
    op = get_op("UniqueValues")
    assert op is not None


def test_unitnormalization_2225():
    op = get_op("UnitNormalization")
    assert op is not None


def test_unoptimized_2226():
    op = get_op("Unoptimized")
    assert op is not None


def test_unravelindex_2227():
    op = get_op("UnravelIndex")
    assert op is not None


def test_unsafechunk_2228():
    op = get_op("UnsafeChunk")
    assert op is not None


def test_unsafesplit_2229():
    op = get_op("UnsafeSplit")
    assert op is not None


def test_unsafesplitwithsizes_2230():
    op = get_op("UnsafeSplitWithSizes")
    assert op is not None


def test_unsatisfiable_2231():
    op = get_op("Unsatisfiable")
    assert op is not None


def test_unsortedsegmentmax_2232():
    op = get_op("UnsortedSegmentMax")
    assert op is not None


def test_unsortedsegmentmean_2233():
    op = get_op("UnsortedSegmentMean")
    assert op is not None


def test_unsortedsegmentmin_2234():
    op = get_op("UnsortedSegmentMin")
    assert op is not None


def test_unsortedsegmentprod_2235():
    op = get_op("UnsortedSegmentProd")
    assert op is not None


def test_unsortedsegmentsqrtn_2236():
    op = get_op("UnsortedSegmentSqrtN")
    assert op is not None


def test_unsortedsegmentsum_2237():
    op = get_op("UnsortedSegmentSum")
    assert op is not None


def test_unspecifiedoutputshapedtype_2238():
    op = get_op("UnspecifiedOutputShapeDtype")
    assert op is not None


def test_unsqueeze_2239():
    op = get_op("Unsqueeze")
    assert op is not None


def test_unsqueezecopy_2240():
    op = get_op("UnsqueezeCopy")
    assert op is not None


def test_unstacked_2241():
    op = get_op("Unstacked")
    assert op is not None


def test_untypedstorage_2242():
    op = get_op("UntypedStorage")
    assert op is not None


def test_upsampling1d_2243():
    op = get_op("UpSampling1d")
    assert op is not None


def test_upsampling2d_2244():
    op = get_op("UpSampling2d")
    assert op is not None


def test_upsampling3d_2245():
    op = get_op("UpSampling3d")
    assert op is not None


def test_updatecontext_2246():
    op = get_op("UpdateContext")
    assert op is not None


def test_updatecontextmanager_2247():
    op = get_op("UpdateContextManager")
    assert op is not None


def test_updatecvstate_2248():
    op = get_op("UpdateCvState")
    assert op is not None


def test_updatefn_2249():
    op = get_op("UpdateFn")
    assert op is not None


def test_updatestate_2250():
    op = get_op("UpdateState")
    assert op is not None


def test_updates_2251():
    op = get_op("Updates")
    assert op is not None


def test_upsamplingbilinear2d_2252():
    op = get_op("UpsamplingBilinear2d")
    assert op is not None


def test_upsamplingnearest2d_2253():
    op = get_op("UpsamplingNearest2d")
    assert op is not None


def test_use_2254():
    op = get_op("Use")
    assert op is not None


def test_usedeterministicalgorithms_2255():
    op = get_op("UseDeterministicAlgorithms")
    assert op is not None


def test_useeagersharding_2256():
    op = get_op("UseEagerSharding")
    assert op is not None


def test_useglobaldeps_2257():
    op = get_op("UseGlobalDeps")
    assert op is not None


def test_usehijax_2258():
    op = get_op("UseHijax")
    assert op is not None


def test_usertldglobal_2259():
    op = get_op("UseRtldGlobal")
    assert op is not None


def test_usingeagersharding_2260():
    op = get_op("UsingEagerSharding")
    assert op is not None


def test_usinghijax_2261():
    op = get_op("UsingHijax")
    assert op is not None


def test_v_2262():
    op = get_op("V")
    assert op is not None


def test_vmem_2263():
    op = get_op("VMEM")
    assert op is not None


def test_vmem_shared_2264():
    op = get_op("VMEM_SHARED")
    assert op is not None


def test_vqngrammer_2265():
    op = get_op("VQNgrammer")
    assert op is not None


def test_validrooflinedtype_2266():
    op = get_op("ValidRooflineDtype")
    assert op is not None


def test_valueandgrad_2267():
    op = get_op("ValueAndGrad")
    assert op is not None


def test_valuemetadata_2268():
    op = get_op("ValueMetadata")
    assert op is not None


def test_valuesite_2269():
    op = get_op("ValueSite")
    assert op is not None


def test_valuesitesforvariable_2270():
    op = get_op("ValueSitesForVariable")
    assert op is not None


def test_valuescopy_2271():
    op = get_op("ValuesCopy")
    assert op is not None


def test_vanillablock_2272():
    op = get_op("VanillaBlock")
    assert op is not None


def test_vanillanet_2273():
    op = get_op("VanillaNet")
    assert op is not None


def test_vardefaults_2274():
    op = get_op("VarDefaults")
    assert op is not None


def test_vardefaultscontext_2275():
    op = get_op("VarDefaultsContext")
    assert op is not None


def test_varmean_2276():
    op = get_op("VarMean")
    assert op is not None


def test_variable_2277():
    op = get_op("Variable")
    assert op is not None


def test_variablecontext_2278():
    op = get_op("VariableContext")
    assert op is not None


def test_variabledef_2279():
    op = get_op("VariableDef")
    assert op is not None


def test_variableeffect_2280():
    op = get_op("VariableEffect")
    assert op is not None


def test_variablekey_2281():
    op = get_op("VariableKey")
    assert op is not None


def test_variablemeta_2282():
    op = get_op("VariableMeta")
    assert op is not None


def test_variablemetadata_2283():
    op = get_op("VariableMetadata")
    assert op is not None


def test_variablenamefromtype_2284():
    op = get_op("VariableNameFromType")
    assert op is not None


def test_variableqdd_2285():
    op = get_op("VariableQDD")
    assert op is not None


def test_variablerepr_2286():
    op = get_op("VariableRepr")
    assert op is not None


def test_variabletype_2287():
    op = get_op("VariableType")
    assert op is not None


def test_variabletypefromname_2288():
    op = get_op("VariableTypeFromName")
    assert op is not None


def test_variance_2289():
    op = get_op("Variance")
    assert op is not None


def test_vdot_2290():
    op = get_op("Vdot")
    assert op is not None


def test_vecdot_2291():
    op = get_op("Vecdot")
    assert op is not None


def test_vectornorm_2292():
    op = get_op("VectorNorm")
    assert op is not None


def test_vectorquantization_2293():
    op = get_op("VectorQuantization")
    assert op is not None


def test_vectorquantizer_2294():
    op = get_op("VectorQuantizer")
    assert op is not None


def test_viewascomplex_2295():
    op = get_op("ViewAsComplex")
    assert op is not None


def test_viewascomplexcopy_2296():
    op = get_op("ViewAsComplexCopy")
    assert op is not None


def test_viewasreal_2297():
    op = get_op("ViewAsReal")
    assert op is not None


def test_viewasrealcopy_2298():
    op = get_op("ViewAsRealCopy")
    assert op is not None


def test_viewcopy_2299():
    op = get_op("ViewCopy")
    assert op is not None


def test_visiontransformer_2300():
    op = get_op("VisionTransformer")
    assert op is not None


def test_vitentrylayers_2301():
    op = get_op("VitEntryLayers")
    assert op is not None


def test_vitexitlayers_2302():
    op = get_op("VitExitLayers")
    assert op is not None


def test_vitalsenabled_2303():
    op = get_op("VitalsEnabled")
    assert op is not None


def test_vmap_2304():
    op = get_op("Vmap")
    assert op is not None


def test_vmapfn_2305():
    op = get_op("VmapFn")
    assert op is not None


def test_void_2306():
    op = get_op("Void")
    assert op is not None


def test_vonmises_2307():
    op = get_op("VonMises")
    assert op is not None


def test_warn_for_unfused_kernels_2308():
    op = get_op("WARN_FOR_UNFUSED_KERNELS")
    assert op is not None


def test_warp_2309():
    op = get_op("WARP")
    assert op is not None


def test_warpgroup_size_2310():
    op = get_op("WARPGROUP_SIZE")
    assert op is not None


def test_warps_in_warpgroup_2311():
    op = get_op("WARPS_IN_WARPGROUP")
    assert op is not None


def test_warp_size_2312():
    op = get_op("WARP_SIZE")
    assert op is not None


def test_wgmmaaccumulator_2313():
    op = get_op("WGMMAAccumulator")
    assert op is not None


def test_wgmma_col_layout_2314():
    op = get_op("WGMMA_COL_LAYOUT")
    assert op is not None


def test_wgmma_layout_2315():
    op = get_op("WGMMA_LAYOUT")
    assert op is not None


def test_wgmma_layout_8bit_2316():
    op = get_op("WGMMA_LAYOUT_8BIT")
    assert op is not None


def test_wgmma_layout_acc_32bit_2317():
    op = get_op("WGMMA_LAYOUT_ACC_32BIT")
    assert op is not None


def test_wgmma_layout_upcast_2x_2318():
    op = get_op("WGMMA_LAYOUT_UPCAST_2X")
    assert op is not None


def test_wgmma_layout_upcast_4x_2319():
    op = get_op("WGMMA_LAYOUT_UPCAST_4X")
    assert op is not None


def test_wgmma_row_layout_2320():
    op = get_op("WGMMA_ROW_LAYOUT")
    assert op is not None


def test_wgmma_transposed_layout_2321():
    op = get_op("WGMMA_TRANSPOSED_LAYOUT")
    assert op is not None


def test_wgsplatfraglayout_2322():
    op = get_op("WGSplatFragLayout")
    assert op is not None


def test_wgstridedfraglayout_2323():
    op = get_op("WGStridedFragLayout")
    assert op is not None


def test_workgroup_nvptx_address_space_2324():
    op = get_op("WORKGROUP_NVPTX_ADDRESS_SPACE")
    assert op is not None


def test_write_dq_2325():
    op = get_op("WRITE_DQ")
    assert op is not None


def test_wait_2326():
    op = get_op("Wait")
    assert op is not None


def test_wald_2327():
    op = get_op("Wald")
    assert op is not None


def test_warpgroup_2328():
    op = get_op("Warpgroup")
    assert op is not None


def test_weibull_2329():
    op = get_op("Weibull")
    assert op is not None


def test_weibullmin_2330():
    op = get_op("WeibullMin")
    assert op is not None


def test_weightnorm_2331():
    op = get_op("WeightNorm")
    assert op is not None


def test_welch_2332():
    op = get_op("Welch")
    assert op is not None


def test_welford_2333():
    op = get_op("Welford")
    assert op is not None


def test_whileloop_2334():
    op = get_op("WhileLoop")
    assert op is not None


def test_whileloopbodyfn_2335():
    op = get_op("WhileLoopBodyFn")
    assert op is not None


def test_whileloopcondfn_2336():
    op = get_op("WhileLoopCondFn")
    assert op is not None


def test_withmetadata_2337():
    op = get_op("WithMetadata")
    assert op is not None


def test_withpartitioning_2338():
    op = get_op("WithPartitioning")
    assert op is not None


def test_withshardingconstraint_2339():
    op = get_op("WithShardingConstraint")
    assert op is not None


def test_withtag_2340():
    op = get_op("WithTag")
    assert op is not None


def test_wrapkeydata_2341():
    op = get_op("WrapKeyData")
    assert op is not None


def test_wrappedschedule_2342():
    op = get_op("WrappedSchedule")
    assert op is not None


def test_wrapper_2343():
    op = get_op("Wrapper")
    assert op is not None


def test_xdivy_2344():
    op = get_op("Xdivy")
    assert op is not None


def test_xlogy__2345():
    op = get_op("Xlogy_")
    assert op is not None


def test_yiqtorgb_2346():
    op = get_op("YiqToRgb")
    assert op is not None


def test_yuvtorgb_2347():
    op = get_op("YuvToRgb")
    assert op is not None


def test_zerofraction_2348():
    op = get_op("ZeroFraction")
    assert op is not None


def test_zeronansstate_2349():
    op = get_op("ZeroNansState")
    assert op is not None


def test_zeropad1d_2350():
    op = get_op("ZeroPad1d")
    assert op is not None


def test_zeropad2d_2351():
    op = get_op("ZeroPad2d")
    assert op is not None


def test_zeropad3d_2352():
    op = get_op("ZeroPad3d")
    assert op is not None


def test_zeropadding1d_2353():
    op = get_op("ZeroPadding1d")
    assert op is not None


def test_zeropadding2d_2354():
    op = get_op("ZeroPadding2d")
    assert op is not None


def test_zeropadding3d_2355():
    op = get_op("ZeroPadding3d")
    assert op is not None


def test_zeroseries_2356():
    op = get_op("ZeroSeries")
    assert op is not None


def test_zeroterm_2357():
    op = get_op("ZeroTerm")
    assert op is not None


def test_zero__2358():
    op = get_op("Zero_")
    assert op is not None


def test_zeros_2359():
    op = get_op("Zeros")
    assert op is not None


def test_zeroslike_2360():
    op = get_op("ZerosLike")
    assert op is not None


def test_zeta_2361():
    op = get_op("Zeta")
    assert op is not None


def test___array_namespace_info___2362():
    op = get_op("__array_namespace_info__")
    assert op is not None


def test_abs2_2363():
    op = get_op("abs2")
    assert op is not None


def test_absolute_2364():
    op = get_op("absolute")
    assert op is not None


def test_acc_2365():
    op = get_op("acc")
    assert op is not None


def test_accuracy_attr_2366():
    op = get_op("accuracy_attr")
    assert op is not None


def test_acos_2367():
    op = get_op("acos")
    assert op is not None


def test_acosh_2368():
    op = get_op("acosh")
    assert op is not None


def test_activate_flash_attention_impl_2369():
    op = get_op("activate_flash_attention_impl")
    assert op is not None


def test_activation_2370():
    op = get_op("activation")
    assert op is not None


def test_activation_relu_or_gelu_2371():
    op = get_op("activation_relu_or_gelu")
    assert op is not None


def test_actual_end_2372():
    op = get_op("actual_end")
    assert op is not None


def test_actual_size_2373():
    op = get_op("actual_size")
    assert op is not None


def test_actual_start_2374():
    op = get_op("actual_start")
    assert op is not None


def test_adapt_2375():
    op = get_op("adapt")
    assert op is not None


def test_adaptive_average_pool_2376():
    op = get_op("adaptive_average_pool")
    assert op is not None


def test_adaptive_avg_pool1d_2377():
    op = get_op("adaptive_avg_pool1d")
    assert op is not None


def test_adaptive_avg_pool2d_2378():
    op = get_op("adaptive_avg_pool2d")
    assert op is not None


def test_adaptive_avg_pool3d_2379():
    op = get_op("adaptive_avg_pool3d")
    assert op is not None


def test_adaptive_grad_clip_2380():
    op = get_op("adaptive_grad_clip")
    assert op is not None


def test_adaptive_max_pool_2381():
    op = get_op("adaptive_max_pool")
    assert op is not None


def test_adaptive_max_pool1d_2382():
    op = get_op("adaptive_max_pool1d")
    assert op is not None


def test_adaptive_max_pool1d_with_indices_2383():
    op = get_op("adaptive_max_pool1d_with_indices")
    assert op is not None


def test_adaptive_max_pool2d_2384():
    op = get_op("adaptive_max_pool2d")
    assert op is not None


def test_adaptive_max_pool2d_with_indices_2385():
    op = get_op("adaptive_max_pool2d_with_indices")
    assert op is not None


def test_adaptive_max_pool3d_2386():
    op = get_op("adaptive_max_pool3d")
    assert op is not None


def test_adaptive_max_pool3d_with_indices_2387():
    op = get_op("adaptive_max_pool3d_with_indices")
    assert op is not None


def test_add_axis_2388():
    op = get_op("add_axis")
    assert op is not None


def test_add_decayed_weights_2389():
    op = get_op("add_decayed_weights")
    assert op is not None


def test_add_module_2390():
    op = get_op("add_module")
    assert op is not None


def test_add_newdoc_2391():
    op = get_op("add_newdoc")
    assert op is not None


def test_add_newdoc_for_scalar_type_2392():
    op = get_op("add_newdoc_for_scalar_type")
    assert op is not None


def test_add_noise_2393():
    op = get_op("add_noise")
    assert op is not None


def test_add_pruning_method_2394():
    op = get_op("add_pruning_method")
    assert op is not None


def test_add_scale_2395():
    op = get_op("add_scale")
    assert op is not None


def test_add_to_call_tf_concrete_function_list_2396():
    op = get_op("add_to_call_tf_concrete_function_list")
    assert op is not None


def test_add_zero_attn_2397():
    op = get_op("add_zero_attn")
    assert op is not None


def test_addf_2398():
    op = get_op("addf")
    assert op is not None


def test_address_2399():
    op = get_op("address")
    assert op is not None


def test_addressable_data_2400():
    op = get_op("addressable_data")
    assert op is not None


def test_affine_2401():
    op = get_op("affine")
    assert op is not None


def test_affine_grid_2402():
    op = get_op("affine_grid")
    assert op is not None


def test_after_all_2403():
    op = get_op("after_all")
    assert op is not None


def test_after_conversion_2404():
    op = get_op("after_conversion")
    assert op is not None


def test_ag_kernel_2405():
    op = get_op("ag_kernel")
    assert op is not None


def test_airy_ai_2406():
    op = get_op("airy_ai")
    assert op is not None


def test_align_corners_2407():
    op = get_op("align_corners")
    assert op is not None


def test_all_2408():
    op = get_op("all")
    assert op is not None


def test_all_gather_2409():
    op = get_op("all_gather")
    assert op is not None


def test_all_gather_done_2410():
    op = get_op("all_gather_done")
    assert op is not None


def test_all_gather_invariant_2411():
    op = get_op("all_gather_invariant")
    assert op is not None


def test_all_gather_lhs_matmul_2412():
    op = get_op("all_gather_lhs_matmul")
    assert op is not None


def test_all_gather_reduced_2413():
    op = get_op("all_gather_reduced")
    assert op is not None


def test_all_gather_start_2414():
    op = get_op("all_gather_start")
    assert op is not None


def test_all_passes_2415():
    op = get_op("all_passes")
    assert op is not None


def test_all_to_all_2416():
    op = get_op("all_to_all")
    assert op is not None


def test_all_weights_2417():
    op = get_op("all_weights")
    assert op is not None


def test_allclose_2418():
    op = get_op("allclose")
    assert op is not None


def test_alpha_2419():
    op = get_op("alpha")
    assert op is not None


def test_alpha_dropout_2420():
    op = get_op("alpha_dropout")
    assert op is not None


def test_amax_2421():
    op = get_op("amax")
    assert op is not None


def test_amin_2422():
    op = get_op("amin")
    assert op is not None


def test_amount_2423():
    op = get_op("amount")
    assert op is not None


def test_and_masks_2424():
    op = get_op("and_masks")
    assert op is not None


def test_angle_2425():
    op = get_op("angle")
    assert op is not None


def test_any_2426():
    op = get_op("any")
    assert op is not None


def test_append_2427():
    op = get_op("append")
    assert op is not None


def test_apply_2428():
    op = get_op("apply")
    assert op is not None


def test_apply_along_axis_2429():
    op = get_op("apply_along_axis")
    assert op is not None


def test_apply_gradients_2430():
    op = get_op("apply_gradients")
    assert op is not None


def test_apply_if_finite_2431():
    op = get_op("apply_if_finite")
    assert op is not None


def test_apply_mask_2432():
    op = get_op("apply_mask")
    assert op is not None


def test_apply_over_axes_2433():
    op = get_op("apply_over_axes")
    assert op is not None


def test_apply_permutation_2434():
    op = get_op("apply_permutation")
    assert op is not None


def test_apply_variable_updates_2435():
    op = get_op("apply_variable_updates")
    assert op is not None


def test_approx_max_k_2436():
    op = get_op("approx_max_k")
    assert op is not None


def test_approx_min_k_2437():
    op = get_op("approx_min_k")
    assert op is not None


def test_approximate_2438():
    op = get_op("approximate")
    assert op is not None


def test_arange_2439():
    op = get_op("arange")
    assert op is not None


def test_arccos_2440():
    op = get_op("arccos")
    assert op is not None


def test_arccosh_2441():
    op = get_op("arccosh")
    assert op is not None


def test_arcsin_2442():
    op = get_op("arcsin")
    assert op is not None


def test_arcsinh_2443():
    op = get_op("arcsinh")
    assert op is not None


def test_arctan_2444():
    op = get_op("arctan")
    assert op is not None


def test_arctan2_2445():
    op = get_op("arctan2")
    assert op is not None


def test_arctanh_2446():
    op = get_op("arctanh")
    assert op is not None


def test_are_hlo_shardings_equal_2447():
    op = get_op("are_hlo_shardings_equal")
    assert op is not None


def test_argmax_2448():
    op = get_op("argmax")
    assert op is not None


def test_argmin_2449():
    op = get_op("argmin")
    assert op is not None


def test_argpartition_2450():
    op = get_op("argpartition")
    assert op is not None


def test_args_2451():
    op = get_op("args")
    assert op is not None


def test_args_specs_2452():
    op = get_op("args_specs")
    assert op is not None


def test_argsort_2453():
    op = get_op("argsort")
    assert op is not None


def test_argwhere_2454():
    op = get_op("argwhere")
    assert op is not None


def test_around_2455():
    op = get_op("around")
    assert op is not None


def test_array_2456():
    op = get_op("array")
    assert op is not None


def test_array2string_2457():
    op = get_op("array2string")
    assert op is not None


def test_array_equal_2458():
    op = get_op("array_equal")
    assert op is not None


def test_array_equiv_2459():
    op = get_op("array_equiv")
    assert op is not None


def test_array_function_dispatch_2460():
    op = get_op("array_function_dispatch")
    assert op is not None


def test_array_function_errmsg_formatter_2461():
    op = get_op("array_function_errmsg_formatter")
    assert op is not None


def test_array_function_from_dispatcher_2462():
    op = get_op("array_function_from_dispatcher")
    assert op is not None


def test_array_mapping_to_axis_resources_2463():
    op = get_op("array_mapping_to_axis_resources")
    assert op is not None


def test_array_repr_2464():
    op = get_op("array_repr")
    assert op is not None


def test_array_split_2465():
    op = get_op("array_split")
    assert op is not None


def test_array_str_2466():
    op = get_op("array_str")
    assert op is not None


def test_array_ufunc_errmsg_formatter_2467():
    op = get_op("array_ufunc_errmsg_formatter")
    assert op is not None


def test_arrays_to_spvalues_2468():
    op = get_op("arrays_to_spvalues")
    assert op is not None


def test_arrival_count_2469():
    op = get_op("arrival_count")
    assert op is not None


def test_arrive_2470():
    op = get_op("arrive")
    assert op is not None


def test_arrive_expect_tx_2471():
    op = get_op("arrive_expect_tx")
    assert op is not None


def test_artificial_shared_memory_limit_2472():
    op = get_op("artificial_shared_memory_limit")
    assert op is not None


def test_as_barrier_memref_2473():
    op = get_op("as_barrier_memref")
    assert op is not None


def test_as_gpu_kernel_2474():
    op = get_op("as_gpu_kernel")
    assert op is not None


def test_as_strided_2475():
    op = get_op("as_strided")
    assert op is not None


def test_as_string_2476():
    op = get_op("as_string")
    assert op is not None


def test_as_tiled_layout_2477():
    op = get_op("as_tiled_layout")
    assert op is not None


def test_as_torch_gpu_kernel_2478():
    op = get_op("as_torch_gpu_kernel")
    assert op is not None


def test_as_tuple_2479():
    op = get_op("as_tuple")
    assert op is not None


def test_asarray_2480():
    op = get_op("asarray")
    assert op is not None


def test_asin_2481():
    op = get_op("asin")
    assert op is not None


def test_asinh_2482():
    op = get_op("asinh")
    assert op is not None


def test_assertsparsearraysequivalent_2483():
    op = get_op("assertSparseArraysEquivalent")
    assert op is not None


def test_assert_equal_2484():
    op = get_op("assert_equal")
    assert op is not None


def test_assert_int_or_pair_2485():
    op = get_op("assert_int_or_pair")
    assert op is not None


def test_assert_is_supported_dtype_2486():
    op = get_op("assert_is_supported_dtype")
    assert op is not None


def test_assign_layouts_2487():
    op = get_op("assign_layouts")
    assert op is not None


def test_assignments_2488():
    op = get_op("assignments")
    assert op is not None


def test_associative_scan_2489():
    op = get_op("associative_scan")
    assert op is not None


def test_astype_2490():
    op = get_op("astype")
    assert op is not None


def test_async_copy_2491():
    op = get_op("async_copy")
    assert op is not None


def test_async_copy_scales_smem_to_tmem_2492():
    op = get_op("async_copy_scales_smem_to_tmem")
    assert op is not None


def test_async_copy_smem_to_tmem_2493():
    op = get_op("async_copy_smem_to_tmem")
    assert op is not None


def test_async_copy_sparse_metadata_smem_to_tmem_2494():
    op = get_op("async_copy_sparse_metadata_smem_to_tmem")
    assert op is not None


def test_async_deserialize_2495():
    op = get_op("async_deserialize")
    assert op is not None


def test_async_prefetch_2496():
    op = get_op("async_prefetch")
    assert op is not None


def test_async_serialize_2497():
    op = get_op("async_serialize")
    assert op is not None


def test_at_2498():
    op = get_op("at")
    assert op is not None


def test_atan_2499():
    op = get_op("atan")
    assert op is not None


def test_atan2_2500():
    op = get_op("atan2")
    assert op is not None


def test_atanh_2501():
    op = get_op("atanh")
    assert op is not None


def test_atleast_1d_2502():
    op = get_op("atleast_1d")
    assert op is not None


def test_atleast_2d_2503():
    op = get_op("atleast_2d")
    assert op is not None


def test_atleast_3d_2504():
    op = get_op("atleast_3d")
    assert op is not None


def test_attention_reference_2505():
    op = get_op("attention_reference")
    assert op is not None


def test_attention_reference_custom_2506():
    op = get_op("attention_reference_custom")
    assert op is not None


def test_attention_with_pipeline_emitter_2507():
    op = get_op("attention_with_pipeline_emitter")
    assert op is not None


def test_attn_forward_kernel_2508():
    op = get_op("attn_forward_kernel")
    assert op is not None


def test_attr_2509():
    op = get_op("attr")
    assert op is not None


def test_attr_element_2510():
    op = get_op("attr_element")
    assert op is not None


def test_attr_get_2511():
    op = get_op("attr_get")
    assert op is not None


def test_auto_barriers_2512():
    op = get_op("auto_barriers")
    assert op is not None


def test_aval_2513():
    op = get_op("aval")
    assert op is not None


def test_avals_in_2514():
    op = get_op("avals_in")
    assert op is not None


def test_avals_out_2515():
    op = get_op("avals_out")
    assert op is not None


def test_average_2516():
    op = get_op("average")
    assert op is not None


def test_average_pool_2517():
    op = get_op("average_pool")
    assert op is not None


def test_avg_pool1d_2518():
    op = get_op("avg_pool1d")
    assert op is not None


def test_avg_pool2d_2519():
    op = get_op("avg_pool2d")
    assert op is not None


def test_avg_pool3d_2520():
    op = get_op("avg_pool3d")
    assert op is not None


def test_await_async_copy_2521():
    op = get_op("await_async_copy")
    assert op is not None


def test_await_cp_async_copy_2522():
    op = get_op("await_cp_async_copy")
    assert op is not None


def test_await_pull_2523():
    op = get_op("await_pull")
    assert op is not None


def test_axes_2524():
    op = get_op("axes")
    assert op is not None


def test_axis_2525():
    op = get_op("axis")
    assert op is not None


def test_axis_index_2526():
    op = get_op("axis_index")
    assert op is not None


def test_axis_size_2527():
    op = get_op("axis_size")
    assert op is not None


def test_backup_keys_2528():
    op = get_op("backup_keys")
    assert op is not None


def test_barrier_2529():
    op = get_op("barrier")
    assert op is not None


def test_barrier_ref_2530():
    op = get_op("barrier_ref")
    assert op is not None


def test_bartlett_2531():
    op = get_op("bartlett")
    assert op is not None


def test_base_2532():
    op = get_op("base")
    assert op is not None


def test_base_address_2533():
    op = get_op("base_address")
    assert op is not None


def test_base_offset_2534():
    op = get_op("base_offset")
    assert op is not None


def test_base_repr_2535():
    op = get_op("base_repr")
    assert op is not None


def test_base_tile_shape_2536():
    op = get_op("base_tile_shape")
    assert op is not None


def test_batch_2537():
    op = get_op("batch")
    assert op is not None


def test_batch_first_2538():
    op = get_op("batch_first")
    assert op is not None


def test_batch_matmul_2539():
    op = get_op("batch_matmul")
    assert op is not None


def test_batch_norm_2540():
    op = get_op("batch_norm")
    assert op is not None


def test_batch_normalization_2541():
    op = get_op("batch_normalization")
    assert op is not None


def test_batch_shape_2542():
    op = get_op("batch_shape")
    assert op is not None


def test_batch_sizes_2543():
    op = get_op("batch_sizes")
    assert op is not None


def test_bcoo_broadcast_in_dim_2544():
    op = get_op("bcoo_broadcast_in_dim")
    assert op is not None


def test_bcoo_concatenate_2545():
    op = get_op("bcoo_concatenate")
    assert op is not None


def test_bcoo_conv_general_dilated_2546():
    op = get_op("bcoo_conv_general_dilated")
    assert op is not None


def test_bcoo_dot_general_2547():
    op = get_op("bcoo_dot_general")
    assert op is not None


def test_bcoo_dot_general_p_2548():
    op = get_op("bcoo_dot_general_p")
    assert op is not None


def test_bcoo_dot_general_sampled_2549():
    op = get_op("bcoo_dot_general_sampled")
    assert op is not None


def test_bcoo_dot_general_sampled_p_2550():
    op = get_op("bcoo_dot_general_sampled_p")
    assert op is not None


def test_bcoo_dynamic_slice_2551():
    op = get_op("bcoo_dynamic_slice")
    assert op is not None


def test_bcoo_eliminate_zeros_2552():
    op = get_op("bcoo_eliminate_zeros")
    assert op is not None


def test_bcoo_extract_2553():
    op = get_op("bcoo_extract")
    assert op is not None


def test_bcoo_extract_p_2554():
    op = get_op("bcoo_extract_p")
    assert op is not None


def test_bcoo_fromdense_2555():
    op = get_op("bcoo_fromdense")
    assert op is not None


def test_bcoo_fromdense_p_2556():
    op = get_op("bcoo_fromdense_p")
    assert op is not None


def test_bcoo_gather_2557():
    op = get_op("bcoo_gather")
    assert op is not None


def test_bcoo_multiply_dense_2558():
    op = get_op("bcoo_multiply_dense")
    assert op is not None


def test_bcoo_multiply_sparse_2559():
    op = get_op("bcoo_multiply_sparse")
    assert op is not None


def test_bcoo_reduce_sum_2560():
    op = get_op("bcoo_reduce_sum")
    assert op is not None


def test_bcoo_reshape_2561():
    op = get_op("bcoo_reshape")
    assert op is not None


def test_bcoo_rev_2562():
    op = get_op("bcoo_rev")
    assert op is not None


def test_bcoo_slice_2563():
    op = get_op("bcoo_slice")
    assert op is not None


def test_bcoo_sort_indices_2564():
    op = get_op("bcoo_sort_indices")
    assert op is not None


def test_bcoo_sort_indices_p_2565():
    op = get_op("bcoo_sort_indices_p")
    assert op is not None


def test_bcoo_spdot_general_p_2566():
    op = get_op("bcoo_spdot_general_p")
    assert op is not None


def test_bcoo_squeeze_2567():
    op = get_op("bcoo_squeeze")
    assert op is not None


def test_bcoo_sum_duplicates_2568():
    op = get_op("bcoo_sum_duplicates")
    assert op is not None


def test_bcoo_sum_duplicates_p_2569():
    op = get_op("bcoo_sum_duplicates_p")
    assert op is not None


def test_bcoo_todense_2570():
    op = get_op("bcoo_todense")
    assert op is not None


def test_bcoo_todense_p_2571():
    op = get_op("bcoo_todense_p")
    assert op is not None


def test_bcoo_transpose_2572():
    op = get_op("bcoo_transpose")
    assert op is not None


def test_bcoo_transpose_p_2573():
    op = get_op("bcoo_transpose_p")
    assert op is not None


def test_bcoo_update_layout_2574():
    op = get_op("bcoo_update_layout")
    assert op is not None


def test_bcsr_broadcast_in_dim_2575():
    op = get_op("bcsr_broadcast_in_dim")
    assert op is not None


def test_bcsr_concatenate_2576():
    op = get_op("bcsr_concatenate")
    assert op is not None


def test_bcsr_dot_general_2577():
    op = get_op("bcsr_dot_general")
    assert op is not None


def test_bcsr_dot_general_p_2578():
    op = get_op("bcsr_dot_general_p")
    assert op is not None


def test_bcsr_eliminate_zeros_2579():
    op = get_op("bcsr_eliminate_zeros")
    assert op is not None


def test_bcsr_extract_2580():
    op = get_op("bcsr_extract")
    assert op is not None


def test_bcsr_extract_p_2581():
    op = get_op("bcsr_extract_p")
    assert op is not None


def test_bcsr_fromdense_2582():
    op = get_op("bcsr_fromdense")
    assert op is not None


def test_bcsr_fromdense_p_2583():
    op = get_op("bcsr_fromdense_p")
    assert op is not None


def test_bcsr_sum_duplicates_2584():
    op = get_op("bcsr_sum_duplicates")
    assert op is not None


def test_bcsr_todense_2585():
    op = get_op("bcsr_todense")
    assert op is not None


def test_bcsr_todense_p_2586():
    op = get_op("bcsr_todense_p")
    assert op is not None


def test_before_conversion_2587():
    op = get_op("before_conversion")
    assert op is not None


def test_below_or_on_diag_2588():
    op = get_op("below_or_on_diag")
    assert op is not None


def test_bessel_i0e_2589():
    op = get_op("bessel_i0e")
    assert op is not None


def test_bessel_i0e_impl_2590():
    op = get_op("bessel_i0e_impl")
    assert op is not None


def test_bessel_i1e_2591():
    op = get_op("bessel_i1e")
    assert op is not None


def test_bessel_j0_2592():
    op = get_op("bessel_j0")
    assert op is not None


def test_bessel_j1_2593():
    op = get_op("bessel_j1")
    assert op is not None


def test_bessel_y0_2594():
    op = get_op("bessel_y0")
    assert op is not None


def test_bessel_y1_2595():
    op = get_op("bessel_y1")
    assert op is not None


def test_beta_2596():
    op = get_op("beta")
    assert op is not None


def test_betainc_2597():
    op = get_op("betainc")
    assert op is not None


def test_betainc_grad_not_implemented_2598():
    op = get_op("betainc_grad_not_implemented")
    assert op is not None


def test_betainc_gradx_2599():
    op = get_op("betainc_gradx")
    assert op is not None


def test_bfloat16_2600():
    op = get_op("bfloat16")
    assert op is not None


def test_bias_2601():
    op = get_op("bias")
    assert op is not None


def test_bias_correction_2602():
    op = get_op("bias_correction")
    assert op is not None


def test_bias_hh_2603():
    op = get_op("bias_hh")
    assert op is not None


def test_bias_ih_2604():
    op = get_op("bias_ih")
    assert op is not None


def test_bias_k_2605():
    op = get_op("bias_k")
    assert op is not None


def test_bias_v_2606():
    op = get_op("bias_v")
    assert op is not None


def test_bidirectional_2607():
    op = get_op("bidirectional")
    assert op is not None


def test_bilinear_2608():
    op = get_op("bilinear")
    assert op is not None


def test_binary_cross_entropy_2609():
    op = get_op("binary_cross_entropy")
    assert op is not None


def test_binary_cross_entropy_with_logits_2610():
    op = get_op("binary_cross_entropy_with_logits")
    assert op is not None


def test_binary_crossentropy_2611():
    op = get_op("binary_crossentropy")
    assert op is not None


def test_binary_focal_crossentropy_2612():
    op = get_op("binary_focal_crossentropy")
    assert op is not None


def test_binary_repr_2613():
    op = get_op("binary_repr")
    assert op is not None


def test_binary_ufunc_2614():
    op = get_op("binary_ufunc")
    assert op is not None


def test_bincount_2615():
    op = get_op("bincount")
    assert op is not None


def test_bind_2616():
    op = get_op("bind")
    assert op is not None


def test_bind_psum_invariant_2617():
    op = get_op("bind_psum_invariant")
    assert op is not None


def test_binomial_2618():
    op = get_op("binomial")
    assert op is not None


def test_bitcast_convert_type_2619():
    op = get_op("bitcast_convert_type")
    assert op is not None


def test_bits_2620():
    op = get_op("bits")
    assert op is not None


def test_bitwidth_2621():
    op = get_op("bitwidth")
    assert op is not None


def test_bitwidth_impl_2622():
    op = get_op("bitwidth_impl")
    assert op is not None


def test_bitwise_and_2623():
    op = get_op("bitwise_and")
    assert op is not None


def test_bitwise_count_2624():
    op = get_op("bitwise_count")
    assert op is not None


def test_bitwise_invert_2625():
    op = get_op("bitwise_invert")
    assert op is not None


def test_bitwise_left_shift_2626():
    op = get_op("bitwise_left_shift")
    assert op is not None


def test_bitwise_not_2627():
    op = get_op("bitwise_not")
    assert op is not None


def test_bitwise_or_2628():
    op = get_op("bitwise_or")
    assert op is not None


def test_bitwise_right_shift_2629():
    op = get_op("bitwise_right_shift")
    assert op is not None


def test_bitwise_xor_2630():
    op = get_op("bitwise_xor")
    assert op is not None


def test_blackman_2631():
    op = get_op("blackman")
    assert op is not None


def test_blank_2632():
    op = get_op("blank")
    assert op is not None


def test_block_2633():
    op = get_op("block")
    assert op is not None


def test_block_b_2634():
    op = get_op("block_b")
    assert op is not None


def test_block_idx_2635():
    op = get_op("block_idx")
    assert op is not None


def test_block_k_2636():
    op = get_op("block_k")
    assert op is not None


def test_block_k_dkv_2637():
    op = get_op("block_k_dkv")
    assert op is not None


def test_block_k_dq_2638():
    op = get_op("block_k_dq")
    assert op is not None


def test_block_k_major_2639():
    op = get_op("block_k_major")
    assert op is not None


def test_block_k_major_dkv_2640():
    op = get_op("block_k_major_dkv")
    assert op is not None


def test_block_k_major_dq_2641():
    op = get_op("block_k_major_dq")
    assert op is not None


def test_block_kv_2642():
    op = get_op("block_kv")
    assert op is not None


def test_block_kv_compute_2643():
    op = get_op("block_kv_compute")
    assert op is not None


def test_block_kv_dkv_2644():
    op = get_op("block_kv_dkv")
    assert op is not None


def test_block_kv_dkv_compute_2645():
    op = get_op("block_kv_dkv_compute")
    assert op is not None


def test_block_kv_dq_2646():
    op = get_op("block_kv_dq")
    assert op is not None


def test_block_mask_2647():
    op = get_op("block_mask")
    assert op is not None


def test_block_q_2648():
    op = get_op("block_q")
    assert op is not None


def test_block_q_dkv_2649():
    op = get_op("block_q_dkv")
    assert op is not None


def test_block_q_dq_2650():
    op = get_op("block_q_dq")
    assert op is not None


def test_block_q_major_dkv_2651():
    op = get_op("block_q_major_dkv")
    assert op is not None


def test_block_start_2652():
    op = get_op("block_start")
    assert op is not None


def test_blocked_iota_2653():
    op = get_op("blocked_iota")
    assert op is not None


def test_bmm_einsum_2654():
    op = get_op("bmm_einsum")
    assert op is not None


def test_bool__2655():
    op = get_op("bool_")
    assert op is not None


def test_broadcast_arrays_2656():
    op = get_op("broadcast_arrays")
    assert op is not None


def test_broadcast_bucket_size_2657():
    op = get_op("broadcast_bucket_size")
    assert op is not None


def test_broadcast_buffers_2658():
    op = get_op("broadcast_buffers")
    assert op is not None


def test_broadcast_coalesced_2659():
    op = get_op("broadcast_coalesced")
    assert op is not None


def test_broadcast_in_dim_2660():
    op = get_op("broadcast_in_dim")
    assert op is not None


def test_broadcast_minor_2661():
    op = get_op("broadcast_minor")
    assert op is not None


def test_broadcast_one_to_all_2662():
    op = get_op("broadcast_one_to_all")
    assert op is not None


def test_broadcast_prefix_2663():
    op = get_op("broadcast_prefix")
    assert op is not None


def test_broadcast_shapes_2664():
    op = get_op("broadcast_shapes")
    assert op is not None


def test_broadcast_shardings_2665():
    op = get_op("broadcast_shardings")
    assert op is not None


def test_broadcast_to_2666():
    op = get_op("broadcast_to")
    assert op is not None


def test_broadcast_to_rank_2667():
    op = get_op("broadcast_to_rank")
    assert op is not None


def test_broadcasted_iota_2668():
    op = get_op("broadcasted_iota")
    assert op is not None


def test_broadcasting_shape_rule_2669():
    op = get_op("broadcasting_shape_rule")
    assert op is not None


def test_broadcasting_sharding_rule_2670():
    op = get_op("broadcasting_sharding_rule")
    assert op is not None


def test_broadcasting_vmap_2671():
    op = get_op("broadcasting_vmap")
    assert op is not None


def test_bucket_bytes_cap_2672():
    op = get_op("bucket_bytes_cap")
    assert op is not None


def test_bucket_bytes_cap_default_2673():
    op = get_op("bucket_bytes_cap_default")
    assert op is not None


def test_buffers_2674():
    op = get_op("buffers")
    assert op is not None


def test_build_2675():
    op = get_op("build")
    assert op is not None


def test_build_func_data_2676():
    op = get_op("build_func_data")
    assert op is not None


def test_busday_count_2677():
    op = get_op("busday_count")
    assert op is not None


def test_busday_offset_2678():
    op = get_op("busday_offset")
    assert op is not None


def test_byte_2679():
    op = get_op("byte")
    assert op is not None


def test_bytes_2680():
    op = get_op("bytes")
    assert op is not None


def test_bytewidth_2681():
    op = get_op("bytewidth")
    assert op is not None


def test_c_void_p_2682():
    op = get_op("c_void_p")
    assert op is not None


def test_cached_2683():
    op = get_op("cached")
    assert op is not None


def test_calculate_gain_2684():
    op = get_op("calculate_gain")
    assert op is not None


def test_call_2685():
    op = get_op("call")
    assert op is not None


def test_call_impl_2686():
    op = get_op("call_impl")
    assert op is not None


def test_call_param_updaters_2687():
    op = get_op("call_param_updaters")
    assert op is not None


def test_call_reduced_rule_2688():
    op = get_op("call_reduced_rule")
    assert op is not None


def test_call_shape_dtype_sharding_rule_2689():
    op = get_op("call_shape_dtype_sharding_rule")
    assert op is not None


def test_call_sharding_rule_2690():
    op = get_op("call_sharding_rule")
    assert op is not None


def test_call_super_init_2691():
    op = get_op("call_super_init")
    assert op is not None


def test_call_tf_2692():
    op = get_op("call_tf")
    assert op is not None


def test_call_tf_effect_2693():
    op = get_op("call_tf_effect")
    assert op is not None


def test_call_tf_ordered_effect_2694():
    op = get_op("call_tf_ordered_effect")
    assert op is not None


def test_call_tf_p_2695():
    op = get_op("call_tf_p")
    assert op is not None


def test_call_unreduced_rule_2696():
    op = get_op("call_unreduced_rule")
    assert op is not None


def test_can_broadcast_to_2697():
    op = get_op("can_broadcast_to")
    assert op is not None


def test_can_cast_2698():
    op = get_op("can_cast")
    assert op is not None


def test_can_relayout_wgmma_2x_to_wgmma_2699():
    op = get_op("can_relayout_wgmma_2x_to_wgmma")
    assert op is not None


def test_can_relayout_wgmma_4x_to_wgmma_2x_2700():
    op = get_op("can_relayout_wgmma_4x_to_wgmma_2x")
    assert op is not None


def test_canonicalize_2701():
    op = get_op("canonicalize")
    assert op is not None


def test_canonicalize_device_to_sharding_2702():
    op = get_op("canonicalize_device_to_sharding")
    assert op is not None


def test_canonicalize_dtype_2703():
    op = get_op("canonicalize_dtype")
    assert op is not None


def test_canonicalize_filename_2704():
    op = get_op("canonicalize_filename")
    assert op is not None


def test_canonicalize_padding_2705():
    op = get_op("canonicalize_padding")
    assert op is not None


def test_canonicalize_precision_2706():
    op = get_op("canonicalize_precision")
    assert op is not None


def test_canonicalize_shape_2707():
    op = get_op("canonicalize_shape")
    assert op is not None


def test_capitalize_2708():
    op = get_op("capitalize")
    assert op is not None


def test_cast_like_2709():
    op = get_op("cast_like")
    assert op is not None


def test_categorical_2710():
    op = get_op("categorical")
    assert op is not None


def test_categorical_crossentropy_2711():
    op = get_op("categorical_crossentropy")
    assert op is not None


def test_categorical_focal_crossentropy_2712():
    op = get_op("categorical_focal_crossentropy")
    assert op is not None


def test_categorical_generalized_cross_entropy_2713():
    op = get_op("categorical_generalized_cross_entropy")
    assert op is not None


def test_categorical_hinge_2714():
    op = get_op("categorical_hinge")
    assert op is not None


def test_causal_2715():
    op = get_op("causal")
    assert op is not None


def test_causal_lower_right_2716():
    op = get_op("causal_lower_right")
    assert op is not None


def test_causal_upper_left_2717():
    op = get_op("causal_upper_left")
    assert op is not None


def test_cbrt_2718():
    op = get_op("cbrt")
    assert op is not None


def test_cdouble_2719():
    op = get_op("cdouble")
    assert op is not None


def test_ceil_2720():
    op = get_op("ceil")
    assert op is not None


def test_ceil_div_2721():
    op = get_op("ceil_div")
    assert op is not None


def test_ceil_mode_2722():
    op = get_op("ceil_mode")
    assert op is not None


def test_celu_2723():
    op = get_op("celu")
    assert op is not None


def test_celu__2724():
    op = get_op("celu_")
    assert op is not None


def test_center_2725():
    op = get_op("center")
    assert op is not None


def test_chain_2726():
    op = get_op("chain")
    assert op is not None


def test_channel_shuffle_2727():
    op = get_op("channel_shuffle")
    assert op is not None


def test_char_2728():
    op = get_op("char")
    assert op is not None


def test_character_2729():
    op = get_op("character")
    assert op is not None


def test_chararray_2730():
    op = get_op("chararray")
    assert op is not None


def test_chebyshev_polynomial_t_2731():
    op = get_op("chebyshev_polynomial_t")
    assert op is not None


def test_chebyshev_polynomial_u_2732():
    op = get_op("chebyshev_polynomial_u")
    assert op is not None


def test_chebyshev_polynomial_v_2733():
    op = get_op("chebyshev_polynomial_v")
    assert op is not None


def test_chebyshev_polynomial_w_2734():
    op = get_op("chebyshev_polynomial_w")
    assert op is not None


def test_check_api_dict_2735():
    op = get_op("check_api_dict")
    assert op is not None


def test_check_api_version_2736():
    op = get_op("check_api_version")
    assert op is not None


def test_check_arraylike_2737():
    op = get_op("check_arraylike")
    assert op is not None


def test_check_arraylike_or_none_2738():
    op = get_op("check_arraylike_or_none")
    assert op is not None


def test_check_collective_2739():
    op = get_op("check_collective")
    assert op is not None


def test_check_consistent_aliasing_2740():
    op = get_op("check_consistent_aliasing")
    assert op is not None


def test_check_for_errors_2741():
    op = get_op("check_for_errors")
    assert op is not None


def test_check_for_prngkeys_2742():
    op = get_op("check_for_prngkeys")
    assert op is not None


def test_check_forward_args_2743():
    op = get_op("check_forward_args")
    assert op is not None


def test_check_hidden_size_2744():
    op = get_op("check_hidden_size")
    assert op is not None


def test_check_input_2745():
    op = get_op("check_input")
    assert op is not None


def test_check_layout_assignment_2746():
    op = get_op("check_layout_assignment")
    assert op is not None


def test_check_no_aliases_2747():
    op = get_op("check_no_aliases")
    assert op is not None


def test_check_no_float0s_2748():
    op = get_op("check_no_float0s")
    assert op is not None


def test_check_pytree_2749():
    op = get_op("check_pytree")
    assert op is not None


def test_check_same_dtypes_2750():
    op = get_op("check_same_dtypes")
    assert op is not None


def test_check_same_variables_2751():
    op = get_op("check_same_variables")
    assert op is not None


def test_check_td_order_2752():
    op = get_op("check_td_order")
    assert op is not None


def test_check_tf_result_2753():
    op = get_op("check_tf_result")
    assert op is not None


def test_check_type_2754():
    op = get_op("check_type")
    assert op is not None


def test_check_where_2755():
    op = get_op("check_where")
    assert op is not None


def test_checkify_2756():
    op = get_op("checkify")
    assert op is not None


def test_children_2757():
    op = get_op("children")
    assert op is not None


def test_chisquare_2758():
    op = get_op("chisquare")
    assert op is not None


def test_chlo_precision_attr_2759():
    op = get_op("chlo_precision_attr")
    assert op is not None


def test_choice_2760():
    op = get_op("choice")
    assert op is not None


def test_cholesky_ex_2761():
    op = get_op("cholesky_ex")
    assert op is not None


def test_cholesky_inverse_2762():
    op = get_op("cholesky_inverse")
    assert op is not None


def test_cholesky_update_2763():
    op = get_op("cholesky_update")
    assert op is not None


def test_choose_2764():
    op = get_op("choose")
    assert op is not None


def test_choose_device_or_out_sharding_2765():
    op = get_op("choose_device_or_out_sharding")
    assert op is not None


def test_chunk_size_2766():
    op = get_op("chunk_size")
    assert op is not None


def test_circle_2767():
    op = get_op("circle")
    assert op is not None


def test_clamp_2768():
    op = get_op("clamp")
    assert op is not None


def test_clean_command_2769():
    op = get_op("clean_command")
    assert op is not None


def test_clear_2770():
    op = get_op("clear")
    assert op is not None


def test_clear_non_graph_nodes_2771():
    op = get_op("clear_non_graph_nodes")
    assert op is not None


def test_clip_2772():
    op = get_op("clip")
    assert op is not None


def test_clip_by_block_rms_2773():
    op = get_op("clip_by_block_rms")
    assert op is not None


def test_clip_by_global_norm_2774():
    op = get_op("clip_by_global_norm")
    assert op is not None


def test_clip_grad_norm_2775():
    op = get_op("clip_grad_norm")
    assert op is not None


def test_clip_grad_norm__2776():
    op = get_op("clip_grad_norm_")
    assert op is not None


def test_clip_grad_value__2777():
    op = get_op("clip_grad_value_")
    assert op is not None


def test_clip_grads_2778():
    op = get_op("clip_grads")
    assert op is not None


def test_clock_2779():
    op = get_op("clock")
    assert op is not None


def test_clone_2780():
    op = get_op("clone")
    assert op is not None


def test_close_2781():
    op = get_op("close")
    assert op is not None


def test_cloud_tpu_init_2782():
    op = get_op("cloud_tpu_init")
    assert op is not None


def test_cls_to_become_2783():
    op = get_op("cls_to_become")
    assert op is not None


def test_cluster_collective_mask_2784():
    op = get_op("cluster_collective_mask")
    assert op is not None


def test_cluster_dimension_2785():
    op = get_op("cluster_dimension")
    assert op is not None


def test_cluster_idx_2786():
    op = get_op("cluster_idx")
    assert op is not None


def test_cluster_mask_2787():
    op = get_op("cluster_mask")
    assert op is not None


def test_cluster_size_2788():
    op = get_op("cluster_size")
    assert op is not None


def test_clz_2789():
    op = get_op("clz")
    assert op is not None


def test_col_2790():
    op = get_op("col")
    assert op is not None


def test_collapse_2791():
    op = get_op("collapse")
    assert op is not None


def test_collect_profile_2792():
    op = get_op("collect_profile")
    assert op is not None


def test_collective_2793():
    op = get_op("collective")
    assert op is not None


def test_collective_dims_2794():
    op = get_op("collective_dims")
    assert op is not None


def test_collective_vma_rule_2795():
    op = get_op("collective_vma_rule")
    assert op is not None


def test_colocated_cpu_devices_2796():
    op = get_op("colocated_cpu_devices")
    assert op is not None


def test_colocated_python_2797():
    op = get_op("colocated_python")
    assert op is not None


def test_colocated_python_class_2798():
    op = get_op("colocated_python_class")
    assert op is not None


def test_colorized_2799():
    op = get_op("colorized")
    assert op is not None


def test_cols_in_shape_2800():
    op = get_op("cols_in_shape")
    assert op is not None


def test_cols_sorted_2801():
    op = get_op("cols_sorted")
    assert op is not None


def test_column_stack_2802():
    op = get_op("column_stack")
    assert op is not None


def test_combine_kvstores_2803():
    op = get_op("combine_kvstores")
    assert op is not None


def test_combine_masks_2804():
    op = get_op("combine_masks")
    assert op is not None


def test_commit_arrive_2805():
    op = get_op("commit_arrive")
    assert op is not None


def test_commit_shared_2806():
    op = get_op("commit_shared")
    assert op is not None


def test_commit_tmem_2807():
    op = get_op("commit_tmem")
    assert op is not None


def test_common_notes_2808():
    op = get_op("common_notes")
    assert op is not None


def test_common_stride_2809():
    op = get_op("common_stride")
    assert op is not None


def test_compact_2810():
    op = get_op("compact")
    assert op is not None


def test_compile_2811():
    op = get_op("compile")
    assert op is not None


def test_compile_fn_2812():
    op = get_op("compile_fn")
    assert op is not None


def test_compile_jaxpr_2813():
    op = get_op("compile_jaxpr")
    assert op is not None


def test_compile_with_env_2814():
    op = get_op("compile_with_env")
    assert op is not None


def test_complete_tx_2815():
    op = get_op("complete_tx")
    assert op is not None


def test_complex_2816():
    op = get_op("complex")
    assert op is not None


def test_complex128_2817():
    op = get_op("complex128")
    assert op is not None


def test_complex64_2818():
    op = get_op("complex64")
    assert op is not None


def test_complex__2819():
    op = get_op("complex_")
    assert op is not None


def test_complexfloating_2820():
    op = get_op("complexfloating")
    assert op is not None


def test_composite_2821():
    op = get_op("composite")
    assert op is not None


def test_composite_jvp_2822():
    op = get_op("composite_jvp")
    assert op is not None


def test_composite_transpose_2823():
    op = get_op("composite_transpose")
    assert op is not None


def test_compress_2824():
    op = get_op("compress")
    assert op is not None


def test_compute_mask_2825():
    op = get_op("compute_mask")
    assert op is not None


def test_compute_scalar_offset_2826():
    op = get_op("compute_scalar_offset")
    assert op is not None


def test_compute_transitively_equal_vars_2827():
    op = get_op("compute_transitively_equal_vars")
    assert op is not None


def test_compute_weight_2828():
    op = get_op("compute_weight")
    assert op is not None


def test_compute_wgs_bwd_2829():
    op = get_op("compute_wgs_bwd")
    assert op is not None


def test_concat_2830():
    op = get_op("concat")
    assert op is not None


def test_concatenate_2831():
    op = get_op("concatenate")
    assert op is not None


def test_conditionally_mask_2832():
    op = get_op("conditionally_mask")
    assert op is not None


def test_conditionally_transform_2833():
    op = get_op("conditionally_transform")
    assert op is not None


def test_conj_2834():
    op = get_op("conj")
    assert op is not None


def test_conjugate_2835():
    op = get_op("conjugate")
    assert op is not None


def test_conjure_assignment_2836():
    op = get_op("conjure_assignment")
    assert op is not None


def test_connect_2837():
    op = get_op("connect")
    assert op is not None


def test_constant__2838():
    op = get_op("constant_")
    assert op is not None


def test_constant_schedule_2839():
    op = get_op("constant_schedule")
    assert op is not None


def test_constraints_2840():
    op = get_op("constraints")
    assert op is not None


def test_consume_prefix_in_state_dict_if_present_2841():
    op = get_op("consume_prefix_in_state_dict_if_present")
    assert op is not None


def test_consumer_operands_2842():
    op = get_op("consumer_operands")
    assert op is not None


def test_control_delta_method_2843():
    op = get_op("control_delta_method")
    assert op is not None


def test_control_variates_jacobians_2844():
    op = get_op("control_variates_jacobians")
    assert op is not None


def test_conv_2845():
    op = get_op("conv")
    assert op is not None


def test_conv1d_input_2846():
    op = get_op("conv1d_input")
    assert op is not None


def test_conv1d_weight_2847():
    op = get_op("conv1d_weight")
    assert op is not None


def test_conv2d_input_2848():
    op = get_op("conv2d_input")
    assert op is not None


def test_conv2d_weight_2849():
    op = get_op("conv2d_weight")
    assert op is not None


def test_conv3d_input_2850():
    op = get_op("conv3d_input")
    assert op is not None


def test_conv3d_weight_2851():
    op = get_op("conv3d_weight")
    assert op is not None


def test_conv_dimension_numbers_2852():
    op = get_op("conv_dimension_numbers")
    assert op is not None


def test_conv_general_dilated_2853():
    op = get_op("conv_general_dilated")
    assert op is not None


def test_conv_general_dilated_local_2854():
    op = get_op("conv_general_dilated_local")
    assert op is not None


def test_conv_general_dilated_patches_2855():
    op = get_op("conv_general_dilated_patches")
    assert op is not None


def test_conv_general_permutations_2856():
    op = get_op("conv_general_permutations")
    assert op is not None


def test_conv_general_shape_tuple_2857():
    op = get_op("conv_general_shape_tuple")
    assert op is not None


def test_conv_shape_tuple_2858():
    op = get_op("conv_shape_tuple")
    assert op is not None


def test_conv_tbc_2859():
    op = get_op("conv_tbc")
    assert op is not None


def test_conv_transpose_2860():
    op = get_op("conv_transpose")
    assert op is not None


def test_conv_transpose1d_2861():
    op = get_op("conv_transpose1d")
    assert op is not None


def test_conv_transpose2d_2862():
    op = get_op("conv_transpose2d")
    assert op is not None


def test_conv_transpose3d_2863():
    op = get_op("conv_transpose3d")
    assert op is not None


def test_conv_transpose_shape_tuple_2864():
    op = get_op("conv_transpose_shape_tuple")
    assert op is not None


def test_conv_with_general_padding_2865():
    op = get_op("conv_with_general_padding")
    assert op is not None


def test_convert_2866():
    op = get_op("convert")
    assert op is not None


def test_convert_conv2d_weight_memory_format_2867():
    op = get_op("convert_conv2d_weight_memory_format")
    assert op is not None


def test_convert_conv3d_weight_memory_format_2868():
    op = get_op("convert_conv3d_weight_memory_format")
    assert op is not None


def test_convert_element_type_2869():
    op = get_op("convert_element_type")
    assert op is not None


def test_convert_kwargs_2870():
    op = get_op("convert_kwargs")
    assert op is not None


def test_convert_sync_batchnorm_2871():
    op = get_op("convert_sync_batchnorm")
    assert op is not None


def test_convert_to_numpy_2872():
    op = get_op("convert_to_numpy")
    assert op is not None


def test_convert_to_tensor_2873():
    op = get_op("convert_to_tensor")
    assert op is not None


def test_convex_kl_divergence_2874():
    op = get_op("convex_kl_divergence")
    assert op is not None


def test_convolution_notes_2875():
    op = get_op("convolution_notes")
    assert op is not None


def test_convolve_2876():
    op = get_op("convolve")
    assert op is not None


def test_coo_fromdense_2877():
    op = get_op("coo_fromdense")
    assert op is not None


def test_coo_fromdense_p_2878():
    op = get_op("coo_fromdense_p")
    assert op is not None


def test_coo_matmat_2879():
    op = get_op("coo_matmat")
    assert op is not None


def test_coo_matmat_p_2880():
    op = get_op("coo_matmat_p")
    assert op is not None


def test_coo_matvec_2881():
    op = get_op("coo_matvec")
    assert op is not None


def test_coo_matvec_p_2882():
    op = get_op("coo_matvec_p")
    assert op is not None


def test_coo_todense_2883():
    op = get_op("coo_todense")
    assert op is not None


def test_coo_todense_p_2884():
    op = get_op("coo_todense_p")
    assert op is not None


def test_copy_2885():
    op = get_op("copy")
    assert op is not None


def test_copy_tiled_2886():
    op = get_op("copy_tiled")
    assert op is not None


def test_copy_to_host_async_2887():
    op = get_op("copy_to_host_async")
    assert op is not None


def test_copysign_2888():
    op = get_op("copysign")
    assert op is not None


def test_copyto_2889():
    op = get_op("copyto")
    assert op is not None


def test_corrcoef_2890():
    op = get_op("corrcoef")
    assert op is not None


def test_correlate_2891():
    op = get_op("correlate")
    assert op is not None


def test_cos_2892():
    op = get_op("cos")
    assert op is not None


def test_cosh_2893():
    op = get_op("cosh")
    assert op is not None


def test_cosine_decay_schedule_2894():
    op = get_op("cosine_decay_schedule")
    assert op is not None


def test_cosine_distance_2895():
    op = get_op("cosine_distance")
    assert op is not None


def test_cosine_embedding_loss_2896():
    op = get_op("cosine_embedding_loss")
    assert op is not None


def test_cosine_onecycle_schedule_2897():
    op = get_op("cosine_onecycle_schedule")
    assert op is not None


def test_cosine_similarity_2898():
    op = get_op("cosine_similarity")
    assert op is not None


def test_cosinesimilarityloss_2899():
    op = get_op("cosinesimilarityloss")
    assert op is not None


def test_cost_dictionary_2900():
    op = get_op("cost_dictionary")
    assert op is not None


def test_count_2901():
    op = get_op("count")
    assert op is not None


def test_count_include_pad_2902():
    op = get_op("count_include_pad")
    assert op is not None


def test_count_nonzero_2903():
    op = get_op("count_nonzero")
    assert op is not None


def test_cov_2904():
    op = get_op("cov")
    assert op is not None


def test_cpp_dict_2905():
    op = get_op("cpp_dict")
    assert op is not None


def test_cpp_module_2906():
    op = get_op("cpp_module")
    assert op is not None


def test_cpu_2907():
    op = get_op("cpu")
    assert op is not None


def test_create_2908():
    op = get_op("create")
    assert op is not None


def test_create_block_mask_2909():
    op = get_op("create_block_mask")
    assert op is not None


def test_create_descriptor_2910():
    op = get_op("create_descriptor")
    assert op is not None


def test_create_instr_descriptor_2911():
    op = get_op("create_instr_descriptor")
    assert op is not None


def test_create_mask_2912():
    op = get_op("create_mask")
    assert op is not None


def test_create_mlir_sourcemap_2913():
    op = get_op("create_mlir_sourcemap")
    assert op is not None


def test_create_path_filters_2914():
    op = get_op("create_path_filters")
    assert op is not None


def test_create_scaled_f4_instr_descriptor_2915():
    op = get_op("create_scaled_f4_instr_descriptor")
    assert op is not None


def test_create_scaled_f8f6f4_instr_descriptor_2916():
    op = get_op("create_scaled_f8f6f4_instr_descriptor")
    assert op is not None


def test_create_token_2917():
    op = get_op("create_token")
    assert op is not None


def test_cross_entropy_2918():
    op = get_op("cross_entropy")
    assert op is not None


def test_csingle_2919():
    op = get_op("csingle")
    assert op is not None


def test_csr_fromdense_2920():
    op = get_op("csr_fromdense")
    assert op is not None


def test_csr_fromdense_p_2921():
    op = get_op("csr_fromdense_p")
    assert op is not None


def test_csr_matmat_2922():
    op = get_op("csr_matmat")
    assert op is not None


def test_csr_matmat_p_2923():
    op = get_op("csr_matmat_p")
    assert op is not None


def test_csr_matvec_2924():
    op = get_op("csr_matvec")
    assert op is not None


def test_csr_matvec_p_2925():
    op = get_op("csr_matvec_p")
    assert op is not None


def test_csr_todense_2926():
    op = get_op("csr_todense")
    assert op is not None


def test_csr_todense_p_2927():
    op = get_op("csr_todense_p")
    assert op is not None


def test_ctc_decode_2928():
    op = get_op("ctc_decode")
    assert op is not None


def test_ctc_loss_2929():
    op = get_op("ctc_loss")
    assert op is not None


def test_ctc_loss_with_forward_probs_2930():
    op = get_op("ctc_loss_with_forward_probs")
    assert op is not None


def test_ctcloss_2931():
    op = get_op("ctcloss")
    assert op is not None


def test_ctx_2932():
    op = get_op("ctx")
    assert op is not None


def test_cuda_2933():
    op = get_op("cuda")
    assert op is not None


def test_cuda_root_2934():
    op = get_op("cuda_root")
    assert op is not None


def test_cumlogsumexp_2935():
    op = get_op("cumlogsumexp")
    assert op is not None


def test_cummax_2936():
    op = get_op("cummax")
    assert op is not None


def test_cummin_2937():
    op = get_op("cummin")
    assert op is not None


def test_cumprod_2938():
    op = get_op("cumprod")
    assert op is not None


def test_cumred_reduce_window_impl_2939():
    op = get_op("cumred_reduce_window_impl")
    assert op is not None


def test_cumsum_2940():
    op = get_op("cumsum")
    assert op is not None


def test_cumulative_prod_2941():
    op = get_op("cumulative_prod")
    assert op is not None


def test_cumulative_sum_2942():
    op = get_op("cumulative_sum")
    assert op is not None


def test_current_context_2943():
    op = get_op("current_context")
    assert op is not None


def test_current_flash_attention_impl_2944():
    op = get_op("current_flash_attention_impl")
    assert op is not None


def test_current_jax_trace_2945():
    op = get_op("current_jax_trace")
    assert op is not None


def test_current_linen_module_2946():
    op = get_op("current_linen_module")
    assert op is not None


def test_current_module_2947():
    op = get_op("current_module")
    assert op is not None


def test_current_update_context_2948():
    op = get_op("current_update_context")
    assert op is not None


def test_custom_from_mask_2949():
    op = get_op("custom_from_mask")
    assert op is not None


def test_custom_gradient_2950():
    op = get_op("custom_gradient")
    assert op is not None


def test_custom_linear_solve_2951():
    op = get_op("custom_linear_solve")
    assert op is not None


def test_custom_root_2952():
    op = get_op("custom_root")
    assert op is not None


def test_custom_vjp_2953():
    op = get_op("custom_vjp")
    assert op is not None


def test_cutoffs_2954():
    op = get_op("cutoffs")
    assert op is not None


def test_d_model_2955():
    op = get_op("d_model")
    assert op is not None


def test_data_2956():
    op = get_op("data")
    assert op is not None


def test_data_next_2957():
    op = get_op("data_next")
    assert op is not None


def test_data_parallel_2958():
    op = get_op("data_parallel")
    assert op is not None


def test_data_ptr_2959():
    op = get_op("data_ptr")
    assert op is not None


def test_data_ref_2960():
    op = get_op("data_ref")
    assert op is not None


def test_dataclass_2961():
    op = get_op("dataclass")
    assert op is not None


def test_datetime_as_string_2962():
    op = get_op("datetime_as_string")
    assert op is not None


def test_dce_jaxpr_xla_metadata_rule_2963():
    op = get_op("dce_jaxpr_xla_metadata_rule")
    assert op is not None


def test_dce_sink_2964():
    op = get_op("dce_sink")
    assert op is not None


def test_debug_print_2965():
    op = get_op("debug_print")
    assert op is not None


def test_decode_2966():
    op = get_op("decode")
    assert op is not None


def test_decode_attn_unbatched_2967():
    op = get_op("decode_attn_unbatched")
    assert op is not None


def test_decoder_2968():
    op = get_op("decoder")
    assert op is not None


def test_def_comp_2969():
    op = get_op("def_comp")
    assert op is not None


def test_def_deriv_2970():
    op = get_op("def_deriv")
    assert op is not None


def test_default_2971():
    op = get_op("default")
    assert op is not None


def test_default_split_fn_2972():
    op = get_op("default_split_fn")
    assert op is not None


def test_default_stream_2973():
    op = get_op("default_stream")
    assert op is not None


def test_deflinear_2974():
    op = get_op("deflinear")
    assert op is not None


def test_defzero_2975():
    op = get_op("defzero")
    assert op is not None


def test_deg2rad_2976():
    op = get_op("deg2rad")
    assert op is not None


def test_degrees_2977():
    op = get_op("degrees")
    assert op is not None


def test_delay_2978():
    op = get_op("delay")
    assert op is not None


def test_delete_2979():
    op = get_op("delete")
    assert op is not None


def test_delta_2980():
    op = get_op("delta")
    assert op is not None


def test_depends_2981():
    op = get_op("depends")
    assert op is not None


def test_depthwise_conv_2982():
    op = get_op("depthwise_conv")
    assert op is not None


def test_deriv_prop_2983():
    op = get_op("deriv_prop")
    assert op is not None


def test_derive_relayout_constraints_2984():
    op = get_op("derive_relayout_constraints")
    assert op is not None


def test_deserialize_2985():
    op = get_op("deserialize")
    assert op is not None


def test_deserialize_and_load_2986():
    op = get_op("deserialize_and_load")
    assert op is not None


def test_deserialize_portable_artifact_2987():
    op = get_op("deserialize_portable_artifact")
    assert op is not None


def test_deserialize_pytreedef_2988():
    op = get_op("deserialize_pytreedef")
    assert op is not None


def test_deserialize_with_paths_2989():
    op = get_op("deserialize_with_paths")
    assert op is not None


def test_deserializeloss_2990():
    op = get_op("deserializeloss")
    assert op is not None


def test_device_2991():
    op = get_op("device")
    assert op is not None


def test_device_collective_metadata_2992():
    op = get_op("device_collective_metadata")
    assert op is not None


def test_device_count_2993():
    op = get_op("device_count")
    assert op is not None


def test_device_id_2994():
    op = get_op("device_id")
    assert op is not None


def test_device_ids_2995():
    op = get_op("device_ids")
    assert op is not None


def test_device_mesh_2996():
    op = get_op("device_mesh")
    assert op is not None


def test_device_ptr_2997():
    op = get_op("device_ptr")
    assert op is not None


def test_device_put_replicated_2998():
    op = get_op("device_put_replicated")
    assert op is not None


def test_device_put_sharded_2999():
    op = get_op("device_put_sharded")
    assert op is not None


def test_device_type_3000():
    op = get_op("device_type")
    assert op is not None


def test_devices_3001():
    op = get_op("devices")
    assert op is not None


def test_diag_3002():
    op = get_op("diag")
    assert op is not None


def test_diag_indices_3003():
    op = get_op("diag_indices")
    assert op is not None


def test_diag_indices_from_3004():
    op = get_op("diag_indices_from")
    assert op is not None


def test_diagflat_3005():
    op = get_op("diagflat")
    assert op is not None


def test_diagonal_3006():
    op = get_op("diagonal")
    assert op is not None


def test_diceloss_3007():
    op = get_op("diceloss")
    assert op is not None


def test_diff_3008():
    op = get_op("diff")
    assert op is not None


def test_digamma_3009():
    op = get_op("digamma")
    assert op is not None


def test_digitize_3010():
    op = get_op("digitize")
    assert op is not None


def test_dilation_3011():
    op = get_op("dilation")
    assert op is not None


def test_dim_3012():
    op = get_op("dim")
    assert op is not None


def test_dimension_numbers_3013():
    op = get_op("dimension_numbers")
    assert op is not None


def test_dims_3014():
    op = get_op("dims")
    assert op is not None


def test_dirac_3015():
    op = get_op("dirac")
    assert op is not None


def test_dirac__3016():
    op = get_op("dirac_")
    assert op is not None


def test_dirichlet_3017():
    op = get_op("dirichlet")
    assert op is not None


def test_display_3018():
    op = get_op("display")
    assert op is not None


def test_distance_function_3019():
    op = get_op("distance_function")
    assert op is not None


def test_div_value_3020():
    op = get_op("div_value")
    assert op is not None


def test_divide_3021():
    op = get_op("divide")
    assert op is not None


def test_divide_no_nan_3022():
    op = get_op("divide_no_nan")
    assert op is not None


def test_divisor_override_3023():
    op = get_op("divisor_override")
    assert op is not None


def test_divmod_3024():
    op = get_op("divmod")
    assert op is not None


def test_dkv_mask_info_3025():
    op = get_op("dkv_mask_info")
    assert op is not None


def test_do_generate_api_3026():
    op = get_op("do_generate_api")
    assert op is not None


def test_do_matmul_3027():
    op = get_op("do_matmul")
    assert op is not None


def test_docstrings_3028():
    op = get_op("docstrings")
    assert op is not None


def test_done_3029():
    op = get_op("done")
    assert op is not None


def test_dot_algorithm_attr_3030():
    op = get_op("dot_algorithm_attr")
    assert op is not None


def test_dot_general_3031():
    op = get_op("dot_general")
    assert op is not None


def test_dot_product_attention_3032():
    op = get_op("dot_product_attention")
    assert op is not None


def test_dot_product_attention_weights_3033():
    op = get_op("dot_product_attention_weights")
    assert op is not None


def test_double_3034():
    op = get_op("double")
    assert op is not None


def test_double_kernel_3035():
    op = get_op("double_kernel")
    assert op is not None


def test_downscale_factor_3036():
    op = get_op("downscale_factor")
    assert op is not None


def test_dq_mask_info_3037():
    op = get_op("dq_mask_info")
    assert op is not None


def test_draw_3038():
    op = get_op("draw")
    assert op is not None


def test_dropout1_3039():
    op = get_op("dropout1")
    assert op is not None


def test_dropout1d_3040():
    op = get_op("dropout1d")
    assert op is not None


def test_dropout2_3041():
    op = get_op("dropout2")
    assert op is not None


def test_dropout2d_3042():
    op = get_op("dropout2d")
    assert op is not None


def test_dropout3_3043():
    op = get_op("dropout3")
    assert op is not None


def test_dropout3d_3044():
    op = get_op("dropout3d")
    assert op is not None


def test_ds_3045():
    op = get_op("ds")
    assert op is not None


def test_dsplit_3046():
    op = get_op("dsplit")
    assert op is not None


def test_dstack_3047():
    op = get_op("dstack")
    assert op is not None


def test_dtype_3048():
    op = get_op("dtype")
    assert op is not None


def test_dtype_from_ctypes_type_3049():
    op = get_op("dtype_from_ctypes_type")
    assert op is not None


def test_dtype_is_implied_3050():
    op = get_op("dtype_is_implied")
    assert op is not None


def test_dtype_of_val_3051():
    op = get_op("dtype_of_val")
    assert op is not None


def test_dtype_short_repr_3052():
    op = get_op("dtype_short_repr")
    assert op is not None


def test_dtype_to_ir_type_3053():
    op = get_op("dtype_to_ir_type")
    assert op is not None


def test_dtype_to_string_3054():
    op = get_op("dtype_to_string")
    assert op is not None


def test_dump_3055():
    op = get_op("dump")
    assert op is not None


def test_dump_patches_3056():
    op = get_op("dump_patches")
    assert op is not None


def test_dump_path_3057():
    op = get_op("dump_path")
    assert op is not None


def test_dyn_dot_3058():
    op = get_op("dyn_dot")
    assert op is not None


def test_dynamic_gcd_3059():
    op = get_op("dynamic_gcd")
    assert op is not None


def test_dynamic_index_in_dim_3060():
    op = get_op("dynamic_index_in_dim")
    assert op is not None


def test_dynamic_slice_3061():
    op = get_op("dynamic_slice")
    assert op is not None


def test_dynamic_slice_in_dim_3062():
    op = get_op("dynamic_slice_in_dim")
    assert op is not None


def test_dynamic_update_index_in_dim_3063():
    op = get_op("dynamic_update_index_in_dim")
    assert op is not None


def test_dynamic_update_slice_3064():
    op = get_op("dynamic_update_slice")
    assert op is not None


def test_dynamic_update_slice_in_dim_3065():
    op = get_op("dynamic_update_slice_in_dim")
    assert op is not None


def test_dynamic_validate_inputs_3066():
    op = get_op("dynamic_validate_inputs")
    assert op is not None


def test_e_3067():
    op = get_op("e")
    assert op is not None


def test_ediff1d_3068():
    op = get_op("ediff1d")
    assert op is not None


def test_eig_jvp_rule_3069():
    op = get_op("eig_jvp_rule")
    assert op is not None


def test_eigvals_3070():
    op = get_op("eigvals")
    assert op is not None


def test_eigvalsh_3071():
    op = get_op("eigvalsh")
    assert op is not None


def test_einsum_path_3072():
    op = get_op("einsum_path")
    assert op is not None


def test_elastic_transform_3073():
    op = get_op("elastic_transform")
    assert op is not None


def test_elementwise_3074():
    op = get_op("elementwise")
    assert op is not None


def test_elementwise_affine_3075():
    op = get_op("elementwise_affine")
    assert op is not None


def test_eliminate_deprecated_list_indexing_3076():
    op = get_op("eliminate_deprecated_list_indexing")
    assert op is not None


def test_elu__3077():
    op = get_op("elu_")
    assert op is not None


def test_ema_3078():
    op = get_op("ema")
    assert op is not None


def test_embed_dim_3079():
    op = get_op("embed_dim")
    assert op is not None


def test_embedding_bag_3080():
    op = get_op("embedding_bag")
    assert op is not None


def test_embedding_dim_3081():
    op = get_op("embedding_dim")
    assert op is not None


def test_emit_tf_embedded_graph_custom_call_3082():
    op = get_op("emit_tf_embedded_graph_custom_call")
    assert op is not None


def test_empty_3083():
    op = get_op("empty")
    assert op is not None


def test_empty2_3084():
    op = get_op("empty2")
    assert op is not None


def test_empty_like_3085():
    op = get_op("empty_like")
    assert op is not None


def test_enable_grad_3086():
    op = get_op("enable_grad")
    assert op is not None


def test_enable_nested_tensor_3087():
    op = get_op("enable_nested_tensor")
    assert op is not None


def test_encode_3088():
    op = get_op("encode")
    assert op is not None


def test_encode_addr_3089():
    op = get_op("encode_addr")
    assert op is not None


def test_encode_descriptor_3090():
    op = get_op("encode_descriptor")
    assert op is not None


def test_encoder_3091():
    op = get_op("encoder")
    assert op is not None


def test_end_dim_3092():
    op = get_op("end_dim")
    assert op is not None


def test_endswith_3093():
    op = get_op("endswith")
    assert op is not None


def test_english_capitalize_3094():
    op = get_op("english_capitalize")
    assert op is not None


def test_english_lower_3095():
    op = get_op("english_lower")
    assert op is not None


def test_english_upper_3096():
    op = get_op("english_upper")
    assert op is not None


def test_ensure_arraylike_3097():
    op = get_op("ensure_arraylike")
    assert op is not None


def test_ensure_arraylike_tuple_3098():
    op = get_op("ensure_arraylike_tuple")
    assert op is not None


def test_ensure_shaped_3099():
    op = get_op("ensure_shaped")
    assert op is not None


def test_entr_3100():
    op = get_op("entr")
    assert op is not None


def test_entries_per_warpgroup_3101():
    op = get_op("entries_per_warpgroup")
    assert op is not None


def test_entries_per_wg_3102():
    op = get_op("entries_per_wg")
    assert op is not None


def test_enumerate_negative_3103():
    op = get_op("enumerate_negative")
    assert op is not None


def test_epi_tile_m_3104():
    op = get_op("epi_tile_m")
    assert op is not None


def test_epi_tile_n_3105():
    op = get_op("epi_tile_n")
    assert op is not None


def test_epilogue_tile_n_3106():
    op = get_op("epilogue_tile_n")
    assert op is not None


def test_eps_3107():
    op = get_op("eps")
    assert op is not None


def test_eq_3108():
    op = get_op("eq")
    assert op is not None


def test_equal_3109():
    op = get_op("equal")
    assert op is not None


def test_erf_3110():
    op = get_op("erf")
    assert op is not None


def test_erf_inv_3111():
    op = get_op("erf_inv")
    assert op is not None


def test_erfc_3112():
    op = get_op("erfc")
    assert op is not None


def test_erfcx_3113():
    op = get_op("erfcx")
    assert op is not None


def test_erfinv_3114():
    op = get_op("erfinv")
    assert op is not None


def test_error_checking_behavior_3115():
    op = get_op("error_checking_behavior")
    assert op is not None


def test_errstate_3116():
    op = get_op("errstate")
    assert op is not None


def test_estimate_control_variate_coefficients_3117():
    op = get_op("estimate_control_variate_coefficients")
    assert op is not None


def test_estimate_read_memory_footprint_3118():
    op = get_op("estimate_read_memory_footprint")
    assert op is not None


def test_euler_gamma_3119():
    op = get_op("euler_gamma")
    assert op is not None


def test_eval_3120():
    op = get_op("eval")
    assert op is not None


def test_eval_polymorphic_shape_3121():
    op = get_op("eval_polymorphic_shape")
    assert op is not None


def test_eval_shape_3122():
    op = get_op("eval_shape")
    assert op is not None


def test_eval_sparse_3123():
    op = get_op("eval_sparse")
    assert op is not None


def test_evaluate_chebyshev_polynomial_3124():
    op = get_op("evaluate_chebyshev_polynomial")
    assert op is not None


def test_exp2_3125():
    op = get_op("exp2")
    assert op is not None


def test_expand_dims_3126():
    op = get_op("expand_dims")
    assert op is not None


def test_expandtabs_3127():
    op = get_op("expandtabs")
    assert op is not None


def test_expit_3128():
    op = get_op("expit")
    assert op is not None


def test_expm1_3129():
    op = get_op("expm1")
    assert op is not None


def test_exponential_3130():
    op = get_op("exponential")
    assert op is not None


def test_exponential_decay_3131():
    op = get_op("exponential_decay")
    assert op is not None


def test_expression_3132():
    op = get_op("expression")
    assert op is not None


def test_extend_3133():
    op = get_op("extend")
    assert op is not None


def test_extend_all_3134():
    op = get_op("extend_all")
    assert op is not None


def test_extra_repr_3135():
    op = get_op("extra_repr")
    assert op is not None


def test_extract_3136():
    op = get_op("extract")
    assert op is not None


def test_extract_assignment_candidates_from_reduce_equation_3137():
    op = get_op("extract_assignment_candidates_from_reduce_equation")
    assert op is not None


def test_extract_sequences_3138():
    op = get_op("extract_sequences")
    assert op is not None


def test_eye_3139():
    op = get_op("eye")
    assert op is not None


def test_eye__3140():
    op = get_op("eye_")
    assert op is not None


def test_f_3141():
    op = get_op("f")
    assert op is not None


def test_fa_m64_collective_layout_3142():
    op = get_op("fa_m64_collective_layout")
    assert op is not None


def test_fabs_3143():
    op = get_op("fabs")
    assert op is not None


def test_fact_3144():
    op = get_op("fact")
    assert op is not None


def test_factory_kwargs_3145():
    op = get_op("factory_kwargs")
    assert op is not None


def test_feature_alpha_dropout_3146():
    op = get_op("feature_alpha_dropout")
    assert op is not None


def test_fence_release_sys_3147():
    op = get_op("fence_release_sys")
    assert op is not None


def test_fft_3148():
    op = get_op("fft")
    assert op is not None


def test_fft2_3149():
    op = get_op("fft2")
    assert op is not None


def test_fft_abstract_eval_3150():
    op = get_op("fft_abstract_eval")
    assert op is not None


def test_fftfreq_3151():
    op = get_op("fftfreq")
    assert op is not None


def test_fftn_3152():
    op = get_op("fftn")
    assert op is not None


def test_fftshift_3153():
    op = get_op("fftshift")
    assert op is not None


def test_file_in_this_dir_3154():
    op = get_op("file_in_this_dir")
    assert op is not None


def test_fill_diagonal_3155():
    op = get_op("fill_diagonal")
    assert op is not None


def test_filter_3156():
    op = get_op("filter")
    assert op is not None


def test_filter_passes_3157():
    op = get_op("filter_passes")
    assert op is not None


def test_filter_rng_streams_3158():
    op = get_op("filter_rng_streams")
    assert op is not None


def test_filter_state_3159():
    op = get_op("filter_state")
    assert op is not None


def test_filters_3160():
    op = get_op("filters")
    assert op is not None


def test_filters_to_predicates_3161():
    op = get_op("filters_to_predicates")
    assert op is not None


def test_finalize_3162():
    op = get_op("finalize")
    assert op is not None


def test_finalize_array_function_like_3163():
    op = get_op("finalize_array_function_like")
    assert op is not None


def test_finalize_size_3164():
    op = get_op("finalize_size")
    assert op is not None


def test_find_3165():
    op = get_op("find")
    assert op is not None


def test_find_assignments_for_3166():
    op = get_op("find_assignments_for")
    assert op is not None


def test_find_comma_decimal_point_locale_3167():
    op = get_op("find_comma_decimal_point_locale")
    assert op is not None


def test_find_duplicate_3168():
    op = get_op("find_duplicate")
    assert op is not None


def test_find_duplicates_3169():
    op = get_op("find_duplicates")
    assert op is not None


def test_find_functions_3170():
    op = get_op("find_functions")
    assert op is not None


def test_find_unused_parameters_3171():
    op = get_op("find_unused_parameters")
    assert op is not None


def test_finfo_3172():
    op = get_op("finfo")
    assert op is not None


def test_first_from_3173():
    op = get_op("first_from")
    assert op is not None


def test_fit_4th_order_polynomial_3174():
    op = get_op("fit_4th_order_polynomial")
    assert op is not None


def test_fix_3175():
    op = get_op("fix")
    assert op is not None


def test_flag_env_3176():
    op = get_op("flag_env")
    assert op is not None


def test_flash_attention_3177():
    op = get_op("flash_attention")
    assert op is not None


def test_flash_attention_kernel_3178():
    op = get_op("flash_attention_kernel")
    assert op is not None


def test_flatnonzero_3179():
    op = get_op("flatnonzero")
    assert op is not None


def test_flatten_3180():
    op = get_op("flatten")
    assert op is not None


def test_flatten_fun_for_sparse_ad_3181():
    op = get_op("flatten_fun_for_sparse_ad")
    assert op is not None


def test_flatten_mapping_3182():
    op = get_op("flatten_mapping")
    assert op is not None


def test_flatten_parameters_3183():
    op = get_op("flatten_parameters")
    assert op is not None


def test_flatten_to_sequence_3184():
    op = get_op("flatten_to_sequence")
    assert op is not None


def test_flex_attention_3185():
    op = get_op("flex_attention")
    assert op is not None


def test_flexible_3186():
    op = get_op("flexible")
    assert op is not None


def test_flip_3187():
    op = get_op("flip")
    assert op is not None


def test_flip_sequences_3188():
    op = get_op("flip_sequences")
    assert op is not None


def test_fliplr_3189():
    op = get_op("fliplr")
    assert op is not None


def test_flipud_3190():
    op = get_op("flipud")
    assert op is not None


def test_float_3191():
    op = get_op("float")
    assert op is not None


def test_float4_e2m1fn_3192():
    op = get_op("float4_e2m1fn")
    assert op is not None


def test_float8_e3m4_3193():
    op = get_op("float8_e3m4")
    assert op is not None


def test_float8_e4m3_3194():
    op = get_op("float8_e4m3")
    assert op is not None


def test_float8_e4m3b11fnuz_3195():
    op = get_op("float8_e4m3b11fnuz")
    assert op is not None


def test_float8_e4m3fn_3196():
    op = get_op("float8_e4m3fn")
    assert op is not None


def test_float8_e4m3fnuz_3197():
    op = get_op("float8_e4m3fnuz")
    assert op is not None


def test_float8_e5m2_3198():
    op = get_op("float8_e5m2")
    assert op is not None


def test_float8_e5m2fnuz_3199():
    op = get_op("float8_e5m2fnuz")
    assert op is not None


def test_float8_e8m0fnu_3200():
    op = get_op("float8_e8m0fnu")
    assert op is not None


def test_float__3201():
    op = get_op("float_")
    assert op is not None


def test_float_power_3202():
    op = get_op("float_power")
    assert op is not None


def test_floating_3203():
    op = get_op("floating")
    assert op is not None


def test_floor_3204():
    op = get_op("floor")
    assert op is not None


def test_floor_divide_3205():
    op = get_op("floor_divide")
    assert op is not None


def test_flops_3206():
    op = get_op("flops")
    assert op is not None


def test_fmax_3207():
    op = get_op("fmax")
    assert op is not None


def test_fmin_3208():
    op = get_op("fmin")
    assert op is not None


def test_fmod_3209():
    op = get_op("fmod")
    assert op is not None


def test_fn_3210():
    op = get_op("fn")
    assert op is not None


def test_foreach_3211():
    op = get_op("foreach")
    assert op is not None


def test_fori_3212():
    op = get_op("fori")
    assert op is not None


def test_fori_loop_3213():
    op = get_op("fori_loop")
    assert op is not None


def test_fork_rngs_3214():
    op = get_op("fork_rngs")
    assert op is not None


def test_format_float_positional_3215():
    op = get_op("format_float_positional")
    assert op is not None


def test_format_float_scientific_3216():
    op = get_op("format_float_scientific")
    assert op is not None


def test_format_parser_3217():
    op = get_op("format_parser")
    assert op is not None


def test_forward_3218():
    op = get_op("forward")
    assert op is not None


def test_fractional_max_pool2d_3219():
    op = get_op("fractional_max_pool2d")
    assert op is not None


def test_fractional_max_pool2d_with_indices_3220():
    op = get_op("fractional_max_pool2d_with_indices")
    assert op is not None


def test_fractional_max_pool3d_3221():
    op = get_op("fractional_max_pool3d")
    assert op is not None


def test_fractional_max_pool3d_with_indices_3222():
    op = get_op("fractional_max_pool3d_with_indices")
    assert op is not None


def test_fragmented_array_to_ir_3223():
    op = get_op("fragmented_array_to_ir")
    assert op is not None


def test_freeze_3224():
    op = get_op("freeze")
    assert op is not None


def test_frexp_3225():
    op = get_op("frexp")
    assert op is not None


def test_from_alloc_3226():
    op = get_op("from_alloc")
    assert op is not None


def test_from_aval_3227():
    op = get_op("from_aval")
    assert op is not None


def test_from_barrier_memref_3228():
    op = get_op("from_barrier_memref")
    assert op is not None


def test_from_bcoo_3229():
    op = get_op("from_bcoo")
    assert op is not None


def test_from_dlpack_3230():
    op = get_op("from_dlpack")
    assert op is not None


def test_from_flat_state_3231():
    op = get_op("from_flat_state")
    assert op is not None


def test_from_head_minor_3232():
    op = get_op("from_head_minor")
    assert op is not None


def test_from_int8_3233():
    op = get_op("from_int8")
    assert op is not None


def test_from_kv_blocks_3234():
    op = get_op("from_kv_blocks")
    assert op is not None


def test_from_layout_attr_3235():
    op = get_op("from_layout_attr")
    assert op is not None


def test_from_pretrained_3236():
    op = get_op("from_pretrained")
    assert op is not None


def test_from_registers_3237():
    op = get_op("from_registers")
    assert op is not None


def test_from_scipy_sparse_3238():
    op = get_op("from_scipy_sparse")
    assert op is not None


def test_from_shaped_type_3239():
    op = get_op("from_shaped_type")
    assert op is not None


def test_from_splat_fragmented_layout_attr_3240():
    op = get_op("from_splat_fragmented_layout_attr")
    assert op is not None


def test_from_strided_fragmented_layout_attr_3241():
    op = get_op("from_strided_fragmented_layout_attr")
    assert op is not None


def test_from_tensor_slices_3242():
    op = get_op("from_tensor_slices")
    assert op is not None


def test_from_tiled_layout_attr_3243():
    op = get_op("from_tiled_layout_attr")
    assert op is not None


def test_from_transform_attr_3244():
    op = get_op("from_transform_attr")
    assert op is not None


def test_from_tree_3245():
    op = get_op("from_tree")
    assert op is not None


def test_fromarrays_3246():
    op = get_op("fromarrays")
    assert op is not None


def test_frombuffer_3247():
    op = get_op("frombuffer")
    assert op is not None


def test_fromdense_3248():
    op = get_op("fromdense")
    assert op is not None


def test_fromfile_3249():
    op = get_op("fromfile")
    assert op is not None


def test_fromfunction_3250():
    op = get_op("fromfunction")
    assert op is not None


def test_fromiter_3251():
    op = get_op("fromiter")
    assert op is not None


def test_fromkeys_3252():
    op = get_op("fromkeys")
    assert op is not None


def test_frompyfunc_3253():
    op = get_op("frompyfunc")
    assert op is not None


def test_fromrecords_3254():
    op = get_op("fromrecords")
    assert op is not None


def test_fromstring_3255():
    op = get_op("fromstring")
    assert op is not None


def test_full_3256():
    op = get_op("full")
    assert op is not None


def test_full_kv_indices_3257():
    op = get_op("full_kv_indices")
    assert op is not None


def test_full_kv_num_blocks_3258():
    op = get_op("full_kv_num_blocks")
    assert op is not None


def test_full_like_3259():
    op = get_op("full_like")
    assert op is not None


def test_full_lower_3260():
    op = get_op("full_lower")
    assert op is not None


def test_full_q_indices_3261():
    op = get_op("full_q_indices")
    assert op is not None


def test_full_q_num_blocks_3262():
    op = get_op("full_q_num_blocks")
    assert op is not None


def test_fullapi_hash_3263():
    op = get_op("fullapi_hash")
    assert op is not None


def test_fun_3264():
    op = get_op("fun")
    assert op is not None


def test_fun_jax_3265():
    op = get_op("fun_jax")
    assert op is not None


def test_fun_signature_3266():
    op = get_op("fun_signature")
    assert op is not None


def test_fun_sourceinfo_3267():
    op = get_op("fun_sourceinfo")
    assert op is not None


def test_func_3268():
    op = get_op("func")
    assert op is not None


def test_functional_3269():
    op = get_op("functional")
    assert op is not None


def test_functional_call_3270():
    op = get_op("functional_call")
    assert op is not None


def test_fuse_conv_bn_eval_3271():
    op = get_op("fuse_conv_bn_eval")
    assert op is not None


def test_fuse_conv_bn_weights_3272():
    op = get_op("fuse_conv_bn_weights")
    assert op is not None


def test_fuse_linear_bn_eval_3273():
    op = get_op("fuse_linear_bn_eval")
    assert op is not None


def test_fuse_linear_bn_weights_3274():
    op = get_op("fuse_linear_bn_weights")
    assert op is not None


def test_fused_3275():
    op = get_op("fused")
    assert op is not None


def test_fused_p_3276():
    op = get_op("fused_p")
    assert op is not None


def test_fwd_mask_info_3277():
    op = get_op("fwd_mask_info")
    assert op is not None


def test_gamma_3278():
    op = get_op("gamma")
    assert op is not None


def test_gammainc_3279():
    op = get_op("gammainc")
    assert op is not None


def test_gammaincc_3280():
    op = get_op("gammaincc")
    assert op is not None


def test_gammaln_3281():
    op = get_op("gammaln")
    assert op is not None


def test_gather_3282():
    op = get_op("gather")
    assert op is not None


def test_gaussian_blur_3283():
    op = get_op("gaussian_blur")
    assert op is not None


def test_gaussian_nll_loss_3284():
    op = get_op("gaussian_nll_loss")
    assert op is not None


def test_gcd_3285():
    op = get_op("gcd")
    assert op is not None


def test_ge_3286():
    op = get_op("ge")
    assert op is not None


def test_gelu_approx_3287():
    op = get_op("gelu_approx")
    assert op is not None


def test_gelu_fast_approx_3288():
    op = get_op("gelu_fast_approx")
    assert op is not None


def test_generate_api_3289():
    op = get_op("generate_api")
    assert op is not None


def test_generate_dump_3290():
    op = get_op("generate_dump")
    assert op is not None


def test_generate_sourcemaps_3291():
    op = get_op("generate_sourcemaps")
    assert op is not None


def test_generate_square_subsequent_mask_3292():
    op = get_op("generate_square_subsequent_mask")
    assert op is not None


def test_generated_code_3293():
    op = get_op("generated_code")
    assert op is not None


def test_generic_3294():
    op = get_op("generic")
    assert op is not None


def test_geometric_3295():
    op = get_op("geometric")
    assert op is not None


def test_geomspace_3296():
    op = get_op("geomspace")
    assert op is not None


def test_geqp3_3297():
    op = get_op("geqp3")
    assert op is not None


def test_geqrf_3298():
    op = get_op("geqrf")
    assert op is not None


def test_get_3299():
    op = get_op("get")
    assert op is not None


def test_get_a_var_3300():
    op = get_op("get_a_var")
    assert op is not None


def test_get_abstract_model_3301():
    op = get_op("get_abstract_model")
    assert op is not None


def test_get_algorithm_compute_types_3302():
    op = get_op("get_algorithm_compute_types")
    assert op is not None


def test_get_all_with_path_3303():
    op = get_op("get_all_with_path")
    assert op is not None


def test_get_annotations_3304():
    op = get_op("get_annotations")
    assert op is not None


def test_get_api_functions_3305():
    op = get_op("get_api_functions")
    assert op is not None


def test_get_api_versions_3306():
    op = get_op("get_api_versions")
    assert op is not None


def test_get_arch_3307():
    op = get_op("get_arch")
    assert op is not None


def test_get_array_function_like_doc_3308():
    op = get_op("get_array_function_like_doc")
    assert op is not None


def test_get_attached_topology_3309():
    op = get_op("get_attached_topology")
    assert op is not None


def test_get_aval_3310():
    op = get_op("get_aval")
    assert op is not None


def test_get_base_3311():
    op = get_op("get_base")
    assert op is not None


def test_get_buffer_3312():
    op = get_op("get_buffer")
    assert op is not None


def test_get_cluster_ptr_3313():
    op = get_op("get_cluster_ptr")
    assert op is not None


def test_get_cluster_ref_3314():
    op = get_op("get_cluster_ref")
    assert op is not None


def test_get_col_name_3315():
    op = get_op("get_col_name")
    assert op is not None


def test_get_contiguous_strides_3316():
    op = get_op("get_contiguous_strides")
    assert op is not None


def test_get_default_3317():
    op = get_op("get_default")
    assert op is not None


def test_get_device_name_3318():
    op = get_op("get_device_name")
    assert op is not None


def test_get_dtype_packing_3319():
    op = get_op("get_dtype_packing")
    assert op is not None


def test_get_expected_cell_size_3320():
    op = get_op("get_expected_cell_size")
    assert op is not None


def test_get_expected_hidden_size_3321():
    op = get_op("get_expected_hidden_size")
    assert op is not None


def test_get_extra_state_3322():
    op = get_op("get_extra_state")
    assert op is not None


def test_get_hlo_sharding_from_serialized_proto_3323():
    op = get_op("get_hlo_sharding_from_serialized_proto")
    assert op is not None


def test_get_item_3324():
    op = get_op("get_item")
    assert op is not None


def test_get_kernel_name_3325():
    op = get_op("get_kernel_name")
    assert op is not None


def test_get_min_heads_per_blk_3326():
    op = get_op("get_min_heads_per_blk")
    assert op is not None


def test_get_min_page_size_3327():
    op = get_op("get_min_page_size")
    assert op is not None


def test_get_named_sharding_3328():
    op = get_op("get_named_sharding")
    assert op is not None


def test_get_neighbor_3329():
    op = get_op("get_neighbor")
    assert op is not None


def test_get_node_impl_3330():
    op = get_op("get_node_impl")
    assert op is not None


def test_get_node_impl_for_type_3331():
    op = get_op("get_node_impl_for_type")
    assert op is not None


def test_get_num_params_in_lstm_3332():
    op = get_op("get_num_params_in_lstm")
    assert op is not None


def test_get_op_sharding_from_serialized_proto_3333():
    op = get_op("get_op_sharding_from_serialized_proto")
    assert op is not None


def test_get_parameter_3334():
    op = get_op("get_parameter")
    assert op is not None


def test_get_partition_spec_3335():
    op = get_op("get_partition_spec")
    assert op is not None


def test_get_printoptions_3336():
    op = get_op("get_printoptions")
    assert op is not None


def test_get_processor_3337():
    op = get_op("get_processor")
    assert op is not None


def test_get_profiled_instructions_proto_3338():
    op = get_op("get_profiled_instructions_proto")
    assert op is not None


def test_get_ptr_3339():
    op = get_op("get_ptr")
    assert op is not None


def test_get_quantization_scales_3340():
    op = get_op("get_quantization_scales")
    assert op is not None


def test_get_remaining_size_3341():
    op = get_op("get_remaining_size")
    assert op is not None


def test_get_repr_3342():
    op = get_op("get_repr")
    assert op is not None


def test_get_serialized_proto_from_hlo_sharding_3343():
    op = get_op("get_serialized_proto_from_hlo_sharding")
    assert op is not None


def test_get_submodule_3344():
    op = get_op("get_submodule")
    assert op is not None


def test_get_tensorstore_spec_3345():
    op = get_op("get_tensorstore_spec")
    assert op is not None


def test_get_thread_local_state_call_tf_concrete_function_list_3346():
    op = get_op("get_thread_local_state_call_tf_concrete_function_list")
    assert op is not None


def test_get_topology_desc_3347():
    op = get_op("get_topology_desc")
    assert op is not None


def test_get_tpu_version_3348():
    op = get_op("get_tpu_version")
    assert op is not None


def test_get_tuned_block_sizes_3349():
    op = get_op("get_tuned_block_sizes")
    assert op is not None


def test_get_var_pspec_3350():
    op = get_op("get_var_pspec")
    assert op is not None


def test_get_versions_hash_3351():
    op = get_op("get_versions_hash")
    assert op is not None


def test_get_vjp_fun_3352():
    op = get_op("get_vjp_fun")
    assert op is not None


def test_get_weight_3353():
    op = get_op("get_weight")
    assert op is not None


def test_getbufsize_3354():
    op = get_op("getbufsize")
    assert op is not None


def test_getelementptr_3355():
    op = get_op("getelementptr")
    assert op is not None


def test_geterr_3356():
    op = get_op("geterr")
    assert op is not None


def test_geterrcall_3357():
    op = get_op("geterrcall")
    assert op is not None


def test_getloss_3358():
    op = get_op("getloss")
    assert op is not None


def test_global_array_to_host_local_array_3359():
    op = get_op("global_array_to_host_local_array")
    assert op is not None


def test_global_array_to_host_local_array_impl_3360():
    op = get_op("global_array_to_host_local_array_impl")
    assert op is not None


def test_global_array_to_host_local_array_p_3361():
    op = get_op("global_array_to_host_local_array_p")
    assert op is not None


def test_global_aval_to_result_handler_3362():
    op = get_op("global_aval_to_result_handler")
    assert op is not None


def test_global_avals_to_results_handler_3363():
    op = get_op("global_avals_to_results_handler")
    assert op is not None


def test_global_result_handlers_3364():
    op = get_op("global_result_handlers")
    assert op is not None


def test_global_unstructured_3365():
    op = get_op("global_unstructured")
    assert op is not None


def test_globaltimer_3366():
    op = get_op("globaltimer")
    assert op is not None


def test_glorot_3367():
    op = get_op("glorot")
    assert op is not None


def test_glorot_normal_3368():
    op = get_op("glorot_normal")
    assert op is not None


def test_glorot_uniform_3369():
    op = get_op("glorot_uniform")
    assert op is not None


def test_glu_3370():
    op = get_op("glu")
    assert op is not None


def test_gmm_3371():
    op = get_op("gmm")
    assert op is not None


def test_gpu_address_space_to_nvptx_3372():
    op = get_op("gpu_address_space_to_nvptx")
    assert op is not None


def test_gqa_3373():
    op = get_op("gqa")
    assert op is not None


def test_gqa_reference_3374():
    op = get_op("gqa_reference")
    assert op is not None


def test_grad_3375():
    op = get_op("grad")
    assert op is not None


def test_gradient_3376():
    op = get_op("gradient")
    assert op is not None


def test_gradient_as_bucket_view_3377():
    op = get_op("gradient_as_bucket_view")
    assert op is not None


def test_graph_pop_3378():
    op = get_op("graph_pop")
    assert op is not None


def test_graphdef_3379():
    op = get_op("graphdef")
    assert op is not None


def test_greater_equal_3380():
    op = get_op("greater_equal")
    assert op is not None


def test_grid_minor_dim_3381():
    op = get_op("grid_minor_dim")
    assert op is not None


def test_grid_sample_3382():
    op = get_op("grid_sample")
    assert op is not None


def test_grid_tile_width_3383():
    op = get_op("grid_tile_width")
    assert op is not None


def test_group_3384():
    op = get_op("group")
    assert op is not None


def test_group_id_3385():
    op = get_op("group_id")
    assert op is not None


def test_group_norm_3386():
    op = get_op("group_norm")
    assert op is not None


def test_group_pred_3387():
    op = get_op("group_pred")
    assert op is not None


def test_group_size_3388():
    op = get_op("group_size")
    assert op is not None


def test_grouped_mm_3389():
    op = get_op("grouped_mm")
    assert op is not None


def test_grouped_query_attention_reference_3390():
    op = get_op("grouped_query_attention_reference")
    assert op is not None


def test_groups_3391():
    op = get_op("groups")
    assert op is not None


def test_gt_3392():
    op = get_op("gt")
    assert op is not None


def test_gtl_abstract_eval_3393():
    op = get_op("gtl_abstract_eval")
    assert op is not None


def test_gumbel_3394():
    op = get_op("gumbel")
    assert op is not None


def test_gumbel_softmax_3395():
    op = get_op("gumbel_softmax")
    assert op is not None


def test_half_3396():
    op = get_op("half")
    assert op is not None


def test_hamming_3397():
    op = get_op("hamming")
    assert op is not None


def test_hanning_3398():
    op = get_op("hanning")
    assert op is not None


def test_hard_shrink_3399():
    op = get_op("hard_shrink")
    assert op is not None


def test_hard_sigmoid_3400():
    op = get_op("hard_sigmoid")
    assert op is not None


def test_hard_silu_3401():
    op = get_op("hard_silu")
    assert op is not None


def test_hard_swish_3402():
    op = get_op("hard_swish")
    assert op is not None


def test_hard_tanh_3403():
    op = get_op("hard_tanh")
    assert op is not None


def test_hardshrink_3404():
    op = get_op("hardshrink")
    assert op is not None


def test_hardsigmoid_3405():
    op = get_op("hardsigmoid")
    assert op is not None


def test_hardtanh_3406():
    op = get_op("hardtanh")
    assert op is not None


def test_hardtanh__3407():
    op = get_op("hardtanh_")
    assert op is not None


def test_has_any_layout_set_3408():
    op = get_op("has_any_layout_set")
    assert op is not None


def test_has_backward_blocks_3409():
    op = get_op("has_backward_blocks")
    assert op is not None


def test_has_data_3410():
    op = get_op("has_data")
    assert op is not None


def test_has_in_layouts_set_3411():
    op = get_op("has_in_layouts_set")
    assert op is not None


def test_has_in_tmem_layouts_set_3412():
    op = get_op("has_in_tmem_layouts_set")
    assert op is not None


def test_has_in_transforms_set_3413():
    op = get_op("has_in_transforms_set")
    assert op is not None


def test_has_keyword_arg_3414():
    op = get_op("has_keyword_arg")
    assert op is not None


def test_has_out_layouts_set_3415():
    op = get_op("has_out_layouts_set")
    assert op is not None


def test_has_out_tmem_layouts_set_3416():
    op = get_op("has_out_tmem_layouts_set")
    assert op is not None


def test_has_out_transforms_set_3417():
    op = get_op("has_out_transforms_set")
    assert op is not None


def test_has_setup_3418():
    op = get_op("has_setup")
    assert op is not None


def test_has_uninitialized_params_3419():
    op = get_op("has_uninitialized_params")
    assert op is not None


def test_hbm_bytes_3420():
    op = get_op("hbm_bytes")
    assert op is not None


def test_he_normal_3421():
    op = get_op("he_normal")
    assert op is not None


def test_he_uniform_3422():
    op = get_op("he_uniform")
    assert op is not None


def test_head_3423():
    op = get_op("head")
    assert op is not None


def test_head_bias_3424():
    op = get_op("head_bias")
    assert op is not None


def test_head_dim_3425():
    op = get_op("head_dim")
    assert op is not None


def test_head_size_3426():
    op = get_op("head_size")
    assert op is not None


def test_heaviside_3427():
    op = get_op("heaviside")
    assert op is not None


def test_hermite_polynomial_h_3428():
    op = get_op("hermite_polynomial_h")
    assert op is not None


def test_hermite_polynomial_he_3429():
    op = get_op("hermite_polynomial_he")
    assert op is not None


def test_hessenberg_3430():
    op = get_op("hessenberg")
    assert op is not None


def test_hfft_3431():
    op = get_op("hfft")
    assert op is not None


def test_hfft2_3432():
    op = get_op("hfft2")
    assert op is not None


def test_hfftn_3433():
    op = get_op("hfftn")
    assert op is not None


def test_hidden_size_3434():
    op = get_op("hidden_size")
    assert op is not None


def test_hinge_embedding_loss_3435():
    op = get_op("hinge_embedding_loss")
    assert op is not None


def test_hinge_loss_3436():
    op = get_op("hinge_loss")
    assert op is not None


def test_hingeloss_3437():
    op = get_op("hingeloss")
    assert op is not None


def test_histogram_3438():
    op = get_op("histogram")
    assert op is not None


def test_histogram2d_3439():
    op = get_op("histogram2d")
    assert op is not None


def test_histogram_bin_edges_3440():
    op = get_op("histogram_bin_edges")
    assert op is not None


def test_histogramdd_3441():
    op = get_op("histogramdd")
    assert op is not None


def test_hlo_to_stablehlo_3442():
    op = get_op("hlo_to_stablehlo")
    assert op is not None


def test_holds_3443():
    op = get_op("holds")
    assert op is not None


def test_host_collective_metadata_3444():
    op = get_op("host_collective_metadata")
    assert op is not None


def test_host_init_3445():
    op = get_op("host_init")
    assert op is not None


def test_host_local_array_to_global_array_3446():
    op = get_op("host_local_array_to_global_array")
    assert op is not None


def test_host_local_array_to_global_array_impl_3447():
    op = get_op("host_local_array_to_global_array_impl")
    assert op is not None


def test_host_local_array_to_global_array_p_3448():
    op = get_op("host_local_array_to_global_array_p")
    assert op is not None


def test_householder_product_3449():
    op = get_op("householder_product")
    assert op is not None


def test_hsplit_3450():
    op = get_op("hsplit")
    assert op is not None


def test_hstack_3451():
    op = get_op("hstack")
    assert op is not None


def test_huber_loss_3452():
    op = get_op("huber_loss")
    assert op is not None


def test_hypot_3453():
    op = get_op("hypot")
    assert op is not None


def test_i0_3454():
    op = get_op("i0")
    assert op is not None


def test_i0e_3455():
    op = get_op("i0e")
    assert op is not None


def test_i1_3456():
    op = get_op("i1")
    assert op is not None


def test_i1e_3457():
    op = get_op("i1e")
    assert op is not None


def test_ici_bytes_3458():
    op = get_op("ici_bytes")
    assert op is not None


def test_ici_latency_3459():
    op = get_op("ici_latency")
    assert op is not None


def test_id_3460():
    op = get_op("id")
    assert op is not None


def test_identity_3461():
    op = get_op("identity")
    assert op is not None


def test_ifft_3462():
    op = get_op("ifft")
    assert op is not None


def test_ifft2_3463():
    op = get_op("ifft2")
    assert op is not None


def test_ifftn_3464():
    op = get_op("ifftn")
    assert op is not None


def test_ifftshift_3465():
    op = get_op("ifftshift")
    assert op is not None


def test_ifrt_programs_3466():
    op = get_op("ifrt_programs")
    assert op is not None


def test_igamma_3467():
    op = get_op("igamma")
    assert op is not None


def test_igamma_grad_a_3468():
    op = get_op("igamma_grad_a")
    assert op is not None


def test_igamma_grad_a_impl_3469():
    op = get_op("igamma_grad_a_impl")
    assert op is not None


def test_igamma_grada_3470():
    op = get_op("igamma_grada")
    assert op is not None


def test_igamma_gradx_3471():
    op = get_op("igamma_gradx")
    assert op is not None


def test_igamma_impl_3472():
    op = get_op("igamma_impl")
    assert op is not None


def test_igammac_3473():
    op = get_op("igammac")
    assert op is not None


def test_igammac_grada_3474():
    op = get_op("igammac_grada")
    assert op is not None


def test_igammac_gradx_3475():
    op = get_op("igammac_gradx")
    assert op is not None


def test_igammac_impl_3476():
    op = get_op("igammac_impl")
    assert op is not None


def test_ignore_index_3477():
    op = get_op("ignore_index")
    assert op is not None


def test_ihfft_3478():
    op = get_op("ihfft")
    assert op is not None


def test_ihfft2_3479():
    op = get_op("ihfft2")
    assert op is not None


def test_ihfftn_3480():
    op = get_op("ihfftn")
    assert op is not None


def test_iinfo_3481():
    op = get_op("iinfo")
    assert op is not None


def test_imag_3482():
    op = get_op("imag")
    assert op is not None


def test_impl_3483():
    op = get_op("impl")
    assert op is not None


def test_in1_features_3484():
    op = get_op("in1_features")
    assert op is not None


def test_in2_features_3485():
    op = get_op("in2_features")
    assert op is not None


def test_in_features_3486():
    op = get_op("in_features")
    assert op is not None


def test_in_layout_for_operand_3487():
    op = get_op("in_layout_for_operand")
    assert op is not None


def test_in_layouts_3488():
    op = get_op("in_layouts")
    assert op is not None


def test_in_proj_bias_3489():
    op = get_op("in_proj_bias")
    assert op is not None


def test_in_proj_weight_3490():
    op = get_op("in_proj_weight")
    assert op is not None


def test_in_specs_leaves_3491():
    op = get_op("in_specs_leaves")
    assert op is not None


def test_in_specs_treedef_3492():
    op = get_op("in_specs_treedef")
    assert op is not None


def test_in_tmem_layouts_3493():
    op = get_op("in_tmem_layouts")
    assert op is not None


def test_in_top_k_3494():
    op = get_op("in_top_k")
    assert op is not None


def test_in_transforms_3495():
    op = get_op("in_transforms")
    assert op is not None


def test_in_transforms_for_operand_3496():
    op = get_op("in_transforms_for_operand")
    assert op is not None


def test_include_last_offset_3497():
    op = get_op("include_last_offset")
    assert op is not None


def test_indent_3498():
    op = get_op("indent")
    assert op is not None


def test_index_3499():
    op = get_op("index")
    assert op is not None


def test_index_in_dim_3500():
    op = get_op("index_in_dim")
    assert op is not None


def test_index_take_3501():
    op = get_op("index_take")
    assert op is not None


def test_indices_3502():
    op = get_op("indices")
    assert op is not None


def test_indices_ref_3503():
    op = get_op("indices_ref")
    assert op is not None


def test_indices_sorted_3504():
    op = get_op("indices_sorted")
    assert op is not None


def test_indptr_3505():
    op = get_op("indptr")
    assert op is not None


def test_indptr_ref_3506():
    op = get_op("indptr_ref")
    assert op is not None


def test_inexact_3507():
    op = get_op("inexact")
    assert op is not None


def test_inf_3508():
    op = get_op("inf")
    assert op is not None


def test_infer_layout_3509():
    op = get_op("infer_layout")
    assert op is not None


def test_init_3510():
    op = get_op("init")
    assert op is not None


def test_init_fn_3511():
    op = get_op("init_fn")
    assert op is not None


def test_init_lstm_weight_3512():
    op = get_op("init_lstm_weight")
    assert op is not None


def test_initial_step_size_3513():
    op = get_op("initial_step_size")
    assert op is not None


def test_initialize_3514():
    op = get_op("initialize")
    assert op is not None


def test_initialize_parameters_3515():
    op = get_op("initialize_parameters")
    assert op is not None


def test_inject_hyperparams_3516():
    op = get_op("inject_hyperparams")
    assert op is not None


def test_inject_stateful_hyperparams_3517():
    op = get_op("inject_stateful_hyperparams")
    assert op is not None


def test_inner_3518():
    op = get_op("inner")
    assert op is not None


def test_inplace_3519():
    op = get_op("inplace")
    assert op is not None


def test_input_dtype_3520():
    op = get_op("input_dtype")
    assert op is not None


def test_input_size_3521():
    op = get_op("input_size")
    assert op is not None


def test_insert_3522():
    op = get_op("insert")
    assert op is not None


def test_insert_collective_pvary_3523():
    op = get_op("insert_collective_pvary")
    assert op is not None


def test_inside_call_tf_3524():
    op = get_op("inside_call_tf")
    assert op is not None


def test_instance_norm_3525():
    op = get_op("instance_norm")
    assert op is not None


def test_int_3526():
    op = get_op("int")
    assert op is not None


def test_int2_3527():
    op = get_op("int2")
    assert op is not None


def test_int4_3528():
    op = get_op("int4")
    assert op is not None


def test_int8_3529():
    op = get_op("int8")
    assert op is not None


def test_int__3530():
    op = get_op("int_")
    assert op is not None


def test_int_dtype_for_dim_3531():
    op = get_op("int_dtype_for_dim")
    assert op is not None


def test_int_dtype_for_shape_3532():
    op = get_op("int_dtype_for_shape")
    assert op is not None


def test_integer_3533():
    op = get_op("integer")
    assert op is not None


def test_integer_pow_3534():
    op = get_op("integer_pow")
    assert op is not None


def test_intern_name_3535():
    op = get_op("intern_name")
    assert op is not None


def test_interned_names_3536():
    op = get_op("interned_names")
    assert op is not None


def test_interp_3537():
    op = get_op("interp")
    assert op is not None


def test_interp_fit_dopri_3538():
    op = get_op("interp_fit_dopri")
    assert op is not None


def test_interpolate_3539():
    op = get_op("interpolate")
    assert op is not None


def test_intersect1d_3540():
    op = get_op("intersect1d")
    assert op is not None


def test_inv_ex_3541():
    op = get_op("inv_ex")
    assert op is not None


def test_inverse_time_decay_3542():
    op = get_op("inverse_time_decay")
    assert op is not None


def test_invert_3543():
    op = get_op("invert")
    assert op is not None


def test_invert_permutation_3544():
    op = get_op("invert_permutation")
    assert op is not None


def test_iota_3545():
    op = get_op("iota")
    assert op is not None


def test_ipu_3546():
    op = get_op("ipu")
    assert op is not None


def test_irfft_3547():
    op = get_op("irfft")
    assert op is not None


def test_irfft2_3548():
    op = get_op("irfft2")
    assert op is not None


def test_irfftn_3549():
    op = get_op("irfftn")
    assert op is not None


def test_is_array_ref_3550():
    op = get_op("is_array_ref")
    assert op is not None


def test_is_available_3551():
    op = get_op("is_available")
    assert op is not None


def test_is_bcoo_3552():
    op = get_op("is_bcoo")
    assert op is not None


def test_is_bcsr_3553():
    op = get_op("is_bcsr")
    assert op is not None


def test_is_busday_3554():
    op = get_op("is_busday")
    assert op is not None


def test_is_collective_kernel_3555():
    op = get_op("is_collective_kernel")
    assert op is not None


def test_is_cuda_3556():
    op = get_op("is_cuda")
    assert op is not None


def test_is_data_3557():
    op = get_op("is_data")
    assert op is not None


def test_is_dense_3558():
    op = get_op("is_dense")
    assert op is not None


def test_is_device_collective_3559():
    op = get_op("is_device_collective")
    assert op is not None


def test_is_dynamic_mask_3560():
    op = get_op("is_dynamic_mask")
    assert op is not None


def test_is_finite_3561():
    op = get_op("is_finite")
    assert op is not None


def test_is_graph_node_3562():
    op = get_op("is_graph_node")
    assert op is not None


def test_is_hlo_sharding_replicated_3563():
    op = get_op("is_hlo_sharding_replicated")
    assert op is not None


def test_is_known_divisible_3564():
    op = get_op("is_known_divisible")
    assert op is not None


def test_is_lazy_3565():
    op = get_op("is_lazy")
    assert op is not None


def test_is_memref_transposed_3566():
    op = get_op("is_memref_transposed")
    assert op is not None


def test_is_mma_layout_3567():
    op = get_op("is_mma_layout")
    assert op is not None


def test_is_multi_device_module_3568():
    op = get_op("is_multi_device_module")
    assert op is not None


def test_is_namedtuple_3569():
    op = get_op("is_namedtuple")
    assert op is not None


def test_is_node_3570():
    op = get_op("is_node")
    assert op is not None


def test_is_node_leaf_3571():
    op = get_op("is_node_leaf")
    assert op is not None


def test_is_node_type_3572():
    op = get_op("is_node_type")
    assert op is not None


def test_is_nvshmem_available_3573():
    op = get_op("is_nvshmem_available")
    assert op is not None


def test_is_nvshmem_used_3574():
    op = get_op("is_nvshmem_used")
    assert op is not None


def test_is_parametrized_3575():
    op = get_op("is_parametrized")
    assert op is not None


def test_is_pinned_3576():
    op = get_op("is_pinned")
    assert op is not None


def test_is_pruned_3577():
    op = get_op("is_pruned")
    assert op is not None


def test_is_pytree_node_3578():
    op = get_op("is_pytree_node")
    assert op is not None


def test_is_remote_storage_3579():
    op = get_op("is_remote_storage")
    assert op is not None


def test_is_signed_3580():
    op = get_op("is_signed")
    assert op is not None


def test_is_single_process_multi_device_topology_3581():
    op = get_op("is_single_process_multi_device_topology")
    assert op is not None


def test_is_smem_ref_3582():
    op = get_op("is_smem_ref")
    assert op is not None


def test_is_sparse_3583():
    op = get_op("is_sparse")
    assert op is not None


def test_is_splat_fragmented_layout_3584():
    op = get_op("is_splat_fragmented_layout")
    assert op is not None


def test_is_strided_fragmented_layout_3585():
    op = get_op("is_strided_fragmented_layout")
    assert op is not None


def test_is_swizzle_transform_3586():
    op = get_op("is_swizzle_transform")
    assert op is not None


def test_is_tensor_3587():
    op = get_op("is_tensor")
    assert op is not None


def test_is_tensorstore_spec_leaf_3588():
    op = get_op("is_tensorstore_spec_leaf")
    assert op is not None


def test_is_terminator_3589():
    op = get_op("is_terminator")
    assert op is not None


def test_is_tile_transform_3590():
    op = get_op("is_tile_transform")
    assert op is not None


def test_is_tiled_layout_3591():
    op = get_op("is_tiled_layout")
    assert op is not None


def test_is_tmem_ref_3592():
    op = get_op("is_tmem_ref")
    assert op is not None


def test_is_tpu_3593():
    op = get_op("is_tpu")
    assert op is not None


def test_is_transformable_smem_memref_3594():
    op = get_op("is_transformable_smem_memref")
    assert op is not None


def test_is_transpose_transform_3595():
    op = get_op("is_transpose_transform")
    assert op is not None


def test_is_tree_node_3596():
    op = get_op("is_tree_node")
    assert op is not None


def test_is_valid_register_layout_assignment_3597():
    op = get_op("is_valid_register_layout_assignment")
    assert op is not None


def test_is_valid_smem_layout_assignment_3598():
    op = get_op("is_valid_smem_layout_assignment")
    assert op is not None


def test_is_valid_tmem_layout_assignment_3599():
    op = get_op("is_valid_tmem_layout_assignment")
    assert op is not None


def test_is_vanilla_variable_3600():
    op = get_op("is_vanilla_variable")
    assert op is not None


def test_is_vector_3601():
    op = get_op("is_vector")
    assert op is not None


def test_isclose_3602():
    op = get_op("isclose")
    assert op is not None


def test_iscomplex_3603():
    op = get_op("iscomplex")
    assert op is not None


def test_iscomplexobj_3604():
    op = get_op("iscomplexobj")
    assert op is not None


def test_isdtype_3605():
    op = get_op("isdtype")
    assert op is not None


def test_isfinite_3606():
    op = get_op("isfinite")
    assert op is not None


def test_isfortran_3607():
    op = get_op("isfortran")
    assert op is not None


def test_isin_3608():
    op = get_op("isin")
    assert op is not None


def test_isinf_3609():
    op = get_op("isinf")
    assert op is not None


def test_isnan_3610():
    op = get_op("isnan")
    assert op is not None


def test_isneginf_3611():
    op = get_op("isneginf")
    assert op is not None


def test_isposinf_3612():
    op = get_op("isposinf")
    assert op is not None


def test_isreal_3613():
    op = get_op("isreal")
    assert op is not None


def test_isrealobj_3614():
    op = get_op("isrealobj")
    assert op is not None


def test_isscalar_3615():
    op = get_op("isscalar")
    assert op is not None


def test_issctype_3616():
    op = get_op("issctype")
    assert op is not None


def test_issubclass__3617():
    op = get_op("issubclass_")
    assert op is not None


def test_issubdtype_3618():
    op = get_op("issubdtype")
    assert op is not None


def test_issubsctype_3619():
    op = get_op("issubsctype")
    assert op is not None


def test_istft_3620():
    op = get_op("istft")
    assert op is not None


def test_item_3621():
    op = get_op("item")
    assert op is not None


def test_items_3622():
    op = get_op("items")
    assert op is not None


def test_itemsize_3623():
    op = get_op("itemsize")
    assert op is not None


def test_iter_bcsr_layouts_3624():
    op = get_op("iter_bcsr_layouts")
    assert op is not None


def test_iter_children_3625():
    op = get_op("iter_children")
    assert op is not None


def test_iter_graph_3626():
    op = get_op("iter_graph")
    assert op is not None


def test_iter_modules_3627():
    op = get_op("iter_modules")
    assert op is not None


def test_iter_sparse_layouts_3628():
    op = get_op("iter_sparse_layouts")
    assert op is not None


def test_iter_subsets_3629():
    op = get_op("iter_subsets")
    assert op is not None


def test_ix__3630():
    op = get_op("ix_")
    assert op is not None


def test_jacfwd_3631():
    op = get_op("jacfwd")
    assert op is not None


def test_jacobian_3632():
    op = get_op("jacobian")
    assert op is not None


def test_jacrev_3633():
    op = get_op("jacrev")
    assert op is not None


def test_jax_buffer_type_3634():
    op = get_op("jax_buffer_type")
    assert op is not None


def test_jax_to_hlo_3635():
    op = get_op("jax_to_hlo")
    assert op is not None


def test_jax_to_ir_3636():
    op = get_op("jax_to_ir")
    assert op is not None


def test_jax_to_nnx_path_3637():
    op = get_op("jax_to_nnx_path")
    assert op is not None


def test_jax_to_tf_3638():
    op = get_op("jax_to_tf")
    assert op is not None


def test_jaxpr_eqn_ctx_3639():
    op = get_op("jaxpr_eqn_ctx")
    assert op is not None


def test_jet_3640():
    op = get_op("jet")
    assert op is not None


def test_jet2_3641():
    op = get_op("jet2")
    assert op is not None


def test_jet_fun_3642():
    op = get_op("jet_fun")
    assert op is not None


def test_jet_rules_3643():
    op = get_op("jet_rules")
    assert op is not None


def test_jet_subtrace_3644():
    op = get_op("jet_subtrace")
    assert op is not None


def test_jit_3645():
    op = get_op("jit")
    assert op is not None


def test_jit_partial_3646():
    op = get_op("jit_partial")
    assert op is not None


def test_join_3647():
    op = get_op("join")
    assert op is not None


def test_join_device_3648():
    op = get_op("join_device")
    assert op is not None


def test_join_hook_3649():
    op = get_op("join_hook")
    assert op is not None


def test_join_process_group_3650():
    op = get_op("join_process_group")
    assert op is not None


def test_join_schedules_3651():
    op = get_op("join_schedules")
    assert op is not None


def test_jvp_3652():
    op = get_op("jvp")
    assert op is not None


def test_k_3653():
    op = get_op("k")
    assert op is not None


def test_k_layout_3654():
    op = get_op("k_layout")
    assert op is not None


def test_k_proj_weight_3655():
    op = get_op("k_proj_weight")
    assert op is not None


def test_kaiming_normal_3656():
    op = get_op("kaiming_normal")
    assert op is not None


def test_kaiming_normal__3657():
    op = get_op("kaiming_normal_")
    assert op is not None


def test_kaiming_uniform_3658():
    op = get_op("kaiming_uniform")
    assert op is not None


def test_kaiming_uniform__3659():
    op = get_op("kaiming_uniform_")
    assert op is not None


def test_kaiser_3660():
    op = get_op("kaiser")
    assert op is not None


def test_kdim_3661():
    op = get_op("kdim")
    assert op is not None


def test_keep_params_nonnegative_3662():
    op = get_op("keep_params_nonnegative")
    assert op is not None


def test_keepdim_3663():
    op = get_op("keepdim")
    assert op is not None


def test_kernel_3664():
    op = get_op("kernel")
    assert op is not None


def test_kernel_size_3665():
    op = get_op("kernel_size")
    assert op is not None


def test_key_3666():
    op = get_op("key")
    assert op is not None


def test_keys_3667():
    op = get_op("keys")
    assert op is not None


def test_kl_div_3668():
    op = get_op("kl_div")
    assert op is not None


def test_kl_divergence_3669():
    op = get_op("kl_divergence")
    assert op is not None


def test_kldivergence_3670():
    op = get_op("kldivergence")
    assert op is not None


def test_kpack_3671():
    op = get_op("kpack")
    assert op is not None


def test_kron_3672():
    op = get_op("kron")
    assert op is not None


def test_kv_3673():
    op = get_op("kv")
    assert op is not None


def test_kv_indices_3674():
    op = get_op("kv_indices")
    assert op is not None


def test_kv_num_blocks_3675():
    op = get_op("kv_num_blocks")
    assert op is not None


def test_kwargs_3676():
    op = get_op("kwargs")
    assert op is not None


def test_kwargs_specs_3677():
    op = get_op("kwargs_specs")
    assert op is not None


def test_l1_loss_3678():
    op = get_op("l1_loss")
    assert op is not None


def test_l1_unstructured_3679():
    op = get_op("l1_unstructured")
    assert op is not None


def test_l2_loss_3680():
    op = get_op("l2_loss")
    assert op is not None


def test_l2_norm_3681():
    op = get_op("l2_norm")
    assert op is not None


def test_label_smoothing_3682():
    op = get_op("label_smoothing")
    assert op is not None


def test_laguerre_polynomial_l_3683():
    op = get_op("laguerre_polynomial_l")
    assert op is not None


def test_lambd_3684():
    op = get_op("lambd")
    assert op is not None


def test_lane_dims_3685():
    op = get_op("lane_dims")
    assert op is not None


def test_lane_indices_3686():
    op = get_op("lane_indices")
    assert op is not None


def test_laplace_3687():
    op = get_op("laplace")
    assert op is not None


def test_launch_context_3688():
    op = get_op("launch_context")
    assert op is not None


def test_layer_norm_3689():
    op = get_op("layer_norm")
    assert op is not None


def test_layer_norm_backward_3690():
    op = get_op("layer_norm_backward")
    assert op is not None


def test_layer_norm_backward_kernel_dw_db_3691():
    op = get_op("layer_norm_backward_kernel_dw_db")
    assert op is not None


def test_layer_norm_backward_kernel_dx_3692():
    op = get_op("layer_norm_backward_kernel_dx")
    assert op is not None


def test_layer_norm_forward_3693():
    op = get_op("layer_norm_forward")
    assert op is not None


def test_layer_norm_forward_kernel_3694():
    op = get_op("layer_norm_forward_kernel")
    assert op is not None


def test_layer_norm_reference_3695():
    op = get_op("layer_norm_reference")
    assert op is not None


def test_layers_3696():
    op = get_op("layers")
    assert op is not None


def test_lazy_init_3697():
    op = get_op("lazy_init")
    assert op is not None


def test_lcm_3698():
    op = get_op("lcm")
    assert op is not None


def test_ldexp_3699():
    op = get_op("ldexp")
    assert op is not None


def test_ldl_factor_3700():
    op = get_op("ldl_factor")
    assert op is not None


def test_ldl_factor_ex_3701():
    op = get_op("ldl_factor_ex")
    assert op is not None


def test_ldl_solve_3702():
    op = get_op("ldl_solve")
    assert op is not None


def test_le_3703():
    op = get_op("le")
    assert op is not None


def test_leaky_relu_3704():
    op = get_op("leaky_relu")
    assert op is not None


def test_leaky_relu__3705():
    op = get_op("leaky_relu_")
    assert op is not None


def test_left_3706():
    op = get_op("left")
    assert op is not None


def test_left_shift_3707():
    op = get_op("left_shift")
    assert op is not None


def test_legendre_polynomial_p_3708():
    op = get_op("legendre_polynomial_p")
    assert op is not None


def test_length_3709():
    op = get_op("length")
    assert op is not None


def test_lentz_thompson_barnett_algorithm_3710():
    op = get_op("lentz_thompson_barnett_algorithm")
    assert op is not None


def test_less_3711():
    op = get_op("less")
    assert op is not None


def test_less_equal_3712():
    op = get_op("less_equal")
    assert op is not None


def test_lexsort_3713():
    op = get_op("lexsort")
    assert op is not None


def test_lgamma_3714():
    op = get_op("lgamma")
    assert op is not None


def test_lhs_3715():
    op = get_op("lhs")
    assert op is not None


def test_lhs_shape_3716():
    op = get_op("lhs_shape")
    assert op is not None


def test_libdevice_path_3717():
    op = get_op("libdevice_path")
    assert op is not None


def test_linalg_primitive_3718():
    op = get_op("linalg_primitive")
    assert op is not None


def test_linalg_shape_rule_3719():
    op = get_op("linalg_shape_rule")
    assert op is not None


def test_linalg_sharding_rule_3720():
    op = get_op("linalg_sharding_rule")
    assert op is not None


def test_linalg_vma_rule_3721():
    op = get_op("linalg_vma_rule")
    assert op is not None


def test_linear1_3722():
    op = get_op("linear1")
    assert op is not None


def test_linear2_3723():
    op = get_op("linear2")
    assert op is not None


def test_linear_onecycle_schedule_3724():
    op = get_op("linear_onecycle_schedule")
    assert op is not None


def test_linear_prop_3725():
    op = get_op("linear_prop")
    assert op is not None


def test_linear_schedule_3726():
    op = get_op("linear_schedule")
    assert op is not None


def test_linear_thread_idxs_3727():
    op = get_op("linear_thread_idxs")
    assert op is not None


def test_linear_to_mel_weight_matrix_3728():
    op = get_op("linear_to_mel_weight_matrix")
    assert op is not None


def test_linen_in_bridge_mdl_3729():
    op = get_op("linen_in_bridge_mdl")
    assert op is not None


def test_linen_rngs_dict_3730():
    op = get_op("linen_rngs_dict")
    assert op is not None


def test_linen_vars_to_nnx_attrs_3731():
    op = get_op("linen_vars_to_nnx_attrs")
    assert op is not None


def test_linspace_3732():
    op = get_op("linspace")
    assert op is not None


def test_list_flash_attention_impls_3733():
    op = get_op("list_flash_attention_impls")
    assert op is not None


def test_live_devices_3734():
    op = get_op("live_devices")
    assert op is not None


def test_ljust_3735():
    op = get_op("ljust")
    assert op is not None


def test_ln_structured_3736():
    op = get_op("ln_structured")
    assert op is not None


def test_load_3737():
    op = get_op("load")
    assert op is not None


def test_load_pytreedef_3738():
    op = get_op("load_pytreedef")
    assert op is not None


def test_load_reduce_untiled_3739():
    op = get_op("load_reduce_untiled")
    assert op is not None


def test_load_state_dict_3740():
    op = get_op("load_state_dict")
    assert op is not None


def test_load_strided_3741():
    op = get_op("load_strided")
    assert op is not None


def test_load_tiled_3742():
    op = get_op("load_tiled")
    assert op is not None


def test_load_untiled_3743():
    op = get_op("load_untiled")
    assert op is not None


def test_lobpcg_standard_3744():
    op = get_op("lobpcg_standard")
    assert op is not None


def test_local_response_norm_3745():
    op = get_op("local_response_norm")
    assert op is not None


def test_log10_3746():
    op = get_op("log10")
    assert op is not None


def test_log1p_3747():
    op = get_op("log1p")
    assert op is not None


def test_log2_3748():
    op = get_op("log2")
    assert op is not None


def test_log_cosh_3749():
    op = get_op("log_cosh")
    assert op is not None


def test_log_input_3750():
    op = get_op("log_input")
    assert op is not None


def test_log_ndtr_3751():
    op = get_op("log_ndtr")
    assert op is not None


def test_log_prob_3752():
    op = get_op("log_prob")
    assert op is not None


def test_log_sigmoid_3753():
    op = get_op("log_sigmoid")
    assert op is not None


def test_log_softmax_3754():
    op = get_op("log_softmax")
    assert op is not None


def test_log_target_3755():
    op = get_op("log_target")
    assert op is not None


def test_logaddexp_3756():
    op = get_op("logaddexp")
    assert op is not None


def test_logaddexp2_3757():
    op = get_op("logaddexp2")
    assert op is not None


def test_logcoshloss_3758():
    op = get_op("logcoshloss")
    assert op is not None


def test_logdet_3759():
    op = get_op("logdet")
    assert op is not None


def test_logger_3760():
    op = get_op("logger")
    assert op is not None


def test_logical_and_3761():
    op = get_op("logical_and")
    assert op is not None


def test_logical_not_3762():
    op = get_op("logical_not")
    assert op is not None


def test_logical_or_3763():
    op = get_op("logical_or")
    assert op is not None


def test_logical_xor_3764():
    op = get_op("logical_xor")
    assert op is not None


def test_logistic_3765():
    op = get_op("logistic")
    assert op is not None


def test_logistic_impl_3766():
    op = get_op("logistic_impl")
    assert op is not None


def test_logit_3767():
    op = get_op("logit")
    assert op is not None


def test_lognormal_3768():
    op = get_op("lognormal")
    assert op is not None


def test_logsigmoid_3769():
    op = get_op("logsigmoid")
    assert op is not None


def test_logspace_3770():
    op = get_op("logspace")
    assert op is not None


def test_logsumexp_3771():
    op = get_op("logsumexp")
    assert op is not None


def test_long_3772():
    op = get_op("long")
    assert op is not None


def test_lower_3773():
    op = get_op("lower")
    assert op is not None


def test_lower_mgpu_dialect_3774():
    op = get_op("lower_mgpu_dialect")
    assert op is not None


def test_lower_op_3775():
    op = get_op("lower_op")
    assert op is not None


def test_lowered_operations_3776():
    op = get_op("lowered_operations")
    assert op is not None


def test_lp_pool1d_3777():
    op = get_op("lp_pool1d")
    assert op is not None


def test_lp_pool2d_3778():
    op = get_op("lp_pool2d")
    assert op is not None


def test_lp_pool3d_3779():
    op = get_op("lp_pool3d")
    assert op is not None


def test_lpmn_3780():
    op = get_op("lpmn")
    assert op is not None


def test_lpmn_values_3781():
    op = get_op("lpmn_values")
    assert op is not None


def test_lse_3782():
    op = get_op("lse")
    assert op is not None


def test_lstm_bwd_3783():
    op = get_op("lstm_bwd")
    assert op is not None


def test_lstm_fwd_3784():
    op = get_op("lstm_fwd")
    assert op is not None


def test_lstm_ref_3785():
    op = get_op("lstm_ref")
    assert op is not None


def test_lstrip_3786():
    op = get_op("lstrip")
    assert op is not None


def test_lt_3787():
    op = get_op("lt")
    assert op is not None


def test_ltg_abstract_eval_3788():
    op = get_op("ltg_abstract_eval")
    assert op is not None


def test_ltg_batcher_3789():
    op = get_op("ltg_batcher")
    assert op is not None


def test_lu_3790():
    op = get_op("lu")
    assert op is not None


def test_lu_factor_3791():
    op = get_op("lu_factor")
    assert op is not None


def test_lu_factor_ex_3792():
    op = get_op("lu_factor_ex")
    assert op is not None


def test_lu_pivots_to_permutation_3793():
    op = get_op("lu_pivots_to_permutation")
    assert op is not None


def test_lu_solve_3794():
    op = get_op("lu_solve")
    assert op is not None


def test_main_3795():
    op = get_op("main")
    assert op is not None


def test_major_3796():
    op = get_op("major")
    assert op is not None


def test_make_arrays_3797():
    op = get_op("make_arrays")
    assert op is not None


def test_make_attention_mask_3798():
    op = get_op("make_attention_mask")
    assert op is not None


def test_make_attention_reference_3799():
    op = get_op("make_attention_reference")
    assert op is not None


def test_make_callable_3800():
    op = get_op("make_callable")
    assert op is not None


def test_make_causal_mask_3801():
    op = get_op("make_causal_mask")
    assert op is not None


def test_make_chunk_attention_mask_3802():
    op = get_op("make_chunk_attention_mask")
    assert op is not None


def test_make_code_3803():
    op = get_op("make_code")
    assert op is not None


def test_make_error_array_3804():
    op = get_op("make_error_array")
    assert op is not None


def test_make_group_metadata_3805():
    op = get_op("make_group_metadata")
    assert op is not None


def test_make_jaxpr_dump_3806():
    op = get_op("make_jaxpr_dump")
    assert op is not None


def test_make_local_attention_mask_3807():
    op = get_op("make_local_attention_mask")
    assert op is not None


def test_make_masked_mha_reference_3808():
    op = get_op("make_masked_mha_reference")
    assert op is not None


def test_make_masked_mqa_reference_3809():
    op = get_op("make_masked_mqa_reference")
    assert op is not None


def test_make_mesh_3810():
    op = get_op("make_mesh")
    assert op is not None


def test_make_random_mask_3811():
    op = get_op("make_random_mask")
    assert op is not None


def test_make_schedule_3812():
    op = get_op("make_schedule")
    assert op is not None


def test_make_splash_mha_3813():
    op = get_op("make_splash_mha")
    assert op is not None


def test_make_splash_mha_single_device_3814():
    op = get_op("make_splash_mha_single_device")
    assert op is not None


def test_make_splash_mqa_3815():
    op = get_op("make_splash_mqa")
    assert op is not None


def test_make_splash_mqa_single_device_3816():
    op = get_op("make_splash_mqa_single_device")
    assert op is not None


def test_make_ufuncs_3817():
    op = get_op("make_ufuncs")
    assert op is not None


def test_maketrans_3818():
    op = get_op("maketrans")
    assert op is not None


def test_manual_sharding_spec_3819():
    op = get_op("manual_sharding_spec")
    assert op is not None


def test_map_params_3820():
    op = get_op("map_params")
    assert op is not None


def test_map_state_3821():
    op = get_op("map_state")
    assert op is not None


def test_mapped_aval_3822():
    op = get_op("mapped_aval")
    assert op is not None


def test_margin_3823():
    op = get_op("margin")
    assert op is not None


def test_margin_ranking_loss_3824():
    op = get_op("margin_ranking_loss")
    assert op is not None


def test_mask_3825():
    op = get_op("mask")
    assert op is not None


def test_mask_check_3826():
    op = get_op("mask_check")
    assert op is not None


def test_mask_indices_3827():
    op = get_op("mask_indices")
    assert op is not None


def test_mask_mod_3828():
    op = get_op("mask_mod")
    assert op is not None


def test_mask_next_3829():
    op = get_op("mask_next")
    assert op is not None


def test_mask_variable_updates_3830():
    op = get_op("mask_variable_updates")
    assert op is not None


def test_masked_3831():
    op = get_op("masked")
    assert op is not None


def test_masks_3832():
    op = get_op("masks")
    assert op is not None


def test_materialize_3833():
    op = get_op("materialize")
    assert op is not None


def test_matmul_kernel_3834():
    op = get_op("matmul_kernel")
    assert op is not None


def test_matrix_exp_3835():
    op = get_op("matrix_exp")
    assert op is not None


def test_matrix_instr_nonkdim_3836():
    op = get_op("matrix_instr_nonkdim")
    assert op is not None


def test_matrix_norm_3837():
    op = get_op("matrix_norm")
    assert op is not None


def test_matrix_power_3838():
    op = get_op("matrix_power")
    assert op is not None


def test_matrix_rank_3839():
    op = get_op("matrix_rank")
    assert op is not None


def test_matrix_transpose_3840():
    op = get_op("matrix_transpose")
    assert op is not None


def test_matvec_3841():
    op = get_op("matvec")
    assert op is not None


def test_max_3842():
    op = get_op("max")
    assert op is not None


def test_max_concurrent_steps_3843():
    op = get_op("max_concurrent_steps")
    assert op is not None


def test_max_norm_3844():
    op = get_op("max_norm")
    assert op is not None


def test_max_pool_3845():
    op = get_op("max_pool")
    assert op is not None


def test_max_pool1d_3846():
    op = get_op("max_pool1d")
    assert op is not None


def test_max_pool1d_with_indices_3847():
    op = get_op("max_pool1d_with_indices")
    assert op is not None


def test_max_pool2d_3848():
    op = get_op("max_pool2d")
    assert op is not None


def test_max_pool2d_with_indices_3849():
    op = get_op("max_pool2d_with_indices")
    assert op is not None


def test_max_pool3d_3850():
    op = get_op("max_pool3d")
    assert op is not None


def test_max_pool3d_with_indices_3851():
    op = get_op("max_pool3d_with_indices")
    assert op is not None


def test_max_scores_3852():
    op = get_op("max_scores")
    assert op is not None


def test_max_unpool1d_3853():
    op = get_op("max_unpool1d")
    assert op is not None


def test_max_unpool2d_3854():
    op = get_op("max_unpool2d")
    assert op is not None


def test_max_unpool3d_3855():
    op = get_op("max_unpool3d")
    assert op is not None


def test_max_val_3856():
    op = get_op("max_val")
    assert op is not None


def test_may_share_memory_3857():
    op = get_op("may_share_memory")
    assert op is not None


def test_mean_error_ratio_3858():
    op = get_op("mean_error_ratio")
    assert op is not None


def test_meanabsoluteerror_3859():
    op = get_op("meanabsoluteerror")
    assert op is not None


def test_meanabsolutepercentageerror_3860():
    op = get_op("meanabsolutepercentageerror")
    assert op is not None


def test_meansquarederror_3861():
    op = get_op("meansquarederror")
    assert op is not None


def test_meansquaredlogarithmicerror_3862():
    op = get_op("meansquaredlogarithmicerror")
    assert op is not None


def test_measure_3863():
    op = get_op("measure")
    assert op is not None


def test_measure_valued_estimation_mean_3864():
    op = get_op("measure_valued_estimation_mean")
    assert op is not None


def test_measure_valued_estimation_std_3865():
    op = get_op("measure_valued_estimation_std")
    assert op is not None


def test_measure_valued_jacobians_3866():
    op = get_op("measure_valued_jacobians")
    assert op is not None


def test_median_3867():
    op = get_op("median")
    assert op is not None


def test_members_3868():
    op = get_op("members")
    assert op is not None


def test_memmap_3869():
    op = get_op("memmap")
    assert op is not None


def test_memory_space_3870():
    op = get_op("memory_space")
    assert op is not None


def test_memref_fold_3871():
    op = get_op("memref_fold")
    assert op is not None


def test_memref_ptr_3872():
    op = get_op("memref_ptr")
    assert op is not None


def test_memref_reshape_3873():
    op = get_op("memref_reshape")
    assert op is not None


def test_memref_slice_3874():
    op = get_op("memref_slice")
    assert op is not None


def test_memref_transpose_3875():
    op = get_op("memref_transpose")
    assert op is not None


def test_memref_unfold_3876():
    op = get_op("memref_unfold")
    assert op is not None


def test_memref_unsqueeze_3877():
    op = get_op("memref_unsqueeze")
    assert op is not None


def test_merge_3878():
    op = get_op("merge")
    assert op is not None


def test_merge_api_dicts_3879():
    op = get_op("merge_api_dicts")
    assert op is not None


def test_merge_context_3880():
    op = get_op("merge_context")
    assert op is not None


def test_merge_divides_constraints_3881():
    op = get_op("merge_divides_constraints")
    assert op is not None


def test_merge_inputs_3882():
    op = get_op("merge_inputs")
    assert op is not None


def test_merge_masks_3883():
    op = get_op("merge_masks")
    assert op is not None


def test_merge_nested_ts_specs_3884():
    op = get_op("merge_nested_ts_specs")
    assert op is not None


def test_merge_state_3885():
    op = get_op("merge_state")
    assert op is not None


def test_merge_tree_node_3886():
    op = get_op("merge_tree_node")
    assert op is not None


def test_mesh_3887():
    op = get_op("mesh")
    assert op is not None


def test_meshgrid_3888():
    op = get_op("meshgrid")
    assert op is not None


def test_metadata_3889():
    op = get_op("metadata")
    assert op is not None


def test_mha_3890():
    op = get_op("mha")
    assert op is not None


def test_mha_backward_kernel_3891():
    op = get_op("mha_backward_kernel")
    assert op is not None


def test_mha_forward_kernel_3892():
    op = get_op("mha_forward_kernel")
    assert op is not None


def test_mha_reference_3893():
    op = get_op("mha_reference")
    assert op is not None


def test_mha_reference_bwd_3894():
    op = get_op("mha_reference_bwd")
    assert op is not None


def test_mha_reference_no_custom_vjp_3895():
    op = get_op("mha_reference_no_custom_vjp")
    assert op is not None


def test_min_3896():
    op = get_op("min")
    assert op is not None


def test_min_scalar_type_3897():
    op = get_op("min_scalar_type")
    assert op is not None


def test_min_val_3898():
    op = get_op("min_val")
    assert op is not None


def test_minor_3899():
    op = get_op("minor")
    assert op is not None


def test_mish_3900():
    op = get_op("mish")
    assert op is not None


def test_mixed_matmul_kernel_3901():
    op = get_op("mixed_matmul_kernel")
    assert op is not None


def test_mixed_precision_3902():
    op = get_op("mixed_precision")
    assert op is not None


def test_mlir_buffer_type_3903():
    op = get_op("mlir_buffer_type")
    assert op is not None


def test_mlir_dtype_3904():
    op = get_op("mlir_dtype")
    assert op is not None


def test_mma_3905():
    op = get_op("mma")
    assert op is not None


def test_mod_3906():
    op = get_op("mod")
    assert op is not None


def test_mode_3907():
    op = get_op("mode")
    assert op is not None


def test_modf_3908():
    op = get_op("modf")
    assert op is not None


def test_modified_bessel_i0_3909():
    op = get_op("modified_bessel_i0")
    assert op is not None


def test_modified_bessel_i1_3910():
    op = get_op("modified_bessel_i1")
    assert op is not None


def test_modified_bessel_k0_3911():
    op = get_op("modified_bessel_k0")
    assert op is not None


def test_modified_bessel_k1_3912():
    op = get_op("modified_bessel_k1")
    assert op is not None


def test_modified_orthogonal_3913():
    op = get_op("modified_orthogonal")
    assert op is not None


def test_module_3914():
    op = get_op("module")
    assert op is not None


def test_modules_3915():
    op = get_op("modules")
    assert op is not None


def test_moments_3916():
    op = get_op("moments")
    assert op is not None


def test_momentum_3917():
    op = get_op("momentum")
    assert op is not None


def test_mosaic_gpu_p_3918():
    op = get_op("mosaic_gpu_p")
    assert op is not None


def test_moveaxis_3919():
    op = get_op("moveaxis")
    assert op is not None


def test_moving_avg_baseline_3920():
    op = get_op("moving_avg_baseline")
    assert op is not None


def test_mqa_3921():
    op = get_op("mqa")
    assert op is not None


def test_mqa_reference_3922():
    op = get_op("mqa_reference")
    assert op is not None


def test_mse_loss_3923():
    op = get_op("mse_loss")
    assert op is not None


def test_mtia_3924():
    op = get_op("mtia")
    assert op is not None


def test_mul_3925():
    op = get_op("mul")
    assert op is not None


def test_mul32_hi_lo_3926():
    op = get_op("mul32_hi_lo")
    assert op is not None


def test_mulf_3927():
    op = get_op("mulf")
    assert op is not None


def test_multi_dot_3928():
    op = get_op("multi_dot")
    assert op is not None


def test_multi_head_attention_forward_3929():
    op = get_op("multi_head_attention_forward")
    assert op is not None


def test_multi_margin_loss_3930():
    op = get_op("multi_margin_loss")
    assert op is not None


def test_multi_mem_space_rule_3931():
    op = get_op("multi_mem_space_rule")
    assert op is not None


def test_multi_transform_3932():
    op = get_op("multi_transform")
    assert op is not None


def test_multigammaln_3933():
    op = get_op("multigammaln")
    assert op is not None


def test_multihead_attn_3934():
    op = get_op("multihead_attn")
    assert op is not None


def test_multilabel_margin_loss_3935():
    op = get_op("multilabel_margin_loss")
    assert op is not None


def test_multilabel_soft_margin_loss_3936():
    op = get_op("multilabel_soft_margin_loss")
    assert op is not None


def test_multimem_load_reduce_3937():
    op = get_op("multimem_load_reduce")
    assert op is not None


def test_multimem_store_3938():
    op = get_op("multimem_store")
    assert op is not None


def test_multivariate_normal_3939():
    op = get_op("multivariate_normal")
    assert op is not None


def test_n_3940():
    op = get_op("n")
    assert op is not None


def test_n_batch_3941():
    op = get_op("n_batch")
    assert op is not None


def test_n_classes_3942():
    op = get_op("n_classes")
    assert op is not None


def test_n_clusters_3943():
    op = get_op("n_clusters")
    assert op is not None


def test_n_dense_3944():
    op = get_op("n_dense")
    assert op is not None


def test_n_power_iterations_3945():
    op = get_op("n_power_iterations")
    assert op is not None


def test_n_sparse_3946():
    op = get_op("n_sparse")
    assert op is not None


def test_name_3947():
    op = get_op("name")
    assert op is not None


def test_name_stack_3948():
    op = get_op("name_stack")
    assert op is not None


def test_named_buffers_3949():
    op = get_op("named_buffers")
    assert op is not None


def test_named_chain_3950():
    op = get_op("named_chain")
    assert op is not None


def test_named_children_3951():
    op = get_op("named_children")
    assert op is not None


def test_named_modules_3952():
    op = get_op("named_modules")
    assert op is not None


def test_named_parameters_3953():
    op = get_op("named_parameters")
    assert op is not None


def test_named_region_3954():
    op = get_op("named_region")
    assert op is not None


def test_nan_3955():
    op = get_op("nan")
    assert op is not None


def test_nan_to_num_3956():
    op = get_op("nan_to_num")
    assert op is not None


def test_nanargmax_3957():
    op = get_op("nanargmax")
    assert op is not None


def test_nanargmin_3958():
    op = get_op("nanargmin")
    assert op is not None


def test_nancumprod_3959():
    op = get_op("nancumprod")
    assert op is not None


def test_nancumsum_3960():
    op = get_op("nancumsum")
    assert op is not None


def test_nanmax_3961():
    op = get_op("nanmax")
    assert op is not None


def test_nanmean_3962():
    op = get_op("nanmean")
    assert op is not None


def test_nanmedian_3963():
    op = get_op("nanmedian")
    assert op is not None


def test_nanmin_3964():
    op = get_op("nanmin")
    assert op is not None


def test_nanpercentile_3965():
    op = get_op("nanpercentile")
    assert op is not None


def test_nanprod_3966():
    op = get_op("nanprod")
    assert op is not None


def test_nanquantile_3967():
    op = get_op("nanquantile")
    assert op is not None


def test_nanstd_3968():
    op = get_op("nanstd")
    assert op is not None


def test_nansum_3969():
    op = get_op("nansum")
    assert op is not None


def test_nanvar_3970():
    op = get_op("nanvar")
    assert op is not None


def test_nary_reduced_rule_3971():
    op = get_op("nary_reduced_rule")
    assert op is not None


def test_naryop_3972():
    op = get_op("naryop")
    assert op is not None


def test_naryop_dtype_rule_3973():
    op = get_op("naryop_dtype_rule")
    assert op is not None


def test_native_channel_shuffle_3974():
    op = get_op("native_channel_shuffle")
    assert op is not None


def test_native_serialization_disabled_checks_3975():
    op = get_op("native_serialization_disabled_checks")
    assert op is not None


def test_native_serialization_platforms_3976():
    op = get_op("native_serialization_platforms")
    assert op is not None


def test_nbytes_3977():
    op = get_op("nbytes")
    assert op is not None


def test_ndarray_3978():
    op = get_op("ndarray")
    assert op is not None


def test_ndim_3979():
    op = get_op("ndim")
    assert op is not None


def test_ndtr_3980():
    op = get_op("ndtr")
    assert op is not None


def test_ndtri_3981():
    op = get_op("ndtri")
    assert op is not None


def test_ne_3982():
    op = get_op("ne")
    assert op is not None


def test_neg_3983():
    op = get_op("neg")
    assert op is not None


def test_negative_slope_3984():
    op = get_op("negative_slope")
    assert op is not None


def test_nesterov_3985():
    op = get_op("nesterov")
    assert op is not None


def test_next_offset_3986():
    op = get_op("next_offset")
    assert op is not None


def test_next_power_of_2_3987():
    op = get_op("next_power_of_2")
    assert op is not None


def test_nextafter_3988():
    op = get_op("nextafter")
    assert op is not None


def test_nfold_vmap_3989():
    op = get_op("nfold_vmap")
    assert op is not None


def test_nhead_3990():
    op = get_op("nhead")
    assert op is not None


def test_nll_loss_3991():
    op = get_op("nll_loss")
    assert op is not None


def test_nm_pack_3992():
    op = get_op("nm_pack")
    assert op is not None


def test_nm_pack_p_3993():
    op = get_op("nm_pack_p")
    assert op is not None


def test_nm_spmm_3994():
    op = get_op("nm_spmm")
    assert op is not None


def test_nm_spmm_p_3995():
    op = get_op("nm_spmm_p")
    assert op is not None


def test_nnx_attrs_to_linen_vars_3996():
    op = get_op("nnx_attrs_to_linen_vars")
    assert op is not None


def test_nnx_in_bridge_mdl_3997():
    op = get_op("nnx_in_bridge_mdl")
    assert op is not None


def test_no_grad_3998():
    op = get_op("no_grad")
    assert op is not None


def test_no_sync_3999():
    op = get_op("no_sync")
    assert op is not None


def test_non_splat_variables_4000():
    op = get_op("non_splat_variables")
    assert op is not None


def test_nonblocking_load_4001():
    op = get_op("nonblocking_load")
    assert op is not None


def test_nonblocking_save_4002():
    op = get_op("nonblocking_save")
    assert op is not None


def test_nonlinearity_4003():
    op = get_op("nonlinearity")
    assert op is not None


def test_nonzero_4004():
    op = get_op("nonzero")
    assert op is not None


def test_noop_mask_4005():
    op = get_op("noop_mask")
    assert op is not None


def test_norm1_4006():
    op = get_op("norm1")
    assert op is not None


def test_norm2_4007():
    op = get_op("norm2")
    assert op is not None


def test_norm3_4008():
    op = get_op("norm3")
    assert op is not None


def test_norm_first_4009():
    op = get_op("norm_first")
    assert op is not None


def test_norm_type_4010():
    op = get_op("norm_type")
    assert op is not None


def test_normal__4011():
    op = get_op("normal_")
    assert op is not None


def test_normalize_4012():
    op = get_op("normalize")
    assert op is not None


def test_normalize_axis_tuple_4013():
    op = get_op("normalize_axis_tuple")
    assert op is not None


def test_normalize_doc_4014():
    op = get_op("normalize_doc")
    assert op is not None


def test_normalized_shape_4015():
    op = get_op("normalized_shape")
    assert op is not None


def test_not_equal_4016():
    op = get_op("not_equal")
    assert op is not None


def test_npy_ctypes_check_4017():
    op = get_op("npy_ctypes_check")
    assert op is not None


def test_nse_4018():
    op = get_op("nse")
    assert op is not None


def test_nsys_path_4019():
    op = get_op("nsys_path")
    assert op is not None


def test_ntensors_4020():
    op = get_op("ntensors")
    assert op is not None


def test_ntxent_4021():
    op = get_op("ntxent")
    assert op is not None


def test_num_barriers_4022():
    op = get_op("num_barriers")
    assert op is not None


def test_num_channels_4023():
    op = get_op("num_channels")
    assert op is not None


def test_num_chunks_4024():
    op = get_op("num_chunks")
    assert op is not None


def test_num_embeddings_4025():
    op = get_op("num_embeddings")
    assert op is not None


def test_num_features_4026():
    op = get_op("num_features")
    assert op is not None


def test_num_groups_4027():
    op = get_op("num_groups")
    assert op is not None


def test_num_heads_4028():
    op = get_op("num_heads")
    assert op is not None


def test_num_layers_4029():
    op = get_op("num_layers")
    assert op is not None


def test_num_parameters_4030():
    op = get_op("num_parameters")
    assert op is not None


def test_num_peers_4031():
    op = get_op("num_peers")
    assert op is not None


def test_num_stages_4032():
    op = get_op("num_stages")
    assert op is not None


def test_num_warps_4033():
    op = get_op("num_warps")
    assert op is not None


def test_number_4034():
    op = get_op("number")
    assert op is not None


def test_numel_4035():
    op = get_op("numel")
    assert op is not None


def test_numeric_type_aliases_4036():
    op = get_op("numeric_type_aliases")
    assert op is not None


def test_nvvm_mbarrier_arrive_expect_tx_4037():
    op = get_op("nvvm_mbarrier_arrive_expect_tx")
    assert op is not None


def test_obj2sctype_4038():
    op = get_op("obj2sctype")
    assert op is not None


def test_object__4039():
    op = get_op("object_")
    assert op is not None


def test_odeint_4040():
    op = get_op("odeint")
    assert op is not None


def test_offset_4041():
    op = get_op("offset")
    assert op is not None


def test_one_hot_4042():
    op = get_op("one_hot")
    assert op is not None


def test_ones_4043():
    op = get_op("ones")
    assert op is not None


def test_ones__4044():
    op = get_op("ones_")
    assert op is not None


def test_ones_init_4045():
    op = get_op("ones_init")
    assert op is not None


def test_ones_like_4046():
    op = get_op("ones_like")
    assert op is not None


def test_op_sharding_to_indices_4047():
    op = get_op("op_sharding_to_indices")
    assert op is not None


def test_operation_4048():
    op = get_op("operation")
    assert op is not None


def test_optimal_step_size_4049():
    op = get_op("optimal_step_size")
    assert op is not None


def test_optimization_barrier_4050():
    op = get_op("optimization_barrier")
    assert op is not None


def test_optimized_generate_dump_4051():
    op = get_op("optimized_generate_dump")
    assert op is not None


def test_or_masks_4052():
    op = get_op("or_masks")
    assert op is not None


def test_order_4053():
    op = get_op("order")
    assert op is not None


def test_order_dict_4054():
    op = get_op("order_dict")
    assert op is not None


def test_original_4055():
    op = get_op("original")
    assert op is not None


def test_original_hlo_generate_dump_4056():
    op = get_op("original_hlo_generate_dump")
    assert op is not None


def test_orthogonal_4057():
    op = get_op("orthogonal")
    assert op is not None


def test_orthogonal__4058():
    op = get_op("orthogonal_")
    assert op is not None


def test_out_channels_4059():
    op = get_op("out_channels")
    assert op is not None


def test_out_features_4060():
    op = get_op("out_features")
    assert op is not None


def test_out_layouts_4061():
    op = get_op("out_layouts")
    assert op is not None


def test_out_proj_4062():
    op = get_op("out_proj")
    assert op is not None


def test_out_specs_fn_4063():
    op = get_op("out_specs_fn")
    assert op is not None


def test_out_specs_leaves_4064():
    op = get_op("out_specs_leaves")
    assert op is not None


def test_out_specs_treedef_4065():
    op = get_op("out_specs_treedef")
    assert op is not None


def test_out_tmem_layouts_4066():
    op = get_op("out_tmem_layouts")
    assert op is not None


def test_out_transforms_4067():
    op = get_op("out_transforms")
    assert op is not None


def test_outer_4068():
    op = get_op("outer")
    assert op is not None


def test_output_device_4069():
    op = get_op("output_device")
    assert op is not None


def test_output_ratio_4070():
    op = get_op("output_ratio")
    assert op is not None


def test_output_size_4071():
    op = get_op("output_size")
    assert op is not None


def test_p_4072():
    op = get_op("p")
    assert op is not None


def test_pack_array_4073():
    op = get_op("pack_array")
    assert op is not None


def test_pack_optimizer_state_4074():
    op = get_op("pack_optimizer_state")
    assert op is not None


def test_pack_padded_sequence_4075():
    op = get_op("pack_padded_sequence")
    assert op is not None


def test_pack_sequence_4076():
    op = get_op("pack_sequence")
    assert op is not None


def test_packbits_4077():
    op = get_op("packbits")
    assert op is not None


def test_packing_4078():
    op = get_op("packing")
    assert op is not None


def test_pad_4079():
    op = get_op("pad")
    assert op is not None


def test_pad_packed_sequence_4080():
    op = get_op("pad_packed_sequence")
    assert op is not None


def test_pad_sequence_4081():
    op = get_op("pad_sequence")
    assert op is not None


def test_padding_4082():
    op = get_op("padding")
    assert op is not None


def test_padding_idx_4083():
    op = get_op("padding_idx")
    assert op is not None


def test_padtype_to_pads_4084():
    op = get_op("padtype_to_pads")
    assert op is not None


def test_paged_attention_4085():
    op = get_op("paged_attention")
    assert op is not None


def test_paged_attention_kernel_4086():
    op = get_op("paged_attention_kernel")
    assert op is not None


def test_paged_attention_reference_4087():
    op = get_op("paged_attention_reference")
    assert op is not None


def test_paged_attention_unbatched_4088():
    op = get_op("paged_attention_unbatched")
    assert op is not None


def test_paged_flash_attention_kernel_4089():
    op = get_op("paged_flash_attention_kernel")
    assert op is not None


def test_paged_flash_attention_kernel_inline_seq_dim_4090():
    op = get_op("paged_flash_attention_kernel_inline_seq_dim")
    assert op is not None


def test_pairwise_distance_4091():
    op = get_op("pairwise_distance")
    assert op is not None


def test_parallel_4092():
    op = get_op("parallel")
    assert op is not None


def test_parallel_apply_4093():
    op = get_op("parallel_apply")
    assert op is not None


def test_parallel_callable_4094():
    op = get_op("parallel_callable")
    assert op is not None


def test_parameters_4095():
    op = get_op("parameters")
    assert op is not None


def test_parameters_to_ignore_4096():
    op = get_op("parameters_to_ignore")
    assert op is not None


def test_parameters_to_vector_4097():
    op = get_op("parameters_to_vector")
    assert op is not None


def test_params_4098():
    op = get_op("params")
    assert op is not None


def test_params_fn_4099():
    op = get_op("params_fn")
    assert op is not None


def test_parent_trace_4100():
    op = get_op("parent_trace")
    assert op is not None


def test_pareto_4101():
    op = get_op("pareto")
    assert op is not None


def test_pargmax_4102():
    op = get_op("pargmax")
    assert op is not None


def test_pargmin_4103():
    op = get_op("pargmin")
    assert op is not None


def test_parse_hlo_dump_4104():
    op = get_op("parse_hlo_dump")
    assert op is not None


def test_parse_indices_4105():
    op = get_op("parse_indices")
    assert op is not None


def test_parse_mlir_locations_4106():
    op = get_op("parse_mlir_locations")
    assert op is not None


def test_parse_shape_str_4107():
    op = get_op("parse_shape_str")
    assert op is not None


def test_parser_4108():
    op = get_op("parser")
    assert op is not None


def test_partial_4109():
    op = get_op("partial")
    assert op is not None


def test_partial_mask_blocks_4110():
    op = get_op("partial_mask_blocks")
    assert op is not None


def test_partition_4111():
    op = get_op("partition")
    assert op is not None


def test_partitioned_lane_dims_4112():
    op = get_op("partitioned_lane_dims")
    assert op is not None


def test_partitioned_warp_dims_4113():
    op = get_op("partitioned_warp_dims")
    assert op is not None


def test_pass_name_4114():
    op = get_op("pass_name")
    assert op is not None


def test_paths_4115():
    op = get_op("paths")
    assert op is not None


def test_pathwise_jacobians_4116():
    op = get_op("pathwise_jacobians")
    assert op is not None


def test_pbroadcast_4117():
    op = get_op("pbroadcast")
    assert op is not None


def test_pcast_4118():
    op = get_op("pcast")
    assert op is not None


def test_pdist_4119():
    op = get_op("pdist")
    assert op is not None


def test_peak_hbm_bytes_4120():
    op = get_op("peak_hbm_bytes")
    assert op is not None


def test_per_example_global_norm_clip_4121():
    op = get_op("per_example_global_norm_clip")
    assert op is not None


def test_per_example_layer_norm_clip_4122():
    op = get_op("per_example_layer_norm_clip")
    assert op is not None


def test_percentile_4123():
    op = get_op("percentile")
    assert op is not None


def test_permutation_4124():
    op = get_op("permutation")
    assert op is not None


def test_permute_dims_4125():
    op = get_op("permute_dims")
    assert op is not None


def test_permute_hidden_4126():
    op = get_op("permute_hidden")
    assert op is not None


def test_persistent_4127():
    op = get_op("persistent")
    assert op is not None


def test_pgle_filename_4128():
    op = get_op("pgle_filename")
    assert op is not None


def test_pgle_folder_4129():
    op = get_op("pgle_folder")
    assert op is not None


def test_phases_4130():
    op = get_op("phases")
    assert op is not None


def test_philox_4x32_4131():
    op = get_op("philox_4x32")
    assert op is not None


def test_philox_4x32_count_4132():
    op = get_op("philox_4x32_count")
    assert op is not None


def test_philox_4x32_kernel_4133():
    op = get_op("philox_4x32_kernel")
    assert op is not None


def test_philox_fold_in_4134():
    op = get_op("philox_fold_in")
    assert op is not None


def test_philox_random_bits_4135():
    op = get_op("philox_random_bits")
    assert op is not None


def test_philox_split_4136():
    op = get_op("philox_split")
    assert op is not None


def test_pi_4137():
    op = get_op("pi")
    assert op is not None


def test_piecewise_4138():
    op = get_op("piecewise")
    assert op is not None


def test_piecewise_constant_4139():
    op = get_op("piecewise_constant")
    assert op is not None


def test_piecewise_constant_schedule_4140():
    op = get_op("piecewise_constant_schedule")
    assert op is not None


def test_piecewise_interpolate_schedule_4141():
    op = get_op("piecewise_interpolate_schedule")
    assert op is not None


def test_pin_lhs_in_vmem_4142():
    op = get_op("pin_lhs_in_vmem")
    assert op is not None


def test_pin_memory_4143():
    op = get_op("pin_memory")
    assert op is not None


def test_pin_rhs_in_vmem_4144():
    op = get_op("pin_rhs_in_vmem")
    assert op is not None


def test_pixel_shuffle_4145():
    op = get_op("pixel_shuffle")
    assert op is not None


def test_pixel_unshuffle_4146():
    op = get_op("pixel_unshuffle")
    assert op is not None


def test_pjit_4147():
    op = get_op("pjit")
    assert op is not None


def test_place_4148():
    op = get_op("place")
    assert op is not None


def test_plan_tiled_transfer_4149():
    op = get_op("plan_tiled_transfer")
    assert op is not None


def test_platform_dependent_4150():
    op = get_op("platform_dependent")
    assert op is not None


def test_plphilox_prng_impl_4151():
    op = get_op("plphilox_prng_impl")
    assert op is not None


def test_plthreefry_prng_impl_4152():
    op = get_op("plthreefry_prng_impl")
    assert op is not None


def test_plthreefry_random_bits_4153():
    op = get_op("plthreefry_random_bits")
    assert op is not None


def test_pmax_4154():
    op = get_op("pmax")
    assert op is not None


def test_pmean_4155():
    op = get_op("pmean")
    assert op is not None


def test_pmin_4156():
    op = get_op("pmin")
    assert op is not None


def test_poisson_nll_loss_4157():
    op = get_op("poisson_nll_loss")
    assert op is not None


def test_poissonloss_4158():
    op = get_op("poissonloss")
    assert op is not None


def test_polar_4159():
    op = get_op("polar")
    assert op is not None


def test_poly_4160():
    op = get_op("poly")
    assert op is not None


def test_polyadd_4161():
    op = get_op("polyadd")
    assert op is not None


def test_polyder_4162():
    op = get_op("polyder")
    assert op is not None


def test_polydiv_4163():
    op = get_op("polydiv")
    assert op is not None


def test_polyfit_4164():
    op = get_op("polyfit")
    assert op is not None


def test_polygamma_4165():
    op = get_op("polygamma")
    assert op is not None


def test_polygamma_gradm_4166():
    op = get_op("polygamma_gradm")
    assert op is not None


def test_polygamma_gradx_4167():
    op = get_op("polygamma_gradx")
    assert op is not None


def test_polyint_4168():
    op = get_op("polyint")
    assert op is not None


def test_polymul_4169():
    op = get_op("polymul")
    assert op is not None


def test_polynomial_decay_4170():
    op = get_op("polynomial_decay")
    assert op is not None


def test_polynomial_schedule_4171():
    op = get_op("polynomial_schedule")
    assert op is not None


def test_polysub_4172():
    op = get_op("polysub")
    assert op is not None


def test_polyval_4173():
    op = get_op("polyval")
    assert op is not None


def test_pop_4174():
    op = get_op("pop")
    assert op is not None


def test_popitem_4175():
    op = get_op("popitem")
    assert op is not None


def test_population_count_4176():
    op = get_op("population_count")
    assert op is not None


def test_pos_weight_4177():
    op = get_op("pos_weight")
    assert op is not None


def test_positive_4178():
    op = get_op("positive")
    assert op is not None


def test_pow_4179():
    op = get_op("pow")
    assert op is not None


def test_ppermute_4180():
    op = get_op("ppermute")
    assert op is not None


def test_pprint_layout_4181():
    op = get_op("pprint_layout")
    assert op is not None


def test_precision_attr_4182():
    op = get_op("precision_attr")
    assert op is not None


def test_precv_4183():
    op = get_op("precv")
    assert op is not None


def test_predict_4184():
    op = get_op("predict")
    assert op is not None


def test_preduced_4185():
    op = get_op("preduced")
    assert op is not None


def test_prelu_4186():
    op = get_op("prelu")
    assert op is not None


def test_preprocess_arg_tf_4187():
    op = get_op("preprocess_arg_tf")
    assert op is not None


def test_primal_4188():
    op = get_op("primal")
    assert op is not None


def test_prime_decomposition_4189():
    op = get_op("prime_decomposition")
    assert op is not None


def test_primitive_4190():
    op = get_op("primitive")
    assert op is not None


def test_printoptions_4191():
    op = get_op("printoptions")
    assert op is not None


def test_proc_4192():
    op = get_op("proc")
    assert op is not None


def test_process_allgather_4193():
    op = get_op("process_allgather")
    assert op is not None


def test_process_call_4194():
    op = get_op("process_call")
    assert op is not None


def test_process_custom_jvp_call_4195():
    op = get_op("process_custom_jvp_call")
    assert op is not None


def test_process_custom_vjp_call_4196():
    op = get_op("process_custom_vjp_call")
    assert op is not None


def test_process_dynamic_mask_4197():
    op = get_op("process_dynamic_mask")
    assert op is not None


def test_process_dynamic_mask_dkv_4198():
    op = get_op("process_dynamic_mask_dkv")
    assert op is not None


def test_process_group_4199():
    op = get_op("process_group")
    assert op is not None


def test_process_mask_4200():
    op = get_op("process_mask")
    assert op is not None


def test_process_mask_dkv_4201():
    op = get_op("process_mask_dkv")
    assert op is not None


def test_process_primitive_4202():
    op = get_op("process_primitive")
    assert op is not None


def test_prod_4203():
    op = get_op("prod")
    assert op is not None


def test_producer_ref_4204():
    op = get_op("producer_ref")
    assert op is not None


def test_producer_result_4205():
    op = get_op("producer_result")
    assert op is not None


def test_profile_folder_4206():
    op = get_op("profile_folder")
    assert op is not None


def test_profiler_4207():
    op = get_op("profiler")
    assert op is not None


def test_proj_size_4208():
    op = get_op("proj_size")
    assert op is not None


def test_promote_args_4209():
    op = get_op("promote_args")
    assert op is not None


def test_promote_args_inexact_4210():
    op = get_op("promote_args_inexact")
    assert op is not None


def test_promote_args_numeric_4211():
    op = get_op("promote_args_numeric")
    assert op is not None


def test_promote_dtype_4212():
    op = get_op("promote_dtype")
    assert op is not None


def test_promote_dtypes_4213():
    op = get_op("promote_dtypes")
    assert op is not None


def test_promote_dtypes_complex_4214():
    op = get_op("promote_dtypes_complex")
    assert op is not None


def test_promote_dtypes_inexact_4215():
    op = get_op("promote_dtypes_inexact")
    assert op is not None


def test_promote_dtypes_numeric_4216():
    op = get_op("promote_dtypes_numeric")
    assert op is not None


def test_promote_shapes_4217():
    op = get_op("promote_shapes")
    assert op is not None


def test_promote_types_4218():
    op = get_op("promote_types")
    assert op is not None


def test_prune_4219():
    op = get_op("prune")
    assert op is not None


def test_psend_4220():
    op = get_op("psend")
    assert op is not None


def test_pshuffle_4221():
    op = get_op("pshuffle")
    assert op is not None


def test_psi_4222():
    op = get_op("psi")
    assert op is not None


def test_psnr_4223():
    op = get_op("psnr")
    assert op is not None


def test_psum_4224():
    op = get_op("psum")
    assert op is not None


def test_psum_scatter_4225():
    op = get_op("psum_scatter")
    assert op is not None


def test_pswapaxes_4226():
    op = get_op("pswapaxes")
    assert op is not None


def test_ptp_4227():
    op = get_op("ptp")
    assert op is not None


def test_ptr_4228():
    op = get_op("ptr")
    assert op is not None


def test_ptr_as_memref_4229():
    op = get_op("ptr_as_memref")
    assert op is not None


def test_pull_4230():
    op = get_op("pull")
    assert op is not None


def test_pure_4231():
    op = get_op("pure")
    assert op is not None


def test_put_4232():
    op = get_op("put")
    assert op is not None


def test_put_along_axis_4233():
    op = get_op("put_along_axis")
    assert op is not None


def test_putmask_4234():
    op = get_op("putmask")
    assert op is not None


def test_pvary_4235():
    op = get_op("pvary")
    assert op is not None


def test_q_4236():
    op = get_op("q")
    assert op is not None


def test_q_indices_4237():
    op = get_op("q_indices")
    assert op is not None


def test_q_layout_4238():
    op = get_op("q_layout")
    assert op is not None


def test_q_num_blocks_4239():
    op = get_op("q_num_blocks")
    assert op is not None


def test_q_proj_weight_4240():
    op = get_op("q_proj_weight")
    assert op is not None


def test_q_sequence_4241():
    op = get_op("q_sequence")
    assert op is not None


def test_qr_jvp_rule_4242():
    op = get_op("qr_jvp_rule")
    assert op is not None


def test_quantile_4243():
    op = get_op("quantile")
    assert op is not None


def test_quantize_4244():
    op = get_op("quantize")
    assert op is not None


def test_quantize_to_int8_4245():
    op = get_op("quantize_to_int8")
    assert op is not None


def test_query_4246():
    op = get_op("query")
    assert op is not None


def test_query_cluster_cancel_4247():
    op = get_op("query_cluster_cancel")
    assert op is not None


def test_query_reports_command_4248():
    op = get_op("query_reports_command")
    assert op is not None


def test_rad2deg_4249():
    op = get_op("rad2deg")
    assert op is not None


def test_radians_4250():
    op = get_op("radians")
    assert op is not None


def test_ragged_all_to_all_4251():
    op = get_op("ragged_all_to_all")
    assert op is not None


def test_ragged_dot_4252():
    op = get_op("ragged_dot")
    assert op is not None


def test_ragged_dot_general_4253():
    op = get_op("ragged_dot_general")
    assert op is not None


def test_ragged_dot_kernel_4254():
    op = get_op("ragged_dot_kernel")
    assert op is not None


def test_ragged_dot_reference_4255():
    op = get_op("ragged_dot_reference")
    assert op is not None


def test_ragged_paged_attention_4256():
    op = get_op("ragged_paged_attention")
    assert op is not None


def test_ragged_paged_attention_kernel_4257():
    op = get_op("ragged_paged_attention_kernel")
    assert op is not None


def test_rand_bcoo_4258():
    op = get_op("rand_bcoo")
    assert op is not None


def test_rand_bcsr_4259():
    op = get_op("rand_bcsr")
    assert op is not None


def test_rand_sparse_4260():
    op = get_op("rand_sparse")
    assert op is not None


def test_random_bcoo_4261():
    op = get_op("random_bcoo")
    assert op is not None


def test_random_gamma_grad_4262():
    op = get_op("random_gamma_grad")
    assert op is not None


def test_random_gamma_grad_impl_4263():
    op = get_op("random_gamma_grad_impl")
    assert op is not None


def test_random_like_4264():
    op = get_op("random_like")
    assert op is not None


def test_random_structured_4265():
    op = get_op("random_structured")
    assert op is not None


def test_random_unstructured_4266():
    op = get_op("random_unstructured")
    assert op is not None


def test_ranges_like_4267():
    op = get_op("ranges_like")
    assert op is not None


def test_rank_4268():
    op = get_op("rank")
    assert op is not None


def test_ravel_4269():
    op = get_op("ravel")
    assert op is not None


def test_ravel_first_arg_4270():
    op = get_op("ravel_first_arg")
    assert op is not None


def test_ravel_first_arg__4271():
    op = get_op("ravel_first_arg_")
    assert op is not None


def test_ravel_multi_index_4272():
    op = get_op("ravel_multi_index")
    assert op is not None


def test_rayleigh_4273():
    op = get_op("rayleigh")
    assert op is not None


def test_reached_preemption_sync_point_4274():
    op = get_op("reached_preemption_sync_point")
    assert op is not None


def test_reader_4275():
    op = get_op("reader")
    assert op is not None


def test_real_4276():
    op = get_op("real")
    assert op is not None


def test_recarray_4277():
    op = get_op("recarray")
    assert op is not None


def test_reciprocal_4278():
    op = get_op("reciprocal")
    assert op is not None


def test_recompute_scale_factor_4279():
    op = get_op("recompute_scale_factor")
    assert op is not None


def test_record_4280():
    op = get_op("record")
    assert op is not None


def test_recursive_map_4281():
    op = get_op("recursive_map")
    assert op is not None


def test_reduce_4282():
    op = get_op("reduce")
    assert op is not None


def test_reduce_add_4283():
    op = get_op("reduce_add")
    assert op is not None


def test_reduce_add_coalesced_4284():
    op = get_op("reduce_add_coalesced")
    assert op is not None


def test_reduce_and_4285():
    op = get_op("reduce_and")
    assert op is not None


def test_reduce_broadcast_expression_4286():
    op = get_op("reduce_broadcast_expression")
    assert op is not None


def test_reduce_constraint_4287():
    op = get_op("reduce_constraint")
    assert op is not None


def test_reduce_expression_4288():
    op = get_op("reduce_expression")
    assert op is not None


def test_reduce_max_4289():
    op = get_op("reduce_max")
    assert op is not None


def test_reduce_min_4290():
    op = get_op("reduce_min")
    assert op is not None


def test_reduce_or_4291():
    op = get_op("reduce_or")
    assert op is not None


def test_reduce_precision_4292():
    op = get_op("reduce_precision")
    assert op is not None


def test_reduce_prod_4293():
    op = get_op("reduce_prod")
    assert op is not None


def test_reduce_reshape_expression_4294():
    op = get_op("reduce_reshape_expression")
    assert op is not None


def test_reduce_scatter_4295():
    op = get_op("reduce_scatter")
    assert op is not None


def test_reduce_sum_4296():
    op = get_op("reduce_sum")
    assert op is not None


def test_reduce_transpose_expression_4297():
    op = get_op("reduce_transpose_expression")
    assert op is not None


def test_reduce_window_4298():
    op = get_op("reduce_window")
    assert op is not None


def test_reduce_window_jvp_4299():
    op = get_op("reduce_window_jvp")
    assert op is not None


def test_reduce_window_shape_tuple_4300():
    op = get_op("reduce_window_shape_tuple")
    assert op is not None


def test_reduce_window_sharding_rule_4301():
    op = get_op("reduce_window_sharding_rule")
    assert op is not None


def test_reduce_xor_4302():
    op = get_op("reduce_xor")
    assert op is not None


def test_reducing_transposes_4303():
    op = get_op("reducing_transposes")
    assert op is not None


def test_ref_4304():
    op = get_op("ref")
    assert op is not None


def test_ref_ragged_paged_attention_4305():
    op = get_op("ref_ragged_paged_attention")
    assert op is not None


def test_ref_transposed_ragged_dot_4306():
    op = get_op("ref_transposed_ragged_dot")
    assert op is not None


def test_reference_4307():
    op = get_op("reference")
    assert op is not None


def test_refine_4308():
    op = get_op("refine")
    assert op is not None


def test_refine_polymorphic_shapes_4309():
    op = get_op("refine_polymorphic_shapes")
    assert op is not None


def test_region_index_4310():
    op = get_op("region_index")
    assert op is not None


def test_register_backend_cache_4311():
    op = get_op("register_backend_cache")
    assert op is not None


def test_register_backward_hook_4312():
    op = get_op("register_backward_hook")
    assert op is not None


def test_register_buffer_4313():
    op = get_op("register_buffer")
    assert op is not None


def test_register_comm_hook_4314():
    op = get_op("register_comm_hook")
    assert op is not None


def test_register_cpu_gpu_lowering_4315():
    op = get_op("register_cpu_gpu_lowering")
    assert op is not None


def test_register_data_type_4316():
    op = get_op("register_data_type")
    assert op is not None


def test_register_flash_attention_impl_4317():
    op = get_op("register_flash_attention_impl")
    assert op is not None


def test_register_forward_hook_4318():
    op = get_op("register_forward_hook")
    assert op is not None


def test_register_forward_pre_hook_4319():
    op = get_op("register_forward_pre_hook")
    assert op is not None


def test_register_full_backward_hook_4320():
    op = get_op("register_full_backward_hook")
    assert op is not None


def test_register_full_backward_pre_hook_4321():
    op = get_op("register_full_backward_pre_hook")
    assert op is not None


def test_register_graph_node_type_4322():
    op = get_op("register_graph_node_type")
    assert op is not None


def test_register_jax_array_methods_4323():
    op = get_op("register_jax_array_methods")
    assert op is not None


def test_register_load_state_dict_post_hook_4324():
    op = get_op("register_load_state_dict_post_hook")
    assert op is not None


def test_register_load_state_dict_pre_hook_4325():
    op = get_op("register_load_state_dict_pre_hook")
    assert op is not None


def test_register_module_4326():
    op = get_op("register_module")
    assert op is not None


def test_register_module_backward_hook_4327():
    op = get_op("register_module_backward_hook")
    assert op is not None


def test_register_module_buffer_registration_hook_4328():
    op = get_op("register_module_buffer_registration_hook")
    assert op is not None


def test_register_module_custom_calls_4329():
    op = get_op("register_module_custom_calls")
    assert op is not None


def test_register_module_forward_hook_4330():
    op = get_op("register_module_forward_hook")
    assert op is not None


def test_register_module_forward_pre_hook_4331():
    op = get_op("register_module_forward_pre_hook")
    assert op is not None


def test_register_module_full_backward_hook_4332():
    op = get_op("register_module_full_backward_hook")
    assert op is not None


def test_register_module_full_backward_pre_hook_4333():
    op = get_op("register_module_full_backward_pre_hook")
    assert op is not None


def test_register_module_module_registration_hook_4334():
    op = get_op("register_module_module_registration_hook")
    assert op is not None


def test_register_module_parameter_registration_hook_4335():
    op = get_op("register_module_parameter_registration_hook")
    assert op is not None


def test_register_parameter_4336():
    op = get_op("register_parameter")
    assert op is not None


def test_register_parametrization_4337():
    op = get_op("register_parametrization")
    assert op is not None


def test_register_pass_4338():
    op = get_op("register_pass")
    assert op is not None


def test_register_pytree_node_type_4339():
    op = get_op("register_pytree_node_type")
    assert op is not None


def test_register_roofline_4340():
    op = get_op("register_roofline")
    assert op is not None


def test_register_standard_roofline_4341():
    op = get_op("register_standard_roofline")
    assert op is not None


def test_register_state_dict_post_hook_4342():
    op = get_op("register_state_dict_post_hook")
    assert op is not None


def test_register_state_dict_pre_hook_4343():
    op = get_op("register_state_dict_pre_hook")
    assert op is not None


def test_register_variable_name_4344():
    op = get_op("register_variable_name")
    assert op is not None


def test_registers_4345():
    op = get_op("registers")
    assert op is not None


def test_registers_element_type_4346():
    op = get_op("registers_element_type")
    assert op is not None


def test_registers_shape_4347():
    op = get_op("registers_shape")
    assert op is not None


def test_regularized_incomplete_beta_impl_4348():
    op = get_op("regularized_incomplete_beta_impl")
    assert op is not None


def test_relu2_4349():
    op = get_op("relu2")
    assert op is not None


def test_relu6_4350():
    op = get_op("relu6")
    assert op is not None


def test_relu__4351():
    op = get_op("relu_")
    assert op is not None


def test_rem_4352():
    op = get_op("rem")
    assert op is not None


def test_remainder_4353():
    op = get_op("remainder")
    assert op is not None


def test_remaining_4354():
    op = get_op("remaining")
    assert op is not None


def test_remat_4355():
    op = get_op("remat")
    assert op is not None


def test_remove_4356():
    op = get_op("remove")
    assert op is not None


def test_remove_axis_4357():
    op = get_op("remove_axis")
    assert op is not None


def test_remove_dimension_4358():
    op = get_op("remove_dimension")
    assert op is not None


def test_remove_parametrizations_4359():
    op = get_op("remove_parametrizations")
    assert op is not None


def test_remove_spectral_norm_4360():
    op = get_op("remove_spectral_norm")
    assert op is not None


def test_remove_weight_norm_4361():
    op = get_op("remove_weight_norm")
    assert op is not None


def test_remove_whitespace_4362():
    op = get_op("remove_whitespace")
    assert op is not None


def test_render_object_constructor_4363():
    op = get_op("render_object_constructor")
    assert op is not None


def test_repeat_4364():
    op = get_op("repeat")
    assert op is not None


def test_replace_4365():
    op = get_op("replace")
    assert op is not None


def test_replace_by_pure_dict_4366():
    op = get_op("replace_by_pure_dict")
    assert op is not None


def test_replicate_4367():
    op = get_op("replicate")
    assert op is not None


def test_replicated_axes_4368():
    op = get_op("replicated_axes")
    assert op is not None


def test_report_name_4369():
    op = get_op("report_name")
    assert op is not None


def test_reports_list_4370():
    op = get_op("reports_list")
    assert op is not None


def test_repr_format_4371():
    op = get_op("repr_format")
    assert op is not None


def test_require_4372():
    op = get_op("require")
    assert op is not None


def test_require_backward_grad_sync_4373():
    op = get_op("require_backward_grad_sync")
    assert op is not None


def test_require_forward_param_sync_4374():
    op = get_op("require_forward_param_sync")
    assert op is not None


def test_requires_grad__4375():
    op = get_op("requires_grad_")
    assert op is not None


def test_reseed_4376():
    op = get_op("reseed")
    assert op is not None


def test_reset_4377():
    op = get_op("reset")
    assert op is not None


def test_reset_parameters_4378():
    op = get_op("reset_parameters")
    assert op is not None


def test_reshape_weight_to_matrix_4379():
    op = get_op("reshape_weight_to_matrix")
    assert op is not None


def test_resize_4380():
    op = get_op("resize")
    assert op is not None


def test_resolve_kwargs_4381():
    op = get_op("resolve_kwargs")
    assert op is not None


def test_restore_int_paths_4382():
    op = get_op("restore_int_paths")
    assert op is not None


def test_restore_rngs_4383():
    op = get_op("restore_rngs")
    assert op is not None


def test_result_4384():
    op = get_op("result")
    assert op is not None


def test_result_type_4385():
    op = get_op("result_type")
    assert op is not None


def test_results_4386():
    op = get_op("results")
    assert op is not None


def test_return_indices_4387():
    op = get_op("return_indices")
    assert op is not None


def test_rev_4388():
    op = get_op("rev")
    assert op is not None


def test_reverse_4389():
    op = get_op("reverse")
    assert op is not None


def test_rewriting_take_4390():
    op = get_op("rewriting_take")
    assert op is not None


def test_rfft_4391():
    op = get_op("rfft")
    assert op is not None


def test_rfft2_4392():
    op = get_op("rfft2")
    assert op is not None


def test_rfftfreq_4393():
    op = get_op("rfftfreq")
    assert op is not None


def test_rfftn_4394():
    op = get_op("rfftn")
    assert op is not None


def test_rfind_4395():
    op = get_op("rfind")
    assert op is not None


def test_rgb_to_grayscale_4396():
    op = get_op("rgb_to_grayscale")
    assert op is not None


def test_rhs_4397():
    op = get_op("rhs")
    assert op is not None


def test_rhs_shape_4398():
    op = get_op("rhs_shape")
    assert op is not None


def test_right_4399():
    op = get_op("right")
    assert op is not None


def test_right_inverse_4400():
    op = get_op("right_inverse")
    assert op is not None


def test_right_shift_4401():
    op = get_op("right_shift")
    assert op is not None


def test_rindex_4402():
    op = get_op("rindex")
    assert op is not None


def test_rint_4403():
    op = get_op("rint")
    assert op is not None


def test_rjust_4404():
    op = get_op("rjust")
    assert op is not None


def test_rms_norm_4405():
    op = get_op("rms_norm")
    assert op is not None


def test_rms_norm_backward_4406():
    op = get_op("rms_norm_backward")
    assert op is not None


def test_rms_norm_backward_kernel_dw_db_4407():
    op = get_op("rms_norm_backward_kernel_dw_db")
    assert op is not None


def test_rms_norm_backward_kernel_dx_4408():
    op = get_op("rms_norm_backward_kernel_dx")
    assert op is not None


def test_rms_norm_forward_4409():
    op = get_op("rms_norm_forward")
    assert op is not None


def test_rms_norm_forward_kernel_4410():
    op = get_op("rms_norm_forward_kernel")
    assert op is not None


def test_rms_norm_reference_4411():
    op = get_op("rms_norm_reference")
    assert op is not None


def test_rms_normalization_4412():
    op = get_op("rms_normalization")
    assert op is not None


def test_rmsprop_momentum_4413():
    op = get_op("rmsprop_momentum")
    assert op is not None


def test_rng_bit_generator_4414():
    op = get_op("rng_bit_generator")
    assert op is not None


def test_rng_uniform_4415():
    op = get_op("rng_uniform")
    assert op is not None


def test_rnn_abstract_eval_4416():
    op = get_op("rnn_abstract_eval")
    assert op is not None


def test_rnn_bwd_abstract_eval_4417():
    op = get_op("rnn_bwd_abstract_eval")
    assert op is not None


def test_rnn_bwd_p_4418():
    op = get_op("rnn_bwd_p")
    assert op is not None


def test_rnn_fwd_p_4419():
    op = get_op("rnn_fwd_p")
    assert op is not None


def test_roll_4420():
    op = get_op("roll")
    assert op is not None


def test_rollaxis_4421():
    op = get_op("rollaxis")
    assert op is not None


def test_roofline_4422():
    op = get_op("roofline")
    assert op is not None


def test_roofline_and_grad_4423():
    op = get_op("roofline_and_grad")
    assert op is not None


def test_roots_4424():
    op = get_op("roots")
    assert op is not None


def test_rot90_4425():
    op = get_op("rot90")
    assert op is not None


def test_round_4426():
    op = get_op("round")
    assert op is not None


def test_round_even_4427():
    op = get_op("round_even")
    assert op is not None


def test_round_up_4428():
    op = get_op("round_up")
    assert op is not None


def test_rounding_4429():
    op = get_op("rounding")
    assert op is not None


def test_row_4430():
    op = get_op("row")
    assert op is not None


def test_row_stack_4431():
    op = get_op("row_stack")
    assert op is not None


def test_rows_sorted_4432():
    op = get_op("rows_sorted")
    assert op is not None


def test_rpartition_4433():
    op = get_op("rpartition")
    assert op is not None


def test_rrelu_4434():
    op = get_op("rrelu")
    assert op is not None


def test_rrelu__4435():
    op = get_op("rrelu_")
    assert op is not None


def test_rsqrt_4436():
    op = get_op("rsqrt")
    assert op is not None


def test_rstrip_4437():
    op = get_op("rstrip")
    assert op is not None


def test_run_fun_tf_4438():
    op = get_op("run_fun_tf")
    assert op is not None


def test_runge_kutta_step_4439():
    op = get_op("runge_kutta_step")
    assert op is not None


def test_safe_softmax_cross_entropy_4440():
    op = get_op("safe_softmax_cross_entropy")
    assert op is not None


def test_sample_group_sizes_4441():
    op = get_op("sample_group_sizes")
    assert op is not None


def test_saturate_cast_4442():
    op = get_op("saturate_cast")
    assert op is not None


def test_saturate_distinct_from_splat_4443():
    op = get_op("saturate_distinct_from_splat")
    assert op is not None


def test_saturate_divides_constraints_for_equal_vars_4444():
    op = get_op("saturate_divides_constraints_for_equal_vars")
    assert op is not None


def test_savez_4445():
    op = get_op("savez")
    assert op is not None


def test_savez_compressed_4446():
    op = get_op("savez_compressed")
    assert op is not None


def test_scalar_4447():
    op = get_op("scalar")
    assert op is not None


def test_scale_4448():
    op = get_op("scale")
    assert op is not None


def test_scale_factor_4449():
    op = get_op("scale_factor")
    assert op is not None


def test_scale_grad_by_freq_4450():
    op = get_op("scale_grad_by_freq")
    assert op is not None


def test_scaled_dot_4451():
    op = get_op("scaled_dot")
    assert op is not None


def test_scaled_dot_product_attention_4452():
    op = get_op("scaled_dot_product_attention")
    assert op is not None


def test_scaled_grouped_mm_4453():
    op = get_op("scaled_grouped_mm")
    assert op is not None


def test_scaled_mm_4454():
    op = get_op("scaled_mm")
    assert op is not None


def test_scaled_modified_bessel_k0_4455():
    op = get_op("scaled_modified_bessel_k0")
    assert op is not None


def test_scaled_modified_bessel_k1_4456():
    op = get_op("scaled_modified_bessel_k1")
    assert op is not None


def test_scales_4457():
    op = get_op("scales")
    assert op is not None


def test_scales_layout_4458():
    op = get_op("scales_layout")
    assert op is not None


def test_scatter_4459():
    op = get_op("scatter")
    assert op is not None


def test_scatter_add_4460():
    op = get_op("scatter_add")
    assert op is not None


def test_scatter_apply_4461():
    op = get_op("scatter_apply")
    assert op is not None


def test_scatter_kwargs_4462():
    op = get_op("scatter_kwargs")
    assert op is not None


def test_scatter_max_4463():
    op = get_op("scatter_max")
    assert op is not None


def test_scatter_min_4464():
    op = get_op("scatter_min")
    assert op is not None


def test_scatter_mul_4465():
    op = get_op("scatter_mul")
    assert op is not None


def test_scatter_sub_4466():
    op = get_op("scatter_sub")
    assert op is not None


def test_scatter_update_4467():
    op = get_op("scatter_update")
    assert op is not None


def test_schedule_4468():
    op = get_op("schedule")
    assert op is not None


def test_scheduling_group_4469():
    op = get_op("scheduling_group")
    assert op is not None


def test_schur_4470():
    op = get_op("schur")
    assert op is not None


def test_score_function_jacobians_4471():
    op = get_op("score_function_jacobians")
    assert op is not None


def test_scratch_4472():
    op = get_op("scratch")
    assert op is not None


def test_sctype2char_4473():
    op = get_op("sctype2char")
    assert op is not None


def test_sdpa_kernel_4474():
    op = get_op("sdpa_kernel")
    assert op is not None


def test_searchsorted_4475():
    op = get_op("searchsorted")
    assert op is not None


def test_seed_4476():
    op = get_op("seed")
    assert op is not None


def test_segment_mask_4477():
    op = get_op("segment_mask")
    assert op is not None


def test_segment_max_4478():
    op = get_op("segment_max")
    assert op is not None


def test_segment_sum_4479():
    op = get_op("segment_sum")
    assert op is not None


def test_select_4480():
    op = get_op("select")
    assert op is not None


def test_select_if_group_4481():
    op = get_op("select_if_group")
    assert op is not None


def test_select_input_dtype_4482():
    op = get_op("select_input_dtype")
    assert op is not None


def test_select_n_4483():
    op = get_op("select_n")
    assert op is not None


def test_selective_transform_4484():
    op = get_op("selective_transform")
    assert op is not None


def test_self_attn_4485():
    op = get_op("self_attn")
    assert op is not None


def test_selu__4486():
    op = get_op("selu_")
    assert op is not None


def test_separable_conv_4487():
    op = get_op("separable_conv")
    assert op is not None


def test_seq_len_kv_4488():
    op = get_op("seq_len_kv")
    assert op is not None


def test_seq_len_q_4489():
    op = get_op("seq_len_q")
    assert op is not None


def test_seq_lengths_4490():
    op = get_op("seq_lengths")
    assert op is not None


def test_serial_4491():
    op = get_op("serial")
    assert op is not None


def test_serialize_4492():
    op = get_op("serialize")
    assert op is not None


def test_serialize_portable_artifact_4493():
    op = get_op("serialize_portable_artifact")
    assert op is not None


def test_serialize_pytreedef_4494():
    op = get_op("serialize_pytreedef")
    assert op is not None


def test_serialize_with_paths_4495():
    op = get_op("serialize_with_paths")
    assert op is not None


def test_serializeloss_4496():
    op = get_op("serializeloss")
    assert op is not None


def test_set_4497():
    op = get_op("set")
    assert op is not None


def test_set_current_trace_4498():
    op = get_op("set_current_trace")
    assert op is not None


def test_set_default_device_4499():
    op = get_op("set_default_device")
    assert op is not None


def test_set_extra_state_4500():
    op = get_op("set_extra_state")
    assert op is not None


def test_set_graph_mode_4501():
    op = get_op("set_graph_mode")
    assert op is not None


def test_set_metadata_4502():
    op = get_op("set_metadata")
    assert op is not None


def test_set_printoptions_4503():
    op = get_op("set_printoptions")
    assert op is not None


def test_set_submodule_4504():
    op = get_op("set_submodule")
    assert op is not None


def test_set_up_flags_4505():
    op = get_op("set_up_flags")
    assert op is not None


def test_set_weight_4506():
    op = get_op("set_weight")
    assert op is not None


def test_setbufsize_4507():
    op = get_op("setbufsize")
    assert op is not None


def test_setdefault_4508():
    op = get_op("setdefault")
    assert op is not None


def test_setdiff1d_4509():
    op = get_op("setdiff1d")
    assert op is not None


def test_seterr_4510():
    op = get_op("seterr")
    assert op is not None


def test_seterrcall_4511():
    op = get_op("seterrcall")
    assert op is not None


def test_setup_tpu_4512():
    op = get_op("setup_tpu")
    assert op is not None


def test_setxor1d_4513():
    op = get_op("setxor1d")
    assert op is not None


def test_sgdr_schedule_4514():
    op = get_op("sgdr_schedule")
    assert op is not None


def test_shape_4515():
    op = get_op("shape")
    assert op is not None


def test_shape_as_value_4516():
    op = get_op("shape_as_value")
    assert op is not None


def test_shape_dependent_4517():
    op = get_op("shape_dependent")
    assert op is not None


def test_shape_from_registers_shape_4518():
    op = get_op("shape_from_registers_shape")
    assert op is not None


def test_shard_args_4519():
    op = get_op("shard_args")
    assert op is not None


def test_shard_inplace_4520():
    op = get_op("shard_inplace")
    assert op is not None


def test_shard_linear_4521():
    op = get_op("shard_linear")
    assert op is not None


def test_shard_map_4522():
    op = get_op("shard_map")
    assert op is not None


def test_share_memory_4523():
    op = get_op("share_memory")
    assert op is not None


def test_share_memory__4524():
    op = get_op("share_memory_")
    assert op is not None


def test_shares_memory_4525():
    op = get_op("shares_memory")
    assert op is not None


def test_shfl_bfly_4526():
    op = get_op("shfl_bfly")
    assert op is not None


def test_shift_left_4527():
    op = get_op("shift_left")
    assert op is not None


def test_shift_right_arithmetic_4528():
    op = get_op("shift_right_arithmetic")
    assert op is not None


def test_shift_right_logical_4529():
    op = get_op("shift_right_logical")
    assert op is not None


def test_shifted_chebyshev_polynomial_t_4530():
    op = get_op("shifted_chebyshev_polynomial_t")
    assert op is not None


def test_shifted_chebyshev_polynomial_u_4531():
    op = get_op("shifted_chebyshev_polynomial_u")
    assert op is not None


def test_shifted_chebyshev_polynomial_v_4532():
    op = get_op("shifted_chebyshev_polynomial_v")
    assert op is not None


def test_shifted_chebyshev_polynomial_w_4533():
    op = get_op("shifted_chebyshev_polynomial_w")
    assert op is not None


def test_short_4534():
    op = get_op("short")
    assert op is not None


def test_shortlist_size_4535():
    op = get_op("shortlist_size")
    assert op is not None


def test_should_have_in_layout_4536():
    op = get_op("should_have_in_layout")
    assert op is not None


def test_should_have_in_tmem_layout_4537():
    op = get_op("should_have_in_tmem_layout")
    assert op is not None


def test_should_have_in_transforms_4538():
    op = get_op("should_have_in_transforms")
    assert op is not None


def test_should_have_layout_4539():
    op = get_op("should_have_layout")
    assert op is not None


def test_should_have_out_layout_4540():
    op = get_op("should_have_out_layout")
    assert op is not None


def test_should_have_out_tmem_layout_4541():
    op = get_op("should_have_out_tmem_layout")
    assert op is not None


def test_should_have_out_transforms_4542():
    op = get_op("should_have_out_transforms")
    assert op is not None


def test_should_have_tmem_layout_4543():
    op = get_op("should_have_tmem_layout")
    assert op is not None


def test_should_have_transforms_4544():
    op = get_op("should_have_transforms")
    assert op is not None


def test_shuffle_4545():
    op = get_op("shuffle")
    assert op is not None


def test_sigmoid_binary_cross_entropy_4546():
    op = get_op("sigmoid_binary_cross_entropy")
    assert op is not None


def test_sigmoid_focal_loss_4547():
    op = get_op("sigmoid_focal_loss")
    assert op is not None


def test_sign_4548():
    op = get_op("sign")
    assert op is not None


def test_signal_4549():
    op = get_op("signal")
    assert op is not None


def test_signal_multimem_4550():
    op = get_op("signal_multimem")
    assert op is not None


def test_signbit_4551():
    op = get_op("signbit")
    assert op is not None


def test_signedinteger_4552():
    op = get_op("signedinteger")
    assert op is not None


def test_simplify_key_4553():
    op = get_op("simplify_key")
    assert op is not None


def test_sin_4554():
    op = get_op("sin")
    assert op is not None


def test_sinc_4555():
    op = get_op("sinc")
    assert op is not None


def test_single_4556():
    op = get_op("single")
    assert op is not None


def test_single_thread_4557():
    op = get_op("single_thread")
    assert op is not None


def test_single_thread_per_block_predicate_4558():
    op = get_op("single_thread_per_block_predicate")
    assert op is not None


def test_single_thread_per_warpgroup_predicate_4559():
    op = get_op("single_thread_per_warpgroup_predicate")
    assert op is not None


def test_single_thread_predicate_4560():
    op = get_op("single_thread_predicate")
    assert op is not None


def test_single_warp_per_block_predicate_4561():
    op = get_op("single_warp_per_block_predicate")
    assert op is not None


def test_sinh_4562():
    op = get_op("sinh")
    assert op is not None


def test_size_4563():
    op = get_op("size")
    assert op is not None


def test_skip_all_reduce_unused_params_4564():
    op = get_op("skip_all_reduce_unused_params")
    assert op is not None


def test_skip_brackets_4565():
    op = get_op("skip_brackets")
    assert op is not None


def test_skip_init_4566():
    op = get_op("skip_init")
    assert op is not None


def test_skip_large_updates_4567():
    op = get_op("skip_large_updates")
    assert op is not None


def test_skip_not_finite_4568():
    op = get_op("skip_not_finite")
    assert op is not None


def test_slice_4569():
    op = get_op("slice")
    assert op is not None


def test_slice_in_dim_4570():
    op = get_op("slice_in_dim")
    assert op is not None


def test_slice_update_4571():
    op = get_op("slice_update")
    assert op is not None


def test_sm3_4572():
    op = get_op("sm3")
    assert op is not None


def test_smem_4573():
    op = get_op("smem")
    assert op is not None


def test_smem_bytes_4574():
    op = get_op("smem_bytes")
    assert op is not None


def test_smem_i32_elements_4575():
    op = get_op("smem_i32_elements")
    assert op is not None


def test_smem_requested_bytes_4576():
    op = get_op("smem_requested_bytes")
    assert op is not None


def test_smid_4577():
    op = get_op("smid")
    assert op is not None


def test_smooth_l1_loss_4578():
    op = get_op("smooth_l1_loss")
    assert op is not None


def test_smooth_labels_4579():
    op = get_op("smooth_labels")
    assert op is not None


def test_snapshot_4580():
    op = get_op("snapshot")
    assert op is not None


def test_soft_margin_loss_4581():
    op = get_op("soft_margin_loss")
    assert op is not None


def test_soft_shrink_4582():
    op = get_op("soft_shrink")
    assert op is not None


def test_softmax_cross_entropy_4583():
    op = get_op("softmax_cross_entropy")
    assert op is not None


def test_softmax_cross_entropy_with_integer_labels_4584():
    op = get_op("softmax_cross_entropy_with_integer_labels")
    assert op is not None


def test_softmin_4585():
    op = get_op("softmin")
    assert op is not None


def test_softshrink_4586():
    op = get_op("softshrink")
    assert op is not None


def test_softsign_4587():
    op = get_op("softsign")
    assert op is not None


def test_solve_ex_4588():
    op = get_op("solve_ex")
    assert op is not None


def test_solve_triangular_4589():
    op = get_op("solve_triangular")
    assert op is not None


def test_sort_4590():
    op = get_op("sort")
    assert op is not None


def test_sort_complex_4591():
    op = get_op("sort_complex")
    assert op is not None


def test_sort_indices_4592():
    op = get_op("sort_indices")
    assert op is not None


def test_sort_key_val_4593():
    op = get_op("sort_key_val")
    assert op is not None


def test_sort_variable_types_4594():
    op = get_op("sort_variable_types")
    assert op is not None


def test_sorted_indices_4595():
    op = get_op("sorted_indices")
    assert op is not None


def test_source_4596():
    op = get_op("source")
    assert op is not None


def test_source_bounds_4597():
    op = get_op("source_bounds")
    assert op is not None


def test_source_map_4598():
    op = get_op("source_map")
    assert op is not None


def test_source_shape_4599():
    op = get_op("source_shape")
    assert op is not None


def test_spacing_4600():
    op = get_op("spacing")
    assert op is not None


def test_sparse_4601():
    op = get_op("sparse")
    assert op is not None


def test_sparse__4602():
    op = get_op("sparse_")
    assert op is not None


def test_sparse_categorical_crossentropy_4603():
    op = get_op("sparse_categorical_crossentropy")
    assert op is not None


def test_sparse_meta_layout_4604():
    op = get_op("sparse_meta_layout")
    assert op is not None


def test_sparse_plus_4605():
    op = get_op("sparse_plus")
    assert op is not None


def test_sparse_rules_bcoo_4606():
    op = get_op("sparse_rules_bcoo")
    assert op is not None


def test_sparse_rules_bcsr_4607():
    op = get_op("sparse_rules_bcsr")
    assert op is not None


def test_sparse_sigmoid_4608():
    op = get_op("sparse_sigmoid")
    assert op is not None


def test_sparsecategoricalcrossentropy_4609():
    op = get_op("sparsecategoricalcrossentropy")
    assert op is not None


def test_sparsify_4610():
    op = get_op("sparsify")
    assert op is not None


def test_sparsify_fun_4611():
    op = get_op("sparsify_fun")
    assert op is not None


def test_sparsify_raw_4612():
    op = get_op("sparsify_raw")
    assert op is not None


def test_sparsify_subtrace_4613():
    op = get_op("sparsify_subtrace")
    assert op is not None


def test_sparsity_4614():
    op = get_op("sparsity")
    assert op is not None


def test_spec_4615():
    op = get_op("spec")
    assert op is not None


def test_spec_to_indices_4616():
    op = get_op("spec_to_indices")
    assert op is not None


def test_spectral_norm_4617():
    op = get_op("spectral_norm")
    assert op is not None


def test_spenv_4618():
    op = get_op("spenv")
    assert op is not None


def test_sph_harm_4619():
    op = get_op("sph_harm")
    assert op is not None


def test_spherical_bessel_j0_4620():
    op = get_op("spherical_bessel_j0")
    assert op is not None


def test_splat_4621():
    op = get_op("splat")
    assert op is not None


def test_splat_is_compatible_with_tiled_4622():
    op = get_op("splat_is_compatible_with_tiled")
    assert op is not None


def test_split_4623():
    op = get_op("split")
    assert op is not None


def test_split_arguments_4624():
    op = get_op("split_arguments")
    assert op is not None


def test_split_context_4625():
    op = get_op("split_context")
    assert op is not None


def test_split_inputs_4626():
    op = get_op("split_inputs")
    assert op is not None


def test_split_key_like_4627():
    op = get_op("split_key_like")
    assert op is not None


def test_split_rngs_4628():
    op = get_op("split_rngs")
    assert op is not None


def test_split_state_4629():
    op = get_op("split_state")
    assert op is not None


def test_split_to_logical_devices_4630():
    op = get_op("split_to_logical_devices")
    assert op is not None


def test_spsolve_4631():
    op = get_op("spsolve")
    assert op is not None


def test_spsolve_p_4632():
    op = get_op("spsolve_p")
    assert op is not None


def test_spvalues_to_arrays_4633():
    op = get_op("spvalues_to_arrays")
    assert op is not None


def test_spvalues_to_avals_4634():
    op = get_op("spvalues_to_avals")
    assert op is not None


def test_square_4635():
    op = get_op("square")
    assert op is not None


def test_squared_error_4636():
    op = get_op("squared_error")
    assert op is not None


def test_squaredhingeloss_4637():
    op = get_op("squaredhingeloss")
    assert op is not None


def test_squareplus_4638():
    op = get_op("squareplus")
    assert op is not None


def test_src_device_obj_4639():
    op = get_op("src_device_obj")
    assert op is not None


def test_stable_hlo_generate_dump_4640():
    op = get_op("stable_hlo_generate_dump")
    assert op is not None


def test_stack_4641():
    op = get_op("stack")
    assert op is not None


def test_stagger_4642():
    op = get_op("stagger")
    assert op is not None


def test_standard_abstract_eval_4643():
    op = get_op("standard_abstract_eval")
    assert op is not None


def test_standard_multi_result_abstract_eval_4644():
    op = get_op("standard_multi_result_abstract_eval")
    assert op is not None


def test_standard_primitive_4645():
    op = get_op("standard_primitive")
    assert op is not None


def test_start_4646():
    op = get_op("start")
    assert op is not None


def test_start_dim_4647():
    op = get_op("start_dim")
    assert op is not None


def test_start_transfer_server_4648():
    op = get_op("start_transfer_server")
    assert op is not None


def test_start_within_block_4649():
    op = get_op("start_within_block")
    assert op is not None


def test_startswith_4650():
    op = get_op("startswith")
    assert op is not None


def test_state_4651():
    op = get_op("state")
    assert op is not None


def test_state_dict_4652():
    op = get_op("state_dict")
    assert op is not None


def test_static_4653():
    op = get_op("static")
    assert op is not None


def test_static_cache_4654():
    op = get_op("static_cache")
    assert op is not None


def test_static_graph_4655():
    op = get_op("static_graph")
    assert op is not None


def test_static_validate_inputs_4656():
    op = get_op("static_validate_inputs")
    assert op is not None


def test_stats_command_4657():
    op = get_op("stats_command")
    assert op is not None


def test_std_4658():
    op = get_op("std")
    assert op is not None


def test_step_4659():
    op = get_op("step")
    assert op is not None


def test_stop_gradient_4660():
    op = get_op("stop_gradient")
    assert op is not None


def test_store_4661():
    op = get_op("store")
    assert op is not None


def test_store_tiled_4662():
    op = get_op("store_tiled")
    assert op is not None


def test_store_tiled_async_4663():
    op = get_op("store_tiled_async")
    assert op is not None


def test_store_untiled_4664():
    op = get_op("store_untiled")
    assert op is not None


def test_str_format_4665():
    op = get_op("str_format")
    assert op is not None


def test_stride_4666():
    op = get_op("stride")
    assert op is not None


def test_strides_4667():
    op = get_op("strides")
    assert op is not None


def test_strip_4668():
    op = get_op("strip")
    assert op is not None


def test_sub_4669():
    op = get_op("sub")
    assert op is not None


def test_subf_4670():
    op = get_op("subf")
    assert op is not None


def test_subjaxprs_4671():
    op = get_op("subjaxprs")
    assert op is not None


def test_subtree_4672():
    op = get_op("subtree")
    assert op is not None


def test_sum_duplicates_4673():
    op = get_op("sum_duplicates")
    assert op is not None


def test_sum_gradients_4674():
    op = get_op("sum_gradients")
    assert op is not None


def test_supported_tmem_transfers_4675():
    op = get_op("supported_tmem_transfers")
    assert op is not None


def test_supports_bfloat16_matmul_4676():
    op = get_op("supports_bfloat16_matmul")
    assert op is not None


def test_supports_color_4677():
    op = get_op("supports_color")
    assert op is not None


def test_supports_cross_device_collectives_4678():
    op = get_op("supports_cross_device_collectives")
    assert op is not None


def test_svdvals_4679():
    op = get_op("svdvals")
    assert op is not None


def test_swap_4680():
    op = get_op("swap")
    assert op is not None


def test_swap_lstm_gates_4681():
    op = get_op("swap_lstm_gates")
    assert op is not None


def test_swapaxes_4682():
    op = get_op("swapaxes")
    assert op is not None


def test_swapcase_4683():
    op = get_op("swapcase")
    assert op is not None


def test_swizzle_4684():
    op = get_op("swizzle")
    assert op is not None


def test_swizzle_and_transforms_from_transforms_attr_4685():
    op = get_op("swizzle_and_transforms_from_transforms_attr")
    assert op is not None


def test_symmetric_product_4686():
    op = get_op("symmetric_product")
    assert op is not None


def test_symmetrize_4687():
    op = get_op("symmetrize")
    assert op is not None


def test_sync_global_devices_4688():
    op = get_op("sync_global_devices")
    assert op is not None


def test_tabulate_4689():
    op = get_op("tabulate")
    assert op is not None


def test_tail_4690():
    op = get_op("tail")
    assert op is not None


def test_take_4691():
    op = get_op("take")
    assert op is not None


def test_take_along_axis_4692():
    op = get_op("take_along_axis")
    assert op is not None


def test_take_current_trace_4693():
    op = get_op("take_current_trace")
    assert op is not None


def test_tan_4694():
    op = get_op("tan")
    assert op is not None


def test_tanh_shrink_4695():
    op = get_op("tanh_shrink")
    assert op is not None


def test_tanhshrink_4696():
    op = get_op("tanhshrink")
    assert op is not None


def test_target_4697():
    op = get_op("target")
    assert op is not None


def test_target_block_shape_4698():
    op = get_op("target_block_shape")
    assert op is not None


def test_target_bounds_4699():
    op = get_op("target_bounds")
    assert op is not None


def test_target_shape_4700():
    op = get_op("target_shape")
    assert op is not None


def test_tensor_4701():
    op = get_op("tensor")
    assert op is not None


def test_tensorinv_4702():
    op = get_op("tensorinv")
    assert op is not None


def test_tensorsolve_4703():
    op = get_op("tensorsolve")
    assert op is not None


def test_terms_4704():
    op = get_op("terms")
    assert op is not None


def test_test_import_4705():
    op = get_op("test_import")
    assert op is not None


def test_tf_wrap_with_input_names_4706():
    op = get_op("tf_wrap_with_input_names")
    assert op is not None


def test_tgmm_4707():
    op = get_op("tgmm")
    assert op is not None


def test_thread_idx_4708():
    op = get_op("thread_idx")
    assert op is not None


def test_thread_idxs_4709():
    op = get_op("thread_idxs")
    assert op is not None


def test_thread_resources_4710():
    op = get_op("thread_resources")
    assert op is not None


def test_threefry_2x32_count_4711():
    op = get_op("threefry_2x32_count")
    assert op is not None


def test_threshold_4712():
    op = get_op("threshold")
    assert op is not None


def test_threshold__4713():
    op = get_op("threshold_")
    assert op is not None


def test_thunk_re_4714():
    op = get_op("thunk_re")
    assert op is not None


def test_tikzpicture_4715():
    op = get_op("tikzpicture")
    assert op is not None


def test_tile_4716():
    op = get_op("tile")
    assert op is not None


def test_tile_index_transforms_4717():
    op = get_op("tile_index_transforms")
    assert op is not None


def test_tile_k_4718():
    op = get_op("tile_k")
    assert op is not None


def test_tile_m_4719():
    op = get_op("tile_m")
    assert op is not None


def test_tile_n_4720():
    op = get_op("tile_n")
    assert op is not None


def test_tile_offset_4721():
    op = get_op("tile_offset")
    assert op is not None


def test_tile_shape_4722():
    op = get_op("tile_shape")
    assert op is not None


def test_tile_strides_4723():
    op = get_op("tile_strides")
    assert op is not None


def test_tiled_copy_smem_gmem_layout_4724():
    op = get_op("tiled_copy_smem_gmem_layout")
    assert op is not None


def test_tiled_memref_shape_4725():
    op = get_op("tiled_memref_shape")
    assert op is not None


def test_tiled_tiling_rank_4726():
    op = get_op("tiled_tiling_rank")
    assert op is not None


def test_tiled_tiling_shape_4727():
    op = get_op("tiled_tiling_shape")
    assert op is not None


def test_tiling_4728():
    op = get_op("tiling")
    assert op is not None


def test_tiling_multiple_4729():
    op = get_op("tiling_multiple")
    assert op is not None


def test_time_ns_4730():
    op = get_op("time_ns")
    assert op is not None


def test_title_4731():
    op = get_op("title")
    assert op is not None


def test_tma_descriptors_4732():
    op = get_op("tma_descriptors")
    assert op is not None


def test_tmem_4733():
    op = get_op("tmem")
    assert op is not None


def test_tmem_alloc_4734():
    op = get_op("tmem_alloc")
    assert op is not None


def test_tmem_alloc_exact_ncols_4735():
    op = get_op("tmem_alloc_exact_ncols")
    assert op is not None


def test_tmem_dealloc_4736():
    op = get_op("tmem_dealloc")
    assert op is not None


def test_tmem_default_layout_4737():
    op = get_op("tmem_default_layout")
    assert op is not None


def test_tmem_half_lane_layout_4738():
    op = get_op("tmem_half_lane_layout")
    assert op is not None


def test_tmem_m64_collective_layout_4739():
    op = get_op("tmem_m64_collective_layout")
    assert op is not None


def test_tmem_native_layout_4740():
    op = get_op("tmem_native_layout")
    assert op is not None


def test_tmem_relinquish_alloc_permit_4741():
    op = get_op("tmem_relinquish_alloc_permit")
    assert op is not None


def test_to_4742():
    op = get_op("to")
    assert op is not None


def test_to_attr_4743():
    op = get_op("to_attr")
    assert op is not None


def test_to_bcoo_4744():
    op = get_op("to_bcoo")
    assert op is not None


def test_to_dense_4745():
    op = get_op("to_dense")
    assert op is not None


def test_to_device_4746():
    op = get_op("to_device")
    assert op is not None


def test_to_empty_4747():
    op = get_op("to_empty")
    assert op is not None


def test_to_flat_state_4748():
    op = get_op("to_flat_state")
    assert op is not None


def test_to_int8_4749():
    op = get_op("to_int8")
    assert op is not None


def test_to_kwargs_4750():
    op = get_op("to_kwargs")
    assert op is not None


def test_to_layout_4751():
    op = get_op("to_layout")
    assert op is not None


def test_to_layout_attr_4752():
    op = get_op("to_layout_attr")
    assert op is not None


def test_to_linen_4753():
    op = get_op("to_linen")
    assert op is not None


def test_to_linen_class_4754():
    op = get_op("to_linen_class")
    assert op is not None


def test_to_linen_var_4755():
    op = get_op("to_linen_var")
    assert op is not None


def test_to_nnx_var_4756():
    op = get_op("to_nnx_var")
    assert op is not None


def test_to_opt_state_4757():
    op = get_op("to_opt_state")
    assert op is not None


def test_to_predicate_4758():
    op = get_op("to_predicate")
    assert op is not None


def test_to_primal_terms_pair_4759():
    op = get_op("to_primal_terms_pair")
    assert op is not None


def test_to_pure_dict_4760():
    op = get_op("to_pure_dict")
    assert op is not None


def test_to_remote_4761():
    op = get_op("to_remote")
    assert op is not None


def test_to_remote_multicast_4762():
    op = get_op("to_remote_multicast")
    assert op is not None


def test_to_sparse_tracer_4763():
    op = get_op("to_sparse_tracer")
    assert op is not None


def test_to_splat_fragmented_layout_attr_4764():
    op = get_op("to_splat_fragmented_layout_attr")
    assert op is not None


def test_to_strided_fragmented_layout_attr_4765():
    op = get_op("to_strided_fragmented_layout_attr")
    assert op is not None


def test_to_string_4766():
    op = get_op("to_string")
    assert op is not None


def test_to_tiled_layout_attr_4767():
    op = get_op("to_tiled_layout_attr")
    assert op is not None


def test_to_transform_attr_4768():
    op = get_op("to_transform_attr")
    assert op is not None


def test_to_tree_4769():
    op = get_op("to_tree")
    assert op is not None


def test_todense_4770():
    op = get_op("todense")
    assert op is not None


def test_todense_p_4771():
    op = get_op("todense_p")
    assert op is not None


def test_tolist_4772():
    op = get_op("tolist")
    assert op is not None


def test_top_k_4773():
    op = get_op("top_k")
    assert op is not None


def test_total_bytes_4774():
    op = get_op("total_bytes")
    assert op is not None


def test_tpu_generation_4775():
    op = get_op("tpu_generation")
    assert op is not None


def test_tpu_kind_4776():
    op = get_op("tpu_kind")
    assert op is not None


def test_trace_and_lower_4777():
    op = get_op("trace_and_lower")
    assert op is not None


def test_traceable_4778():
    op = get_op("traceable")
    assert op is not None


def test_train_4779():
    op = get_op("train")
    assert op is not None


def test_training_4780():
    op = get_op("training")
    assert op is not None


def test_transfer_parametrizations_and_params_4781():
    op = get_op("transfer_parametrizations_and_params")
    assert op is not None


def test_transfer_strided_4782():
    op = get_op("transfer_strided")
    assert op is not None


def test_transfer_tiled_4783():
    op = get_op("transfer_tiled")
    assert op is not None


def test_transform_index_4784():
    op = get_op("transform_index")
    assert op is not None


def test_transform_shape_4785():
    op = get_op("transform_shape")
    assert op is not None


def test_transform_strides_4786():
    op = get_op("transform_strides")
    assert op is not None


def test_transform_type_4787():
    op = get_op("transform_type")
    assert op is not None


def test_translate_4788():
    op = get_op("translate")
    assert op is not None


def test_transposed_ragged_dot_4789():
    op = get_op("transposed_ragged_dot")
    assert op is not None


def test_trapezoid_4790():
    op = get_op("trapezoid")
    assert op is not None


def test_traverse_jaxpr_params_4791():
    op = get_op("traverse_jaxpr_params")
    assert op is not None


def test_traverse_op_4792():
    op = get_op("traverse_op")
    assert op is not None


def test_tree_add_scalar_mul_4793():
    op = get_op("tree_add_scalar_mul")
    assert op is not None


def test_tree_flatten_4794():
    op = get_op("tree_flatten")
    assert op is not None


def test_tree_l1_norm_4795():
    op = get_op("tree_l1_norm")
    assert op is not None


def test_tree_l2_norm_4796():
    op = get_op("tree_l2_norm")
    assert op is not None


def test_tree_linf_norm_4797():
    op = get_op("tree_linf_norm")
    assert op is not None


def test_tree_map_4798():
    op = get_op("tree_map")
    assert op is not None


def test_tree_map_params_4799():
    op = get_op("tree_map_params")
    assert op is not None


def test_tree_scalar_mul_4800():
    op = get_op("tree_scalar_mul")
    assert op is not None


def test_tree_unflatten_4801():
    op = get_op("tree_unflatten")
    assert op is not None


def test_treemap_copy_args_4802():
    op = get_op("treemap_copy_args")
    assert op is not None


def test_tri_4803():
    op = get_op("tri")
    assert op is not None


def test_triangular_4804():
    op = get_op("triangular")
    assert op is not None


def test_triangular_solve_4805():
    op = get_op("triangular_solve")
    assert op is not None


def test_tridiagonal_4806():
    op = get_op("tridiagonal")
    assert op is not None


def test_tridiagonal_solve_4807():
    op = get_op("tridiagonal_solve")
    assert op is not None


def test_tril_4808():
    op = get_op("tril")
    assert op is not None


def test_tril_indices_4809():
    op = get_op("tril_indices")
    assert op is not None


def test_tril_indices_from_4810():
    op = get_op("tril_indices_from")
    assert op is not None


def test_trim_zeros_4811():
    op = get_op("trim_zeros")
    assert op is not None


def test_trim_zeros_tol_4812():
    op = get_op("trim_zeros_tol")
    assert op is not None


def test_triplet_margin_loss_4813():
    op = get_op("triplet_margin_loss")
    assert op is not None


def test_triplet_margin_with_distance_loss_4814():
    op = get_op("triplet_margin_with_distance_loss")
    assert op is not None


def test_triu_4815():
    op = get_op("triu")
    assert op is not None


def test_triu_indices_4816():
    op = get_op("triu_indices")
    assert op is not None


def test_triu_indices_from_4817():
    op = get_op("triu_indices_from")
    assert op is not None


def test_true_divide_4818():
    op = get_op("true_divide")
    assert op is not None


def test_trunc_4819():
    op = get_op("trunc")
    assert op is not None


def test_trunc_normal__4820():
    op = get_op("trunc_normal_")
    assert op is not None


def test_truncated_normal_4821():
    op = get_op("truncated_normal")
    assert op is not None


def test_try_cluster_cancel_4822():
    op = get_op("try_cluster_cancel")
    assert op is not None


def test_tuple_delete_4823():
    op = get_op("tuple_delete")
    assert op is not None


def test_tverskyloss_4824():
    op = get_op("tverskyloss")
    assert op is not None


def test_type_4825():
    op = get_op("type")
    assert op is not None


def test_type_before_parametrizations_4826():
    op = get_op("type_before_parametrizations")
    assert op is not None


def test_ufunc_4827():
    op = get_op("ufunc")
    assert op is not None


def test_uint_4828():
    op = get_op("uint")
    assert op is not None


def test_uint16_4829():
    op = get_op("uint16")
    assert op is not None


def test_uint2_4830():
    op = get_op("uint2")
    assert op is not None


def test_uint32_4831():
    op = get_op("uint32")
    assert op is not None


def test_uint4_4832():
    op = get_op("uint4")
    assert op is not None


def test_uint64_4833():
    op = get_op("uint64")
    assert op is not None


def test_unary_ufunc_4834():
    op = get_op("unary_ufunc")
    assert op is not None


def test_unflatten_4835():
    op = get_op("unflatten")
    assert op is not None


def test_unflatten_mapping_4836():
    op = get_op("unflatten_mapping")
    assert op is not None


def test_unflattened_size_4837():
    op = get_op("unflattened_size")
    assert op is not None


def test_unfold_4838():
    op = get_op("unfold")
    assert op is not None


def test_unfused_flops_4839():
    op = get_op("unfused_flops")
    assert op is not None


def test_unfused_hbm_bytes_4840():
    op = get_op("unfused_hbm_bytes")
    assert op is not None


def test_uniform__4841():
    op = get_op("uniform_")
    assert op is not None


def test_union1d_4842():
    op = get_op("union1d")
    assert op is not None


def test_unique_4843():
    op = get_op("unique")
    assert op is not None


def test_unique_all_4844():
    op = get_op("unique_all")
    assert op is not None


def test_unique_counts_4845():
    op = get_op("unique_counts")
    assert op is not None


def test_unique_indices_4846():
    op = get_op("unique_indices")
    assert op is not None


def test_unique_inverse_4847():
    op = get_op("unique_inverse")
    assert op is not None


def test_unique_values_4848():
    op = get_op("unique_values")
    assert op is not None


def test_unknowns_4849():
    op = get_op("unknowns")
    assert op is not None


def test_unmapped_aval_4850():
    op = get_op("unmapped_aval")
    assert op is not None


def test_unop_4851():
    op = get_op("unop")
    assert op is not None


def test_unop_dtype_rule_4852():
    op = get_op("unop_dtype_rule")
    assert op is not None


def test_unop_reduced_rule_4853():
    op = get_op("unop_reduced_rule")
    assert op is not None


def test_unpack_lstm_weights_4854():
    op = get_op("unpack_lstm_weights")
    assert op is not None


def test_unpack_optimizer_state_4855():
    op = get_op("unpack_optimizer_state")
    assert op is not None


def test_unpack_sequence_4856():
    op = get_op("unpack_sequence")
    assert op is not None


def test_unpackbits_4857():
    op = get_op("unpackbits")
    assert op is not None


def test_unpad_sequence_4858():
    op = get_op("unpad_sequence")
    assert op is not None


def test_unquantize_from_int8_4859():
    op = get_op("unquantize_from_int8")
    assert op is not None


def test_unravel_index_4860():
    op = get_op("unravel_index")
    assert op is not None


def test_unreduced_psum_4861():
    op = get_op("unreduced_psum")
    assert op is not None


def test_unreduced_psum_scatter_4862():
    op = get_op("unreduced_psum_scatter")
    assert op is not None


def test_unsafe_4863():
    op = get_op("unsafe")
    assert op is not None


def test_unsignedinteger_4864():
    op = get_op("unsignedinteger")
    assert op is not None


def test_unsorted_indices_4865():
    op = get_op("unsorted_indices")
    assert op is not None


def test_unstack_4866():
    op = get_op("unstack")
    assert op is not None


def test_unwrap_4867():
    op = get_op("unwrap")
    assert op is not None


def test_unwrap_random_key_data_4868():
    op = get_op("unwrap_random_key_data")
    assert op is not None


def test_unwrap_transformed_memref_4869():
    op = get_op("unwrap_transformed_memref")
    assert op is not None


def test_update_4870():
    op = get_op("update")
    assert op is not None


def test_update_carry_variables_4871():
    op = get_op("update_carry_variables")
    assert op is not None


def test_update_context_4872():
    op = get_op("update_context")
    assert op is not None


def test_update_fn_4873():
    op = get_op("update_fn")
    assert op is not None


def test_update_infinity_moment_4874():
    op = get_op("update_infinity_moment")
    assert op is not None


def test_update_layout_4875():
    op = get_op("update_layout")
    assert op is not None


def test_update_moment_4876():
    op = get_op("update_moment")
    assert op is not None


def test_update_moment_per_elem_norm_4877():
    op = get_op("update_moment_per_elem_norm")
    assert op is not None


def test_update_parities_4878():
    op = get_op("update_parities")
    assert op is not None


def test_updates_and_snapshot_4879():
    op = get_op("updates_and_snapshot")
    assert op is not None


def test_upper_4880():
    op = get_op("upper")
    assert op is not None


def test_upsample_4881():
    op = get_op("upsample")
    assert op is not None


def test_upsample_bilinear_4882():
    op = get_op("upsample_bilinear")
    assert op is not None


def test_upsample_cubic_4883():
    op = get_op("upsample_cubic")
    assert op is not None


def test_upsample_linear_4884():
    op = get_op("upsample_linear")
    assert op is not None


def test_upsample_nearest_4885():
    op = get_op("upsample_nearest")
    assert op is not None


def test_upscale_factor_4886():
    op = get_op("upscale_factor")
    assert op is not None


def test_use_eager_sharding_4887():
    op = get_op("use_eager_sharding")
    assert op is not None


def test_use_fused_bwd_kernel_4888():
    op = get_op("use_fused_bwd_kernel")
    assert op is not None


def test_use_nested_tensor_4889():
    op = get_op("use_nested_tensor")
    assert op is not None


def test_use_schedule_barrier_4890():
    op = get_op("use_schedule_barrier")
    assert op is not None


def test_use_side_stream_for_tensor_copies_4891():
    op = get_op("use_side_stream_for_tensor_copies")
    assert op is not None


def test_uses_collective_metadata_4892():
    op = get_op("uses_collective_metadata")
    assert op is not None


def test_using_eager_sharding_4893():
    op = get_op("using_eager_sharding")
    assert op is not None


def test_v_layout_4894():
    op = get_op("v_layout")
    assert op is not None


def test_v_proj_weight_4895():
    op = get_op("v_proj_weight")
    assert op is not None


def test_value_4896():
    op = get_op("value")
    assert op is not None


def test_value_and_grad_4897():
    op = get_op("value_and_grad")
    assert op is not None


def test_value_sites_for_variable_4898():
    op = get_op("value_sites_for_variable")
    assert op is not None


def test_values_4899():
    op = get_op("values")
    assert op is not None


def test_vander_4900():
    op = get_op("vander")
    assert op is not None


def test_var_4901():
    op = get_op("var")
    assert op is not None


def test_var_defaults_4902():
    op = get_op("var_defaults")
    assert op is not None


def test_variable_for_value_site_4903():
    op = get_op("variable_for_value_site")
    assert op is not None


def test_variable_name_from_type_4904():
    op = get_op("variable_name_from_type")
    assert op is not None


def test_variable_type_from_name_4905():
    op = get_op("variable_type_from_name")
    assert op is not None


def test_variables_4906():
    op = get_op("variables")
    assert op is not None


def test_variant_4907():
    op = get_op("variant")
    assert op is not None


def test_varlen_attn_4908():
    op = get_op("varlen_attn")
    assert op is not None


def test_vars_as_4909():
    op = get_op("vars_as")
    assert op is not None


def test_vary_unreduced_cast_4910():
    op = get_op("vary_unreduced_cast")
    assert op is not None


def test_vdim_4911():
    op = get_op("vdim")
    assert op is not None


def test_vec_size_4912():
    op = get_op("vec_size")
    assert op is not None


def test_vecdot_4913():
    op = get_op("vecdot")
    assert op is not None


def test_vecmat_4914():
    op = get_op("vecmat")
    assert op is not None


def test_vector_concat_4915():
    op = get_op("vector_concat")
    assert op is not None


def test_vector_dim_4916():
    op = get_op("vector_dim")
    assert op is not None


def test_vector_length_4917():
    op = get_op("vector_length")
    assert op is not None


def test_vector_norm_4918():
    op = get_op("vector_norm")
    assert op is not None


def test_vector_slice_4919():
    op = get_op("vector_slice")
    assert op is not None


def test_vector_to_parameters_4920():
    op = get_op("vector_to_parameters")
    assert op is not None


def test_vector_value_sites_4921():
    op = get_op("vector_value_sites")
    assert op is not None


def test_vectorize_4922():
    op = get_op("vectorize")
    assert op is not None


def test_vectorized_map_4923():
    op = get_op("vectorized_map")
    assert op is not None


def test_verify_matching_signatures_4924():
    op = get_op("verify_matching_signatures")
    assert op is not None


def test_verify_tensorstore_spec_4925():
    op = get_op("verify_tensorstore_spec")
    assert op is not None


def test_view_4926():
    op = get_op("view")
    assert op is not None


def test_view_as_complex_4927():
    op = get_op("view_as_complex")
    assert op is not None


def test_view_as_real_4928():
    op = get_op("view_as_real")
    assert op is not None


def test_view_info_4929():
    op = get_op("view_info")
    assert op is not None


def test_vjp_4930():
    op = get_op("vjp")
    assert op is not None


def test_vsplit_4931():
    op = get_op("vsplit")
    assert op is not None


def test_vstack_4932():
    op = get_op("vstack")
    assert op is not None


def test_wait_and_get_loaded_4933():
    op = get_op("wait_and_get_loaded")
    assert op is not None


def test_wait_load_tmem_4934():
    op = get_op("wait_load_tmem")
    assert op is not None


def test_wait_parity_4935():
    op = get_op("wait_parity")
    assert op is not None


def test_wait_until_finished_4936():
    op = get_op("wait_until_finished")
    assert op is not None


def test_wald_4937():
    op = get_op("wald")
    assert op is not None


def test_warmup_constant_schedule_4938():
    op = get_op("warmup_constant_schedule")
    assert op is not None


def test_warmup_cosine_decay_schedule_4939():
    op = get_op("warmup_cosine_decay_schedule")
    assert op is not None


def test_warmup_exponential_decay_schedule_4940():
    op = get_op("warmup_exponential_decay_schedule")
    assert op is not None


def test_warp_barrier_4941():
    op = get_op("warp_barrier")
    assert op is not None


def test_warp_dims_4942():
    op = get_op("warp_dims")
    assert op is not None


def test_warp_idx_4943():
    op = get_op("warp_idx")
    assert op is not None


def test_warp_indices_4944():
    op = get_op("warp_indices")
    assert op is not None


def test_warp_tree_reduce_4945():
    op = get_op("warp_tree_reduce")
    assert op is not None


def test_warpgroup_barrier_4946():
    op = get_op("warpgroup_barrier")
    assert op is not None


def test_warpgroup_idx_4947():
    op = get_op("warpgroup_idx")
    assert op is not None


def test_waves_per_eu_4948():
    op = get_op("waves_per_eu")
    assert op is not None


def test_weight_4949():
    op = get_op("weight")
    assert op is not None


def test_weight_hh_4950():
    op = get_op("weight_hh")
    assert op is not None


def test_weight_ih_4951():
    op = get_op("weight_ih")
    assert op is not None


def test_weight_norm_4952():
    op = get_op("weight_norm")
    assert op is not None


def test_wg_dimension_4953():
    op = get_op("wg_dimension")
    assert op is not None


def test_wgmma_4954():
    op = get_op("wgmma")
    assert op is not None


def test_wgmma_fence_4955():
    op = get_op("wgmma_fence")
    assert op is not None


def test_wgmma_m64_4956():
    op = get_op("wgmma_m64")
    assert op is not None


def test_when_4957():
    op = get_op("when")
    assert op is not None


def test_where_4958():
    op = get_op("where")
    assert op is not None


def test_while_loop_4959():
    op = get_op("while_loop")
    assert op is not None


def test_will_sync_module_buffers_4960():
    op = get_op("will_sync_module_buffers")
    assert op is not None


def test_window_size_4961():
    op = get_op("window_size")
    assert op is not None


def test_with_attributes_4962():
    op = get_op("with_attributes")
    assert op is not None


def test_with_metadata_4963():
    op = get_op("with_metadata")
    assert op is not None


def test_with_partitioning_4964():
    op = get_op("with_partitioning")
    assert op is not None


def test_wrap_class_4965():
    op = get_op("wrap_class")
    assert op is not None


def test_wrap_in_custom_primitive_4966():
    op = get_op("wrap_in_custom_primitive")
    assert op is not None


def test_wrap_init_4967():
    op = get_op("wrap_init")
    assert op is not None


def test_wrap_transformed_memref_4968():
    op = get_op("wrap_transformed_memref")
    assert op is not None


def test_write_code_4969():
    op = get_op("write_code")
    assert op is not None


def test_write_file_4970():
    op = get_op("write_file")
    assert op is not None


def test_xavier_normal_4971():
    op = get_op("xavier_normal")
    assert op is not None


def test_xavier_normal__4972():
    op = get_op("xavier_normal_")
    assert op is not None


def test_xavier_uniform_4973():
    op = get_op("xavier_uniform")
    assert op is not None


def test_xavier_uniform__4974():
    op = get_op("xavier_uniform_")
    assert op is not None


def test_xla_metadata_call_4975():
    op = get_op("xla_metadata_call")
    assert op is not None


def test_xla_metadata_call_p_4976():
    op = get_op("xla_metadata_call_p")
    assert op is not None


def test_xla_pmap_p_4977():
    op = get_op("xla_pmap_p")
    assert op is not None


def test_xlog1py_4978():
    op = get_op("xlog1py")
    assert op is not None


def test_xlogy_4979():
    op = get_op("xlogy")
    assert op is not None


def test_xpu_4980():
    op = get_op("xpu")
    assert op is not None


def test_zero_4981():
    op = get_op("zero")
    assert op is not None


def test_zero_grad_4982():
    op = get_op("zero_grad")
    assert op is not None


def test_zero_infinity_4983():
    op = get_op("zero_infinity")
    assert op is not None


def test_zero_nans_4984():
    op = get_op("zero_nans")
    assert op is not None


def test_zero_prop_4985():
    op = get_op("zero_prop")
    assert op is not None


def test_zero_series_4986():
    op = get_op("zero_series")
    assert op is not None


def test_zero_term_4987():
    op = get_op("zero_term")
    assert op is not None


def test_zeros_4988():
    op = get_op("zeros")
    assert op is not None


def test_zeros__4989():
    op = get_op("zeros_")
    assert op is not None


def test_zeros_init_4990():
    op = get_op("zeros_init")
    assert op is not None


def test_zeros_like_4991():
    op = get_op("zeros_like")
    assert op is not None


def test_zeros_like_shaped_array_4992():
    op = get_op("zeros_like_shaped_array")
    assert op is not None


def test_zeta_4993():
    op = get_op("zeta")
    assert op is not None


def test_zfill_4994():
    op = get_op("zfill")
    assert op is not None


def test_zip_4995():
    op = get_op("zip")
    assert op is not None
