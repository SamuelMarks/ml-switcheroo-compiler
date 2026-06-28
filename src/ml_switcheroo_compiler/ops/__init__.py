# ruff: noqa: F822, ANN401, E402
"""Auto-generated ml_switcheroo_compiler.ops module exports. Refactored into registries."""

import typing

from ml_switcheroo_compiler.ops import _math_registry
from ml_switcheroo_compiler.ops import _nn_registry
from ml_switcheroo_compiler.ops import _core_registry
from ml_switcheroo_compiler.ops import _vision_registry
from . import lax_ops

from .distributed import pbroadcast as pbroadcast
from .distributed import pdot as pdot
from .distributed import ppermute as ppermute
from .distributed import pshuffle as pshuffle
from .distributed import infeed as infeed
from .distributed import outfeed as outfeed
from .distributed import axis_index as axis_index
from .distributed import with_sharding_constraint as with_sharding_constraint
from .distributed import Infeed, Outfeed, AxisIndex, WithShardingConstraint


def __getattr__(name: str) -> object:
    import ml_switcheroo_compiler.random as _random_mod

    if hasattr(_math_registry, name):
        return getattr(_math_registry, name)
    if hasattr(_nn_registry, name):
        return getattr(_nn_registry, name)
    if hasattr(_core_registry, name):
        return getattr(_core_registry, name)
    if hasattr(_vision_registry, name):
        return getattr(_vision_registry, name)  # pragma: no cover
    if hasattr(lax_ops, name):
        return getattr(lax_ops, name)  # pragma: no cover
    if hasattr(_random_mod, name):
        return getattr(_random_mod, name)  # pragma: no cover
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def reducer_batcher(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function reducer_batcher."""
    raise NotImplementedError("reducer_batcher not implemented")  # pragma: no cover


def logcdf(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function logcdf."""
    raise NotImplementedError("logcdf not implemented")  # pragma: no cover


def initialize(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function initialize."""
    raise NotImplementedError("initialize not implemented")  # pragma: no cover


def is_constant_dim(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function is_constant_dim."""
    raise NotImplementedError("is_constant_dim not implemented")  # pragma: no cover


def leaked_tracer_error(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function leaked_tracer_error."""
    raise NotImplementedError("leaked_tracer_error not implemented")  # pragma: no cover


def ResAvalUpdater(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function ResAvalUpdater."""
    raise NotImplementedError("ResAvalUpdater not implemented")  # pragma: no cover


def backend_pjrt_c_api_version(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function backend_pjrt_c_api_version."""
    raise NotImplementedError("backend_pjrt_c_api_version not implemented")  # pragma: no cover


def call_padding_rule(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function call_padding_rule."""
    raise NotImplementedError("call_padding_rule not implemented")  # pragma: no cover


def sequential_vmap(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function sequential_vmap."""
    raise NotImplementedError("sequential_vmap not implemented")  # pragma: no cover


def raise_as_much_as_possible(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function raise_as_much_as_possible."""
    raise NotImplementedError("raise_as_much_as_possible not implemented")  # pragma: no cover


def add_jaxvals(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function add_jaxvals."""
    raise NotImplementedError("add_jaxvals not implemented")  # pragma: no cover


def jvp_jaxpr(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function jvp_jaxpr."""
    raise NotImplementedError("jvp_jaxpr not implemented")  # pragma: no cover


def heap_profile(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function heap_profile."""
    raise NotImplementedError("heap_profile not implemented")  # pragma: no cover


def batch_custom_vjp_bwd(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function batch_custom_vjp_bwd."""
    raise NotImplementedError("batch_custom_vjp_bwd not implemented")  # pragma: no cover


def jvp(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function jvp."""
    raise NotImplementedError("jvp not implemented")  # pragma: no cover


def clear_all_weakref_lru_caches(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function clear_all_weakref_lru_caches."""
    raise NotImplementedError("clear_all_weakref_lru_caches not implemented")  # pragma: no cover


def PartialEvalCustomRule(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function PartialEvalCustomRule."""
    raise NotImplementedError("PartialEvalCustomRule not implemented")  # pragma: no cover


def wrap_name(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function wrap_name."""
    raise NotImplementedError("wrap_name not implemented")  # pragma: no cover


def rearrange_binders(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function rearrange_binders."""
    raise NotImplementedError("rearrange_binders not implemented")  # pragma: no cover


def instantiate_const_at(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function instantiate_const_at."""
    raise NotImplementedError("instantiate_const_at not implemented")  # pragma: no cover


def nonzero_tangent_outputs(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function nonzero_tangent_outputs."""
    raise NotImplementedError("nonzero_tangent_outputs not implemented")  # pragma: no cover


def to_elt(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function to_elt."""
    raise NotImplementedError("to_elt not implemented")  # pragma: no cover


def close_jaxpr(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function close_jaxpr."""
    raise NotImplementedError("close_jaxpr not implemented")  # pragma: no cover


def batch_jaxpr(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function batch_jaxpr."""
    raise NotImplementedError("batch_jaxpr not implemented")  # pragma: no cover


def used_axis_names(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function used_axis_names."""
    raise NotImplementedError("used_axis_names not implemented")  # pragma: no cover


def Elt(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function Elt."""
    raise NotImplementedError("Elt not implemented")  # pragma: no cover


def split_dict(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function split_dict."""
    raise NotImplementedError("split_dict not implemented")  # pragma: no cover


def flatten_lowering_ir_args(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function flatten_lowering_ir_args."""
    raise NotImplementedError("flatten_lowering_ir_args not implemented")  # pragma: no cover


def f_jvp_traceable(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function f_jvp_traceable."""
    raise NotImplementedError("f_jvp_traceable not implemented")  # pragma: no cover


def tuple_update(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function tuple_update."""
    raise NotImplementedError("tuple_update not implemented")  # pragma: no cover


def debug_info(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function debug_info."""
    raise NotImplementedError("debug_info not implemented")  # pragma: no cover


def clear_all_caches(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function clear_all_caches."""
    raise NotImplementedError("clear_all_caches not implemented")  # pragma: no cover


def dctn(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function dctn."""
    raise NotImplementedError("dctn not implemented")  # pragma: no cover


def do_subst_axis_names_jaxpr(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function do_subst_axis_names_jaxpr."""
    raise NotImplementedError("do_subst_axis_names_jaxpr not implemented")  # pragma: no cover


def aval_to_ir_type(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function aval_to_ir_type."""
    raise NotImplementedError("aval_to_ir_type not implemented")  # pragma: no cover


def subs_list2(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function subs_list2."""
    raise NotImplementedError("subs_list2 not implemented")  # pragma: no cover


def distributed_debug_log(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function distributed_debug_log."""
    raise NotImplementedError("distributed_debug_log not implemented")  # pragma: no cover


def curry(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function curry."""
    raise NotImplementedError("curry not implemented")  # pragma: no cover


def stable_unique(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function stable_unique."""
    raise NotImplementedError("stable_unique not implemented")  # pragma: no cover


def to_dlpack(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function to_dlpack."""
    raise NotImplementedError("to_dlpack not implemented")  # pragma: no cover


def bicgstab(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function bicgstab."""
    raise NotImplementedError("bicgstab not implemented")  # pragma: no cover


def ir_constants(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function ir_constants."""
    raise NotImplementedError("ir_constants not implemented")  # pragma: no cover


def zero_jvp(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function zero_jvp."""
    raise NotImplementedError("zero_jvp not implemented")  # pragma: no cover


def recast_to_float0(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function recast_to_float0."""
    raise NotImplementedError("recast_to_float0 not implemented")  # pragma: no cover


def canonicalize_dtype(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function canonicalize_dtype."""
    raise NotImplementedError("canonicalize_dtype not implemented")  # pragma: no cover


def instantiate_zeros(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function instantiate_zeros."""
    raise NotImplementedError("instantiate_zeros not implemented")  # pragma: no cover


def unpair_pval(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function unpair_pval."""
    raise NotImplementedError("unpair_pval not implemented")  # pragma: no cover


def move_binders_to_back(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function move_binders_to_back."""
    raise NotImplementedError("move_binders_to_back not implemented")  # pragma: no cover


def are_op_shardings_equal(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function are_op_shardings_equal."""
    raise NotImplementedError("are_op_shardings_equal not implemented")  # pragma: no cover


def overload(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function overload."""
    raise NotImplementedError("overload not implemented")  # pragma: no cover


def map_bind_with_continuation(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function map_bind_with_continuation."""
    raise NotImplementedError("map_bind_with_continuation not implemented")  # pragma: no cover


def lower_jaxpr_to_module(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function lower_jaxpr_to_module."""
    raise NotImplementedError("lower_jaxpr_to_module not implemented")  # pragma: no cover


def register_custom_call_handler(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function register_custom_call_handler."""
    raise NotImplementedError("register_custom_call_handler not implemented")  # pragma: no cover


def defbilinear(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function defbilinear."""
    raise NotImplementedError("defbilinear not implemented")  # pragma: no cover


def xla_computation_to_mlir_module(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function xla_computation_to_mlir_module."""
    raise NotImplementedError("xla_computation_to_mlir_module not implemented")  # pragma: no cover


def concretization_function_error(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function concretization_function_error."""
    raise NotImplementedError("concretization_function_error not implemented")  # pragma: no cover


def factorial(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function factorial."""
    raise NotImplementedError("factorial not implemented")  # pragma: no cover


def unflatten(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function unflatten."""
    raise NotImplementedError("unflatten not implemented")  # pragma: no cover


def exp1(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function exp1."""
    raise NotImplementedError("exp1 not implemented")  # pragma: no cover


def register_event_listener(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function register_event_listener."""
    raise NotImplementedError("register_event_listener not implemented")  # pragma: no cover


def batch_custom_jvp_subtrace(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function batch_custom_jvp_subtrace."""
    raise NotImplementedError("batch_custom_jvp_subtrace not implemented")  # pragma: no cover


def min_dim(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function min_dim."""
    raise NotImplementedError("min_dim not implemented")  # pragma: no cover


def as_hashable_function(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function as_hashable_function."""
    raise NotImplementedError("as_hashable_function not implemented")  # pragma: no cover


def tuple_insert(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function tuple_insert."""
    raise NotImplementedError("tuple_insert not implemented")  # pragma: no cover


def jvpfun(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function jvpfun."""
    raise NotImplementedError("jvpfun not implemented")  # pragma: no cover


def arg_info_all(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function arg_info_all."""
    raise NotImplementedError("arg_info_all not implemented")  # pragma: no cover


def ndtr(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function ndtr."""
    raise NotImplementedError("ndtr not implemented")  # pragma: no cover


def make_convolution_dimension_numbers(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function make_convolution_dimension_numbers."""
    raise NotImplementedError(
        "make_convolution_dimension_numbers not implemented"
    )  # pragma: no cover


def new_sublevel(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function new_sublevel."""
    raise NotImplementedError("new_sublevel not implemented")  # pragma: no cover


def extend_axis_env(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function extend_axis_env."""
    raise NotImplementedError("extend_axis_env not implemented")  # pragma: no cover


def logsf(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function logsf."""
    raise NotImplementedError("logsf not implemented")  # pragma: no cover


def closed_call_partial_eval_custom_rule(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function closed_call_partial_eval_custom_rule."""
    raise NotImplementedError(
        "closed_call_partial_eval_custom_rule not implemented"
    )  # pragma: no cover


def cho_solve(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function cho_solve."""
    raise NotImplementedError("cho_solve not implemented")  # pragma: no cover


def ceil_of_ratio(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function ceil_of_ratio."""
    raise NotImplementedError("ceil_of_ratio not implemented")  # pragma: no cover


def batch_subtrace(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function batch_subtrace."""
    raise NotImplementedError("batch_subtrace not implemented")  # pragma: no cover


def call_transpose(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function call_transpose."""
    raise NotImplementedError("call_transpose not implemented")  # pragma: no cover


def concrete_or_error(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function concrete_or_error."""
    raise NotImplementedError("concrete_or_error not implemented")  # pragma: no cover


def canonicalize_platform(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function canonicalize_platform."""
    raise NotImplementedError("canonicalize_platform not implemented")  # pragma: no cover


def join_effects(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function join_effects."""
    raise NotImplementedError("join_effects not implemented")  # pragma: no cover


def array_mapping_to_axis_resources(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function array_mapping_to_axis_resources."""
    raise NotImplementedError("array_mapping_to_axis_resources not implemented")  # pragma: no cover


def merge_lists(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function merge_lists."""
    raise NotImplementedError("merge_lists not implemented")  # pragma: no cover


def device_memory_profile(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function device_memory_profile."""
    raise NotImplementedError("device_memory_profile not implemented")  # pragma: no cover


def detrend(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function detrend."""
    raise NotImplementedError("detrend not implemented")  # pragma: no cover


def ToEltHandler(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function ToEltHandler."""
    raise NotImplementedError("ToEltHandler not implemented")  # pragma: no cover


def gmres(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function gmres."""
    raise NotImplementedError("gmres not implemented")  # pragma: no cover


def use_cpp_class(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function use_cpp_class."""
    raise NotImplementedError("use_cpp_class not implemented")  # pragma: no cover


def clear_event_listeners(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function clear_event_listeners."""
    raise NotImplementedError("clear_event_listeners not implemented")  # pragma: no cover


def cg(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function cg."""
    raise NotImplementedError("cg not implemented")  # pragma: no cover


def Value(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function Value."""
    raise NotImplementedError("Value not implemented")  # pragma: no cover


def extend_axis_env_nd(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function extend_axis_env_nd."""
    raise NotImplementedError("extend_axis_env_nd not implemented")  # pragma: no cover


def trace_to_subjaxpr_nounits(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function trace_to_subjaxpr_nounits."""
    raise NotImplementedError("trace_to_subjaxpr_nounits not implemented")  # pragma: no cover


def scalar_type_of(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function scalar_type_of."""
    raise NotImplementedError("scalar_type_of not implemented")  # pragma: no cover


def correlate2d(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function correlate2d."""
    raise NotImplementedError("correlate2d not implemented")  # pragma: no cover


def jvp_subtrace_aux(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function jvp_subtrace_aux."""
    raise NotImplementedError("jvp_subtrace_aux not implemented")  # pragma: no cover


def extend_jaxpr_stack(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function extend_jaxpr_stack."""
    raise NotImplementedError("extend_jaxpr_stack not implemented")  # pragma: no cover


def register_plugin_callbacks(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function register_plugin_callbacks."""
    raise NotImplementedError("register_plugin_callbacks not implemented")  # pragma: no cover


def substitute_vars_in_output_ty(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function substitute_vars_in_output_ty."""
    raise NotImplementedError("substitute_vars_in_output_ty not implemented")  # pragma: no cover


def using_pjrt_c_api(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function using_pjrt_c_api."""
    raise NotImplementedError("using_pjrt_c_api not implemented")  # pragma: no cover


def ppf(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function ppf."""
    raise NotImplementedError("ppf not implemented")  # pragma: no cover


def rebase_donate_argnums(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function rebase_donate_argnums."""
    raise NotImplementedError("rebase_donate_argnums not implemented")  # pragma: no cover


def shard_args(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function shard_args."""
    raise NotImplementedError("shard_args not implemented")  # pragma: no cover


def subvals(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function subvals."""
    raise NotImplementedError("subvals not implemented")  # pragma: no cover


def flatten_fun_nokwargs(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function flatten_fun_nokwargs."""
    raise NotImplementedError("flatten_fun_nokwargs not implemented")  # pragma: no cover


def LoadedExecutable_execute_with_token(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function LoadedExecutable_execute_with_token."""
    raise NotImplementedError(
        "LoadedExecutable_execute_with_token not implemented"
    )  # pragma: no cover


def gammainc(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function gammainc."""
    raise NotImplementedError("gammainc not implemented")  # pragma: no cover


def DCERule(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function DCERule."""
    raise NotImplementedError("DCERule not implemented")  # pragma: no cover


def subst_axis_names_var(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function subst_axis_names_var."""
    raise NotImplementedError("subst_axis_names_var not implemented")  # pragma: no cover


def recipe_to_eqn(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function recipe_to_eqn."""
    raise NotImplementedError("recipe_to_eqn not implemented")  # pragma: no cover


def dce_jaxpr_call_rule(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function dce_jaxpr_call_rule."""
    raise NotImplementedError("dce_jaxpr_call_rule not implemented")  # pragma: no cover


def custom_call(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function custom_call."""
    raise NotImplementedError("custom_call not implemented")  # pragma: no cover


def record_event(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function record_event."""
    raise NotImplementedError("record_event not implemented")  # pragma: no cover


def apply_todos(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function apply_todos."""
    raise NotImplementedError("apply_todos not implemented")  # pragma: no cover


def batch(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function batch."""
    raise NotImplementedError("batch not implemented")  # pragma: no cover


def subst_axis_names(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function subst_axis_names."""
    raise NotImplementedError("subst_axis_names not implemented")  # pragma: no cover


def apply_primitive(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function apply_primitive."""
    raise NotImplementedError("apply_primitive not implemented")  # pragma: no cover


def unregister_vmappable(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function unregister_vmappable."""
    raise NotImplementedError("unregister_vmappable not implemented")  # pragma: no cover


def linearize(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function linearize."""
    raise NotImplementedError("linearize not implemented")  # pragma: no cover


def op_sharding_to_indices(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function op_sharding_to_indices."""
    raise NotImplementedError("op_sharding_to_indices not implemented")  # pragma: no cover


def initialize_pjrt_plugin(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function initialize_pjrt_plugin."""
    raise NotImplementedError("initialize_pjrt_plugin not implemented")  # pragma: no cover


def vectorized_batcher(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function vectorized_batcher."""
    raise NotImplementedError("vectorized_batcher not implemented")  # pragma: no cover


def infer_lambda_input_type(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function infer_lambda_input_type."""
    raise NotImplementedError("infer_lambda_input_type not implemented")  # pragma: no cover


def make_pjrt_tpu_topology(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function make_pjrt_tpu_topology."""
    raise NotImplementedError("make_pjrt_tpu_topology not implemented")  # pragma: no cover


def zeros_like_jaxval(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function zeros_like_jaxval."""
    raise NotImplementedError("zeros_like_jaxval not implemented")  # pragma: no cover


def get_compile_options(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function get_compile_options."""
    raise NotImplementedError("get_compile_options not implemented")  # pragma: no cover


def split_list_checked(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function split_list_checked."""
    raise NotImplementedError("split_list_checked not implemented")  # pragma: no cover


def convert_envvars_to_constvars(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function convert_envvars_to_constvars."""
    raise NotImplementedError("convert_envvars_to_constvars not implemented")  # pragma: no cover


def fun_name(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function fun_name."""
    raise NotImplementedError("fun_name not implemented")  # pragma: no cover


def jaxpr_uses_outfeed(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function jaxpr_uses_outfeed."""
    raise NotImplementedError("jaxpr_uses_outfeed not implemented")  # pragma: no cover


def register_lowering(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function register_lowering."""
    raise NotImplementedError("register_lowering not implemented")  # pragma: no cover


def matchaxis(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function matchaxis."""
    raise NotImplementedError("matchaxis not implemented")  # pragma: no cover


def shape_from_pyval(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function shape_from_pyval."""
    raise NotImplementedError("shape_from_pyval not implemented")  # pragma: no cover


def dce_jaxpr_consts(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function dce_jaxpr_consts."""
    raise NotImplementedError("dce_jaxpr_consts not implemented")  # pragma: no cover


def token_type(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function token_type."""
    raise NotImplementedError("token_type not implemented")  # pragma: no cover


def backend_xla_version(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function backend_xla_version."""
    raise NotImplementedError("backend_xla_version not implemented")  # pragma: no cover


def safe_map(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function safe_map."""
    raise NotImplementedError("safe_map not implemented")  # pragma: no cover


def new_main(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function new_main."""
    raise NotImplementedError("new_main not implemented")  # pragma: no cover


def is_gpu(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function is_gpu."""
    raise NotImplementedError("is_gpu not implemented")  # pragma: no cover


def load_pjrt_plugin_with_c_api(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function load_pjrt_plugin_with_c_api."""
    raise NotImplementedError("load_pjrt_plugin_with_c_api not implemented")  # pragma: no cover


def subst_axis_names_jaxpr(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function subst_axis_names_jaxpr."""
    raise NotImplementedError("subst_axis_names_jaxpr not implemented")  # pragma: no cover


def get_backend(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function get_backend."""
    raise NotImplementedError("get_backend not implemented")  # pragma: no cover


def valid_jaxtype(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function valid_jaxtype."""
    raise NotImplementedError("valid_jaxtype not implemented")  # pragma: no cover


def log_ndtr(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function log_ndtr."""
    raise NotImplementedError("log_ndtr not implemented")  # pragma: no cover


def ConstFoldRule(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function ConstFoldRule."""
    raise NotImplementedError("ConstFoldRule not implemented")  # pragma: no cover


def AxisSize(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function AxisSize."""
    raise NotImplementedError("AxisSize not implemented")  # pragma: no cover


def replace_float0s(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function replace_float0s."""
    raise NotImplementedError("replace_float0s not implemented")  # pragma: no cover


def check_eqn(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function check_eqn."""
    raise NotImplementedError("check_eqn not implemented")  # pragma: no cover


def visualize_array_sharding(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function visualize_array_sharding."""
    raise NotImplementedError("visualize_array_sharding not implemented")  # pragma: no cover


def make_replica_groups(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function make_replica_groups."""
    raise NotImplementedError("make_replica_groups not implemented")  # pragma: no cover


def call(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function call."""
    raise NotImplementedError("call not implemented")  # pragma: no cover


def defreducer(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function defreducer."""
    raise NotImplementedError("defreducer not implemented")  # pragma: no cover


def dense_bool_elements(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function dense_bool_elements."""
    raise NotImplementedError("dense_bool_elements not implemented")  # pragma: no cover


def pjrt_plugin_loaded(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function pjrt_plugin_loaded."""
    raise NotImplementedError("pjrt_plugin_loaded not implemented")  # pragma: no cover


def make_pjrt_topology(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function make_pjrt_topology."""
    raise NotImplementedError("make_pjrt_topology not implemented")  # pragma: no cover


def broadcast_batcher(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function broadcast_batcher."""
    raise NotImplementedError("broadcast_batcher not implemented")  # pragma: no cover


def Const(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function Const."""
    raise NotImplementedError("Const not implemented")  # pragma: no cover


def rel_entr(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function rel_entr."""
    raise NotImplementedError("rel_entr not implemented")  # pragma: no cover


def csd(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function csd."""
    raise NotImplementedError("csd not implemented")  # pragma: no cover


def stop_server(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function stop_server."""
    raise NotImplementedError("stop_server not implemented")  # pragma: no cover


def process_index(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function process_index."""
    raise NotImplementedError("process_index not implemented")  # pragma: no cover


def mode(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function mode."""
    raise NotImplementedError("mode not implemented")  # pragma: no cover


def logpdf(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function logpdf."""
    raise NotImplementedError("logpdf not implemented")  # pragma: no cover


def convert_constvars_jaxpr(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function convert_constvars_jaxpr."""
    raise NotImplementedError("convert_constvars_jaxpr not implemented")  # pragma: no cover


def trace_state_clean(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function trace_state_clean."""
    raise NotImplementedError("trace_state_clean not implemented")  # pragma: no cover


def annotate_function(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function annotate_function."""
    raise NotImplementedError("annotate_function not implemented")  # pragma: no cover


def entr(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function entr."""
    raise NotImplementedError("entr not implemented")  # pragma: no cover


def poch(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function poch."""
    raise NotImplementedError("poch not implemented")  # pragma: no cover


def convolve2d(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function convolve2d."""
    raise NotImplementedError("convolve2d not implemented")  # pragma: no cover


def stash_axis_env(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function stash_axis_env."""
    raise NotImplementedError("stash_axis_env not implemented")  # pragma: no cover


def debug_info_final(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function debug_info_final."""
    raise NotImplementedError("debug_info_final not implemented")  # pragma: no cover


def DTypeLike(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function DTypeLike."""
    raise NotImplementedError("DTypeLike not implemented")  # pragma: no cover


def jaxpr_subcomp(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function jaxpr_subcomp."""
    raise NotImplementedError("jaxpr_subcomp not implemented")  # pragma: no cover


def backends_are_initialized(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function backends_are_initialized."""
    raise NotImplementedError("backends_are_initialized not implemented")  # pragma: no cover


def include_frame(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function include_frame."""
    raise NotImplementedError("include_frame not implemented")  # pragma: no cover


def ForwardingRule(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function ForwardingRule."""
    raise NotImplementedError("ForwardingRule not implemented")  # pragma: no cover


def closure_convert(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function closure_convert."""
    raise NotImplementedError("closure_convert not implemented")  # pragma: no cover


def primal_dtype_to_tangent_dtype(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function primal_dtype_to_tangent_dtype."""
    raise NotImplementedError("primal_dtype_to_tangent_dtype not implemented")  # pragma: no cover


def shape_tensor(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function shape_tensor."""
    raise NotImplementedError("shape_tensor not implemented")  # pragma: no cover


def register_exclusion(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function register_exclusion."""
    raise NotImplementedError("register_exclusion not implemented")  # pragma: no cover


def logpmf(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function logpmf."""
    raise NotImplementedError("logpmf not implemented")  # pragma: no cover


def abstractify(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function abstractify."""
    raise NotImplementedError("abstractify not implemented")  # pragma: no cover


def visualize_sharding(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function visualize_sharding."""
    raise NotImplementedError("visualize_sharding not implemented")  # pragma: no cover


def max_dim(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function max_dim."""
    raise NotImplementedError("max_dim not implemented")  # pragma: no cover


def register_constant_handler(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function register_constant_handler."""
    raise NotImplementedError("register_constant_handler not implemented")  # pragma: no cover


def sph_harm(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function sph_harm."""
    raise NotImplementedError("sph_harm not implemented")  # pragma: no cover


def BatchingRule(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function BatchingRule."""
    raise NotImplementedError("BatchingRule not implemented")  # pragma: no cover


def api_boundary(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function api_boundary."""
    raise NotImplementedError("api_boundary not implemented")  # pragma: no cover


def make_jaxpr_effects(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function make_jaxpr_effects."""
    raise NotImplementedError("make_jaxpr_effects not implemented")  # pragma: no cover


def vq(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function vq."""
    raise NotImplementedError("vq not implemented")  # pragma: no cover


def stop_trace(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function stop_trace."""
    raise NotImplementedError("stop_trace not implemented")  # pragma: no cover


def merge_mlir_modules(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function merge_mlir_modules."""
    raise NotImplementedError("merge_mlir_modules not implemented")  # pragma: no cover


def flatten_fun(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function flatten_fun."""
    raise NotImplementedError("flatten_fun not implemented")  # pragma: no cover


def expand_platform_alias(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function expand_platform_alias."""
    raise NotImplementedError("expand_platform_alias not implemented")  # pragma: no cover


def add_tangents(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function add_tangents."""
    raise NotImplementedError("add_tangents not implemented")  # pragma: no cover


def jaxpr_as_fun(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function jaxpr_as_fun."""
    raise NotImplementedError("jaxpr_as_fun not implemented")  # pragma: no cover


def unzip2(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function unzip2."""
    raise NotImplementedError("unzip2 not implemented")  # pragma: no cover


def lru_cache(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function lru_cache."""
    raise NotImplementedError("lru_cache not implemented")  # pragma: no cover


def module_to_bytecode(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function module_to_bytecode."""
    raise NotImplementedError("module_to_bytecode not implemented")  # pragma: no cover


def JaxprTracerRecipe(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function JaxprTracerRecipe."""
    raise NotImplementedError("JaxprTracerRecipe not implemented")  # pragma: no cover


def expn(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function expn."""
    raise NotImplementedError("expn not implemented")  # pragma: no cover


def is_vmappable(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function is_vmappable."""
    raise NotImplementedError("is_vmappable not implemented")  # pragma: no cover


def hilbert(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function hilbert."""
    raise NotImplementedError("hilbert not implemented")  # pragma: no cover


def parallel_callable(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function parallel_callable."""
    raise NotImplementedError("parallel_callable not implemented")  # pragma: no cover


def pdf(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function pdf."""
    raise NotImplementedError("pdf not implemented")  # pragma: no cover


def jvp_subtrace(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function jvp_subtrace."""
    raise NotImplementedError("jvp_subtrace not implemented")  # pragma: no cover


def process_env_traces_map(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function process_env_traces_map."""
    raise NotImplementedError("process_env_traces_map not implemented")  # pragma: no cover


def emit_python_callback(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function emit_python_callback."""
    raise NotImplementedError("emit_python_callback not implemented")  # pragma: no cover


def GetIdx(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function GetIdx."""
    raise NotImplementedError("GetIdx not implemented")  # pragma: no cover


def pmf(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function pmf."""
    raise NotImplementedError("pmf not implemented")  # pragma: no cover


def backward_pass_internal(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function backward_pass_internal."""
    raise NotImplementedError("backward_pass_internal not implemented")  # pragma: no cover


def module_to_string(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function module_to_string."""
    raise NotImplementedError("module_to_string not implemented")  # pragma: no cover


def register_vmappable(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function register_vmappable."""
    raise NotImplementedError("register_vmappable not implemented")  # pragma: no cover


def trace_to_jaxpr_final2(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function trace_to_jaxpr_final2."""
    raise NotImplementedError("trace_to_jaxpr_final2 not implemented")  # pragma: no cover


def load_pjrt_plugin_dynamically(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function load_pjrt_plugin_dynamically."""
    raise NotImplementedError("load_pjrt_plugin_dynamically not implemented")  # pragma: no cover


def find_top_trace(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function find_top_trace."""
    raise NotImplementedError("find_top_trace not implemented")  # pragma: no cover


def tpu_client_timer_callback(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function tpu_client_timer_callback."""
    raise NotImplementedError("tpu_client_timer_callback not implemented")  # pragma: no cover


def typecheck(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function typecheck."""
    raise NotImplementedError("typecheck not implemented")  # pragma: no cover


def register_custom_call_target(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function register_custom_call_target."""
    raise NotImplementedError("register_custom_call_target not implemented")  # pragma: no cover


def partial_eval_jaxpr_nounits(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function partial_eval_jaxpr_nounits."""
    raise NotImplementedError("partial_eval_jaxpr_nounits not implemented")  # pragma: no cover


def defjvp2(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function defjvp2."""
    raise NotImplementedError("defjvp2 not implemented")  # pragma: no cover


def AbstractedAxesSpec(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function AbstractedAxesSpec."""
    raise NotImplementedError("AbstractedAxesSpec not implemented")  # pragma: no cover


def sem(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function sem."""
    raise NotImplementedError("sem not implemented")  # pragma: no cover


def donation_vector(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function donation_vector."""
    raise NotImplementedError("donation_vector not implemented")  # pragma: no cover


def move_binders_to_front(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function move_binders_to_front."""
    raise NotImplementedError("move_binders_to_front not implemented")  # pragma: no cover


def global_aval_to_result_handler(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function global_aval_to_result_handler."""
    raise NotImplementedError("global_aval_to_result_handler not implemented")  # pragma: no cover


def partition_list(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function partition_list."""
    raise NotImplementedError("partition_list not implemented")  # pragma: no cover


def map_transpose(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function map_transpose."""
    raise NotImplementedError("map_transpose not implemented")  # pragma: no cover


def concrete_aval(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function concrete_aval."""
    raise NotImplementedError("concrete_aval not implemented")  # pragma: no cover


def str_eqn_compact(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function str_eqn_compact."""
    raise NotImplementedError("str_eqn_compact not implemented")  # pragma: no cover


def has_megascale_address(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function has_megascale_address."""
    raise NotImplementedError("has_megascale_address not implemented")  # pragma: no cover


def unzip3(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function unzip3."""
    raise NotImplementedError("unzip3 not implemented")  # pragma: no cover


def axis_frame(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function axis_frame."""
    raise NotImplementedError("axis_frame not implemented")  # pragma: no cover


def lower_fun(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function lower_fun."""
    raise NotImplementedError("lower_fun not implemented")  # pragma: no cover


def bdim_at_front(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function bdim_at_front."""
    raise NotImplementedError("bdim_at_front not implemented")  # pragma: no cover


def assert_unreachable(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function assert_unreachable."""
    raise NotImplementedError("assert_unreachable not implemented")  # pragma: no cover


def spec_to_indices(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function spec_to_indices."""
    raise NotImplementedError("spec_to_indices not implemented")  # pragma: no cover


def from_elt(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function from_elt."""
    raise NotImplementedError("from_elt not implemented")  # pragma: no cover


def maybe_find_leaked_tracers(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function maybe_find_leaked_tracers."""
    raise NotImplementedError("maybe_find_leaked_tracers not implemented")  # pragma: no cover


def backward_pass(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function backward_pass."""
    raise NotImplementedError("backward_pass not implemented")  # pragma: no cover


def get_referent(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function get_referent."""
    raise NotImplementedError("get_referent not implemented")  # pragma: no cover


def format_exception_only(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function format_exception_only."""
    raise NotImplementedError("format_exception_only not implemented")  # pragma: no cover


def process_env_traces_call(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function process_env_traces_call."""
    raise NotImplementedError("process_env_traces_call not implemented")  # pragma: no cover


def standard_jvp(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function standard_jvp."""
    raise NotImplementedError("standard_jvp not implemented")  # pragma: no cover


def trace_to_subjaxpr_dynamic2(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function trace_to_subjaxpr_dynamic2."""
    raise NotImplementedError("trace_to_subjaxpr_dynamic2 not implemented")  # pragma: no cover


def dedup_referents(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function dedup_referents."""
    raise NotImplementedError("dedup_referents not implemented")  # pragma: no cover


def vjp(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function vjp."""
    raise NotImplementedError("vjp not implemented")  # pragma: no cover


def linear_transpose2(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function linear_transpose2."""
    raise NotImplementedError("linear_transpose2 not implemented")  # pragma: no cover


def flatten_fun_for_vmap(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function flatten_fun_for_vmap."""
    raise NotImplementedError("flatten_fun_for_vmap not implemented")  # pragma: no cover


def betaln(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function betaln."""
    raise NotImplementedError("betaln not implemented")  # pragma: no cover


def typecompat(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function typecompat."""
    raise NotImplementedError("typecompat not implemented")  # pragma: no cover


def batch_jaxpr_axes(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function batch_jaxpr_axes."""
    raise NotImplementedError("batch_jaxpr_axes not implemented")  # pragma: no cover


def lower_jaxpr_to_fun(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function lower_jaxpr_to_fun."""
    raise NotImplementedError("lower_jaxpr_to_fun not implemented")  # pragma: no cover


def i1(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function i1."""
    raise NotImplementedError("i1 not implemented")  # pragma: no cover


def make_padding_config(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function make_padding_config."""
    raise NotImplementedError("make_padding_config not implemented")  # pragma: no cover


def aval_to_ir_types(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function aval_to_ir_types."""
    raise NotImplementedError("aval_to_ir_types not implemented")  # pragma: no cover


def get_aval(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function get_aval."""
    raise NotImplementedError("get_aval not implemented")  # pragma: no cover


def partial_eval_jaxpr_custom_rule_not_implemented(
    *args: typing.Any, **kwargs: typing.Any
) -> typing.Any:
    """Not implemented function partial_eval_jaxpr_custom_rule_not_implemented."""
    raise NotImplementedError(
        "partial_eval_jaxpr_custom_rule_not_implemented not implemented"
    )  # pragma: no cover


def make_cpu_client(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function make_cpu_client."""
    raise NotImplementedError("make_cpu_client not implemented")  # pragma: no cover


def lpmn(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function lpmn."""
    raise NotImplementedError("lpmn not implemented")  # pragma: no cover


def cache(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function cache."""
    raise NotImplementedError("cache not implemented")  # pragma: no cover


def make_tfrt_tpu_c_api_client(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function make_tfrt_tpu_c_api_client."""
    raise NotImplementedError("make_tfrt_tpu_c_api_client not implemented")  # pragma: no cover


def trace_to_subjaxpr_nounits_fwd(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function trace_to_subjaxpr_nounits_fwd."""
    raise NotImplementedError("trace_to_subjaxpr_nounits_fwd not implemented")  # pragma: no cover


def shutdown(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function shutdown."""
    raise NotImplementedError("shutdown not implemented")  # pragma: no cover


def tuple_delete(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function tuple_delete."""
    raise NotImplementedError("tuple_delete not implemented")  # pragma: no cover


def rankdata(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function rankdata."""
    raise NotImplementedError("rankdata not implemented")  # pragma: no cover


def reset_trace_state(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function reset_trace_state."""
    raise NotImplementedError("reset_trace_state not implemented")  # pragma: no cover


def unmapped_aval(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function unmapped_aval."""
    raise NotImplementedError("unmapped_aval not implemented")  # pragma: no cover


def TopologyFactory(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function TopologyFactory."""
    raise NotImplementedError("TopologyFactory not implemented")  # pragma: no cover


def get_tpu_library_path(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function get_tpu_library_path."""
    raise NotImplementedError("get_tpu_library_path not implemented")  # pragma: no cover


def traceable(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function traceable."""
    raise NotImplementedError("traceable not implemented")  # pragma: no cover


def traverse_jaxpr_params(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function traverse_jaxpr_params."""
    raise NotImplementedError("traverse_jaxpr_params not implemented")  # pragma: no cover


def call_partial_eval_custom_rule(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function call_partial_eval_custom_rule."""
    raise NotImplementedError("call_partial_eval_custom_rule not implemented")  # pragma: no cover


def defvectorized(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function defvectorized."""
    raise NotImplementedError("defvectorized not implemented")  # pragma: no cover


def i32_attr(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function i32_attr."""
    raise NotImplementedError("i32_attr not implemented")  # pragma: no cover


def expi(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function expi."""
    raise NotImplementedError("expi not implemented")  # pragma: no cover


def CurrentSourceInfoMetadata(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function CurrentSourceInfoMetadata."""
    raise NotImplementedError("CurrentSourceInfoMetadata not implemented")  # pragma: no cover


def pad_jaxpr(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function pad_jaxpr."""
    raise NotImplementedError("pad_jaxpr not implemented")  # pragma: no cover


def new_jaxpr_eqn(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function new_jaxpr_eqn."""
    raise NotImplementedError("new_jaxpr_eqn not implemented")  # pragma: no cover


def hyp1f1(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function hyp1f1."""
    raise NotImplementedError("hyp1f1 not implemented")  # pragma: no cover


def shaped_abstractify(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function shaped_abstractify."""
    raise NotImplementedError("shaped_abstractify not implemented")  # pragma: no cover


def num_available_tpu_chips_and_device_id(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function num_available_tpu_chips_and_device_id."""
    raise NotImplementedError(
        "num_available_tpu_chips_and_device_id not implemented"
    )  # pragma: no cover


def is_constant_shape(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function is_constant_shape."""
    raise NotImplementedError("is_constant_shape not implemented")  # pragma: no cover


def make_c_api_client(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function make_c_api_client."""
    raise NotImplementedError("make_c_api_client not implemented")  # pragma: no cover


def linear_transpose(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function linear_transpose."""
    raise NotImplementedError("linear_transpose not implemented")  # pragma: no cover


def host_count(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function host_count."""
    raise NotImplementedError("host_count not implemented")  # pragma: no cover


def full_lower(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function full_lower."""
    raise NotImplementedError("full_lower not implemented")  # pragma: no cover


def deflinear(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function deflinear."""
    raise NotImplementedError("deflinear not implemented")  # pragma: no cover


def as_named_shape(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function as_named_shape."""
    raise NotImplementedError("as_named_shape not implemented")  # pragma: no cover


def partial_eval_wrapper_nounits(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function partial_eval_wrapper_nounits."""
    raise NotImplementedError("partial_eval_wrapper_nounits not implemented")  # pragma: no cover


def defjvp(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function defjvp."""
    raise NotImplementedError("defjvp not implemented")  # pragma: no cover


def custom_vjp_primal_tree_values(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function custom_vjp_primal_tree_values."""
    raise NotImplementedError("custom_vjp_primal_tree_values not implemented")  # pragma: no cover


def make_dot_dimension_numbers(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function make_dot_dimension_numbers."""
    raise NotImplementedError("make_dot_dimension_numbers not implemented")  # pragma: no cover


def convert_invars_to_constvars(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function convert_invars_to_constvars."""
    raise NotImplementedError("convert_invars_to_constvars not implemented")  # pragma: no cover


def i64_attr(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function i64_attr."""
    raise NotImplementedError("i64_attr not implemented")  # pragma: no cover


def dense_int_elements(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function dense_int_elements."""
    raise NotImplementedError("dense_int_elements not implemented")  # pragma: no cover


def host_ids(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function host_ids."""
    raise NotImplementedError("host_ids not implemented")  # pragma: no cover


def nonzero_outputs(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function nonzero_outputs."""
    raise NotImplementedError("nonzero_outputs not implemented")  # pragma: no cover


def block_diag(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function block_diag."""
    raise NotImplementedError("block_diag not implemented")  # pragma: no cover


def record_event_duration_secs(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function record_event_duration_secs."""
    raise NotImplementedError("record_event_duration_secs not implemented")  # pragma: no cover


def make_ir_context(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function make_ir_context."""
    raise NotImplementedError("make_ir_context not implemented")  # pragma: no cover


def split_merge(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function split_merge."""
    raise NotImplementedError("split_merge not implemented")  # pragma: no cover


def trace_to_jaxpr(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function trace_to_jaxpr."""
    raise NotImplementedError("trace_to_jaxpr not implemented")  # pragma: no cover


def check_type(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function check_type."""
    raise NotImplementedError("check_type not implemented")  # pragma: no cover


def weakref_lru_cache(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function weakref_lru_cache."""
    raise NotImplementedError("weakref_lru_cache not implemented")  # pragma: no cover


def check_toposort(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function check_toposort."""
    raise NotImplementedError("check_toposort not implemented")  # pragma: no cover


def FromEltHandler(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function FromEltHandler."""
    raise NotImplementedError("FromEltHandler not implemented")  # pragma: no cover


def window_padding_type_to_pad_values(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function window_padding_type_to_pad_values."""
    raise NotImplementedError(
        "window_padding_type_to_pad_values not implemented"
    )  # pragma: no cover


def get_primitive_transpose(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function get_primitive_transpose."""
    raise NotImplementedError("get_primitive_transpose not implemented")  # pragma: no cover


def backends(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function backends."""
    raise NotImplementedError("backends not implemented")  # pragma: no cover


def make_gpu_client(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function make_gpu_client."""
    raise NotImplementedError("make_gpu_client not implemented")  # pragma: no cover


def eval_jaxpr(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function eval_jaxpr."""
    raise NotImplementedError("eval_jaxpr not implemented")  # pragma: no cover


def make_tfrt_tpu_c_api_device_topology(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function make_tfrt_tpu_c_api_device_topology."""
    raise NotImplementedError(
        "make_tfrt_tpu_c_api_device_topology not implemented"
    )  # pragma: no cover


def is_known_platform(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function is_known_platform."""
    raise NotImplementedError("is_known_platform not implemented")  # pragma: no cover


def local_device_count(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function local_device_count."""
    raise NotImplementedError("local_device_count not implemented")  # pragma: no cover


def start_trace(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function start_trace."""
    raise NotImplementedError("start_trace not implemented")  # pragma: no cover


def pjrt_plugin_initialized(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function pjrt_plugin_initialized."""
    raise NotImplementedError("pjrt_plugin_initialized not implemented")  # pragma: no cover


def gammaincc(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function gammaincc."""
    raise NotImplementedError("gammaincc not implemented")  # pragma: no cover


def mapped_aval(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function mapped_aval."""
    raise NotImplementedError("mapped_aval not implemented")  # pragma: no cover


def join_named_shapes(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function join_named_shapes."""
    raise NotImplementedError("join_named_shapes not implemented")  # pragma: no cover


def abstract_eval_fun(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function abstract_eval_fun."""
    raise NotImplementedError("abstract_eval_fun not implemented")  # pragma: no cover


def make_tpu_client(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function make_tpu_client."""
    raise NotImplementedError("make_tpu_client not implemented")  # pragma: no cover


def call_bind_with_continuation(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function call_bind_with_continuation."""
    raise NotImplementedError("call_bind_with_continuation not implemented")  # pragma: no cover


def canonicalize_axis(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function canonicalize_axis."""
    raise NotImplementedError("canonicalize_axis not implemented")  # pragma: no cover


def sig_info(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function sig_info."""
    raise NotImplementedError("sig_info not implemented")  # pragma: no cover


def zeros_like_aval(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function zeros_like_aval."""
    raise NotImplementedError("zeros_like_aval not implemented")  # pragma: no cover


def get_device_backend(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function get_device_backend."""
    raise NotImplementedError("get_device_backend not implemented")  # pragma: no cover


def trace_to_jaxpr_dynamic(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function trace_to_jaxpr_dynamic."""
    raise NotImplementedError("trace_to_jaxpr_dynamic not implemented")  # pragma: no cover


def dtype_to_etype(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function dtype_to_etype."""
    raise NotImplementedError("dtype_to_etype not implemented")  # pragma: no cover


def check_valid_jaxtype(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function check_valid_jaxtype."""
    raise NotImplementedError("check_valid_jaxtype not implemented")  # pragma: no cover


def register_plugin(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function register_plugin."""
    raise NotImplementedError("register_plugin not implemented")  # pragma: no cover


def sf(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function sf."""
    raise NotImplementedError("sf not implemented")  # pragma: no cover


def MapSpec(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function MapSpec."""
    raise NotImplementedError("MapSpec not implemented")  # pragma: no cover


def devices(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function devices."""
    raise NotImplementedError("devices not implemented")  # pragma: no cover


def map_bind(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function map_bind."""
    raise NotImplementedError("map_bind not implemented")  # pragma: no cover


def tracebacks(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function tracebacks."""
    raise NotImplementedError("tracebacks not implemented")  # pragma: no cover


def dense_bool_array(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function dense_bool_array."""
    raise NotImplementedError("dense_bool_array not implemented")  # pragma: no cover


def toposort(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function toposort."""
    raise NotImplementedError("toposort not implemented")  # pragma: no cover


def cur_sublevel(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function cur_sublevel."""
    raise NotImplementedError("cur_sublevel not implemented")  # pragma: no cover


def gensym(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function gensym."""
    raise NotImplementedError("gensym not implemented")  # pragma: no cover


def discover_pjrt_plugins(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function discover_pjrt_plugins."""
    raise NotImplementedError("discover_pjrt_plugins not implemented")  # pragma: no cover


def LoadedExecutable_execute(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function LoadedExecutable_execute."""
    raise NotImplementedError("LoadedExecutable_execute not implemented")  # pragma: no cover


def inspect_array_sharding(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function inspect_array_sharding."""
    raise NotImplementedError("inspect_array_sharding not implemented")  # pragma: no cover


def new_eqn_recipe(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function new_eqn_recipe."""
    raise NotImplementedError("new_eqn_recipe not implemented")  # pragma: no cover


def local_devices(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function local_devices."""
    raise NotImplementedError("local_devices not implemented")  # pragma: no cover


def generate_pjrt_gpu_plugin_options(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function generate_pjrt_gpu_plugin_options."""
    raise NotImplementedError(
        "generate_pjrt_gpu_plugin_options not implemented"
    )  # pragma: no cover


def lattice_join(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function lattice_join."""
    raise NotImplementedError("lattice_join not implemented")  # pragma: no cover


def jaxprs_in_params(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function jaxprs_in_params."""
    raise NotImplementedError("jaxprs_in_params not implemented")  # pragma: no cover


def expm_frechet(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function expm_frechet."""
    raise NotImplementedError("expm_frechet not implemented")  # pragma: no cover


def use_cpp_method(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function use_cpp_method."""
    raise NotImplementedError("use_cpp_method not implemented")  # pragma: no cover


def cho_factor(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function cho_factor."""
    raise NotImplementedError("cho_factor not implemented")  # pragma: no cover


def set_module(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function set_module."""
    raise NotImplementedError("set_module not implemented")  # pragma: no cover


def welch(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function welch."""
    raise NotImplementedError("welch not implemented")  # pragma: no cover


def raise_to_shaped(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function raise_to_shaped."""
    raise NotImplementedError("raise_to_shaped not implemented")  # pragma: no cover


def eval_context(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function eval_context."""
    raise NotImplementedError("eval_context not implemented")  # pragma: no cover


def standard_jvp2(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function standard_jvp2."""
    raise NotImplementedError("standard_jvp2 not implemented")  # pragma: no cover


def linear_jvp(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function linear_jvp."""
    raise NotImplementedError("linear_jvp not implemented")  # pragma: no cover


def deflinear2(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function deflinear2."""
    raise NotImplementedError("deflinear2 not implemented")  # pragma: no cover


def used_axis_names_jaxpr(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function used_axis_names_jaxpr."""
    raise NotImplementedError("used_axis_names_jaxpr not implemented")  # pragma: no cover


def trace_to_jaxpr_nounits(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function trace_to_jaxpr_nounits."""
    raise NotImplementedError("trace_to_jaxpr_nounits not implemented")  # pragma: no cover


def host_id(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function host_id."""
    raise NotImplementedError("host_id not implemented")  # pragma: no cover


def partition_pvals(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function partition_pvals."""
    raise NotImplementedError("partition_pvals not implemented")  # pragma: no cover


def save_device_memory_profile(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function save_device_memory_profile."""
    raise NotImplementedError("save_device_memory_profile not implemented")  # pragma: no cover


def isf(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function isf."""
    raise NotImplementedError("isf not implemented")  # pragma: no cover


def scale_and_translate(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function scale_and_translate."""
    raise NotImplementedError("scale_and_translate not implemented")  # pragma: no cover


def funm(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function funm."""
    raise NotImplementedError("funm not implemented")  # pragma: no cover


def i0e(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function i0e."""
    raise NotImplementedError("i0e not implemented")  # pragma: no cover


def bilinear_transpose(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function bilinear_transpose."""
    raise NotImplementedError("bilinear_transpose not implemented")  # pragma: no cover


def safe_zip(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function safe_zip."""
    raise NotImplementedError("safe_zip not implemented")  # pragma: no cover


def linear_call(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function linear_call."""
    raise NotImplementedError("linear_call not implemented")  # pragma: no cover


def BackendFactory(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function BackendFactory."""
    raise NotImplementedError("BackendFactory not implemented")  # pragma: no cover


def trace_to_subjaxpr(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function trace_to_subjaxpr."""
    raise NotImplementedError("trace_to_subjaxpr not implemented")  # pragma: no cover


def MeshAxisName(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function MeshAxisName."""
    raise NotImplementedError("MeshAxisName not implemented")  # pragma: no cover


def argnums_partial(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function argnums_partial."""
    raise NotImplementedError("argnums_partial not implemented")  # pragma: no cover


def register_pjrt_plugin_factories_from_env(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function register_pjrt_plugin_factories_from_env."""
    raise NotImplementedError(
        "register_pjrt_plugin_factories_from_env not implemented"
    )  # pragma: no cover


def device_count(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function device_count."""
    raise NotImplementedError("device_count not implemented")  # pragma: no cover


def dce_jaxpr(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function dce_jaxpr."""
    raise NotImplementedError("dce_jaxpr not implemented")  # pragma: no cover


def i1e(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function i1e."""
    raise NotImplementedError("i1e not implemented")  # pragma: no cover


def is_undefined_primal(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function is_undefined_primal."""
    raise NotImplementedError("is_undefined_primal not implemented")  # pragma: no cover


def rsf2csf(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function rsf2csf."""
    raise NotImplementedError("rsf2csf not implemented")  # pragma: no cover


def call_impl(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function call_impl."""
    raise NotImplementedError("call_impl not implemented")  # pragma: no cover


def tpu_enhanced_barrier_supported(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function tpu_enhanced_barrier_supported."""
    raise NotImplementedError("tpu_enhanced_barrier_supported not implemented")  # pragma: no cover


def tracers_to_jaxpr(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function tracers_to_jaxpr."""
    raise NotImplementedError("tracers_to_jaxpr not implemented")  # pragma: no cover


def escaped_tracer_error(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function escaped_tracer_error."""
    raise NotImplementedError("escaped_tracer_error not implemented")  # pragma: no cover


def dense_int_array(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function dense_int_array."""
    raise NotImplementedError("dense_int_array not implemented")  # pragma: no cover


def make_c_api_device_topology(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function make_c_api_device_topology."""
    raise NotImplementedError("make_c_api_device_topology not implemented")  # pragma: no cover


def dtype_to_ir_type(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function dtype_to_ir_type."""
    raise NotImplementedError("dtype_to_ir_type not implemented")  # pragma: no cover


def multigammaln(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function multigammaln."""
    raise NotImplementedError("multigammaln not implemented")  # pragma: no cover


def trace_to_jaxpr_final(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function trace_to_jaxpr_final."""
    raise NotImplementedError("trace_to_jaxpr_final not implemented")  # pragma: no cover


def check_jaxpr(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function check_jaxpr."""
    raise NotImplementedError("check_jaxpr not implemented")  # pragma: no cover


def wraps(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function wraps."""
    raise NotImplementedError("wraps not implemented")  # pragma: no cover


def make_iota(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function make_iota."""
    raise NotImplementedError("make_iota not implemented")  # pragma: no cover


def closed_backward_pass(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function closed_backward_pass."""
    raise NotImplementedError("closed_backward_pass not implemented")  # pragma: no cover


def ArrayLike(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function ArrayLike."""
    raise NotImplementedError("ArrayLike not implemented")  # pragma: no cover


def get_metadata(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function get_metadata."""
    raise NotImplementedError("get_metadata not implemented")  # pragma: no cover


def split_list(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function split_list."""
    raise NotImplementedError("split_list not implemented")  # pragma: no cover


def defjvp_zero(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function defjvp_zero."""
    raise NotImplementedError("defjvp_zero not implemented")  # pragma: no cover


def typematch(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function typematch."""
    raise NotImplementedError("typematch not implemented")  # pragma: no cover


def core_call_lowering(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function core_call_lowering."""
    raise NotImplementedError("core_call_lowering not implemented")  # pragma: no cover


def result_info(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function result_info."""
    raise NotImplementedError("result_info not implemented")  # pragma: no cover


def trivial_ctx(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function trivial_ctx."""
    raise NotImplementedError("trivial_ctx not implemented")  # pragma: no cover


def execute_with_python_values(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function execute_with_python_values."""
    raise NotImplementedError("execute_with_python_values not implemented")  # pragma: no cover


def ensure_compile_time_eval(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function ensure_compile_time_eval."""
    raise NotImplementedError("ensure_compile_time_eval not implemented")  # pragma: no cover


def start_server(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function start_server."""
    raise NotImplementedError("start_server not implemented")  # pragma: no cover


def partial_eval_jaxpr_custom(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function partial_eval_jaxpr_custom."""
    raise NotImplementedError("partial_eval_jaxpr_custom not implemented")  # pragma: no cover


def def_trivial_padding(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function def_trivial_padding."""
    raise NotImplementedError("def_trivial_padding not implemented")  # pragma: no cover


def process_count(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function process_count."""
    raise NotImplementedError("process_count not implemented")  # pragma: no cover


def gammaln(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function gammaln."""
    raise NotImplementedError("gammaln not implemented")  # pragma: no cover


def register_backend_factory(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function register_backend_factory."""
    raise NotImplementedError("register_backend_factory not implemented")  # pragma: no cover


def Vmappable(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function Vmappable."""
    raise NotImplementedError("Vmappable not implemented")  # pragma: no cover


def vtile(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function vtile."""
    raise NotImplementedError("vtile not implemented")  # pragma: no cover


def memoize(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function memoize."""
    raise NotImplementedError("memoize not implemented")  # pragma: no cover


def primitive_uses_outfeed(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function primitive_uses_outfeed."""
    raise NotImplementedError("primitive_uses_outfeed not implemented")  # pragma: no cover


def Index(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function Index."""
    raise NotImplementedError("Index not implemented")  # pragma: no cover


def expit(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function expit."""
    raise NotImplementedError("expit not implemented")  # pragma: no cover


def Atom(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function Atom."""
    raise NotImplementedError("Atom not implemented")  # pragma: no cover


def subjaxprs(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function subjaxprs."""
    raise NotImplementedError("subjaxprs not implemented")  # pragma: no cover


def fftconvolve(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function fftconvolve."""
    raise NotImplementedError("fftconvolve not implemented")  # pragma: no cover


def ParamsUpdater(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function ParamsUpdater."""
    raise NotImplementedError("ParamsUpdater not implemented")  # pragma: no cover


def new_base_main(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function new_base_main."""
    raise NotImplementedError("new_base_main not implemented")  # pragma: no cover


def MakeIotaHandler(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function MakeIotaHandler."""
    raise NotImplementedError("MakeIotaHandler not implemented")  # pragma: no cover


def is_op_sharding_replicated(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function is_op_sharding_replicated."""
    raise NotImplementedError("is_op_sharding_replicated not implemented")  # pragma: no cover


def register_event_duration_secs_listener(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function register_event_duration_secs_listener."""
    raise NotImplementedError(
        "register_event_duration_secs_listener not implemented"
    )  # pragma: no cover


def default_backend(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function default_backend."""
    raise NotImplementedError("default_backend not implemented")  # pragma: no cover


def batch_jaxpr2(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function batch_jaxpr2."""
    raise NotImplementedError("batch_jaxpr2 not implemented")  # pragma: no cover


def maybe_named_axis(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function maybe_named_axis."""
    raise NotImplementedError("maybe_named_axis not implemented")  # pragma: no cover


def toeplitz(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function toeplitz."""
    raise NotImplementedError("toeplitz not implemented")  # pragma: no cover


def global_avals_to_results_handler(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function global_avals_to_results_handler."""
    raise NotImplementedError("global_avals_to_results_handler not implemented")  # pragma: no cover


def idctn(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function idctn."""
    raise NotImplementedError("idctn not implemented")  # pragma: no cover


def cdf(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function cdf."""
    raise NotImplementedError("cdf not implemented")  # pragma: no cover


def subst_axis_names_eqn(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function subst_axis_names_eqn."""
    raise NotImplementedError("subst_axis_names_eqn not implemented")  # pragma: no cover


def dce_jaxpr_closed_call_rule(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function dce_jaxpr_closed_call_rule."""
    raise NotImplementedError("dce_jaxpr_closed_call_rule not implemented")  # pragma: no cover


def AxisContext(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function AxisContext."""
    raise NotImplementedError("AxisContext not implemented")  # pragma: no cover


def kl_div(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function kl_div."""
    raise NotImplementedError("kl_div not implemented")  # pragma: no cover


def gammasgn(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function gammasgn."""
    raise NotImplementedError("gammasgn not implemented")  # pragma: no cover


def trace_to_subjaxpr_nounits_dyn(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function trace_to_subjaxpr_nounits_dyn."""
    raise NotImplementedError("trace_to_subjaxpr_nounits_dyn not implemented")  # pragma: no cover


def filter_traceback(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function filter_traceback."""
    raise NotImplementedError("filter_traceback not implemented")  # pragma: no cover


def trace_to_subjaxpr_dynamic(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function trace_to_subjaxpr_dynamic."""
    raise NotImplementedError("trace_to_subjaxpr_dynamic not implemented")  # pragma: no cover


def defbroadcasting(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function defbroadcasting."""
    raise NotImplementedError("defbroadcasting not implemented")  # pragma: no cover


def execute_with_python_values_replicated(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function execute_with_python_values_replicated."""
    raise NotImplementedError(
        "execute_with_python_values_replicated not implemented"
    )  # pragma: no cover


def trace_to_jaxpr_dynamic2(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function trace_to_jaxpr_dynamic2."""
    raise NotImplementedError("trace_to_jaxpr_dynamic2 not implemented")  # pragma: no cover


def subs_list(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function subs_list."""
    raise NotImplementedError("subs_list not implemented")  # pragma: no cover


def get_tpu_env_value(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function get_tpu_env_value."""
    raise NotImplementedError("get_tpu_env_value not implemented")  # pragma: no cover


def flatten_axes(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function flatten_axes."""
    raise NotImplementedError("flatten_axes not implemented")  # pragma: no cover


def lpmn_values(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function lpmn_values."""
    raise NotImplementedError("lpmn_values not implemented")  # pragma: no cover


def ir_constant(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function ir_constant."""
    raise NotImplementedError("ir_constant not implemented")  # pragma: no cover


def callback(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function callback."""
    raise NotImplementedError("callback not implemented")  # pragma: no cover


def reduce(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function reduce."""
    raise NotImplementedError("reduce not implemented")  # pragma: no cover


def structure(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function structure."""
    raise NotImplementedError("structure not implemented")  # pragma: no cover


def leaves(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Not implemented function leaves."""
    raise NotImplementedError("leaves not implemented")  # pragma: no cover


__all__ = [
    "AbstractedAxesSpec",
    "AdjustBrightness",
    "AdjustContrast",
    "AdjustHue",
    "AdjustSaturation",
    "AffineConfig",
    "AffineGenerator",
    "AffineGrid",
    "AffineTransform",
    "AllGatherOp",
    "AllReduceOp",
    "AllToAll",
    "Any",
    "Arange",
    "ArgSort",
    "ArrayLike",
    "AsStringConfig",
    "AssertOp",
    "Assign",
    "AssignAdd",
    "AssignSub",
    "AssignVariable",
    "AssociativeScan",
    "Atom",
    "AttentionConfig",
    "AttentionInputs",
    "AugMix",
    "AutoContrast",
    "AxisContext",
    "AxisIndex",
    "AxisSize",
    "BackendFactory",
    "BatchNormConfig",
    "BatchingRule",
    "BidirectionalConfig",
    "BidirectionalInputs",
    "BroadcastInDim",
    "BroadcastTo",
    "Callable",
    "ColumnStack",
    "ComplexWarning",
    "Concatenate",
    "Const",
    "ConstFoldRule",
    "ConvGeneralDilated",
    "ConvLSTMConfig",
    "CreationOp",
    "Crop",
    "CropAndResize",
    "CropImages",
    "CurrentSourceInfoMetadata",
    "CustomLinearSolve",
    "Cutmix",
    "DCERule",
    "DType",
    "DTypeLike",
    "Degeneration",
    "DevicePutReplicated",
    "DevicePutSharded",
    "Dot",
    "DotGeneral",
    "DotProductAttentionConfig",
    "DrawBoundingBoxes",
    "Dsplit",
    "Dstack",
    "DynamicSlice",
    "DynamicUpdateSlice",
    "Einsum",
    "ElasticTransform",
    "Elt",
    "Equalization",
    "Expand",
    "ExtractBoundingBoxes",
    "ExtractPatches",
    "Fft",
    "Flatten",
    "FlipLeftRight",
    "FlipUpDown",
    "ForwardingRule",
    "FromEltHandler",
    "Full",
    "Gather",
    "GatherNd",
    "GaussianBlur",
    "GenericConvConfig",
    "GetIdx",
    "GridSample",
    "Hsplit",
    "Hstack",
    "HsvToRgb",
    "Index",
    "Infeed",
    "Invert",
    "IoU",
    "JaxprTracerRecipe",
    "LoadedExecutable_execute",
    "LoadedExecutable_execute_with_token",
    "MAGIC_VAL_3",
    "MAGIC_VAL_4",
    "MakeIotaHandler",
    "MapCoordinates",
    "MapSpec",
    "Matmul",
    "MedianFilter",
    "MeshAxisName",
    "Meshgrid",
    "Mixup",
    "Moveaxis",
    "NonMaxSuppression",
    "NormConfig",
    "Ones",
    "OpDef",
    "Outfeed",
    "PadImages",
    "PadToBoundingBox",
    "ParamsUpdater",
    "PartialEvalCustomRule",
    "Pbroadcast",
    "Pdot",
    "Permute",
    "PerspectiveTransform",
    "Pmax",
    "Pmean",
    "Pmin",
    "Posterize",
    "Ppermute",
    "Pshuffle",
    "Psum",
    "PsumScatter",
    "Pswapaxes",
    "RNNCellDeviceWrapper",
    "RNNCellDropoutWrapper",
    "RNNCellResidualWrapper",
    "RNNConfig",
    "RNNWeights",
    "RaggedAdd",
    "RaggedDynamicBroadcast",
    "RaggedGather",
    "RaggedMatMul",
    "RaggedTensorToDense",
    "RandAugment",
    "RandomColorJitter",
    "RandomCropOp",
    "RandomElasticTransformOp",
    "RandomErasing",
    "RandomFlipOp",
    "RandomGaussianBlurOp",
    "RandomPerspectiveOp",
    "RandomRotationOp",
    "RandomSharpnessOp",
    "RandomShearOp",
    "RandomTranslationOp",
    "RandomZoomOp",
    "RawConv2D",
    "RawMatMul",
    "RawMerge",
    "RawOp",
    "RawSwitch",
    "ReadVariable",
    "ReduceScatterOp",
    "ReduceWindow",
    "Repeat",
    "ResAvalUpdater",
    "Reshape",
    "Resize",
    "ResizeBicubic",
    "ResizeBilinear",
    "ResizeLanczos3",
    "ResizeLanczos5",
    "ResizeNearest",
    "Rfft",
    "RgbToGrayscaleOp",
    "RgbToHsv",
    "RgbToYiq",
    "RgbToYuv",
    "Roll",
    "RowStack",
    "ScanConfig",
    "Scatter",
    "ScatterAdd",
    "ScatterNd",
    "SearchSorted",
    "Select",
    "ShardTensorOp",
    "Sharpen",
    "Slice",
    "SobolSample",
    "Solarize",
    "Sort",
    "SpaceConfig",
    "SparseAdd",
    "SparseDenseMatMul",
    "SparseReduceMax",
    "SparseReduceSum",
    "SparseSoftmax",
    "Split",
    "Squeeze",
    "Stack",
    "StridedSlice",
    "Swapaxes",
    "Take",
    "TakeAlongAxis",
    "Tensor",
    "TensorArrayRead",
    "TensorArrayStack",
    "TensorArrayWrite",
    "TensorConfig",
    "TensorScatterUpdate",
    "Tile",
    "ToEltHandler",
    "TopK",
    "TopologyFactory",
    "Transpose",
    "Tril",
    "Triu",
    "Value",
    "Vdot",
    "Vecdot",
    "Vmappable",
    "Vsplit",
    "Vstack",
    "Where",
    "WithShardingConstraint",
    "YiqToRgb",
    "YuvToRgb",
    "Zeros",
    "abs",
    "absolute",
    "abstract_eval_fun",
    "abstractify",
    "accumulate_n",
    "acos",
    "acosh",
    "activity_regularization",
    "add",
    "add_jaxvals",
    "add_n",
    "add_tangents",
    "adjust_brightness",
    "adjust_contrast",
    "adjust_hue",
    "adjust_saturation",
    "affine_generator",
    "affine_grid",
    "affine_transform",
    "all",
    "all_candidate_sampler",
    "all_gather",
    "all_reduce",
    "all_to_all",
    "allclose",
    "amax",
    "amin",
    "angle",
    "annotate_function",
    "annotations",
    "any",
    "api_boundary",
    "append",
    "apply_along_axis",
    "apply_over_axes",
    "apply_primitive",
    "apply_todos",
    "approx_max_k",
    "approx_min_k",
    "arange",
    "arccos",
    "arccosh",
    "arcsin",
    "arcsinh",
    "arctan",
    "arctan2",
    "arctanh",
    "are_op_shardings_equal",
    "arg_info_all",
    "argmax",
    "argmin",
    "argnums_partial",
    "argpartition",
    "argsort",
    "argwhere",
    "around",
    "array",
    "array_equal",
    "array_equiv",
    "array_mapping_to_axis_resources",
    "array_repr",
    "array_split",
    "array_str",
    "as_hashable_function",
    "as_named_shape",
    "as_string",
    "asarray",
    "asin",
    "asinh",
    "assert_unreachable",
    "assert_value",
    "assert_value_eager",
    "assert_value_tracing",
    "associative_scan",
    "astype",
    "atan",
    "atan2",
    "atanh",
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
    "attention",
    "augmix",
    "auto_contrast",
    "aval_to_ir_type",
    "aval_to_ir_types",
    "average",
    "average_pool",
    "avg_pool",
    "axis_frame",
    "axis_index",
    "backend_pjrt_c_api_version",
    "backend_xla_version",
    "backends",
    "backends_are_initialized",
    "backward_pass",
    "backward_pass_internal",
    "bartlett",
    "batch",
    "batch_custom_jvp_subtrace",
    "batch_custom_vjp_bwd",
    "batch_jaxpr",
    "batch_jaxpr2",
    "batch_jaxpr_axes",
    "batch_normalization",
    "batch_subtrace",
    "bdim_at_front",
    "bessel_i0",
    "bessel_i0e",
    "bessel_i1",
    "bessel_i1e",
    "bessel_j0",
    "bessel_j1",
    "bessel_k0",
    "bessel_k0e",
    "bessel_k1",
    "bessel_k1e",
    "bessel_y0",
    "bessel_y1",
    "betainc",
    "betaln",
    "bfloat16",
    "bicgstab",
    "bidirectional",
    "bilinear_transpose",
    "binary_crossentropy",
    "bincount",
    "bitcast",
    "bitwise_and",
    "bitwise_count",
    "bitwise_invert",
    "bitwise_left_shift",
    "bitwise_not",
    "bitwise_or",
    "bitwise_right_shift",
    "bitwise_xor",
    "blackman",
    "block",
    "block_diag",
    "bool",
    "bool_",
    "boolean_mask",
    "broadcast",
    "broadcast_arrays",
    "broadcast_batcher",
    "broadcast_in_dim",
    "broadcast_shapes",
    "broadcast_to",
    "c_",
    "cache",
    "call",
    "call_bind_with_continuation",
    "call_impl",
    "call_padding_rule",
    "call_partial_eval_custom_rule",
    "call_transpose",
    "callback",
    "can_cast",
    "canonicalize_axis",
    "canonicalize_dtype",
    "canonicalize_platform",
    "case",
    "cast",
    "categorical_crossentropy",
    "categorical_generalized_cross_entropy",
    "cbrt",
    "cdf",
    "cdouble",
    "ceil",
    "ceil_of_ratio",
    "celu",
    "cg",
    "character",
    "check_eqn",
    "check_jaxpr",
    "check_toposort",
    "check_type",
    "check_valid_jaxtype",
    "cho_factor",
    "cho_solve",
    "cholesky",
    "choose",
    "circle_loss",
    "clamp",
    "clear_all_caches",
    "clear_all_weakref_lru_caches",
    "clear_event_listeners",
    "clip",
    "close_jaxpr",
    "closed_backward_pass",
    "closed_call_partial_eval_custom_rule",
    "closure_convert",
    "column_stack",
    "complex",
    "complex128",
    "complex64",
    "complex_",
    "complexfloating",
    "compress",
    "compute_accidental_hits",
    "concat",
    "concatenate",
    "concrete_aval",
    "concrete_or_error",
    "concretization_function_error",
    "cond",
    "cond_eager",
    "cond_tracing",
    "config",
    "conj",
    "conjugate",
    "conv",
    "conv1d",
    "conv1d_lstm_cell",
    "conv1d_transpose",
    "conv2d",
    "conv2d_lstm_cell",
    "conv2d_transpose",
    "conv3d",
    "conv3d_lstm_cell",
    "conv3d_transpose",
    "conv_general_dilated",
    "conv_lstm_cell",
    "conv_transpose",
    "convert_constvars_jaxpr",
    "convert_envvars_to_constvars",
    "convert_invars_to_constvars",
    "convert_to_numpy",
    "convert_to_tensor",
    "convolve",
    "convolve2d",
    "copy",
    "copysign",
    "core_call_lowering",
    "core_config",
    "corrcoef",
    "correlate",
    "correlate2d",
    "cos",
    "cosh",
    "cosine_similarity_loss",
    "count_nonzero",
    "cov",
    "create_eager_alias",
    "crelu",
    "crop",
    "crop_and_resize",
    "crop_images",
    "crop_to_bounding_box",
    "cross",
    "csd",
    "csingle",
    "ctc_decode",
    "ctc_loss",
    "cumprod",
    "cumsum",
    "cumulative_logsumexp",
    "cumulative_sum",
    "cur_sublevel",
    "curry",
    "custom_call",
    "custom_gradient",
    "custom_linear_solve",
    "custom_vjp_primal_tree_values",
    "cutmix",
    "dataclass",
    "dawsn",
    "dce_jaxpr",
    "dce_jaxpr_call_rule",
    "dce_jaxpr_closed_call_rule",
    "dce_jaxpr_consts",
    "dct",
    "dctn",
    "debug_info",
    "debug_info_final",
    "debug_infs",
    "debug_nans",
    "dedup_referents",
    "def_trivial_padding",
    "default_backend",
    "defbilinear",
    "defbroadcasting",
    "defjvp",
    "defjvp2",
    "defjvp_zero",
    "deflinear",
    "deflinear2",
    "defreducer",
    "defvectorized",
    "deg2rad",
    "degeneration",
    "degrees",
    "delete",
    "dense_bool_array",
    "dense_bool_elements",
    "dense_int_array",
    "dense_int_elements",
    "depth_to_space",
    "depthwise_conv",
    "depthwise_conv1d",
    "depthwise_conv2d",
    "det",
    "detrend",
    "device_count",
    "device_memory_profile",
    "device_put_replicated",
    "device_put_sharded",
    "devices",
    "diag",
    "diag_indices",
    "diag_indices_from",
    "diagflat",
    "diagonal",
    "dice_loss",
    "diff",
    "digamma",
    "digitize",
    "discover_pjrt_plugins",
    "dispatch_eager",
    "distributed_debug_log",
    "divide",
    "divide_no_nan",
    "divmod",
    "do_subst_axis_names_jaxpr",
    "donation_vector",
    "dot",
    "dot_general",
    "dot_product_attention",
    "double",
    "draw_bounding_boxes",
    "dropout",
    "dsplit",
    "dstack",
    "dtype",
    "dtype_to_etype",
    "dtype_to_ir_type",
    "dynamic_partition",
    "dynamic_shape",
    "dynamic_slice",
    "dynamic_stitch",
    "dynamic_update_slice",
    "e",
    "eager",
    "ediff1d",
    "edit_distance",
    "eig",
    "eigh",
    "eigvalsh",
    "einsum",
    "einsum_path",
    "elastic_transform",
    "elu",
    "embedding",
    "embedding_lookup",
    "embedding_lookup_sparse",
    "emit_ir_node",
    "emit_python_callback",
    "empty",
    "empty_like",
    "ensure_compile_time_eval",
    "entr",
    "equal",
    "equalization",
    "erf",
    "erfc",
    "erfcinv",
    "erfinv",
    "escaped_tracer_error",
    "euler_gamma",
    "eval_context",
    "eval_jaxpr",
    "execute_with_python_values",
    "execute_with_python_values_replicated",
    "exp",
    "exp1",
    "exp2",
    "expand",
    "expand_dims",
    "expand_platform_alias",
    "expi",
    "expint",
    "expit",
    "expm1",
    "expm_frechet",
    "expn",
    "extend_axis_env",
    "extend_axis_env_nd",
    "extend_jaxpr_stack",
    "extract",
    "extract_bounding_boxes",
    "extract_patches",
    "extract_sequences",
    "extract_volume_patches",
    "eye",
    "f_jvp_traceable",
    "fabs",
    "factorial",
    "fft",
    "fft2",
    "fft2d",
    "fft3",
    "fft3d",
    "fftconvolve",
    "fftnd",
    "fftshift",
    "fill_diagonal",
    "filter_traceback",
    "find_top_trace",
    "finfo",
    "fix",
    "fixed_unigram_candidate_sampler",
    "flatnonzero",
    "flatten",
    "flatten_axes",
    "flatten_fun",
    "flatten_fun_for_vmap",
    "flatten_fun_nokwargs",
    "flatten_lowering_ir_args",
    "flexible",
    "flip",
    "flip_left_right",
    "flip_up_down",
    "fliplr",
    "flipud",
    "float",
    "float16",
    "float32",
    "float64",
    "float8_e4m3b11fnuz",
    "float8_e4m3fn",
    "float8_e4m3fnuz",
    "float8_e5m2",
    "float8_e5m2fnuz",
    "float_",
    "float_power",
    "floating",
    "floor",
    "floor_divide",
    "fmax",
    "fmin",
    "fmod",
    "fori_loop",
    "format_exception_only",
    "frame",
    "fresnel_cos",
    "fresnel_sin",
    "frexp",
    "from_dlpack",
    "from_elt",
    "frombuffer",
    "fromfile",
    "fromfunction",
    "fromiter",
    "frompyfunc",
    "fromstring",
    "full",
    "full_like",
    "full_lower",
    "fun_name",
    "funm",
    "gammainc",
    "gammaincc",
    "gammaln",
    "gammasgn",
    "gather",
    "gather_nd",
    "gaussian_blur",
    "gaussian_nll_loss",
    "gcd",
    "gelu",
    "generate_pjrt_gpu_plugin_options",
    "generic",
    "gensym",
    "geomspace",
    "get_active_backend",
    "get_aval",
    "get_backend",
    "get_compile_options",
    "get_device_backend",
    "get_item",
    "get_metadata",
    "get_op",
    "get_primitive_transpose",
    "get_printoptions",
    "get_referent",
    "get_tpu_env_value",
    "get_tpu_library_path",
    "global_aval_to_result_handler",
    "global_avals_to_results_handler",
    "global_config",
    "glu",
    "gmres",
    "gradient",
    "greater",
    "greater_equal",
    "grid_sample",
    "group_mean",
    "group_norm",
    "group_variance",
    "gru_cell",
    "hamming",
    "hamming_window",
    "hann_window",
    "hanning",
    "hard_shrink",
    "hard_sigmoid",
    "hard_silu",
    "hard_swish",
    "hard_tanh",
    "has_megascale_address",
    "heap_profile",
    "heaviside",
    "hilbert",
    "hinge_loss",
    "histogram",
    "histogram2d",
    "histogram_bin_edges",
    "histogramdd",
    "host_count",
    "host_id",
    "host_ids",
    "hsplit",
    "hstack",
    "hsv_to_rgb",
    "huber_loss",
    "hyp1f1",
    "hypot",
    "i0",
    "i0e",
    "i1",
    "i1e",
    "i32_attr",
    "i64_attr",
    "idct",
    "idctn",
    "identity",
    "ifft",
    "ifft2",
    "ifft2d",
    "ifft3",
    "ifft3d",
    "ifftnd",
    "ifftshift",
    "igamma",
    "igammac",
    "iinfo",
    "imag",
    "image_resize",
    "in_top_k",
    "include_frame",
    "index_exp",
    "indices",
    "inexact",
    "inf",
    "infeed",
    "infer_lambda_input_type",
    "initialize",
    "initialize_pjrt_plugin",
    "inner",
    "insert",
    "inspect_array_sharding",
    "instantiate_const_at",
    "instantiate_zeros",
    "int",
    "int16",
    "int32",
    "int4",
    "int64",
    "int8",
    "int_",
    "integer",
    "interp",
    "intersect1d",
    "inv",
    "inverse_mdct",
    "inverse_stft",
    "inverse_stft_window_fn",
    "invert",
    "invert_permutation",
    "iou",
    "ir_constant",
    "ir_constants",
    "irfft",
    "irfft2d",
    "irfft3d",
    "irfftnd",
    "is_constant_dim",
    "is_constant_shape",
    "is_gpu",
    "is_known_platform",
    "is_non_decreasing",
    "is_op_sharding_replicated",
    "is_strictly_increasing",
    "is_tensor",
    "is_undefined_primal",
    "is_vmappable",
    "isclose",
    "iscomplex",
    "iscomplexobj",
    "isdtype",
    "isf",
    "isfinite",
    "isin",
    "isinf",
    "isnan",
    "isneginf",
    "isotonic_regression",
    "isposinf",
    "isreal",
    "isrealobj",
    "isscalar",
    "issubdtype",
    "istft",
    "iterable",
    "ix_",
    "jaxpr_as_fun",
    "jaxpr_subcomp",
    "jaxpr_uses_outfeed",
    "jaxprs_in_params",
    "join_effects",
    "join_named_shapes",
    "jvp",
    "jvp_jaxpr",
    "jvp_subtrace",
    "jvp_subtrace_aux",
    "jvpfun",
    "kaiser",
    "kaiser_bessel_derived_window",
    "kaiser_window",
    "kl_div",
    "kl_div_loss",
    "kron",
    "l1_loss",
    "l2_loss",
    "l2_normalize",
    "lattice_join",
    "lbeta",
    "lcm",
    "ldexp",
    "leaked_tracer_error",
    "leaky_relu",
    "learned_unigram_candidate_sampler",
    "leaves",
    "left_shift",
    "less",
    "less_equal",
    "lexsort",
    "lgamma",
    "linear_call",
    "linear_jvp",
    "linear_to_mel_weight_matrix",
    "linear_transpose",
    "linear_transpose2",
    "linearize",
    "linspace",
    "load",
    "load_pjrt_plugin_dynamically",
    "load_pjrt_plugin_with_c_api",
    "local_device_count",
    "local_devices",
    "local_response_normalization",
    "log",
    "log10",
    "log1p",
    "log2",
    "log_cosh_loss",
    "log_ndtr",
    "log_poisson_loss",
    "log_sigmoid",
    "log_softmax",
    "log_uniform_candidate_sampler",
    "logaddexp",
    "logaddexp2",
    "logcdf",
    "logdet",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "logit",
    "logpdf",
    "logpmf",
    "logsf",
    "logspace",
    "logsumexp",
    "lookup",
    "lower_fun",
    "lower_jaxpr_to_fun",
    "lower_jaxpr_to_module",
    "lpmn",
    "lpmn_values",
    "lru_cache",
    "lstm_cell",
    "lstsq",
    "lu",
    "lu_factor",
    "lu_solve",
    "make_c_api_client",
    "make_c_api_device_topology",
    "make_convolution_dimension_numbers",
    "make_cpu_client",
    "make_dot_dimension_numbers",
    "make_gpu_client",
    "make_iota",
    "make_ir_context",
    "make_jaxpr_effects",
    "make_padding_config",
    "make_pjrt_topology",
    "make_pjrt_tpu_topology",
    "make_replica_groups",
    "make_tfrt_tpu_c_api_client",
    "make_tfrt_tpu_c_api_device_topology",
    "make_tpu_client",
    "manual_seed",
    "map",
    "map_bind",
    "map_bind_with_continuation",
    "map_coordinates",
    "map_fn",
    "map_fn_eager",
    "map_fn_tracing",
    "map_transpose",
    "mapped_aval",
    "margin_ranking_loss",
    "mask_indices",
    "matchaxis",
    "math",
    "matmul",
    "matrix_exponential",
    "matrix_power",
    "matrix_transpose",
    "max",
    "max_dim",
    "max_pool",
    "maximum",
    "maybe_find_leaked_tracers",
    "maybe_named_axis",
    "mdct",
    "mean",
    "median",
    "median_filter",
    "mel_filterbank",
    "mel_spectrogram",
    "memoize",
    "merge_lists",
    "merge_mlir_modules",
    "meshgrid",
    "mfcc",
    "mfccs_from_log_mel_spectrograms",
    "mgrid",
    "min",
    "min_dim",
    "minimum",
    "mixup",
    "mod",
    "mode",
    "modf",
    "module_to_bytecode",
    "module_to_string",
    "moments",
    "move_binders_to_back",
    "move_binders_to_front",
    "moveaxis",
    "mse_loss",
    "multi_hot",
    "multigammaln",
    "multiply",
    "multiply_no_nan",
    "mvlgamma",
    "nan_to_num",
    "nanargmax",
    "nanargmin",
    "nancumprod",
    "nancumsum",
    "nanmean",
    "nanmedian",
    "nanpercentile",
    "nanquantile",
    "nanstd",
    "nanvar",
    "nce_loss",
    "ndarray",
    "ndim",
    "ndtr",
    "ndtri",
    "negative",
    "new_base_main",
    "new_eqn_recipe",
    "new_jaxpr_eqn",
    "new_main",
    "new_sublevel",
    "newaxis",
    "nextafter",
    "nll_loss",
    "non_max_suppression",
    "nonzero",
    "nonzero_outputs",
    "nonzero_tangent_outputs",
    "norm",
    "normalize",
    "normalize_moments",
    "not_equal",
    "num_available_tpu_chips_and_device_id",
    "number",
    "numpy",
    "object_",
    "ogrid",
    "one_hot",
    "ones",
    "ones_like",
    "op_sharding_to_indices",
    "outer",
    "outfeed",
    "overlap_and_add",
    "overload",
    "packbits",
    "pad",
    "pad_images",
    "pad_jaxpr",
    "pad_to_bounding_box",
    "parallel_callable",
    "partial_eval_jaxpr_custom",
    "partial_eval_jaxpr_custom_rule_not_implemented",
    "partial_eval_jaxpr_nounits",
    "partial_eval_wrapper_nounits",
    "partition",
    "partition_list",
    "partition_pvals",
    "pbroadcast",
    "pdf",
    "pdot",
    "percentile",
    "permute",
    "permute_dims",
    "perspective_transform",
    "pi",
    "piecewise",
    "pinv",
    "pjrt_plugin_initialized",
    "pjrt_plugin_loaded",
    "place",
    "pmap",
    "pmap_eager",
    "pmap_tracing",
    "pmax",
    "pmean",
    "pmf",
    "pmin",
    "poch",
    "polar",
    "poly",
    "polyadd",
    "polyder",
    "polydiv",
    "polyfit",
    "polygamma",
    "polyint",
    "polymul",
    "polysub",
    "polyval",
    "pool1d",
    "pool2d",
    "pool3d",
    "positive",
    "posterize",
    "pow",
    "power",
    "power_iteration",
    "ppermute",
    "ppf",
    "primal_dtype_to_tangent_dtype",
    "primitive_uses_outfeed",
    "printoptions",
    "process_count",
    "process_env_traces_call",
    "process_env_traces_map",
    "process_index",
    "prod",
    "promote_types",
    "pshuffle",
    "psnr",
    "psum",
    "psum_scatter",
    "pswapaxes",
    "ptp",
    "put",
    "qr",
    "quantile",
    "r_",
    "rad2deg",
    "radians",
    "raise_as_much_as_possible",
    "raise_to_shaped",
    "rand",
    "rand_augment",
    "randint",
    "randn",
    "random_color_jitter",
    "random_crop",
    "random_elastic_transform",
    "random_erasing",
    "random_flip",
    "random_gaussian_blur",
    "random_perspective",
    "random_rotation",
    "random_sharpness",
    "random_shear",
    "random_translation",
    "random_zoom",
    "rankdata",
    "ravel",
    "ravel_multi_index",
    "real",
    "rearrange",
    "rearrange_binders",
    "rebase_donate_argnums",
    "recast_to_float0",
    "recipe_to_eqn",
    "reciprocal",
    "reciprocal_no_nan",
    "record_event",
    "record_event_duration_secs",
    "reduce",
    "reduce_euclidean_norm",
    "reduce_logsumexp",
    "reduce_scatter",
    "reduce_window",
    "reducer_batcher",
    "regex_full_match",
    "regex_replace",
    "register_backend_factory",
    "register_constant_handler",
    "register_custom_call_handler",
    "register_custom_call_target",
    "register_event_duration_secs_listener",
    "register_event_listener",
    "register_exclusion",
    "register_lowering",
    "register_op",
    "register_pjrt_plugin_factories_from_env",
    "register_plugin",
    "register_plugin_callbacks",
    "register_vmappable",
    "rel_entr",
    "relu",
    "relu6",
    "remainder",
    "repeat",
    "replace_float0s",
    "reset_trace_state",
    "reshape",
    "resize",
    "resize_bicubic",
    "resize_bilinear",
    "resize_lanczos3",
    "resize_nearest",
    "result_info",
    "result_type",
    "reverse",
    "rfft",
    "rfft2d",
    "rfft3d",
    "rfftnd",
    "rgb_to_grayscale",
    "rgb_to_hsv",
    "rgb_to_yiq",
    "rgb_to_yuv",
    "right_shift",
    "rint",
    "rms_normalization",
    "rnn",
    "roll",
    "rollaxis",
    "roots",
    "rot90",
    "round",
    "round_",
    "rsf2csf",
    "rsqrt",
    "s_",
    "safe_embedding_lookup_sparse",
    "safe_map",
    "safe_zip",
    "sampled_softmax_loss",
    "saturate_cast",
    "save",
    "save_device_memory_profile",
    "savez",
    "scalar_mul",
    "scalar_type_of",
    "scale_and_translate",
    "scale_regularization_loss",
    "scan",
    "scan_bind",
    "scan_eager",
    "scan_tracing",
    "scatter",
    "scatter_add",
    "scatter_nd",
    "scatter_update",
    "searchsorted",
    "segment_max",
    "segment_mean",
    "segment_min",
    "segment_prod",
    "segment_sum",
    "select",
    "selu",
    "sem",
    "separable_conv",
    "separable_conv1d",
    "separable_conv2d",
    "sequential_vmap",
    "set_module",
    "set_printoptions",
    "setdiff1d",
    "setxor1d",
    "sf",
    "shape_from_pyval",
    "shape_tensor",
    "shaped_abstractify",
    "shard_args",
    "shard_tensor",
    "sharpen",
    "shutdown",
    "sig_info",
    "sigmoid",
    "sign",
    "signbit",
    "signedinteger",
    "silu",
    "simple_rnn_cell",
    "sin",
    "sinc",
    "single",
    "sinh",
    "size",
    "slice",
    "slice_update",
    "slogdet",
    "smooth_l1_loss",
    "sobol_sample",
    "soft_shrink",
    "softmax",
    "softplus",
    "softsign",
    "solarize",
    "solve",
    "solve_triangular",
    "sort",
    "sort_complex",
    "space_to_batch",
    "space_to_depth",
    "sparse_categorical_crossentropy",
    "sparse_plus",
    "sparse_sigmoid",
    "sparsemax",
    "spec_to_indices",
    "special",
    "spectral_normalization",
    "spence",
    "sph_harm",
    "split",
    "split_dict",
    "split_list",
    "split_list_checked",
    "split_merge",
    "sqrt",
    "square",
    "squared_difference",
    "squareplus",
    "squeeze",
    "stable_unique",
    "stack",
    "standard_jvp",
    "standard_jvp2",
    "start_server",
    "start_trace",
    "stash_axis_env",
    "std",
    "stft",
    "stop_gradient",
    "stop_gradient_eager",
    "stop_gradient_tracing",
    "stop_server",
    "stop_trace",
    "str_eqn_compact",
    "strided_slice",
    "string_join",
    "string_length",
    "string_lower",
    "string_split",
    "string_substr",
    "string_to_hash",
    "string_to_number",
    "string_upper",
    "structure",
    "subjaxprs",
    "subs_list",
    "subs_list2",
    "subst_axis_names",
    "subst_axis_names_eqn",
    "subst_axis_names_jaxpr",
    "subst_axis_names_var",
    "substitute_vars_in_output_ty",
    "subtract",
    "subvals",
    "sufficient_statistics",
    "sum",
    "svd",
    "swapaxes",
    "swish",
    "switch",
    "switch_case",
    "take",
    "take_along_axis",
    "tan",
    "tanh",
    "tanh_shrink",
    "tensor_scatter_add",
    "tensor_scatter_max",
    "tensor_scatter_min",
    "tensor_scatter_sub",
    "tensor_scatter_update",
    "tensordot",
    "text_vectorization",
    "threshold",
    "tile",
    "time_distributed",
    "to_dlpack",
    "to_elt",
    "toeplitz",
    "token_type",
    "top_k",
    "toposort",
    "tpu_client_timer_callback",
    "tpu_enhanced_barrier_supported",
    "trace",
    "trace_state_clean",
    "trace_to_jaxpr",
    "trace_to_jaxpr_dynamic",
    "trace_to_jaxpr_dynamic2",
    "trace_to_jaxpr_final",
    "trace_to_jaxpr_final2",
    "trace_to_jaxpr_nounits",
    "trace_to_subjaxpr",
    "trace_to_subjaxpr_dynamic",
    "trace_to_subjaxpr_dynamic2",
    "trace_to_subjaxpr_nounits",
    "trace_to_subjaxpr_nounits_dyn",
    "trace_to_subjaxpr_nounits_fwd",
    "traceable",
    "tracebacks",
    "tracers_to_jaxpr",
    "tracing",
    "transpose",
    "trapezoid",
    "traverse_jaxpr_params",
    "tri",
    "tril",
    "tril_indices",
    "tril_indices_from",
    "trim_zeros",
    "triplet_loss",
    "triu",
    "triu_indices",
    "triu_indices_from",
    "trivial_ctx",
    "true_divide",
    "truediv",
    "trunc",
    "truncatediv",
    "truncatemod",
    "tuple_delete",
    "tuple_insert",
    "tuple_update",
    "typecheck",
    "typecompat",
    "typematch",
    "ufunc",
    "uint",
    "uint16",
    "uint32",
    "uint4",
    "uint64",
    "uint8",
    "unflatten",
    "uniform_candidate_sampler",
    "union1d",
    "unique",
    "unique_all",
    "unique_counts",
    "unique_inverse",
    "unique_values",
    "unmapped_aval",
    "unpackbits",
    "unpair_pval",
    "unravel_index",
    "unregister_vmappable",
    "unsignedinteger",
    "unsorted_segment_max",
    "unsorted_segment_mean",
    "unsorted_segment_min",
    "unsorted_segment_prod",
    "unsorted_segment_sqrt_n",
    "unsorted_segment_sum",
    "unsqueeze",
    "unstack",
    "unwrap",
    "unzip2",
    "unzip3",
    "update_slice",
    "use_cpp_class",
    "use_cpp_method",
    "used_axis_names",
    "used_axis_names_jaxpr",
    "using_pjrt_c_api",
    "valid_jaxtype",
    "vander",
    "var",
    "variance",
    "vdot",
    "vecdot",
    "vectorize",
    "vectorized_batcher",
    "vectorized_map",
    "view_as_complex",
    "view_as_real",
    "visualize_array_sharding",
    "visualize_sharding",
    "vjp",
    "vmap",
    "vorbis_window",
    "vq",
    "vsplit",
    "vstack",
    "vtile",
    "weakref_lru_cache",
    "weighted_moments",
    "welch",
    "where",
    "while_loop",
    "while_loop_eager",
    "while_loop_tracing",
    "window_padding_type_to_pad_values",
    "with_sharding_constraint",
    "with_space_to_batch",
    "wrap_name",
    "wraps",
    "xdivy",
    "xla_computation_to_mlir_module",
    "xlog1py",
    "xlogy",
    "yiq_to_rgb",
    "yuv_to_rgb",
    "zero_fraction",
    "zero_jvp",
    "zeros",
    "zeros_like",
    "zeros_like_aval",
    "zeros_like_jaxval",
    "zeta",
]
