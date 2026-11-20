"""WASM Target Emission with Native v128 SIMD Intrinsics and Remainder Loop Peeling."""

from __future__ import annotations

import os

import yaml
from ml_switcheroo_ir import IRGraph

from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    coerce_int,
    coerce_str,
)
from ml_switcheroo_compiler.backends.edge._wasm_base import (
    WasmAttrType as WasmAttrType,
)
from ml_switcheroo_compiler.backends.edge._wasm_base import (
    _as_wasm_attr_val as _as_wasm_attr_val,
)
from ml_switcheroo_compiler.backends.edge._wasm_base import (
    _as_wasm_d1 as _as_wasm_d1,
)
from ml_switcheroo_compiler.backends.edge._wasm_base import (
    _as_wasm_leaf as _as_wasm_leaf,
)
from ml_switcheroo_compiler.backends.edge._wasm_base import (
    _node_attrs as _node_attrs,
)
from ml_switcheroo_compiler.backends.edge._wasm_base import (
    _str_list_attr as _str_list_attr,
)
from ml_switcheroo_compiler.backends.edge._wasm_base import (
    _WasmD1 as _WasmD1,
)
from ml_switcheroo_compiler.backends.edge._wasm_base import (
    _WasmGenBase as _WasmGenBase,
)
from ml_switcheroo_compiler.backends.edge._wasm_base import (
    _WasmLeaf as _WasmLeaf,
)
from ml_switcheroo_compiler.backends.edge._wasm_codegen_core import _WasmCodeGenCore as _WasmCodeGenCore
from ml_switcheroo_compiler.backends.protocols import adapt_backend_module
from ml_switcheroo_compiler.backends.registry import register_backend
from ml_switcheroo_compiler.core.errors import UnimplementedMathError as UnimplementedMathError


@register_backend("edge_wasm")
@register_backend("edge_wasm_simd")
class WasmCodeGenerator(_WasmCodeGenCore):
    """WASM Code Generator for emitting vectorizable, highly optimized WASM-SIMD C++ source code."""

    def compile_wasm(self, output_dir: str = ".") -> tuple[str, str] | None:
        """Automatically compile the generated C++ source code into WASM and JS binary files using emcc or clang.

        Args:
            output_dir (str): Directory where output files (.wasm and .js) should be saved.

        Returns:
            Optional[tuple[str, str]]: Paths to (js_loader_path, wasm_binary_path) if compilation succeeded, else None.

        Raises:
            CompilationError: If compilation or binary verification fails in `compile_wasm`.
        """
        import os
        import shutil
        import subprocess
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".cpp", delete=False, mode="w") as temp_file:
            temp_file.write(self.generate())
            temp_file_path: str = temp_file.name

        try:
            emcc_bin = shutil.which("emcc")
            if emcc_bin:
                js_out: str = os.path.join(output_dir, "kernel.js")
                wasm_out: str = os.path.join(output_dir, "kernel.wasm")
                cmd: list[str] = [
                    emcc_bin,
                    "-O3",
                    "-msimd128",
                    "-s",
                    "EXPORTED_FUNCTIONS=['_main_kernel']",
                    "-s",
                    "STANDALONE_WASM",
                    temp_file_path,
                    "-o",
                    js_out,
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                return js_out, wasm_out

            clang_bin = shutil.which("clang")
            if clang_bin:
                wasm_out = os.path.join(output_dir, "kernel.wasm")
                cmd = [
                    clang_bin,
                    "--target=wasm32",
                    "-O3",
                    "-msimd128",
                    "-nostdlib",
                    "-Wl,--no-entry",
                    "-Wl,--export=main_kernel",
                    temp_file_path,
                    "-o",
                    wasm_out,
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                return "", wasm_out

            from ml_switcheroo_compiler.core.errors import CompilationError

            raise CompilationError("Neither emcc nor clang found for WASM compilation.")

        except subprocess.CalledProcessError as e:
            from ml_switcheroo_compiler.core.errors import CompilationError

            err_msg: str = e.stderr.decode("utf-8") if e.stderr else str(e)
            raise CompilationError(f"WASM compilation failed: {err_msg}") from e
        except (OSError, subprocess.SubprocessError, ValueError, TypeError, RuntimeError) as e:
            from ml_switcheroo_compiler.core.errors import CompilationError

            raise CompilationError(f"WASM compilation failed with unknown error: {e}") from e
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def _compile_aot_impl(self, graph: IRGraph, **kwargs: BackendArg) -> tuple[str, str]:
        """Compile IRGraph into a standalone .wasm binary via compile_wasm.

        Args:
            graph (IRGraph): Target computational graph.
            **kwargs (BackendArg): Compiler options.

        Returns:
            tuple[str, str]: Paths to generated js loader and wasm binary files.
        """
        if graph != self.graph:
            self.graph = graph
            from ml_switcheroo_compiler.core.utils.graph_utils import topological_sort

            self.sorted_nodes = topological_sort(graph)

        out_dir = str(kwargs.get("output_dir", "."))
        res = self.compile_wasm(output_dir=out_dir)
        return res if res is not None else ("", "")

    def export_graph_payload(self) -> dict[str, list[str] | list[dict[str, str | list[str] | list[int]]]]:
        """Export serialized graph definition for browser-side WASM execution.

        Returns:
            dict[str, Union[list[str], list[dict[str, Union[str, list[str], list[int]]]]]]: Graph payload containing
                nodes, inputs, outputs, and shape metadata.
        """
        nodes_list: list[dict[str, str | list[str] | list[int]]] = []
        for node in self.sorted_nodes:
            raw_sm = adapt_backend_module(node).get_symbol("shape_metadata", ()) or ()
            sm_seq = raw_sm if isinstance(raw_sm, (list, tuple)) else ()
            nodes_list.append(
                {
                    "id": str(adapt_backend_module(node).get_symbol("id", "")),
                    "op_type": str(adapt_backend_module(node).get_symbol("op_type", "")),
                    "inputs": _str_list_attr(node, "inputs"),
                    "shape_metadata": [int(s) for s in sm_seq if isinstance(s, (int, float))],
                }
            )
        return {
            "inputs": _str_list_attr(self.graph, "inputs"),
            "outputs": _str_list_attr(self.graph, "outputs"),
            "nodes": nodes_list,
        }

    @staticmethod
    def _record_symbolic_dims_from_concrete(
        meta: tuple[BackendArg, ...] | list[BackendArg] | None,
        concrete: tuple[BackendArg, ...] | list[BackendArg],
        solved_env: dict[str, int],
    ) -> None:
        """Record solved symbolic dimension bindings from a node's existing metadata and concrete shape.

        Args:
            meta (tuple[BackendArg, ...] | list[BackendArg] | None): Existing symbolic shape metadata.
            concrete (tuple[BackendArg, ...] | list[BackendArg]): Concrete shape sequence.
            solved_env (dict[str, int]): Mutable environment mapping symbol names to solved integers.
        """
        if meta and len(meta) == len(concrete):
            for m_dim, c_dim in zip(meta, concrete):
                if adapt_backend_module(m_dim).has_symbol("node"):
                    solved_env[str(adapt_backend_module(m_dim).require_symbol("node"))] = int(coerce_int(c_dim))
                elif isinstance(m_dim, str):
                    solved_env[m_dim] = int(coerce_int(c_dim))

    @staticmethod
    def _evaluate_refined_dim(dim: BackendArg, solved_env: dict[str, int]) -> int | None:
        """Evaluate a single dimension entry against the solved symbolic environment.

        Args:
            dim (BackendArg): Symbolic or concrete dimension entry.
            solved_env (dict[str, int]): Mapping of symbol names to concrete integer sizes.

        Returns:
            int | None: Resolved integer dimension or None if unresolvable.
        """
        if adapt_backend_module(dim).has_symbol("node") and adapt_backend_module(
            adapt_backend_module(dim).require_symbol("node")
        ).has_symbol("eval"):
            try:
                return int(coerce_int(adapt_backend_module(dim).require_symbol("node").eval(solved_env)))
            except (
                AttributeError,
                ImportError,
                LookupError,
                OSError,
                ArithmeticError,
                RuntimeError,
                SyntaxError,
                TypeError,
                ValueError,
            ):
                return int(dim) if isinstance(dim, (int, float)) else 1
        if isinstance(dim, str) and dim in solved_env:
            return solved_env[dim]
        if isinstance(dim, (int, float)):
            return int(dim)
        return None

    def apply_shape_telemetry(self, telemetry: dict[str, BackendArg] | BackendArg) -> None:
        """Update IRGraph node shape metadata and refine symbolic dimensions from client telemetry.

        Args:
            telemetry (dict[str, BackendArg] | BackendArg): Mapping or payload containing learned concrete shapes.
        """
        shape_data = telemetry.get("shapes", telemetry) if isinstance(telemetry, dict) else {}
        if not isinstance(shape_data, dict):
            return

        solved_env: dict[str, int] = {}
        for node in self.sorted_nodes:
            nid: str = coerce_str(adapt_backend_module(node).get_symbol("id", ""))
            concrete = shape_data.get(nid)
            if isinstance(concrete, (list, tuple)):
                raw_meta = adapt_backend_module(node).get_symbol("shape_metadata", None)
                meta_seq = raw_meta if isinstance(raw_meta, (list, tuple)) else None
                self._record_symbolic_dims_from_concrete(meta_seq, concrete, solved_env)
                node.shape_metadata = tuple(int(x) for x in concrete if isinstance(x, (int, float)))

        for node in self.sorted_nodes:
            meta = adapt_backend_module(node).get_symbol("shape_metadata", None)
            if isinstance(meta, (list, tuple)):
                refined = [r for d in meta if (r := self._evaluate_refined_dim(d, solved_env)) is not None]
                if refined:
                    node.shape_metadata = tuple(refined)

    def generate_wat(self) -> str:
        """Generate valid WebAssembly Text (WAT) module with SIMD-128 instructions.

        Returns:
            str: WAT module string suitable for wat2wasm compilation.
        """
        yaml_path: str = os.path.join(os.path.dirname(__file__), "wasm_simd", "wasm_opcodes.yaml")
        op_map: dict[str, dict[str, BackendArg]] = {}
        if os.path.exists(yaml_path):
            with open(yaml_path, encoding="utf-8") as f:
                raw = yaml.safe_load(f)
            if isinstance(raw, dict) and "opcodes" in raw and isinstance(raw["opcodes"], dict):
                op_map = raw["opcodes"]

        nodes = [
            n for n in self.sorted_nodes if coerce_str(adapt_backend_module(n).get_symbol("op_type", "")) != "Input"
        ]
        wat_lines: list[str] = [
            "(module",
            '  (memory (export "memory") 1)',
            '  (func (export "compute") (param $in i32) (param $len i32) (param $out i32)',
            "    (local $i i32)",
            "    (local.set $i (i32.const 0))",
            "    (block $B",
            "      (loop $L",
            "        (br_if $B (i32.ge_u (local.get $i) (local.get $len)))",
        ]

        for node in nodes:
            op_type = coerce_str(adapt_backend_module(node).get_symbol("op_type", "")).lower()
            spec = op_map.get(op_type, {})
            simd_inst: str = str(spec.get("instruction", "f32x4.mul"))
            arity: int = int(
                coerce_int(
                    spec.get(
                        "arity",
                        1
                        if simd_inst
                        in (
                            "f32x4.neg",
                            "f32x4.sqrt",
                            "f32x4.abs",
                            "f32x4.ceil",
                            "f32x4.floor",
                            "f32x4.trunc",
                            "f32x4.nearest",
                        )
                        else 2,
                    )
                )
            )

            if arity == 1:
                wat_lines.extend(
                    [
                        "        (v128.store",
                        "          (i32.add (local.get $out) (i32.shl (local.get $i) (i32.const 2)))",
                        f"          ({simd_inst}",
                        "            (v128.load (i32.add (local.get $in) (i32.shl (local.get $i) (i32.const 2))))",
                        "          )",
                        "        )",
                    ]
                )
            else:
                wat_lines.extend(
                    [
                        "        (v128.store",
                        "          (i32.add (local.get $out) (i32.shl (local.get $i) (i32.const 2)))",
                        f"          ({simd_inst}",
                        "            (v128.load (i32.add (local.get $in) (i32.shl (local.get $i) (i32.const 2))))",
                        "            (v128.load (i32.add (local.get $in) (i32.shl (local.get $i) (i32.const 2))))",
                        "          )",
                        "        )",
                    ]
                )

        wat_lines.extend(
            [
                "        (local.set $i (i32.add (local.get $i) (i32.const 4)))",
                "        (br $L)",
                "      )",
                "    )",
                "  )",
                ")",
            ]
        )
        return "\n".join(wat_lines)
