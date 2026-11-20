"""Browser JavaScript memory arena, bind groups, compute pass, and readback emitter."""

from __future__ import annotations

import typing  # noqa: F401

from ml_switcheroo_ir import IRNode

from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    coerce_float,
    coerce_int,
    coerce_str,
    require_callable,
)
from ml_switcheroo_compiler.backends.edge._webgpu_base import (
    _as_wgsl_visit_result,
    _node_attrs_dict,
    _node_inputs_list,
)
from ml_switcheroo_compiler.backends.edge._webgpu_collectives_cf import _WebGPUCollectivesCF
from ml_switcheroo_compiler.backends.protocols import adapt_backend_module


class _WebGPUJsOrchestrator(_WebGPUCollectivesCF):
    """WebGPU JavaScript host orchestration and arena buffer planner."""

    def _compute_max_io_counts(self) -> tuple[int, int]:
        """Compute maximum input and output tensor counts across all graph nodes.

        Args:
            self (WebGPUCodeGenerator): The code generator instance.

        Returns:
            tuple[int, int]: Maximum number of inputs (at least 3) and outputs (at least 1).
        """
        all_inputs_lens: list[int] = [len(_node_inputs_list(n)) for n in self.sorted_nodes]
        max_inputs: int = max(all_inputs_lens + [3]) if all_inputs_lens else 3
        max_outputs: int = 1
        for n in self.sorted_nodes:
            raw_outs = adapt_backend_module(n).get_symbol("outputs", None) or _node_attrs_dict(n).get("outputs")
            if isinstance(raw_outs, (list, tuple)):
                max_outputs = max(max_outputs, len(raw_outs))
        return max_inputs, max_outputs

    def _build_arena_allocations(self) -> tuple[dict[str, str], dict[str, int]]:
        """Map graph nodes to shared memory arenas and compute total byte size per arena.

        Args:
            self (WebGPUCodeGenerator): The code generator instance.

        Returns:
            tuple[dict[str, str], dict[str, int]]: Node-to-arena mapping and arena-to-byte-size mapping.
        """
        node_arenas: dict[str, str] = {}
        for idx, node in enumerate(self.sorted_nodes):
            nid: str = coerce_str(adapt_backend_module(node).get_symbol("id", ""))
            raw_arena = _node_attrs_dict(node).get("buffer_id")
            if raw_arena is not None:
                arena_id: str = str(raw_arena)
            elif coerce_str(adapt_backend_module(node).get_symbol("op_type", "")) == "Output" and _node_inputs_list(
                node
            ):
                arena_id = node_arenas.get(node.inputs[0], str(idx))
            else:
                arena_id = str(idx)
            node_arenas[nid] = arena_id

        arenas: dict[str, int] = {}
        for node in self.sorted_nodes:
            if coerce_str(adapt_backend_module(node).get_symbol("op_type", "")) == "Output":
                continue
            nid = coerce_str(adapt_backend_module(node).get_symbol("id", ""))
            arena_id = node_arenas.get(nid, "0")
            offset: int = int(coerce_int(_node_attrs_dict(node).get("buffer_offset", 0)))
            shape, _ = self._get_shape_and_strides(node)
            size: int = self._num_elements(shape) * 4 if shape else 4
            arenas[arena_id] = max(arenas.get(arena_id, 0), offset + size)
        return node_arenas, arenas

    def _emit_js_arena_and_input_setup(
        self,
        js: list[str],
        node_arenas: dict[str, str],
        arenas: dict[str, int],
        output_ids: list[str],
        webrtc_ops_found: bool,
    ) -> None:
        """Emit JS arena allocations, optional WebRTC init, input writes, and staging buffers.

        Args:
            js (list[str]): Accumulator list of JavaScript orchestration lines.
            node_arenas (dict[str, str]): Mapping from node ID to arena ID.
            arenas (dict[str, int]): Mapping from arena ID to required byte size.
            output_ids (list[str]): Graph output node IDs.
            webrtc_ops_found (bool): Whether distributed WebRTC collective ops are present.

        Returns:
            None: Appends initialization statements to js in-place.
        """
        from ml_switcheroo_compiler.backends.edge.webgpu_webrtc import emit_webrtc_init
        from ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider import get_js_orchestration_template

        js.append("  // Allocate shared storage arenas")
        for arena_id, total_size in arenas.items():
            js.append(get_js_orchestration_template("allocate_arena").format(arena_id=arena_id, total_size=total_size))

        if webrtc_ops_found:
            init_str: str = emit_webrtc_init()
            if init_str:
                js.append("  // WebRTC Distributed Initialization")
                for line in init_str.split("\n"):
                    js.append("  " + line)

        js.append("  // Write inputs to arenas")
        for node in self.sorted_nodes:
            if coerce_str(adapt_backend_module(node).get_symbol("op_type", "")) == "Input":
                nid: str = coerce_str(adapt_backend_module(node).get_symbol("id", ""))
                arena_id = node_arenas.get(nid, "0")
                offset: int = int(coerce_int(_node_attrs_dict(node).get("buffer_offset", 0)))
                js.append(
                    get_js_orchestration_template("write_input").format(nid=nid, arena_id=arena_id, offset=offset)
                )

        js.append("")
        for i, out_id in enumerate(output_ids):
            out_node: IRNode | None = next(
                (n for n in self.sorted_nodes if adapt_backend_module(n).get_symbol("id", None) == out_id), None
            )
            shape, _ = self._get_shape_and_strides(out_node) if out_node else ([], [])
            nelem: int = self._num_elements(shape) if shape else 1
            js.append(get_js_orchestration_template("create_staging").format(i=i, size=nelem * 4))

    def _build_node_bind_entries(
        self,
        node: IRNode,
        inputs: list[str],
        node_arenas: dict[str, str],
    ) -> list[str]:
        """Build JavaScript bind group entry strings for a node's inputs and outputs.

        Args:
            node (IRNode): The operation node.
            inputs (list[str]): Input node IDs for the operation.
            node_arenas (dict[str, str]): Mapping from node ID to arena ID.

        Returns:
            list[str]: Formatted JS bind group resource entry literals.
        """
        entries: list[str] = []
        for j, inp in enumerate(inputs):
            inp_arena_id: str = node_arenas.get(inp, "0")
            b_idx: int = self._get_binding_for_input(j)
            entries.append(f"{{ binding: {b_idx}, resource: {{ buffer: buf_arena_{inp_arena_id} }} }}")

        raw_outs = _node_attrs_dict(node).get("outputs") or adapt_backend_module(node).get_symbol("outputs", None)
        out_nodes: list[str] = (
            [str(o) for o in raw_outs]
            if isinstance(raw_outs, (list, tuple))
            else [coerce_str(adapt_backend_module(node).get_symbol("id", "0"))]
        )

        for k, out_id in enumerate(out_nodes):
            out_arena_id: str = node_arenas.get(out_id, node_arenas.get(node.id, "0"))
            b_idx = self._get_binding_for_output(k)
            entries.append(f"{{ binding: {b_idx}, resource: {{ buffer: buf_arena_{out_arena_id} }} }}")
        return entries

    def _emit_compute_passes(
        self,
        wgsl: list[str],
        js: list[str],
        node_arenas: dict[str, str],
    ) -> None:
        """Emit WGSL compute shaders and JavaScript compute pass dispatches for all active nodes.

        Args:
            wgsl (list[str]): Accumulator list of WGSL shader lines.
            js (list[str]): Accumulator list of JavaScript orchestration lines.
            node_arenas (dict[str, str]): Mapping from node ID to arena ID.

        Returns:
            None: Appends WGSL kernels and JS compute passes in-place.
        """
        from ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider import get_js_orchestration_template

        js.append("  const commandEncoder = device.createCommandEncoder();")

        for node in self.sorted_nodes:
            op_type: str = coerce_str(adapt_backend_module(node).get_symbol("op_type", ""))
            resolved_op: str = self._resolve_wgsl_op_name(op_type)
            if resolved_op in ("Input", "Output"):
                continue
            nid: str = coerce_str(adapt_backend_module(node).get_symbol("id", ""))
            clean_id: str = nid.replace("-", "_")

            inputs: list[str] = _node_inputs_list(node)
            shape, _ = self._get_shape_and_strides(node)
            nelem: int = self._num_elements(shape) if shape else 1

            visitor = adapt_backend_module(self).get_symbol(f"visit_{op_type}", None)
            if visitor is not None:
                op_wgsl, dispatch_x, dispatch_y, dispatch_z = _as_wgsl_visit_result(
                    require_callable(visitor)(node, inputs, shape=shape, nelem=nelem, clean_id=clean_id)
                )
            else:
                op_wgsl, dispatch_x, dispatch_y, dispatch_z = self._get_wgsl_for_op(node, shape, nelem, clean_id)
            wgsl.extend(op_wgsl)

            entries: list[str] = self._build_node_bind_entries(node, inputs, node_arenas)
            js.append(
                get_js_orchestration_template("compute_pass").format(
                    clean_id=clean_id,
                    entries=", ".join(entries),
                    dispatch_x=dispatch_x,
                    dispatch_y=dispatch_y,
                    dispatch_z=dispatch_z,
                )
            )
            js.append("")

    def _emit_dynamic_memory_orchestration(self, js: list[str]) -> None:
        """Emit JavaScript dynamic runtime offset calculations and buffer resize logic.

        Args:
            js (list[str]): Accumulator list of JavaScript orchestration lines.

        Returns:
            None: Appends dynamic memory orchestration lines to js in-place.
        """
        import os

        import yaml

        from ml_switcheroo_compiler.backends.edge.config_models import MemorySchemasConfig

        raw_g_attrs = adapt_backend_module(self.graph).get_symbol("attributes", {})
        graph_attrs: dict[str, BackendArg] = (
            {str(k): v for k, v in raw_g_attrs.items()} if isinstance(raw_g_attrs, dict) else {}
        )
        raw_schema = graph_attrs.get("dynamic_memory_schema", {})
        dynamic_schema: dict[str, BackendArg] = dict(raw_schema) if isinstance(raw_schema, dict) else {}
        raw_offsets = dynamic_schema.get("dynamic_offsets", [])
        dynamic_offsets: list[dict[str, str]] = (
            [{str(k): str(v) for k, v in item.items()} for item in raw_offsets if isinstance(item, dict)]
            if isinstance(raw_offsets, list)
            else []
        )
        if not dynamic_offsets:
            return

        yaml_path: str = os.path.join(os.path.dirname(__file__), "memory_schemas.yaml")
        if not os.path.exists(yaml_path):
            return

        with open(yaml_path) as f:
            schemas: dict[str, BackendArg] = MemorySchemasConfig(**yaml.safe_load(f)).model_dump()

        raw_schemas_sub = schemas.get("schemas", {})
        raw_tpls = raw_schemas_sub.get("js_orchestration_templates", {}) if isinstance(raw_schemas_sub, dict) else {}
        tpls: dict[str, str] = {str(k): str(v) for k, v in raw_tpls.items()} if isinstance(raw_tpls, dict) else {}
        resize_tpl: str = tpls.get("dynamic_resize", "")
        offset_tpl: str = tpls.get("runtime_offset_calc", "")

        js.append("  // Dynamic Runtime Offsets")
        js.append("  let current_offset = 0;")
        total_computed_size: list[str] = []
        for entry in dynamic_offsets:
            js.append(
                offset_tpl.format(
                    var_name=entry["var_name"], symbolic_math=entry["symbolic_math"], byte_alignment=4
                ).strip()
            )
            total_computed_size.append(f"({entry['symbolic_math']} * 4)")

        if resize_tpl:
            computed_size_str: str = " + ".join(total_computed_size) if total_computed_size else "0"
            resize_logic: str = (
                "device.createBuffer({ size: new_cap, usage: GPUBufferUsage.STORAGE | "
                "GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });"
            )
            default_sub = raw_schemas_sub.get("default", {}) if isinstance(raw_schemas_sub, dict) else {}
            growth_raw = default_sub.get("growth_multiplier", 1.5) if isinstance(default_sub, dict) else 1.5
            growth: float = float(coerce_float(growth_raw))
            resize_block: str = resize_tpl.format(
                computed_size=computed_size_str, arena_id=0, growth_multiplier=growth, resize_logic=resize_logic
            )
            for line in resize_block.split("\n"):
                js.append("  " + line)

    def _emit_webrtc_and_output_readback(
        self,
        js: list[str],
        node_arenas: dict[str, str],
        output_ids: list[str],
        webrtc_ops_found: bool,
    ) -> None:
        """Emit WebRTC collective operations, staging buffer copies, queue submission, and readback.

        Args:
            js (list[str]): Accumulator list of JavaScript orchestration lines.
            node_arenas (dict[str, str]): Mapping from node ID to arena ID.
            output_ids (list[str]): Graph output node IDs.
            webrtc_ops_found (bool): Whether distributed WebRTC collective ops are present.

        Returns:
            None: Appends output readback and return statements to js in-place.
        """
        from ml_switcheroo_compiler.backends.edge.webgpu_webrtc import emit_webrtc_op
        from ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider import get_js_orchestration_template

        if webrtc_ops_found:
            collective_ops: tuple[str, ...] = ("AllReduce", "AllGather", "AllToAll", "ReduceScatter", "Broadcast")
            for node in self.sorted_nodes:
                op_type: str = coerce_str(adapt_backend_module(node).get_symbol("op_type", ""))
                if op_type in collective_ops:
                    op_id: str = coerce_str(adapt_backend_module(node).get_symbol("id", ""))
                    node_inputs: list[str] = _node_inputs_list(node)
                    in0: str = node_inputs[0] if node_inputs else "dummy"
                    op_str: str = emit_webrtc_op(op_type, f"buf_arena_{in0}", op_id)
                    if op_str:
                        js.append(f"  // Collectives {op_type}")
                        for line in op_str.split("\n"):
                            js.append("  " + line)

        js.append("  // Copy outputs to staging")
        for i, out_id in enumerate(output_ids):
            out_node: IRNode | None = next(
                (n for n in self.sorted_nodes if adapt_backend_module(n).get_symbol("id", None) == out_id), None
            )
            arena_id: str = node_arenas.get(out_id, "0")
            offset: int = int(coerce_int(_node_attrs_dict(out_node).get("buffer_offset", 0))) if out_node else 0
            shape, _ = self._get_shape_and_strides(out_node) if out_node else ([], [])
            nelem: int = self._num_elements(shape) if shape else 1
            js.append(
                get_js_orchestration_template("copy_output").format(
                    arena_id=arena_id, offset=offset, i=i, size=nelem * 4
                )
            )

        js.append("  device.queue.submit([commandEncoder.finish()]);")

        ret_entries: list[str] = []
        for i, out_id in enumerate(output_ids):
            js.append(get_js_orchestration_template("read_output").format(i=i))
            ret_entries.append(f'    "{out_id}": out_{i}_array,')

        js.append(get_js_orchestration_template("return_dict").format(returns="\n".join(ret_entries)))
        js.append("}")

    def generate(self) -> str:
        r"""Generate WebGPU WGSL compute shader module code enclosed in a JavaScript orchestrator.

        Returns:
            str: Complete, executable JavaScript orchestration code wrapper around WGSL compute shader.
        """
        from ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider import get_js_orchestration_template

        raw_outs = adapt_backend_module(self.graph).get_symbol("outputs", []) or []
        output_ids: list[str] = [str(x) for x in raw_outs] if isinstance(raw_outs, (list, tuple)) else []
        max_inputs, max_outputs = self._compute_max_io_counts()

        wgsl: list[str] = [self._generate_bind_group_declarations(max_inputs, max_outputs), ""]
        js: list[str] = [get_js_orchestration_template("init")]

        node_arenas, arenas = self._build_arena_allocations()
        collective_ops: tuple[str, ...] = ("AllReduce", "AllGather", "AllToAll", "ReduceScatter", "Broadcast")
        webrtc_ops_found: bool = any(
            coerce_str(adapt_backend_module(n).get_symbol("op_type", "")) in collective_ops for n in self.sorted_nodes
        )

        self._emit_js_arena_and_input_setup(js, node_arenas, arenas, output_ids, webrtc_ops_found)
        self._emit_compute_passes(wgsl, js, node_arenas)
        self._emit_dynamic_memory_orchestration(js)
        self._emit_webrtc_and_output_readback(js, node_arenas, output_ids, webrtc_ops_found)

        js_str: str = "\n".join(js)
        wgsl_str: str = "\n".join(wgsl)
        return f"const shaderCode = `{wgsl_str}`;\n{js_str}"
