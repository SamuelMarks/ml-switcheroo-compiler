const test = require('node:test');
const assert = require('node:assert');
const { execSync } = require('child_process');
const fs = require('fs');

test('WASM can evaluate AD gradient graphs', async (t) => {
    const script = `
import ml_switcheroo_compiler as compiler
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.ops.control_flow_utils import _trace_function
from ml_switcheroo_compiler.transforms.autodiff import grad as graph_grad
from ml_switcheroo_compiler.ir.core import LogicalGraph

def simple_fn(x):
    return x * x

x = Tensor([2.0, 3.0, 4.0, 5.0], TensorConfig((4,), "float32", "cpu"))

with ConfigContext(backend="numpy"):
    # trace forward pass
    block = _trace_function(simple_fn, (x,), "fwd")
    fwd_graph = LogicalGraph(name="fwd")
    nodes_iter = block.nodes.values() if isinstance(block.nodes, dict) else block.nodes
    for node in nodes_iter:
        fwd_graph.nodes[node.id] = node
    fwd_graph.inputs = block.inputs
    fwd_graph.outputs = block.outputs

    # Get grad graph
    grad_graph = graph_grad(fwd_graph, fwd_graph.inputs, fwd_graph.outputs[0])

    gen = WasmCodeGenerator(grad_graph)
    code = gen.generate()

    with open("temp_wasm_code.js", "w") as f:
        f.write(code)

    # Directly emit WAT module from compiler
    fwd_wat = WasmCodeGenerator(fwd_graph).generate_wat()
    bwd_wat = WasmCodeGenerator(grad_graph).generate_wat()
    with open("temp_fwd_simd.wat", "w") as f:
        f.write(fwd_wat)
    with open("temp_bwd_simd.wat", "w") as f:
        f.write(bwd_wat)
`;

    fs.writeFileSync('temp_wasm_test.py', script);
    execSync('python3 temp_wasm_test.py');
    fs.unlinkSync('temp_wasm_test.py');

    const wasmCppCode = fs.readFileSync('temp_wasm_code.js', 'utf8');
    fs.unlinkSync('temp_wasm_code.js');

    assert.ok(wasmCppCode.includes('extern "C"'));

    // Automated Compilation: Compile compiler-emitted WAT binaries using wat2wasm
    execSync('wat2wasm temp_fwd_simd.wat -o temp_fwd_simd.wasm');
    execSync('wat2wasm temp_bwd_simd.wat -o temp_bwd_simd.wasm');

    const fwdBytes = fs.readFileSync('temp_fwd_simd.wasm');
    const bwdBytes = fs.readFileSync('temp_bwd_simd.wasm');
    fs.unlinkSync('temp_fwd_simd.wat');
    fs.unlinkSync('temp_bwd_simd.wat');
    fs.unlinkSync('temp_fwd_simd.wasm');
    fs.unlinkSync('temp_bwd_simd.wasm');

    // Numerically verify forward pass: inputs [2.0, 3.0, 4.0, 5.0]
    const fwdInstance = (await WebAssembly.instantiate(fwdBytes)).instance;
    const fwdMem = new Float32Array(fwdInstance.exports.memory.buffer);
    fwdMem.set([2.0, 3.0, 4.0, 5.0], 0);
    fwdInstance.exports.compute(0, 4, 32); // output at offset 32 bytes (index 8)
    const fwdResult = Array.from(fwdMem.slice(8, 12));

    // Analytical NumPy reference: [2^2, 3^2, 4^2, 5^2] = [4.0, 9.0, 16.0, 25.0]
    assert.deepStrictEqual(fwdResult, [4.0, 9.0, 16.0, 25.0]);

    // Numerically verify backward pass (VJP gradient): 2x = [4.0, 6.0, 8.0, 10.0]
    const bwdInstance = (await WebAssembly.instantiate(bwdBytes)).instance;
    const bwdMem = new Float32Array(bwdInstance.exports.memory.buffer);
    bwdMem.set([2.0, 3.0, 4.0, 5.0], 0);
    bwdInstance.exports.compute(0, 4, 32);
    const bwdResult = Array.from(bwdMem.slice(8, 12));

    // Analytical reference: 2 * x = [4.0, 6.0, 8.0, 10.0]
    assert.deepStrictEqual(bwdResult, [4.0, 6.0, 8.0, 10.0]);

    console.log("WASM AD Graph numerical verification successful with real SIMD-128 instructions.");
});

test('WASM AD Graph with dynamic shape learning feedback loop on arbitrary tensor shapes', async () => {
    const script = `
import ml_switcheroo_compiler as compiler
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.ops.control_flow_utils import _trace_function
from ml_switcheroo_compiler.transforms.autodiff import grad as graph_grad
from ml_switcheroo_compiler.ir.core import LogicalGraph
from ml_switcheroo_compiler.transforms.passes.shape_inference import shape_inference_pass, annotate_learned_shapes

def simple_fn(x):
    return x * x

x = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], TensorConfig((8,), "float32", "cpu"))

with ConfigContext(backend="numpy"):
    block = _trace_function(simple_fn, (x,), "fwd8")
    graph = LogicalGraph(name="fwd8")
    nodes_iter = block.nodes.values() if isinstance(block.nodes, dict) else block.nodes
    for node in nodes_iter:
        graph.nodes[node.id] = node
    graph.inputs = block.inputs
    graph.outputs = block.outputs

    # Simulate dynamic shape feedback loop
    shape_inference_pass(graph)
    annotate_learned_shapes(graph, {"fwd8_0": [8], "fwd8_1": [8]})

    grad_graph = graph_grad(graph, graph.inputs, graph.outputs[0])
    gen = WasmCodeGenerator(grad_graph)
    wat = gen.generate_wat()
    with open("temp_arbitrary_shape.wat", "w") as f:
        f.write(wat)
`;
    fs.writeFileSync('temp_arbitrary_test.py', script);
    execSync('python3 temp_arbitrary_test.py');
    fs.unlinkSync('temp_arbitrary_test.py');

    execSync('wat2wasm temp_arbitrary_shape.wat -o temp_arbitrary_shape.wasm');
    const wasmBytes = fs.readFileSync('temp_arbitrary_shape.wasm');
    fs.unlinkSync('temp_arbitrary_shape.wat');
    fs.unlinkSync('temp_arbitrary_shape.wasm');

    const instance = (await WebAssembly.instantiate(wasmBytes)).instance;
    const mem = new Float32Array(instance.exports.memory.buffer);
    mem.set([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 0);
    // Execute gradient across 8 elements: write to offset 64 bytes (index 16)
    instance.exports.compute(0, 8, 64);
    const gradResult = Array.from(mem.slice(16, 24));
    assert.deepStrictEqual(gradResult, [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);
});
