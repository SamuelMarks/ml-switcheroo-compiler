const test = require('node:test');
const assert = require('node:assert');
const { execSync } = require('child_process');
const fs = require('fs');
const http = require('http');
let puppeteer;
try {
    puppeteer = require('puppeteer');
} catch {
    puppeteer = null;
}

test('WebGPU can evaluate AD gradient graphs (forward, VJP, JVP)', async (t) => {
    if (!puppeteer) {
        t.skip('puppeteer not installed in node_modules');
        return;
    }
    const chromePath = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
    if (!fs.existsSync(chromePath)) {
        t.skip("Chrome not found for WebGPU testing");
        return;
    }

    const script = `
import ml_switcheroo_compiler as compiler
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.backends.edge.webgpu import WebGPUCodeGenerator
from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.ops.control_flow_utils import _trace_function
from ml_switcheroo_compiler.transforms.autodiff import grad as graph_grad, jvp as graph_jvp
from ml_switcheroo_compiler.ir.core import LogicalGraph, IRNode

def simple_fn(x):
    return x * x

x = Tensor([3.0], TensorConfig((1,), "float32", "cpu"))

with ConfigContext(backend="numpy"):
    # trace forward pass
    block = _trace_function(simple_fn, (x,), "fwd")
    fwd_graph = LogicalGraph(name="fwd")
    nodes_iter = block.nodes.values() if isinstance(block.nodes, dict) else block.nodes
    for node in nodes_iter:
        fwd_graph.nodes[node.id] = node
    fwd_graph.inputs = block.inputs
    fwd_graph.outputs = block.outputs

    # Get VJP (reverse-mode grad) graph
    vjp_graph = graph_grad(fwd_graph, fwd_graph.inputs, fwd_graph.outputs[0])
    vjp_graph.inputs = [nid for nid, n in vjp_graph.nodes.items() if n.op_type == "Input"]

    # Get JVP (forward-mode grad) graph
    tan_node = IRNode(id="tan_x", op_type="Input")
    fwd_graph.nodes["tan_x"] = tan_node
    jvp_graph = graph_jvp(fwd_graph, fwd_graph.inputs, ["tan_x"], fwd_graph.outputs)
    jvp_graph.inputs = [fwd_graph.inputs[0], "tan_x"]

    fwd_code = WebGPUCodeGenerator(fwd_graph).generate()
    vjp_code = WebGPUCodeGenerator(vjp_graph).generate()
    jvp_code = WebGPUCodeGenerator(jvp_graph).generate()

    with open("temp_fwd_webgpu.js", "w") as f:
        f.write(fwd_code)
    with open("temp_vjp_webgpu.js", "w") as f:
        f.write(vjp_code)
    with open("temp_jvp_webgpu.js", "w") as f:
        f.write(jvp_code)
`;

    fs.writeFileSync('temp_webgpu_test.py', script);
    execSync('python3 temp_webgpu_test.py');
    fs.unlinkSync('temp_webgpu_test.py');

    const fwdCode = fs.readFileSync('temp_fwd_webgpu.js', 'utf8');
    const vjpCode = fs.readFileSync('temp_vjp_webgpu.js', 'utf8');
    const jvpCode = fs.readFileSync('temp_jvp_webgpu.js', 'utf8');
    fs.unlinkSync('temp_fwd_webgpu.js');
    fs.unlinkSync('temp_vjp_webgpu.js');
    fs.unlinkSync('temp_jvp_webgpu.js');

    assert.ok(fwdCode.includes('var<storage, read>'));
    assert.ok(vjpCode.includes('device.createCommandEncoder()'));

    // Execute on real WebGPU in headless Chrome
    const server = http.createServer((req, res) => {
        res.writeHead(200, { 'Content-Type': 'text/html' });
        res.end('<!DOCTYPE html><html><body></body></html>');
    });

    await new Promise((resolve) => server.listen(8984, resolve));

    try {
        const browser = await puppeteer.launch({
            executablePath: chromePath,
            headless: 'new',
            args: [
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--enable-unsafe-webgpu',
                '--use-angle=swiftshader'
            ]
        });
        const page = await browser.newPage();
        await page.goto('http://localhost:8984');

        const results = await page.evaluate(async (fwdSrc, vjpSrc, jvpSrc) => {
            const adapter = await navigator.gpu.requestAdapter();
            const device = await adapter.requestDevice();

            // Run forward pass: f(3.0) = 9.0
            const fwdFn = new Function('device', 'inputs', fwdSrc + '\nreturn execute(device, inputs);');
            const fwdMatches = Array.from(fwdSrc.matchAll(/inputs\["([^"]+)"\]/g));
            const fwdInputs = {};
            for (const m of fwdMatches) fwdInputs[m[1]] = new Float32Array([3.0]);
            const fwdRes = await fwdFn(device, fwdInputs);

            // Run VJP pass: df/dx at 3.0 = 2 * 3.0 = 6.0
            const vjpFn = new Function('device', 'inputs', vjpSrc + '\nreturn execute(device, inputs);');
            const vjpMatches = Array.from(vjpSrc.matchAll(/inputs\["([^"]+)"\]/g));
            const vjpInputs = {};
            for (const m of vjpMatches) vjpInputs[m[1]] = new Float32Array([3.0]);
            const vjpRes = await vjpFn(device, vjpInputs);

            // Run JVP pass: df/dx * v at 3.0 with v=1.0 = 2 * 3.0 * 1.0 = 6.0
            const jvpFn = new Function('device', 'inputs', jvpSrc + '\nreturn execute(device, inputs);');
            const jvpMatches = Array.from(jvpSrc.matchAll(/inputs\["([^"]+)"\]/g));
            const jvpInputs = {};
            for (const m of jvpMatches) {
                if (m[1] === 'tan_x') {
                    jvpInputs[m[1]] = new Float32Array([1.0]);
                } else {
                    jvpInputs[m[1]] = new Float32Array([3.0]);
                }
            }
            const jvpRes = await jvpFn(device, jvpInputs);

            return {
                fwdVal: Array.from(Object.values(fwdRes)[0])[0],
                vjpVal: Array.from(Object.values(vjpRes)[0])[0],
                jvpVal: Array.from(Object.values(jvpRes)[0])[0]
            };
        }, fwdCode, vjpCode, jvpCode);

        // Numerically assert WebGPU forward pass against reference NumPy outputs: 3.0 * 3.0 = 9.0
        assert.strictEqual(results.fwdVal, 9.0);

        // Numerically assert WebGPU reverse-mode (VJP) gradient values: 2 * 3.0 = 6.0
        assert.strictEqual(results.vjpVal, 6.0);

        // Numerically assert WebGPU forward-mode (JVP) gradient values: 2 * 3.0 * 1.0 = 6.0
        assert.strictEqual(results.jvpVal, 6.0);

        console.log("WebGPU AD Graph numerical verification successful: forward=9.0, VJP=6.0, JVP=6.0");
        await browser.close();
    } catch (err) {
        console.error("Test error:", err);
        throw err;
    } finally {
        if (server.closeAllConnections) {
            server.closeAllConnections();
        }
        server.close();
    }
});
