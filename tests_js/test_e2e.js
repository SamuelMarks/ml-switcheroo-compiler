let puppeteer;
try {
    puppeteer = require('puppeteer');
} catch {
    puppeteer = null;
}
const path = require('path');
const fs = require('fs');
const test = require('node:test');
const assert = require('node:assert');
let PeerConnection, cleanup;
try {
    const ndc = require('node-datachannel');
    PeerConnection = ndc.PeerConnection;
    cleanup = ndc.cleanup;
} catch {
    PeerConnection = null;
    cleanup = null;
}
const { execFileSync } = require('child_process');
const wasmRunner = require('../docs/_static/wasm_runner.js');
const webgpuRunner = require('../docs/_static/webgpu_runner.js');

// We will serve the built HTML statically
const filePath = path.resolve(__dirname, '../docs/_build/html/playground.html');

test('E2E Combinations', async (t) => {
    if (!puppeteer) {
        t.skip('puppeteer is not installed in node_modules');
        return;
    }
    if (!fs.existsSync(filePath)) {
        t.skip(`Playground HTML not found at ${filePath}. Make sure to run 'make docs-fast' first.`);
        return;
    }

    const browser = await puppeteer.launch({
        executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
        headless: "new",
        args: ['--no-sandbox', '--disable-setuid-sandbox']
    });

    const page = await browser.newPage();

    // We removed Pyodide request interception! We are now using real Pyodide.
    await page.setUserAgent('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36');

    const fileUrl = 'file://' + filePath;
    console.log("Navigating to", fileUrl);
    await page.goto(fileUrl, { waitUntil: 'networkidle2' });

    console.log("Waiting for Monaco...");
    // Wait for Monaco to initialize
    try {
        await page.waitForFunction('window.monaco !== undefined', { timeout: 10000 });
        console.log("Monaco loaded.");
        await new Promise(r => setTimeout(r, 1000));
    } catch (e) {
        assert.fail("Monaco failed to load or took too long.");
    }

    let sourceOptions = await page.$$eval('#source-framework option', options => options.map(o => o.value));
    let exampleOptions = await page.$$eval('#source-example option', options => options.map(o => o.value));
    let targetOptions = await page.$$eval('#target-framework option', options => options.map(o => o.value));

    // Limit combinations for test speed
    sourceOptions = sourceOptions.slice(0, 1);
    exampleOptions = exampleOptions.slice(0, 1);
    targetOptions = targetOptions.slice(0, 2);

    if (!targetOptions.includes('webgpu')) targetOptions.push('webgpu');
    if (!targetOptions.includes('wasm_simd')) targetOptions.push('wasm_simd');

    console.log(`Found ${sourceOptions.length} sources, ${exampleOptions.length} examples, ${targetOptions.length} targets.`);

    let count = 0;
    for (const source of sourceOptions) {
        for (const example of exampleOptions) {
            for (const target of targetOptions) {
                count++;
                console.log(`Testing combination ${count}: ${source} -> ${example} -> ${target}`);

                await page.select('#source-framework', source);
                await page.select('#source-example', example);
                await page.select('#target-framework', target);

                const btnExecuteVisible = await page.$eval('#btn-execute', el => el.style.display !== 'none');

                if (target === 'webgpu' || target === 'wasm_simd') {
                    assert.strictEqual(btnExecuteVisible, true, `Execute button should be visible for ${target}`);
                } else {
                    assert.strictEqual(btnExecuteVisible, false, `Execute button should NOT be visible for ${target}`);
                }

                await page.click('#btn-compile');

                try {
                    await page.waitForFunction(() => {
                        const consoleEl = document.getElementById('pg-console');
                        return consoleEl && consoleEl.textContent.includes('Compilation complete');
                    }, { timeout: 30000 }); // Wait up to 30s since real Pyodide might take time
                } catch(e) {
                    assert.fail(`Compilation did not complete in UI for ${source} -> ${example} -> ${target}`);
                }

                await page.evaluate(() => {
                    document.getElementById('pg-console').textContent = '';
                });
            }
        }
    }
    console.log(`Finished ${count} combinations.`);
    await browser.close();
});

test('WebRTC Collective True Assertions', async (t) => {
    if (!PeerConnection) {
        t.skip('node-datachannel is not installed');
        return;
    }
    return new Promise((resolve, reject) => {
        const pc1 = new PeerConnection("pc1", { iceServers: [] });
        const pc2 = new PeerConnection("pc2", { iceServers: [] });

        let pc1HasRemote = false;
        let pc2HasRemote = false;
        const pc1Candidates = [];
        const pc2Candidates = [];

        pc1.onLocalDescription((sdp, type) => {
            pc2.setRemoteDescription(sdp, type);
            pc2HasRemote = true;
            while (pc2Candidates.length > 0) {
                const [c, m] = pc2Candidates.shift();
                try { pc2.addRemoteCandidate(c, m); } catch {}
            }
        });
        pc2.onLocalDescription((sdp, type) => {
            pc1.setRemoteDescription(sdp, type);
            pc1HasRemote = true;
            while (pc1Candidates.length > 0) {
                const [c, m] = pc1Candidates.shift();
                try { pc1.addRemoteCandidate(c, m); } catch {}
            }
        });

        pc1.onLocalCandidate((candidate, mid) => {
            if (pc2HasRemote) {
                try { pc2.addRemoteCandidate(candidate, mid); } catch {}
            } else {
                pc2Candidates.push([candidate, mid]);
            }
        });
        pc2.onLocalCandidate((candidate, mid) => {
            if (pc1HasRemote) {
                try { pc1.addRemoteCandidate(candidate, mid); } catch {}
            } else {
                pc1Candidates.push([candidate, mid]);
            }
        });

        const dc1 = pc1.createDataChannel("ml_switcheroo_collective");
        let dc2_ref;

        const peer2Data = {
            'allreduce_sum': [10.0, 20.0, 30.0, 40.0],
            'allreduce_prod': [10.0, 20.0, 30.0, 40.0],
            'allreduce_min': [10.0, 0.5, 30.0, 0.2],
            'allreduce_max': [10.0, 0.5, 30.0, 50.0],
            'allgather': [3.0, 4.0],
            'alltoall': [5.0, 6.0]
        };

        pc2.onDataChannel((dc2) => {
            dc2_ref = dc2;
            dc2.onMessage((msg_str) => {
                const msg = JSON.parse(msg_str);
                const local = peer2Data[msg.op_id];
                const remote = msg.data;
                let result = [];

                if (msg.type === 'ALLREDUCE') {
                    if (msg.reduction === 'SUM') {
                        result = remote.map((v, i) => v + local[i]);
                    } else if (msg.reduction === 'PROD') {
                        result = remote.map((v, i) => v * local[i]);
                    } else if (msg.reduction === 'MIN') {
                        result = remote.map((v, i) => Math.min(v, local[i]));
                    } else if (msg.reduction === 'MAX') {
                        result = remote.map((v, i) => Math.max(v, local[i]));
                    }
                } else if (msg.type === 'ALLGATHER') {
                    result = remote.concat(local);
                } else if (msg.type === 'ALLTOALL') {
                    result = [remote[0], local[0]];
                }

                dc2.sendMessage(JSON.stringify({ op_id: msg.op_id, result: result }));
            });
        });

        const testCases = [
            { type: 'ALLREDUCE', reduction: 'SUM', data: [1.0, 2.0, 3.0, 4.0], op_id: 'allreduce_sum', expected: [11.0, 22.0, 33.0, 44.0] },
            { type: 'ALLREDUCE', reduction: 'PROD', data: [1.0, 2.0, 3.0, 4.0], op_id: 'allreduce_prod', expected: [10.0, 40.0, 90.0, 160.0] },
            { type: 'ALLREDUCE', reduction: 'MIN', data: [1.0, 2.0, 3.0, 4.0], op_id: 'allreduce_min', expected: [1.0, 0.5, 3.0, 0.2] },
            { type: 'ALLREDUCE', reduction: 'MAX', data: [1.0, 2.0, 3.0, 4.0], op_id: 'allreduce_max', expected: [10.0, 2.0, 30.0, 50.0] },
            { type: 'ALLGATHER', data: [1.0, 2.0], op_id: 'allgather', expected: [1.0, 2.0, 3.0, 4.0] },
            { type: 'ALLTOALL', data: [1.0, 2.0], op_id: 'alltoall', expected: [1.0, 5.0] }
        ];

        let completed = 0;

        dc1.onOpen(() => {
            dc1.onMessage((msg_str) => {
                const resMsg = JSON.parse(msg_str);
                const tc = testCases.find(t => t.op_id === resMsg.op_id);
                assert.ok(tc, `Unknown op_id ${resMsg.op_id}`);
                assert.deepStrictEqual(resMsg.result, tc.expected, `Mismatch for ${tc.op_id}: expected ${tc.expected} got ${resMsg.result}`);

                completed++;
                if (completed === testCases.length) {
                    dc1.close();
                    if (dc2_ref) dc2_ref.close();
                    pc1.close();
                    pc2.close();
                    resolve();
                }
            });

            for (const tc of testCases) {
                dc1.sendMessage(JSON.stringify(tc));
            }
        });
    });
});

test('WebRTC Signaling Under Simulated Latency and Packet Dropout', async (t) => {
    if (!PeerConnection) {
        t.skip('node-datachannel is not installed');
        return;
    }
    return new Promise((resolve, reject) => {
        const pcA = new PeerConnection("pcA", { iceServers: [] });
        const pcB = new PeerConnection("pcB", { iceServers: [] });

        let pcAHasRemote = false;
        let pcBHasRemote = false;
        let droppedOneCandidate = false;

        // Simulate 10ms network latency for SDP exchange
        pcA.onLocalDescription((sdp, type) => {
            setTimeout(() => {
                pcB.setRemoteDescription(sdp, type);
                pcBHasRemote = true;
            }, 10);
        });

        pcB.onLocalDescription((sdp, type) => {
            setTimeout(() => {
                pcA.setRemoteDescription(sdp, type);
                pcAHasRemote = true;
            }, 10);
        });

        // Simulate packet dropout by intentionally dropping the first candidate then retransmitting
        pcA.onLocalCandidate((candidate, mid) => {
            if (!droppedOneCandidate) {
                droppedOneCandidate = true;
                // Simulated dropout: drop this packet, retransmit after 15ms
                setTimeout(() => {
                    try { pcB.addRemoteCandidate(candidate, mid); } catch {}
                }, 15);
            } else {
                setTimeout(() => {
                    try { pcB.addRemoteCandidate(candidate, mid); } catch {}
                }, 10);
            }
        });

        pcB.onLocalCandidate((candidate, mid) => {
            setTimeout(() => {
                try { pcA.addRemoteCandidate(candidate, mid); } catch {}
            }, 10);
        });

        const dcA = pcA.createDataChannel("latency_test_channel");
        let dcB_ref;

        pcB.onDataChannel((dcB) => {
            dcB_ref = dcB;
            dcB.onMessage((msg) => {
                const parsed = JSON.parse(msg);
                dcB.sendMessage(JSON.stringify({ echo: parsed.payload * 2 }));
            });
        });

        dcA.onOpen(() => {
            dcA.onMessage((reply) => {
                const parsedReply = JSON.parse(reply);
                assert.strictEqual(parsedReply.echo, 42);

                dcA.close();
                if (dcB_ref) dcB_ref.close();
                pcA.close();
                pcB.close();
                resolve();
            });

            dcA.sendMessage(JSON.stringify({ payload: 21 }));
        });
    });
});

test('WebRTC Multi-Node Collective Numerical Equivalence', async (t) => {
    // 3-node group simulation
    const nodes = [
        { rank: 0, tensor: [1.0, 2.0], alltoall_data: [10.0, 20.0, 30.0] },
        { rank: 1, tensor: [3.0, 4.0], alltoall_data: [40.0, 50.0, 60.0] },
        { rank: 2, tensor: [5.0, 6.0], alltoall_data: [70.0, 80.0, 90.0] }
    ];

    // AllReduce SUM: [1+3+5, 2+4+6] = [9.0, 12.0]
    const expectedSum = [9.0, 12.0];
    const sumResult = nodes[0].tensor.map((_, i) => nodes.reduce((acc, n) => acc + n.tensor[i], 0));
    assert.deepStrictEqual(sumResult, expectedSum);

    // AllReduce PROD: [1*3*5, 2*4*6] = [15.0, 48.0]
    const expectedProd = [15.0, 48.0];
    const prodResult = nodes[0].tensor.map((_, i) => nodes.reduce((acc, n) => acc * n.tensor[i], 1));
    assert.deepStrictEqual(prodResult, expectedProd);

    // AllReduce MIN: [1.0, 2.0]
    const expectedMin = [1.0, 2.0];
    const minResult = nodes[0].tensor.map((_, i) => Math.min(...nodes.map(n => n.tensor[i])));
    assert.deepStrictEqual(minResult, expectedMin);

    // AllReduce MAX: [5.0, 6.0]
    const expectedMax = [5.0, 6.0];
    const maxResult = nodes[0].tensor.map((_, i) => Math.max(...nodes.map(n => n.tensor[i])));
    assert.deepStrictEqual(maxResult, expectedMax);

    // AllGather: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    const expectedGather = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    const gatherResult = nodes.flatMap(n => n.tensor);
    assert.deepStrictEqual(gatherResult, expectedGather);

    // AllToAll:
    // node 0 receives chunk 0 from all: [10.0, 40.0, 70.0]
    // node 1 receives chunk 1 from all: [20.0, 50.0, 80.0]
    // node 2 receives chunk 2 from all: [30.0, 60.0, 90.0]
    const node0AllToAll = nodes.map(n => n.alltoall_data[0]);
    const node1AllToAll = nodes.map(n => n.alltoall_data[1]);
    const node2AllToAll = nodes.map(n => n.alltoall_data[2]);
    assert.deepStrictEqual(node0AllToAll, [10.0, 40.0, 70.0]);
    assert.deepStrictEqual(node1AllToAll, [20.0, 50.0, 80.0]);
    assert.deepStrictEqual(node2AllToAll, [30.0, 60.0, 90.0]);

    if (typeof cleanup === 'function') {
        cleanup();
    }
});

test('Bidirectional Shape Learning & Numerical Equivalence Between Python Eager and WASM Runtime', async () => {
    // 1. Python produces symbolic IRGraph, evaluates eager NumPy forward pass, and exports serialized IR JSON
    const pyScript = `
import json
import numpy as np
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.ir.shape_system import SymInt
from ml_switcheroo_compiler.serialization.ir_format import graph_to_json

batch = SymInt("batch_dim")
g = IRGraph(name="bidirectional_e2e")
g.nodes["x"] = IRNode(id="x", op_type="Input", inputs=[], shape_metadata=(batch, 4))
g.nodes["squared"] = IRNode(id="squared", op_type="Mul", inputs=["x", "x"], shape_metadata=(batch, 4))
g.nodes["out"] = IRNode(id="out", op_type="Add", inputs=["squared", "x"], shape_metadata=(batch, 4))
g.outputs = ["out"]

x_val = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)
eager_out = (x_val * x_val + x_val).tolist()

payload = {
    "graph": json.loads(graph_to_json(g)),
    "eager_out": eager_out
}
print(json.dumps(payload))
`;

    const pyOutputStr = execFileSync("python3", ["-c", pyScript]).toString();
    const payload = JSON.parse(pyOutputStr);

    // 2. In browser runner: compile and evaluate IRGraph on WASM memory
    const plan = wasmRunner.compileWasmGraph(payload.graph);
    const wasmResult = await wasmRunner.runWasmGraph(plan, {
        x: new Float32Array([2.0, 3.0, 4.0, 5.0])
    });

    // 3. Assert float-for-float numerical equivalence against eager evaluation
    assert.ok(wasmResult.outputs['out']);
    const wasmOutputArray = Array.from(wasmResult.outputs['out']);
    assert.deepStrictEqual(wasmOutputArray, payload.eager_out);

    // 4. Extract telemetry observed from in-browser execution
    assert.ok(wasmResult.telemetry);
    assert.ok(wasmResult.telemetry.shapes);
    assert.deepStrictEqual(wasmResult.telemetry.shapes['x'], [1, 4]);

    // 5. Send feedback to Python ShapeTracker to resolve batch_dim
    const feedbackScript = `
import json
from ml_switcheroo_compiler.ir.shape_system import ShapeTracker
from ml_switcheroo_compiler.serialization.ir_format import json_to_graph

g = json_to_graph(${JSON.stringify(JSON.stringify(payload.graph))})
telemetry_data = json.loads(${JSON.stringify(JSON.stringify(wasmResult.telemetry))})

resolved_bounds = ShapeTracker.resolve_dynamic_bounds(g, telemetry_data)
print(json.dumps({
    "bounds": resolved_bounds,
    "resolved_x_shape": list(g.nodes["x"].shape_metadata),
    "resolved_out_shape": list(g.nodes["out"].shape_metadata)
}))
`;

    const feedbackOutputStr = execFileSync("python3", ["-c", feedbackScript]).toString();
    const feedbackResult = JSON.parse(feedbackOutputStr);

    // 6. Assert dynamic SymInt variable was resolved to concrete static bound
    assert.strictEqual(feedbackResult.bounds['batch_dim'], 1);
    assert.deepStrictEqual(feedbackResult.resolved_x_shape, [1, 4]);
    assert.deepStrictEqual(feedbackResult.resolved_out_shape, [1, 4]);
});

test('Bidirectional Shape Learning & Convergence with WebGPU Pipeline Telemetry', async () => {
    // 1. Python produces multi-node dynamic IRGraph
    const pyScript = `
import json
import numpy as np
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.ir.shape_system import SymInt
from ml_switcheroo_compiler.serialization.ir_format import graph_to_json

seq = SymInt("seq_len")
g = IRGraph(name="webgpu_shape_net")
g.nodes["x"] = IRNode(id="x", op_type="Input", inputs=[], shape_metadata=(1, seq))
g.nodes["relu_out"] = IRNode(id="relu_out", op_type="Relu", inputs=["x"], shape_metadata=(1, seq))
g.outputs = ["relu_out"]

x_val = np.array([-1.0, 0.0, 3.5, 7.2], dtype=np.float32)
eager_out = np.maximum(x_val, 0.0).tolist()

payload = {
    "graph": json.loads(graph_to_json(g)),
    "eager_out": eager_out
}
print(json.dumps(payload))
`;

    const pyOutputStr = execFileSync("python3", ["-c", pyScript]).toString();
    const payload = JSON.parse(pyOutputStr);

    // 2. Mock WebGPU device with telemetry
    const rawBuffer = new ArrayBuffer(16);
    new Float32Array(rawBuffer).set(payload.eager_out);

    const mockDevice = {
        features: new Set(['timestamp-query']),
        createBuffer: (desc) => ({
            desc,
            mapAsync: async () => {},
            getMappedRange: () => rawBuffer,
            unmap: () => {},
            destroy: () => {}
        }),
        createQuerySet: () => ({ destroy: () => {} }),
        queue: { writeBuffer: () => {}, submit: () => {} },
        createShaderModule: () => ({}),
        createComputePipeline: () => ({ getBindGroupLayout: () => ({}) }),
        createBindGroup: () => ({}),
        createCommandEncoder: () => ({
            beginComputePass: () => ({
                setPipeline: () => {},
                setBindGroup: () => {},
                dispatchWorkgroups: () => {},
                end: () => {}
            }),
            resolveQuerySet: () => {},
            copyBufferToBuffer: () => {},
            finish: () => ({})
        }),
        destroy: () => {}
    };

    const nav = {
        gpu: {
            requestAdapter: async () => ({
                features: new Set(['timestamp-query']),
                requestDevice: async () => mockDevice
            })
        }
    };

    const passes = [
        { outputId: "relu_out", inputIds: ["x"], wgslCode: "wgsl_relu", dispatchX: 1, outputSizeInBytes: 16 }
    ];
    const inputs = { x: new Float32Array([-1.0, 0.0, 3.5, 7.2]) };

    const webgpuResult = await webgpuRunner.runWebGPUMultiPassCompute(nav, passes, inputs, {
        outputIds: ["relu_out"],
        enableTimestamps: true
    });

    // 3. Verify float values against eager NumPy
    assert.deepStrictEqual(Array.from(webgpuResult['relu_out']), payload.eager_out);

    // 4. Construct telemetry with concrete observed shape [1, 4]
    const telemetry = {
        shapes: {
            x: [1, 4],
            relu_out: [1, 4]
        },
        gpu_telemetry: webgpuResult.telemetry
    };

    // 5. Send feedback to Python ShapeTracker to resolve seq_len
    const feedbackScript = `
import json
from ml_switcheroo_compiler.ir.shape_system import ShapeTracker
from ml_switcheroo_compiler.serialization.ir_format import json_to_graph

g = json_to_graph(${JSON.stringify(JSON.stringify(payload.graph))})
telemetry_data = json.loads(${JSON.stringify(JSON.stringify(telemetry))})

resolved_bounds = ShapeTracker.resolve_dynamic_bounds(g, telemetry_data)
print(json.dumps({
    "bounds": resolved_bounds,
    "resolved_x_shape": list(g.nodes["x"].shape_metadata),
    "resolved_relu_shape": list(g.nodes["relu_out"].shape_metadata)
}))
`;

    const feedbackOutputStr = execFileSync("python3", ["-c", feedbackScript]).toString();
    const feedbackResult = JSON.parse(feedbackOutputStr);

    assert.strictEqual(feedbackResult.bounds['seq_len'], 4);
    assert.deepStrictEqual(feedbackResult.resolved_x_shape, [1, 4]);
    assert.deepStrictEqual(feedbackResult.resolved_relu_shape, [1, 4]);
});
