const test = require('node:test');
const assert = require('node:assert');

const webgpuModule = require('../docs/_static/webgpu_runner.js');

test('initWebGPU throws if not supported', async () => {
    const nav = {}; // no gpu
    await assert.rejects(
        () => webgpuModule.initWebGPU(nav),
        /WebGPU is not supported/
    );
});

test('initWebGPU throws if adapter request fails', async () => {
    const nav = {
        gpu: {
            requestAdapter: async () => null
        }
    };
    await assert.rejects(
        () => webgpuModule.initWebGPU(nav),
        /Failed to request WebGPU adapter/
    );
});

test('initWebGPU succeeds with valid mock', async () => {
    const mockDevice = { id: 'device' };
    const mockAdapter = {
        requestDevice: async () => mockDevice
    };
    const nav = {
        gpu: {
            requestAdapter: async () => mockAdapter
        }
    };

    const { adapter, device } = await webgpuModule.initWebGPU(nav);
    assert.strictEqual(adapter, mockAdapter);
    assert.strictEqual(device, mockDevice);
});

test('createComputePipeline uses correct configuration', () => {
    const mockPipeline = { id: 'pipeline' };
    let capturedCode = null;
    let capturedLayout = null;
    let capturedModule = null;

    const mockDevice = {
        createShaderModule: (descriptor) => {
            capturedCode = descriptor.code;
            return { id: 'module' };
        },
        createComputePipeline: (descriptor) => {
            capturedLayout = descriptor.layout;
            capturedModule = descriptor.compute.module;
            return mockPipeline;
        }
    };

    const wgsl = 'fn main() {}';
    const pipeline = webgpuModule.createComputePipeline(mockDevice, wgsl);

    assert.strictEqual(pipeline, mockPipeline);
    assert.strictEqual(capturedCode, wgsl);
    assert.strictEqual(capturedLayout, 'auto');
    assert.strictEqual(capturedModule.id, 'module');
});

// Polyfill WebGPU Constants
global.GPUBufferUsage = {
    STORAGE: 1,
    COPY_DST: 2,
    COPY_SRC: 4,
    MAP_READ: 8
};
global.GPUMapMode = {
    READ: 1
};

test('runWebGPUCompute flows correctly and maps memory', async () => {
    // We mock the entire GPU pipeline since we are running in headless Node.js
    // without a native WGPU backend available in this test environment.
    let buffersDestroyed = 0;

    class MockBuffer {
        constructor() {
            this.destroyed = false;
        }
        async mapAsync() { return Promise.resolve(); }
        getMappedRange() { return new Float32Array([42]).buffer; }
        unmap() {}
        destroy() { buffersDestroyed++; }
    }

    const mockDevice = {
        createBuffer: () => new MockBuffer(),
        queue: {
            writeBuffer: () => {},
            submit: () => {}
        },
        createShaderModule: () => ({}),
        createComputePipeline: () => ({
            getBindGroupLayout: () => ({})
        }),
        createBindGroup: () => ({}),
        createCommandEncoder: () => ({
            beginComputePass: () => ({
                setPipeline: () => {},
                setBindGroup: () => {},
                dispatchWorkgroups: () => {},
                end: () => {}
            }),
            copyBufferToBuffer: () => {},
            finish: () => ({})
        }),
        destroy: () => {}
    };

    const nav = {
        gpu: {
            requestAdapter: async () => ({
                requestDevice: async () => mockDevice
            })
        }
    };

    const inputData = new Float32Array([1, 2, 3]);
    const result = await webgpuModule.runWebGPUCompute(nav, "wgsl code", inputData, 4);

    assert.strictEqual(result.length, 1);
    assert.strictEqual(result[0], 42); // Asserts our mock buffer value surfaced
    assert.strictEqual(buffersDestroyed, 3); // input, output, staging must be cleaned up
});

test('runWebGPUCompute handles multi-buffer inputs and data types', async () => {
    let buffersDestroyed = 0;

    class MockBuffer {
        constructor() {
            this.destroyed = false;
        }
        async mapAsync() { return Promise.resolve(); }
        getMappedRange() { return new Int32Array([10, 20]).buffer; }
        unmap() {}
        destroy() { buffersDestroyed++; }
    }

    const mockDevice = {
        createBuffer: () => new MockBuffer(),
        queue: {
            writeBuffer: () => {},
            submit: () => {}
        },
        createShaderModule: () => ({}),
        createComputePipeline: () => ({
            getBindGroupLayout: () => ({})
        }),
        createBindGroup: () => ({}),
        createCommandEncoder: () => ({
            beginComputePass: () => ({
                setPipeline: () => {},
                setBindGroup: () => {},
                dispatchWorkgroups: () => {},
                end: () => {}
            }),
            copyBufferToBuffer: () => {},
            finish: () => ({})
        }),
        destroy: () => {}
    };

    const nav = {
        gpu: {
            requestAdapter: async () => ({
                requestDevice: async () => mockDevice
            })
        }
    };

    const inputs = [new Float32Array([1, 2]), new Float32Array([3, 4])];
    const intResult = await webgpuModule.runWebGPUCompute(nav, "wgsl", inputs, 8, {
        inputBindings: [0, 1],
        outputBinding: 2,
        outputType: 'int32',
        outputShape: [2]
    });

    assert.ok(intResult instanceof Int32Array);
    assert.strictEqual(intResult.length, 2);
    assert.strictEqual(intResult[0], 10);
    assert.deepStrictEqual(intResult.shape, [2]);
    assert.strictEqual(buffersDestroyed, 4); // 2 inputs + output + staging

    const uintResult = await webgpuModule.runWebGPUCompute(nav, "wgsl", inputs, 8, {
        outputType: 'uint32'
    });
    assert.ok(uintResult instanceof Uint32Array);

    const emptyResult = await webgpuModule.runWebGPUCompute(nav, "wgsl", [], 4);
    assert.ok(emptyResult instanceof Float32Array);
});

test('createUniformBuffer constructs buffer with shape and strides metadata', () => {
    let bufferWritten = null;
    const mockDevice = {
        createBuffer: (desc) => ({ desc, destroy: () => {} }),
        queue: {
            writeBuffer: (buf, off, data) => {
                bufferWritten = data;
            }
        }
    };

    const buf = webgpuModule.createUniformBuffer(mockDevice, [2, 4], [4, 1], 64);
    assert.ok(buf);
    assert.ok(bufferWritten instanceof Uint32Array);
    assert.strictEqual(bufferWritten[0], 2); // rank
    assert.strictEqual(bufferWritten[1], 64); // offset
    assert.strictEqual(bufferWritten[2], 2); // dim 0
    assert.strictEqual(bufferWritten[3], 4); // dim 1
    assert.strictEqual(bufferWritten[8], 4); // stride 0
    assert.strictEqual(bufferWritten[9], 1); // stride 1
});

test('runWebGPUMultiPassCompute orchestrates multi-pass pipeline and backward execution', async () => {
    const rawBuffer = new ArrayBuffer(16);
    new Float32Array(rawBuffer).set([4.0, 9.0, 16.0, 25.0]);

    const mockDevice = {
        createBuffer: () => ({
            mapAsync: async () => {},
            getMappedRange: () => rawBuffer,
            unmap: () => {},
            destroy: () => {}
        }),
        queue: {
            writeBuffer: () => {},
            submit: () => {}
        },
        createShaderModule: () => ({}),
        createComputePipeline: () => ({
            getBindGroupLayout: () => ({})
        }),
        createBindGroup: () => ({}),
        createCommandEncoder: () => ({
            beginComputePass: () => ({
                setPipeline: () => {},
                setBindGroup: () => {},
                dispatchWorkgroups: () => {},
                end: () => {}
            }),
            copyBufferToBuffer: () => {},
            finish: () => ({})
        }),
        destroy: () => {}
    };

    const nav = {
        gpu: {
            requestAdapter: async () => ({
                requestDevice: async () => mockDevice
            })
        }
    };

    const passes = [
        { outputId: "inter", inputIds: ["x"], wgslCode: "code1", dispatchX: 1, outputSizeInBytes: 16 },
        { outputId: "out", inputIds: ["inter"], wgslCode: "code2", dispatchX: 1, outputSizeInBytes: 16 }
    ];
    const inputs = { x: new Float32Array([2.0, 3.0, 4.0, 5.0]) };

    const results = await webgpuModule.runWebGPUMultiPassCompute(nav, passes, inputs, { outputIds: ["out"] });
    assert.ok(results["out"]);
    assert.strictEqual(results["out"][0], 4.0);

    // Test backward pass execution
    const bwdPasses = [
        { outputId: "grad_x", inputIds: ["primal_x", "grad_out"], wgslCode: "bwd_code", dispatchX: 1, outputSizeInBytes: 16 }
    ];
    const bwdResults = await webgpuModule.runWebGPUBackward(nav, bwdPasses, { primal_x: inputs.x }, { grad_out: new Float32Array([1.0, 1.0, 1.0, 1.0]) });
    assert.ok(bwdResults["grad_x"]);
});

test('real WebGPU compute via Chrome/Puppeteer with real GPU/SwiftShader', async (t) => {
    const fs = require('fs');
    const http = require('http');
    let puppeteer;
    try {
        puppeteer = require('puppeteer');
    } catch {
        t.skip('puppeteer not installed');
        return;
    }

    const chromePath = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
    if (!fs.existsSync(chromePath)) {
        t.skip("Chrome not found for WebGPU testing");
        return;
    }

    const runnerSrc = fs.readFileSync('docs/_static/webgpu_runner.js', 'utf8');
    const server = http.createServer((req, res) => {
        res.writeHead(200, { 'Content-Type': 'text/html' });
        res.end('<!DOCTYPE html><html><body></body></html>');
    });

    await new Promise((resolve) => server.listen(8983, resolve));

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
        await page.goto('http://localhost:8983');
        await page.addScriptTag({ content: runnerSrc });

        const results = await page.evaluate(async () => {
            // Forward operations verification against NumPy baseline
            // 1. Elementwise square (Multiply)
            const squareShader = `
                @group(0) @binding(0) var<storage, read> inBuf: array<f32>;
                @group(0) @binding(1) var<storage, read_write> outBuf: array<f32>;
                @compute @workgroup_size(64, 1, 1)
                fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                    let i = id.x;
                    if (i < 3u) {
                        outBuf[i] = inBuf[i] * inBuf[i];
                    }
                }
            `;
            const sqIn = new Float32Array([2.0, 3.0, 4.0]);
            const sqOut = await runWebGPUCompute(navigator, squareShader, sqIn, 3 * 4);

            // 2. Elementwise Add (Multi-buffer input)
            const addShader = `
                @group(0) @binding(0) var<storage, read> in0: array<f32>;
                @group(0) @binding(1) var<storage, read> in1: array<f32>;
                @group(0) @binding(2) var<storage, read_write> outBuf: array<f32>;
                @compute @workgroup_size(64, 1, 1)
                fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                    let i = id.x;
                    if (i < 2u) {
                        outBuf[i] = in0[i] + in1[i];
                    }
                }
            `;
            const addIn0 = new Float32Array([1.0, 2.0]);
            const addIn1 = new Float32Array([10.0, 20.0]);
            const addOut = await runWebGPUCompute(navigator, addShader, [addIn0, addIn1], 2 * 4, {
                inputBindings: [0, 1],
                outputBinding: 2
            });

            // 3. Int32 data type and shape
            const intShader = `
                @group(0) @binding(0) var<storage, read> inBuf: array<i32>;
                @group(0) @binding(1) var<storage, read_write> outBuf: array<i32>;
                @compute @workgroup_size(64, 1, 1)
                fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                    let i = id.x;
                    if (i < 2u) {
                        outBuf[i] = inBuf[i] * 2i;
                    }
                }
            `;
            const intIn = new Int32Array([5, 15]);
            const intOut = await runWebGPUCompute(navigator, intShader, intIn, 2 * 4, {
                outputType: 'int32',
                outputShape: [2]
            });

            return {
                sq: Array.from(sqOut),
                add: Array.from(addOut),
                int: Array.from(intOut),
                intShape: intOut.shape
            };
        });

        // Numerical assertions against reference values
        assert.deepStrictEqual(results.sq, [4.0, 9.0, 16.0]);
        assert.deepStrictEqual(results.add, [11.0, 22.0]);
        assert.deepStrictEqual(results.int, [10, 30]);
        assert.deepStrictEqual(results.intShape, [2]);

        await browser.close();
    } finally {
        if (server.closeAllConnections) {
            server.closeAllConnections();
        }
        server.close();
    }
});

test('updateUniformBuffer updates buffer with new shape and strides dynamically', () => {
    let written = null;
    const mockDevice = {
        queue: {
            writeBuffer: (buf, off, data) => {
                written = data;
            }
        }
    };
    const mockBuf = {};
    webgpuModule.updateUniformBuffer(mockDevice, mockBuf, [3, 5, 7], [35, 7, 1], 128, 4);

    assert.ok(written instanceof Uint32Array);
    assert.strictEqual(written[0], 3); // rank
    assert.strictEqual(written[1], 128); // offset
    assert.strictEqual(written[2], 3); // dim 0
    assert.strictEqual(written[3], 5); // dim 1
    assert.strictEqual(written[4], 7); // dim 2
    assert.strictEqual(written[8], 35); // stride 0
    assert.strictEqual(written[9], 7); // stride 1
    assert.strictEqual(written[10], 1); // stride 2
    assert.strictEqual(written[14], 4); // batchSize
});

test('WebGPUPipelineChain manages chained passes and dynamic uniform buffers', async () => {
    const rawBuffer = new ArrayBuffer(16);
    new Float32Array(rawBuffer).set([10.0, 20.0, 30.0, 40.0]);

    let destroyedCount = 0;
    const mockDevice = {
        createBuffer: (desc) => ({
            desc,
            mapAsync: async () => {},
            getMappedRange: () => rawBuffer,
            unmap: () => {},
            destroy: () => { destroyedCount++; }
        }),
        queue: {
            writeBuffer: () => {},
            submit: () => {}
        },
        createShaderModule: () => ({}),
        createComputePipeline: () => ({
            getBindGroupLayout: () => ({})
        }),
        createBindGroup: () => ({}),
        createCommandEncoder: () => ({
            beginComputePass: () => ({
                setPipeline: () => {},
                setBindGroup: () => {},
                dispatchWorkgroups: () => {},
                end: () => {}
            }),
            copyBufferToBuffer: () => {},
            finish: () => ({})
        })
    };

    const chain = new webgpuModule.WebGPUPipelineChain(mockDevice);
    chain.setStorageBuffer('x', new Float32Array([1.0, 2.0, 3.0, 4.0]));
    chain.setStorageBuffer('out', 16);
    chain.setUniform('u0', [4], [1], 0, 1);
    // Update existing uniform
    chain.setUniform('u0', [4], [1], 0, 2);

    chain.addPass({
        wgslCode: 'compute_code',
        inputIds: ['x'],
        uniformIds: ['u0'],
        outputId: 'out',
        dispatchX: 1
    });

    const results = await chain.execute(['out']);
    assert.ok(results['out']);
    assert.strictEqual(results['out'][0], 10.0);

    chain.destroy();
    assert.ok(destroyedCount > 0);
});

test('WebGPUPipelineChain supports pipeline caching across iterative steps, buffer recycling, and arena aliasing', async () => {
    let pipelineCreateCount = 0;
    let bufferCreateCount = 0;

    const mockDevice = {
        createBuffer: ({ size }) => {
            bufferCreateCount++;
            return {
                size: size,
                destroy: () => {},
                mapAsync: async () => {},
                getMappedRange: () => new ArrayBuffer(size),
                unmap: () => {}
            };
        },
        queue: {
            writeBuffer: () => {},
            submit: () => {}
        },
        createShaderModule: () => ({}),
        createComputePipeline: () => {
            pipelineCreateCount++;
            return {
                getBindGroupLayout: () => ({})
            };
        },
        createBindGroup: () => ({}),
        createCommandEncoder: () => ({
            beginComputePass: () => ({
                setPipeline: () => {},
                setBindGroup: () => {},
                dispatchWorkgroups: () => {},
                end: () => {}
            }),
            copyBufferToBuffer: () => {},
            finish: () => ({})
        })
    };

    const chain = new webgpuModule.WebGPUPipelineChain(mockDevice);
    chain.setStorageBuffer('in0', new Float32Array([1, 2, 3, 4]));
    chain.setStorageBuffer('intermediate', 64);
    chain.setStorageBuffer('out1', 64);
    chain.setStorageBuffer('out2', 64);

    // Test arena aliasing
    chain.aliasBuffer('in0_alias', 'in0');
    assert.strictEqual(chain.getStorageBuffer('in0_alias'), chain.getStorageBuffer('in0'));

    chain.addPass({
        pipelineId: 'step1',
        wgslCode: 'step1_wgsl',
        inputIds: ['in0_alias'],
        outputId: 'intermediate',
        recycleInputs: []
    });

    chain.addPass({
        pipelineId: 'step2',
        wgslCode: 'step2_wgsl',
        inputIds: ['intermediate'],
        outputIds: ['out1', 'out2'],
        recycleInputs: ['intermediate']
    });

    // Iteration 1
    await chain.execute(['out1', 'out2']);
    assert.strictEqual(pipelineCreateCount, 2);

    // Intermediate buffer should have been recycled
    assert.strictEqual(chain.recycledBuffers.length, 1);

    // Buffer allocation should reuse the recycled intermediate buffer
    const bufCountBefore = bufferCreateCount;
    chain.setStorageBuffer('reused_buf', 32);
    // Buffer count should NOT have increased because recycled buffer was reused
    assert.strictEqual(bufferCreateCount, bufCountBefore);

    // Iteration 2 - pipelines should be reused from cache!
    await chain.execute(['out1', 'out2']);
    assert.strictEqual(pipelineCreateCount, 2, 'Pipelines must be reused across iterative steps without recompilation');
    assert.ok(chain.pipelineCacheHits >= 2);

    chain.destroy();
    assert.strictEqual(chain.storageBuffers.size, 0);
    assert.strictEqual(chain.recycledBuffers.length, 0);
});

test('runWebGPUMultiPassCompute supports timestamp query telemetry', async () => {
    const rawBuffer = new ArrayBuffer(16);
    new Float32Array(rawBuffer).set([1.0, 2.0, 3.0, 4.0]);

    const timestampBuffer = new ArrayBuffer(16);
    const tsView = new BigInt64Array(timestampBuffer);
    tsView[0] = 1000000n;
    tsView[1] = 1050000n;

    const mockDevice = {
        features: new Set(['timestamp-query']),
        createBuffer: (desc) => {
            const isTimestamp = (desc.label === 'timestamp_staging');
            return {
                desc,
                mapAsync: async () => {},
                getMappedRange: () => isTimestamp ? timestampBuffer : rawBuffer,
                unmap: () => {},
                destroy: () => {}
            };
        },
        createQuerySet: () => ({
            destroy: () => {}
        }),
        queue: {
            writeBuffer: () => {},
            submit: () => {}
        },
        createShaderModule: () => ({}),
        createComputePipeline: () => ({
            getBindGroupLayout: () => ({})
        }),
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
        { outputId: "out", inputIds: ["x"], wgslCode: "pass_code", dispatchX: 1, outputSizeInBytes: 16 }
    ];
    const inputs = { x: new Float32Array([1.0, 2.0, 3.0, 4.0]) };

    const results = await webgpuModule.runWebGPUMultiPassCompute(nav, passes, inputs, {
        outputIds: ["out"],
        enableTimestamps: true
    });

    assert.ok(results.telemetry);
    assert.strictEqual(results.telemetry.passes_executed, 1);
    assert.strictEqual(results.telemetry.is_gpu_timestamp, true);
    assert.strictEqual(results.telemetry.duration_ns, 50000);
    assert.strictEqual(results.telemetry.duration_ms, 0.05);
});

test('webgpu_runner full branch and function coverage', async () => {
    const mockDevice = {
        features: new Set(),
        createBuffer: (desc) => ({
            desc,
            size: desc.size,
            mapAsync: async () => {},
            getMappedRange: () => new Float32Array([1, 2, 3, 4]).buffer,
            unmap: () => {},
            destroy: () => {}
        }),
        queue: {
            writeBuffer: () => {},
            submit: () => {}
        },
        createShaderModule: () => ({}),
        createComputePipeline: () => ({
            getBindGroupLayout: () => ({})
        }),
        createBindGroup: () => ({}),
        createCommandEncoder: () => ({
            beginComputePass: () => ({
                setPipeline: () => {},
                setBindGroup: () => {},
                dispatchWorkgroups: () => {},
                end: () => {}
            }),
            copyBufferToBuffer: () => {},
            finish: () => ({})
        }),
        destroy: () => {}
    };

    // 1. createUniformBuffer & updateUniformBuffer with null shape/strides
    const nullU = webgpuModule.createUniformBuffer(mockDevice, null, null);
    webgpuModule.updateUniformBuffer(mockDevice, nullU, null, null);

    // 2. WebGPUPipelineChain branches (getStorageBuffer alias loop, recycleBuffer alias loop, setStorageBuffer write data into recycled, captureRuntimeShapes, destroy recycled)
    const chain = new webgpuModule.WebGPUPipelineChain(mockDevice);
    chain.setStorageBuffer('root_buf', new Float32Array([1, 2, 3, 4]));
    chain.aliasBuffer('alias_1', 'root_buf');
    chain.aliasBuffer('alias_2', 'alias_1');
    assert.ok(chain.getStorageBuffer('alias_2'));
    assert.strictEqual(chain.getStorageBuffer('nonexistent'), null);

    chain.setUniform('u1', [2, 2], [2, 1]);
    chain.setUniform('u1', [4, 1], [1, 1]); // update existing uniform
    const captured = webgpuModule.captureRuntimeShapes(chain);
    assert.deepStrictEqual(captured.u1, [4, 1]);
    assert.deepStrictEqual(captured.root_buf, [4]);

    // Execute pass with missing inputId/uniformId/outputId and no explicit outputIds
    chain.addPass({
        wgslCode: '@compute @workgroup_size(1) fn main() {}',
        inputIds: ['missing_in'],
        uniformIds: ['missing_u'],
        outputIds: ['missing_out']
    });
    chain.addPass({
        wgslCode: '@compute @workgroup_size(1) fn main2() {}'
    });
    await chain.execute();
    await chain.execute(['missing_target']);

    chain.recycleBuffer('alias_2');
    chain.recycleBuffer('nonexistent');
    // Re-use recycled buffer with TypedArray payload
    chain.setStorageBuffer('reused_with_data', new Float32Array([5, 6, 7, 8]));
    chain.recycleBuffer('reused_with_data');
    chain.destroy();

    // 3. captureRuntimeShapes with Map, Object, and null
    assert.deepStrictEqual(webgpuModule.captureRuntimeShapes(null), {});
    assert.deepStrictEqual(webgpuModule.captureRuntimeShapes('primitive'), {});
    const mapShapes = new Map([
        ['m1', { shape: [3, 3] }],
        ['m2', { size: 16 }],
        ['m3', null]
    ]);
    assert.deepStrictEqual(webgpuModule.captureRuntimeShapes(mapShapes), { m1: [3, 3], m2: [4] });
    assert.deepStrictEqual(webgpuModule.captureRuntimeShapes({
        o1: [2, 4],
        o2: { shape: [5] },
        o3: { size: 12 },
        o4: null
    }), { o1: [2, 4], o2: [5], o3: [3] });

    // 3. runWebGPUMultiPassCompute with uniform config and runWebGPUBackward
    const nav = {
        gpu: {
            requestAdapter: async () => ({
                features: new Set(),
                requestDevice: async () => mockDevice
            })
        }
    };
    const passes = [{
        outputId: 'y',
        inputIds: ['x'],
        wgslCode: '@compute @workgroup_size(1) fn main() {}',
        uniform: { shape: [4], strides: [1] }
    }, {
        outputId: 'z',
        wgslCode: '@compute @workgroup_size(1) fn main2() {}'
    }];
    const bwdRes = await webgpuModule.runWebGPUBackward(
        nav,
        passes,
        { x: new Float32Array([1, 2, 3, 4]) },
        { dy: new Float32Array([1, 1, 1, 1]) }
    );
    assert.ok(bwdRes.y);

    // 4. WebRTCCollectiveClient in webgpu_runner.js
    const { WebRTCCollectiveClient } = webgpuModule;
    const c0 = new WebRTCCollectiveClient(0, 2, { chunkSize: 8 });
    const c1 = new WebRTCCollectiveClient(1, 2, { chunkSize: 8 });
    let ch1OnMsg = null;
    let toggleView = false;
    const ch0 = {
        readyState: 'open',
        send: (msg) => {
            if (typeof msg === 'string') {
                c1.handleMessage(0, { data: msg });
            } else if (ch1OnMsg) {
                toggleView = !toggleView;
                ch1OnMsg({ data: toggleView ? new Uint8Array(msg) : msg });
            }
        }
    };
    const ch1 = {
        set onmessage(fn) { ch1OnMsg = fn; },
        get onmessage() { return ch1OnMsg; },
        send: (msg) => {
            c0.handleMessage(1, { data: msg });
        }
    };
    c0.registerDataChannel(1, ch0);
    c1.registerDataChannel(0, ch1);

    await Promise.all([c0.barrier('b1'), c1.barrier('b1')]);

    let recvOut = null;
    c1.opWaiters.set('ar01', { resolve: (r) => { recvOut = r; } });
    await c0.allReduce('ar01', new Float32Array([1, 2, 3, 4]));
    assert.ok(recvOut instanceof Float32Array);

    c0.registerDataChannel(1, {
        readyState: 'open',
        send: () => { throw new Error('send err'); }
    });
    c0.broadcastControl({ type: 'PING' });
    await c0.allReduce('ar02', new Float32Array([1, 2]));

    const ag = await c0.allGather('ag1', new Float32Array([1, 2]));
    assert.strictEqual(ag.length, 4);
    const rs = await c0.reduceScatter('rs1', new Float32Array([1, 2, 3, 4]));
    assert.strictEqual(rs.length, 2);
    const a2a = await c0.allToAll('a2a1', new Float32Array([1, 2]));
    assert.strictEqual(a2a.length, 2);
    const bc = await c0.broadcast('bc1', new Float32Array([1, 2]), 0);
    assert.strictEqual(bc.length, 2);

    c1.handleMessage(0, { data: '{bad' });
    c1.handleMessage(0, { data: 12345 });
    assert.strictEqual(c1.finalizeCollective(1, 'missing').length, 0);

    const single = new WebRTCCollectiveClient(0, 1);
    await single.barrier();
    await single.allReduce('s1', [1, 2]);
    await single.allGather('s2', [1, 2]);
    await single.reduceScatter('s3', [1, 2]);
    await single.allToAll('s4', [1, 2]);
    await single.broadcast('s5', [1, 2]);
});
