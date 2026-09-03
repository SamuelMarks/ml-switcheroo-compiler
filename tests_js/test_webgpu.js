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
