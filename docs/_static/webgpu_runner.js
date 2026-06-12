/**
 * WebGPU Execution Module for ML Switcheroo
 * Handles WebGPU device initialization, buffer allocation, and compute pipeline execution.
 * @module WebGPURunner
 */

/**
 * Initializes the WebGPU device.
 * @param {Navigator} nav - The navigator object.
 * @returns {Promise<{adapter: GPUAdapter, device: GPUDevice}>} The adapter and device.
 * @throws {Error} If WebGPU is not supported or device request fails.
 */
async function initWebGPU(nav) {
    if (!nav.gpu) {
        throw new Error("WebGPU is not supported in this browser.");
    }
    const adapter = await nav.gpu.requestAdapter();
    if (!adapter) {
        throw new Error("Failed to request WebGPU adapter.");
    }
    const device = await adapter.requestDevice();
    return { adapter, device };
}

/**
 * Creates a compute pipeline.
 * @param {GPUDevice} device - The WebGPU device.
 * @param {string} wgslCode - The WGSL shader code.
 * @returns {GPUComputePipeline} The compiled compute pipeline.
 */
function createComputePipeline(device, wgslCode) {
    const shaderModule = device.createShaderModule({ code: wgslCode });
    return device.createComputePipeline({
        layout: 'auto',
        compute: {
            module: shaderModule,
            entryPoint: 'main'
        }
    });
}

/**
 * Runs a WebGPU compute shader with provided inputs and output size.
 * For simplicity, we assume one input buffer and one output buffer.
 * @param {Navigator} nav - The navigator object.
 * @param {string} wgslCode - The WGSL code.
 * @param {Float32Array} inputData - The input data.
 * @param {number} outputSizeInBytes - The size of the output buffer.
 * @returns {Promise<Float32Array>} The computed output data.
 */
async function runWebGPUCompute(nav, wgslCode, inputData, outputSizeInBytes) {
    const { device } = await initWebGPU(nav);

    // Create input buffer
    const inputBuffer = device.createBuffer({
        size: inputData.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(inputBuffer, 0, inputData);

    // Create output buffer (storage for compute, source for copy)
    const outputBuffer = device.createBuffer({
        size: outputSizeInBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });

    // Create staging buffer (for reading back to CPU)
    const stagingBuffer = device.createBuffer({
        size: outputSizeInBytes,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });

    const pipeline = createComputePipeline(device, wgslCode);

    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: inputBuffer } },
            { binding: 1, resource: { buffer: outputBuffer } }
        ]
    });

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);

    // Calculate workgroups (assuming 1D workgroup of size 64 for simple examples)
    const workgroupCount = Math.ceil(inputData.length / 64);
    pass.dispatchWorkgroups(workgroupCount);
    pass.end();

    // Copy from output buffer to staging buffer
    encoder.copyBufferToBuffer(outputBuffer, 0, stagingBuffer, 0, outputSizeInBytes);

    // Submit commands
    device.queue.submit([encoder.finish()]);

    // Map staging buffer to read results
    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const arrayBuffer = stagingBuffer.getMappedRange();
    const result = new Float32Array(arrayBuffer.slice(0)); // Copy out

    // Cleanup
    stagingBuffer.unmap();
    inputBuffer.destroy();
    outputBuffer.destroy();
    stagingBuffer.destroy();
    device.destroy();

    return result;
}

// Export for testing
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        initWebGPU,
        createComputePipeline,
        runWebGPUCompute
    };
}
