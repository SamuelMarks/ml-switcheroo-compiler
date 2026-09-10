/**
 * WebGPU Execution Module for ML Switcheroo
 * Handles WebGPU device initialization, dynamic uniform buffers, chained pipelines,
 * timestamp queries, and compute pass orchestration.
 * @module WebGPURunner
 */

if (typeof globalThis !== 'undefined') {
    if (!globalThis.GPUBufferUsage) {
        globalThis.GPUBufferUsage = {
            MAP_READ: 1,
            MAP_WRITE: 2,
            COPY_SRC: 4,
            COPY_DST: 8,
            INDEX: 16,
            VERTEX: 32,
            UNIFORM: 64,
            STORAGE: 128,
            INDIRECT: 256,
            QUERY_RESOLVE: 512
        };
    }
    if (!globalThis.GPUMapMode) {
        globalThis.GPUMapMode = {
            READ: 1,
            WRITE: 2
        };
    }
}

/**
 * Initializes the WebGPU device with optional feature requests such as timestamp-query.
 * @param {Navigator} nav - The navigator object.
 * @param {Array<string>} [requiredFeatures=[]] - Optional array of WebGPU feature names.
 * @returns {Promise<{adapter: GPUAdapter, device: GPUDevice}>} The adapter and device.
 * @throws {Error} If WebGPU is not supported or device request fails.
 */
async function initWebGPU(nav, requiredFeatures = []) {
    if (!nav.gpu) {
        throw new Error("WebGPU is not supported in this browser.");
    }
    const adapter = await nav.gpu.requestAdapter();
    if (!adapter) {
        throw new Error("Failed to request WebGPU adapter.");
    }

    const availableFeatures = [];
    if (requiredFeatures && requiredFeatures.length > 0 && adapter.features) {
        for (const feat of requiredFeatures) {
            if (adapter.features.has(feat)) {
                availableFeatures.push(feat);
            }
        }
    }

    const deviceDesc = (availableFeatures.length > 0) ? { requiredFeatures: availableFeatures } : undefined;
    const device = await adapter.requestDevice(deviceDesc);
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
 * Constructs a WebGPU uniform buffer encoding tensor metadata.
 * Layout:
 * word 0: rank (u32)
 * word 1: offset (u32)
 * word 2..7: shape dimensions 0..5 (u32)
 * word 8..13: stride dimensions 0..5 (u32)
 * word 14: batch_size (u32)
 * word 15: reserved (u32)
 * @param {GPUDevice} device - WebGPU device.
 * @param {Array<number>} shape - Tensor shape dimensions.
 * @param {Array<number>} strides - Tensor dimension strides.
 * @param {number} [offset=0] - Starting memory word offset.
 * @param {number} [batchSize=1] - Batch size.
 * @returns {GPUBuffer} Initialized uniform buffer.
 */
function createUniformBuffer(device, shape, strides, offset = 0, batchSize = 1) {
    const data = new Uint32Array(16);
    data[0] = shape ? shape.length : 0;
    data[1] = offset;
    if (shape) {
        for (let i = 0; i < shape.length && i < 6; i++) {
            data[2 + i] = shape[i];
        }
    }
    if (strides) {
        for (let i = 0; i < strides.length && i < 6; i++) {
            data[8 + i] = strides[i];
        }
    }
    data[14] = batchSize;
    data[15] = 0;

    const buf = device.createBuffer({
        size: data.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(buf, 0, data);
    return buf;
}

/**
 * Dynamically updates an existing uniform buffer with new shape, stride, and batch dimensions
 * without recompiling WGSL compute pipelines.
 * @param {GPUDevice} device - WebGPU device.
 * @param {GPUBuffer} buffer - Target uniform buffer.
 * @param {Array<number>} shape - Updated shape dimensions.
 * @param {Array<number>} strides - Updated stride dimensions.
 * @param {number} [offset=0] - Starting memory word offset.
 * @param {number} [batchSize=1] - Batch size.
 */
function updateUniformBuffer(device, buffer, shape, strides, offset = 0, batchSize = 1) {
    const data = new Uint32Array(16);
    data[0] = shape ? shape.length : 0;
    data[1] = offset;
    if (shape) {
        for (let i = 0; i < shape.length && i < 6; i++) {
            data[2 + i] = shape[i];
        }
    }
    if (strides) {
        for (let i = 0; i < strides.length && i < 6; i++) {
            data[8 + i] = strides[i];
        }
    }
    data[14] = batchSize;
    data[15] = 0;
    device.queue.writeBuffer(buffer, 0, data);
}

/**
 * Runs a WebGPU compute shader with provided inputs and output size.
 * Handles single or multiple input buffers, dynamic binding layouts,
 * and arbitrary output tensor types and shapes.
 * @param {Navigator} nav - The navigator object.
 * @param {string} wgslCode - The WGSL code.
 * @param {Float32Array|Array<Float32Array|Int32Array|Uint32Array>} inputData - The input data or array of inputs.
 * @param {number} outputSizeInBytes - The size of the output buffer.
 * @param {Object} [options={}] - Additional execution options.
 * @param {Array<number>} [options.inputBindings] - Explicit binding indices for inputs.
 * @param {number} [options.outputBinding] - Explicit binding index for output.
 * @param {string} [options.outputType='float32'] - Output data type ('float32', 'int32', 'uint32').
 * @param {Array<number>} [options.outputShape] - Optional output tensor shape.
 * @returns {Promise<Float32Array|Int32Array|Uint32Array>} The computed output data.
 */
async function runWebGPUCompute(nav, wgslCode, inputData, outputSizeInBytes, options = {}) {
    const { device } = await initWebGPU(nav);

    const inputs = Array.isArray(inputData) ? inputData : [inputData];

    // Create input buffers
    const inputBuffers = inputs.map(data => {
        const buf = device.createBuffer({
            size: data.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        device.queue.writeBuffer(buf, 0, data);
        return buf;
    });

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

    const entries = inputBuffers.map((buf, idx) => ({
        binding: (options.inputBindings && options.inputBindings[idx] !== undefined)
            ? options.inputBindings[idx]
            : idx,
        resource: { buffer: buf }
    }));
    const outBinding = (options.outputBinding !== undefined) ? options.outputBinding : inputs.length;
    entries.push({
        binding: outBinding,
        resource: { buffer: outputBuffer }
    });

    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: entries
    });

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);

    // Calculate workgroups (assuming 1D workgroup of size 64 for simple examples)
    const primaryLen = inputs[0] ? inputs[0].length : 1;
    const workgroupCount = Math.max(1, Math.ceil(primaryLen / 64));
    pass.dispatchWorkgroups(workgroupCount);
    pass.end();

    // Copy from output buffer to staging buffer
    encoder.copyBufferToBuffer(outputBuffer, 0, stagingBuffer, 0, outputSizeInBytes);

    // Submit commands
    device.queue.submit([encoder.finish()]);

    // Map staging buffer to read results
    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const arrayBuffer = stagingBuffer.getMappedRange();

    const outputType = options.outputType || 'float32';
    let result;
    if (outputType === 'int32') {
        result = new Int32Array(arrayBuffer.slice(0));
    } else if (outputType === 'uint32') {
        result = new Uint32Array(arrayBuffer.slice(0));
    } else {
        result = new Float32Array(arrayBuffer.slice(0));
    }
    if (options.outputShape) {
        result.shape = options.outputShape;
    }

    // Cleanup
    stagingBuffer.unmap();
    inputBuffers.forEach(buf => buf.destroy());
    outputBuffer.destroy();
    stagingBuffer.destroy();
    device.destroy();

    return result;
}

/**
 * Chained multi-kernel compute pipeline orchestrator.
 */
class WebGPUPipelineChain {
    /**
     * @param {GPUDevice} device - WebGPU device.
     */
    constructor(device) {
        this.device = device;
        this.storageBuffers = new Map();
        this.uniformBuffers = new Map();
        this.pipelines = new Map();
        this.passes = [];
    }

    /**
     * Registers or updates a storage buffer.
     * @param {string} id - Buffer identifier.
     * @param {Float32Array|Int32Array|Uint32Array|number} dataOrSize - Buffer payload or byte size.
     * @returns {GPUBuffer} Created storage buffer.
     */
    setStorageBuffer(id, dataOrSize) {
        const isNum = (typeof dataOrSize === 'number');
        const size = isNum ? dataOrSize : dataOrSize.byteLength;
        const buf = this.device.createBuffer({
            size: Math.max(16, size),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
        });
        if (!isNum) {
            this.device.queue.writeBuffer(buf, 0, dataOrSize);
        }
        this.storageBuffers.set(id, { buffer: buf, size: Math.max(16, size) });
        return buf;
    }

    /**
     * Registers or updates a dynamic uniform buffer with shape and strides.
     * @param {string} id - Uniform identifier.
     * @param {Array<number>} shape - Tensor shape dimensions.
     * @param {Array<number>} strides - Tensor dimension strides.
     * @param {number} [offset=0] - Byte word offset.
     * @param {number} [batchSize=1] - Batch size.
     * @returns {GPUBuffer} Initialized or updated uniform buffer.
     */
    setUniform(id, shape, strides, offset = 0, batchSize = 1) {
        if (this.uniformBuffers.has(id)) {
            const existing = this.uniformBuffers.get(id);
            updateUniformBuffer(this.device, existing.buffer, shape, strides, offset, batchSize);
            existing.shape = Array.from(shape);
            return existing.buffer;
        }
        const buf = createUniformBuffer(this.device, shape, strides, offset, batchSize);
        this.uniformBuffers.set(id, { buffer: buf, size: 64, shape: Array.from(shape) });
        return buf;
    }

    /**
     * Inspects and captures concrete runtime shapes across registered uniform and storage buffers.
     * @returns {Object.<string, Array<number>>} Observed concrete shapes.
     */
    captureRuntimeShapes() {
        const shapes = {};
        for (const [id, entry] of this.uniformBuffers.entries()) {
            if (entry && entry.shape) {
                shapes[id] = Array.from(entry.shape);
            }
        }
        for (const [id, entry] of this.storageBuffers.entries()) {
            if (!shapes[id] && entry && entry.size) {
                shapes[id] = [Math.floor(entry.size / 4)];
            }
        }
        return shapes;
    }

    /**
     * Adds an execution pass to the pipeline chain.
     * @param {Object} passConfig - Configuration for the compute pass.
     * @param {string} passConfig.wgslCode - Compute shader WGSL code.
     * @param {Array<string>} [passConfig.inputIds=[]] - Storage buffer inputs.
     * @param {string} passConfig.outputId - Storage buffer output.
     * @param {Array<string>} [passConfig.uniformIds=[]] - Uniform buffer inputs.
     * @param {number} [passConfig.dispatchX=1] - Workgroups in X.
     * @param {number} [passConfig.dispatchY=1] - Workgroups in Y.
     * @param {number} [passConfig.dispatchZ=1] - Workgroups in Z.
     */
    addPass(passConfig) {
        this.passes.push(passConfig);
    }

    /**
     * Executes all chained passes sequentially and reads back results.
     * @param {Array<string>} [outputIds=[]] - Designated output IDs to retrieve.
     * @returns {Promise<Object.<string, Float32Array>>} Computed outputs.
     */
    async execute(outputIds = []) {
        const encoder = this.device.createCommandEncoder();

        for (const passConfig of this.passes) {
            let pipeline = this.pipelines.get(passConfig.wgslCode);
            if (!pipeline) {
                pipeline = createComputePipeline(this.device, passConfig.wgslCode);
                this.pipelines.set(passConfig.wgslCode, pipeline);
            }

            const entries = [];
            let bindingIdx = 0;

            for (const inId of (passConfig.inputIds || [])) {
                const meta = this.storageBuffers.get(inId);
                if (meta) {
                    entries.push({ binding: bindingIdx++, resource: { buffer: meta.buffer } });
                }
            }

            for (const uId of (passConfig.uniformIds || [])) {
                const uMeta = this.uniformBuffers.get(uId);
                if (uMeta) {
                    entries.push({ binding: bindingIdx++, resource: { buffer: uMeta.buffer } });
                }
            }

            const outMeta = this.storageBuffers.get(passConfig.outputId);
            if (outMeta) {
                entries.push({ binding: bindingIdx++, resource: { buffer: outMeta.buffer } });
            }

            const bindGroup = this.device.createBindGroup({
                layout: pipeline.getBindGroupLayout(0),
                entries: entries
            });

            const pass = encoder.beginComputePass();
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bindGroup);
            pass.dispatchWorkgroups(passConfig.dispatchX || 1, passConfig.dispatchY || 1, passConfig.dispatchZ || 1);
            pass.end();
        }

        const targets = (outputIds && outputIds.length > 0) ? outputIds : Array.from(this.storageBuffers.keys());
        const stagingBuffers = {};

        for (const outId of targets) {
            const meta = this.storageBuffers.get(outId);
            if (meta) {
                const staging = this.device.createBuffer({
                    size: meta.size,
                    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
                });
                encoder.copyBufferToBuffer(meta.buffer, 0, staging, 0, meta.size);
                stagingBuffers[outId] = { staging, size: meta.size };
            }
        }

        this.device.queue.submit([encoder.finish()]);

        const results = {};
        for (const [outId, meta] of Object.entries(stagingBuffers)) {
            await meta.staging.mapAsync(GPUMapMode.READ);
            results[outId] = new Float32Array(meta.staging.getMappedRange().slice(0));
            meta.staging.unmap();
            meta.staging.destroy();
        }

        return results;
    }

    /**
     * Cleans up all device buffers and resources.
     */
    destroy() {
        for (const meta of this.storageBuffers.values()) {
            meta.buffer.destroy();
        }
        for (const uMeta of this.uniformBuffers.values()) {
            uMeta.buffer.destroy();
        }
        this.storageBuffers.clear();
        this.uniformBuffers.clear();
    }
}

/**
 * Runs a multi-stage WebGPU compute pipeline with intermediate buffers and timestamp benchmarking.
 * @param {Navigator} nav - The navigator object.
 * @param {Array<Object>} passes - List of compute pass definitions.
 * @param {Object.<string, Float32Array>} initialInputs - Initial input tensors.
 * @param {Object} [options={}] - Execution options.
 * @param {Array<string>} [options.outputIds] - Designated outputs to read back.
 * @param {boolean} [options.enableTimestamps=false] - Whether to use WebGPU timestamp queries.
 * @returns {Promise<Object.<string, Float32Array>>} Computed output tensors with attached telemetry.
 */
async function runWebGPUMultiPassCompute(nav, passes, initialInputs, options = {}) {
    const requiredFeatures = options.enableTimestamps ? ['timestamp-query'] : [];
    const { device } = await initWebGPU(nav, requiredFeatures);
    const bufferRegistry = {};
    const uniformRegistry = {};

    // Populate initial inputs into device storage buffers
    for (const [id, data] of Object.entries(initialInputs)) {
        const buf = device.createBuffer({
            size: Math.max(16, data.byteLength),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
        });
        device.queue.writeBuffer(buf, 0, data);
        bufferRegistry[id] = { buffer: buf, size: Math.max(16, data.byteLength) };
    }

    const encoder = device.createCommandEncoder();

    // Check for hardware timestamp query support
    const hasTimestampQuery = device.features && device.features.has('timestamp-query');
    let querySet = null;
    let queryResolveBuffer = null;
    let queryStagingBuffer = null;
    const queryCount = passes.length * 2;

    if (hasTimestampQuery && options.enableTimestamps) {
        querySet = device.createQuerySet({
            type: 'timestamp',
            count: queryCount
        });
        queryResolveBuffer = device.createBuffer({
            size: queryCount * 8,
            usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC
        });
        queryStagingBuffer = device.createBuffer({
            label: 'timestamp_staging',
            size: queryCount * 8,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        });
    }

    const cpuStartTime = (typeof performance !== 'undefined') ? performance.now() : 0;

    // Encode multi-stage passes
    for (let i = 0; i < passes.length; i++) {
        const passConfig = passes[i];
        const outId = passConfig.outputId;
        const outSize = passConfig.outputSizeInBytes || 16;
        if (!bufferRegistry[outId]) {
            const outBuf = device.createBuffer({
                size: outSize,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
            });
            bufferRegistry[outId] = { buffer: outBuf, size: outSize };
        }

        // Automatic uniform buffer creation if pass provides metadata
        let uBuf = null;
        if (passConfig.uniform) {
            const uKey = `_uniform_${i}`;
            uBuf = createUniformBuffer(
                device,
                passConfig.uniform.shape,
                passConfig.uniform.strides,
                passConfig.uniform.offset || 0,
                passConfig.uniform.batchSize || 1
            );
            uniformRegistry[uKey] = uBuf;
        }

        const pipeline = createComputePipeline(device, passConfig.wgslCode);
        const entries = (passConfig.inputIds || []).map((inpId, idx) => ({
            binding: idx,
            resource: { buffer: bufferRegistry[inpId].buffer }
        }));

        let bindingIdx = entries.length;
        if (uBuf) {
            entries.push({
                binding: bindingIdx++,
                resource: { buffer: uBuf }
            });
        }

        entries.push({
            binding: bindingIdx,
            resource: { buffer: bufferRegistry[outId].buffer }
        });

        const bindGroup = device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: entries
        });

        const passDesc = {};
        if (querySet) {
            passDesc.timestampWrites = {
                querySet: querySet,
                beginningOfPassWriteIndex: i * 2,
                endOfPassWriteIndex: i * 2 + 1
            };
        }

        const pass = encoder.beginComputePass(passDesc);
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(passConfig.dispatchX || 1, passConfig.dispatchY || 1, passConfig.dispatchZ || 1);
        pass.end();
    }

    if (querySet && queryResolveBuffer && queryStagingBuffer) {
        encoder.resolveQuerySet(querySet, 0, queryCount, queryResolveBuffer, 0);
        encoder.copyBufferToBuffer(queryResolveBuffer, 0, queryStagingBuffer, 0, queryCount * 8);
    }

    // Readback designated outputs
    const requestedOutputs = options.outputIds || Object.keys(bufferRegistry);
    const stagingBuffers = {};
    for (const outId of requestedOutputs) {
        const meta = bufferRegistry[outId];
        if (meta) {
            const staging = device.createBuffer({
                size: meta.size,
                usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
            });
            encoder.copyBufferToBuffer(meta.buffer, 0, staging, 0, meta.size);
            stagingBuffers[outId] = { staging, size: meta.size };
        }
    }

    device.queue.submit([encoder.finish()]);

    const results = {};
    for (const [outId, meta] of Object.entries(stagingBuffers)) {
        await meta.staging.mapAsync(GPUMapMode.READ);
        results[outId] = new Float32Array(meta.staging.getMappedRange().slice(0));
        meta.staging.unmap();
        meta.staging.destroy();
    }

    let durationNs = 0;
    let isGpuTimestamp = false;
    if (queryStagingBuffer) {
        await queryStagingBuffer.mapAsync(GPUMapMode.READ);
        const bigIntView = new BigInt64Array(queryStagingBuffer.getMappedRange().slice(0));
        queryStagingBuffer.unmap();
        queryStagingBuffer.destroy();
        queryResolveBuffer.destroy();
        querySet.destroy();

        if (bigIntView.length >= 2) {
            durationNs = Number(bigIntView[bigIntView.length - 1] - bigIntView[0]);
            isGpuTimestamp = true;
        }
    } else {
        const cpuEndTime = (typeof performance !== 'undefined') ? performance.now() : 0;
        durationNs = Math.round((cpuEndTime - cpuStartTime) * 1e6);
    }

    const recordedShapes = {};
    for (const [id, bufMeta] of Object.entries(bufferRegistry)) {
        if (bufMeta && bufMeta.shape) {
            recordedShapes[id] = bufMeta.shape;
        }
    }
    for (const [id, uBufMeta] of Object.entries(uniformRegistry)) {
        if (uBufMeta && uBufMeta.shape) {
            recordedShapes[id] = uBufMeta.shape;
        }
    }

    results.telemetry = {
        duration_ns: durationNs,
        duration_ms: durationNs / 1e6,
        is_gpu_timestamp: isGpuTimestamp,
        passes_executed: passes.length,
        shapes: recordedShapes
    };

    for (const meta of Object.values(bufferRegistry)) {
        meta.buffer.destroy();
    }
    for (const uBuf of Object.values(uniformRegistry)) {
        uBuf.destroy();
    }
    device.destroy();

    return results;
}

/**
 * Executes a client-side WebGPU backward pass using multi-pass shader execution.
 * @param {Navigator} nav - The navigator object.
 * @param {Array<Object>} bwdPasses - Reverse-mode gradient compute passes.
 * @param {Object.<string, Float32Array>} primals - Forward pass activation tensors.
 * @param {Object.<string, Float32Array>} gradOutputs - Upstream gradient tensors.
 * @returns {Promise<Object.<string, Float32Array>>} Computed input gradients.
 */
async function runWebGPUBackward(nav, bwdPasses, primals, gradOutputs) {
    const combined = { ...primals, ...gradOutputs };
    return await runWebGPUMultiPassCompute(nav, bwdPasses, combined);
}

/**
 * Captures runtime tensor shapes from WebGPU compute execution.
 * @param {WebGPUPipelineChain|Map|Object} pipelineOrBuffers - Active pipeline or buffer metadata.
 * @returns {Object.<string, Array<number>>} Observed concrete shapes.
 */
function captureRuntimeShapes(pipelineOrBuffers) {
    if (!pipelineOrBuffers) return {};
    if (typeof pipelineOrBuffers.captureRuntimeShapes === 'function') {
        return pipelineOrBuffers.captureRuntimeShapes();
    }
    const shapes = {};
    if (pipelineOrBuffers instanceof Map) {
        for (const [k, v] of pipelineOrBuffers.entries()) {
            if (v && v.shape) {
                shapes[k] = Array.from(v.shape);
            } else if (v && typeof v.size === 'number') {
                shapes[k] = [Math.floor(v.size / 4)];
            }
        }
    } else if (typeof pipelineOrBuffers === 'object') {
        for (const [k, v] of Object.entries(pipelineOrBuffers)) {
            if (Array.isArray(v)) {
                shapes[k] = v;
            } else if (v && v.shape) {
                shapes[k] = Array.from(v.shape);
            } else if (v && typeof v.size === 'number') {
                shapes[k] = [Math.floor(v.size / 4)];
            }
        }
    }
    return shapes;
}

// Export for browser
/* c8 ignore next 14 */
if (typeof window !== 'undefined') {
    window.initWebGPU = initWebGPU;
    window.createComputePipeline = createComputePipeline;
    window.runWebGPUCompute = runWebGPUCompute;
    window.createUniformBuffer = createUniformBuffer;
    window.updateUniformBuffer = updateUniformBuffer;
    window.WebGPUPipelineChain = WebGPUPipelineChain;
    window.runWebGPUMultiPassCompute = runWebGPUMultiPassCompute;
    window.runWebGPUBackward = runWebGPUBackward;
    window.captureRuntimeShapes = captureRuntimeShapes;
}

// Export for testing
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        initWebGPU,
        createComputePipeline,
        runWebGPUCompute,
        createUniformBuffer,
        updateUniformBuffer,
        WebGPUPipelineChain,
        runWebGPUMultiPassCompute,
        runWebGPUBackward,
        captureRuntimeShapes
    };
}
