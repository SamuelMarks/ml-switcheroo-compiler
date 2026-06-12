/**
 * WASM SIMD Execution Module for ML Switcheroo
 * Handles WASM instantiation, memory management, and execution.
 * @module WasmRunner
 */

/**
 * Validates memory bounds.
 * @param {WebAssembly.Memory} memory - The WASM memory instance.
 * @param {number} offset - The starting offset in bytes.
 * @param {number} lengthInBytes - The length of the array in bytes.
 * @throws {Error} If out of bounds.
 */
function validateMemoryBounds(memory, offset, lengthInBytes) {
    if (offset < 0 || offset + lengthInBytes > memory.buffer.byteLength) {
        throw new Error(`Memory bounds violation: offset=${offset}, length=${lengthInBytes}, buffer size=${memory.buffer.byteLength}`);
    }
}

/**
 * Runs a WASM computation with provided inputs.
 * The WASM module is expected to export a `compute` function and optionally a `memory`.
 * We use SIMD by default if the browser supports it and the module is compiled with it.
 * @param {Uint8Array} wasmBinary - The compiled WASM binary.
 * @param {Float32Array} inputData - The input data to populate in memory.
 * @param {number} expectedOutputLength - The expected number of Float32 elements in the output.
 * @returns {Promise<Float32Array>} The computed output data.
 */
async function runWasmCompute(wasmBinary, inputData, expectedOutputLength) {
    if (!WebAssembly || !WebAssembly.instantiate) {
        throw new Error("WebAssembly is not supported in this browser.");
    }

    // Attempt to instantiate the WASM module
    let wasmModule;
    try {
        const result = await WebAssembly.instantiate(wasmBinary, {});
        wasmModule = result.instance;
    } catch (e) {
        throw new Error(`Failed to instantiate WASM module: ${e.message}`);
    }

    const exports = wasmModule.exports;

    // Ensure the module exported a memory and a compute function
    if (!exports.memory || !(exports.memory instanceof WebAssembly.Memory)) {
        throw new Error("WASM module must export 'memory'.");
    }
    if (typeof exports.compute !== 'function') {
        throw new Error("WASM module must export a 'compute' function.");
    }

    const memory = exports.memory;
    const inputByteLength = inputData.length * Float32Array.BYTES_PER_ELEMENT;
    const outputByteLength = expectedOutputLength * Float32Array.BYTES_PER_ELEMENT;

    // For simplicity, input at offset 0, output right after
    const inputOffset = 0;
    const outputOffset = inputByteLength;

    // Validate bounds
    validateMemoryBounds(memory, inputOffset, inputByteLength);
    validateMemoryBounds(memory, outputOffset, outputByteLength);

    // Populate input
    const memFloat32 = new Float32Array(memory.buffer);
    memFloat32.set(inputData, inputOffset / Float32Array.BYTES_PER_ELEMENT);

    // Run compute
    exports.compute(inputOffset, inputData.length, outputOffset);

    // Read back output
    const outputData = new Float32Array(expectedOutputLength);
    outputData.set(memFloat32.subarray(
        outputOffset / Float32Array.BYTES_PER_ELEMENT,
        (outputOffset + outputByteLength) / Float32Array.BYTES_PER_ELEMENT
    ));

    return outputData;
}

// Export for testing
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        validateMemoryBounds,
        runWasmCompute
    };
}
