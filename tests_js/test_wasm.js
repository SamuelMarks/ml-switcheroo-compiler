const test = require('node:test');
const assert = require('node:assert');

const wasmModule = require('../docs/_static/wasm_runner.js');

test('validateMemoryBounds throws on invalid bounds', () => {
    const memory = { buffer: new ArrayBuffer(16) }; // 16 bytes
    assert.throws(() => wasmModule.validateMemoryBounds(memory, 0, 20), /bounds violation/);
    assert.throws(() => wasmModule.validateMemoryBounds(memory, -1, 10), /bounds violation/);
});

test('validateMemoryBounds checks exactly on bounds', () => {
    const memory = { buffer: new ArrayBuffer(16) }; // 16 bytes
    assert.throws(() => wasmModule.validateMemoryBounds(memory, 10, 10), /bounds violation/);
    assert.doesNotThrow(() => wasmModule.validateMemoryBounds(memory, 10, 6)); // Exactly fits
});

test('runWasmCompute validates instantiation properly', async () => {
    // Inject a valid mock WASM to verify the JavaScript orchestration
    // Since building LLVM WASM locally in CI is unstable, we focus on the runtime API contract

    const mockMemory = new WebAssembly.Memory({ initial: 1 });
    const originalInstantiate = WebAssembly.instantiate;

    let computedValues = false;

    WebAssembly.instantiate = async () => {
        return {
            instance: {
                exports: {
                    memory: mockMemory,
                    compute: (inputOffset, inputLength, outputOffset) => {
                        // Mock computation: just copy and add 1
                        const memFloat32 = new Float32Array(mockMemory.buffer);
                        for (let i=0; i<inputLength; i++) {
                            const val = memFloat32[inputOffset/4 + i];
                            memFloat32[outputOffset/4 + i] = val + 1;
                        }
                        computedValues = true;
                    }
                }
            }
        };
    };

    const inputData = new Float32Array([10, 20, 30]);
    const output = await wasmModule.runWasmCompute(new Uint8Array(), inputData, 3);

    assert.strictEqual(computedValues, true);
    assert.strictEqual(output.length, 3);
    assert.strictEqual(output[0], 11);
    assert.strictEqual(output[1], 21);
    assert.strictEqual(output[2], 31);

    // restore
    WebAssembly.instantiate = originalInstantiate;
});

test('runWasmCompute throws if no memory exported', async () => {
    const originalInstantiate = WebAssembly.instantiate;
    WebAssembly.instantiate = async () => ({
        instance: { exports: { compute: () => {} } }
    });

    await assert.rejects(
        () => wasmModule.runWasmCompute(new Uint8Array(), new Float32Array(1), 1),
        /must export 'memory'/
    );
    WebAssembly.instantiate = originalInstantiate;
});
