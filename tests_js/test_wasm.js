const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const jsCode = fs.readFileSync(path.join(__dirname, '../docs/_static/wasm_runner.js'), 'utf8');

let wasmModule;
try {
    const fn = new Function('module', 'exports', jsCode);
    const m = { exports: {} };
    fn(m, m.exports);
    wasmModule = m.exports;
} catch (e) {
    console.error("Error loading wasm module:", e);
}

test('validateMemoryBounds throws on invalid', () => {
    const memory = { buffer: new ArrayBuffer(16) }; // 16 bytes
    assert.throws(() => wasmModule.validateMemoryBounds(memory, 0, 20), /bounds violation/);
    assert.throws(() => wasmModule.validateMemoryBounds(memory, -1, 10), /bounds violation/);
});

test('validateMemoryBounds passes on valid', () => {
    const memory = { buffer: new ArrayBuffer(16) }; // 16 bytes
    assert.doesNotThrow(() => wasmModule.validateMemoryBounds(memory, 0, 16));
    assert.doesNotThrow(() => wasmModule.validateMemoryBounds(memory, 4, 12));
});

test('runWasmCompute runs correctly', async () => {
    const mockMemory = new WebAssembly.Memory({ initial: 1 });
    
    // We mock global WebAssembly.instantiate
    const originalInstantiate = WebAssembly.instantiate;
    
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
                    }
                }
            }
        };
    };
    
    const inputData = new Float32Array([10, 20, 30]);
    const output = await wasmModule.runWasmCompute(new Uint8Array(), inputData, 3);
    
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

test('runWasmCompute throws if no compute exported', async () => {
    const originalInstantiate = WebAssembly.instantiate;
    WebAssembly.instantiate = async () => ({
        instance: { exports: { memory: new WebAssembly.Memory({initial: 1}) } }
    });
    
    await assert.rejects(
        () => wasmModule.runWasmCompute(new Uint8Array(), new Float32Array(1), 1),
        /must export a 'compute' function/
    );
    WebAssembly.instantiate = originalInstantiate;
});
