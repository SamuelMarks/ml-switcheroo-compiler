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

test('encodeWasmSection encodes id and payload length', () => {
    const encoded = wasmModule.encodeWasmSection(1, [10, 20]);
    assert.deepStrictEqual(encoded, [1, 2, 10, 20]);
});

test('compileWasmKernel produces valid WASM binaries for all ops', async () => {
    const ops = ['mul', 'add', 'sub', 'div', 'copy'];
    for (const op of ops) {
        const bin = wasmModule.compileWasmKernel(op);
        assert.ok(bin instanceof Uint8Array);
        assert.ok(bin.length > 20);
    }
});

test('compileWasmFromCode detects op from string', () => {
    assert.ok(wasmModule.compileWasmFromCode('x + y') instanceof Uint8Array);
    assert.ok(wasmModule.compileWasmFromCode('x - y') instanceof Uint8Array);
    assert.ok(wasmModule.compileWasmFromCode('x / y') instanceof Uint8Array);
    assert.ok(wasmModule.compileWasmFromCode('matmul(x, y)') instanceof Uint8Array);
    assert.ok(wasmModule.compileWasmFromCode(null) instanceof Uint8Array);
});

test('runWasmCompute executes real numerical tensors without mocks', async () => {
    const mulKernel = wasmModule.compileWasmKernel('mul');
    const input = new Float32Array([2.0, 3.0, 4.0]);
    const output = await wasmModule.runWasmCompute(mulKernel, input, 3);
    assert.strictEqual(output.length, 3);
    assert.strictEqual(output[0], 4.0);
    assert.strictEqual(output[1], 9.0);
    assert.strictEqual(output[2], 16.0);

    const addKernel = wasmModule.compileWasmKernel('add');
    const addOutput = await wasmModule.runWasmCompute(addKernel, input, 3);
    assert.strictEqual(addOutput[0], 4.0);
    assert.strictEqual(addOutput[1], 6.0);
    assert.strictEqual(addOutput[2], 8.0);

    const copyKernel = wasmModule.compileWasmKernel('copy');
    const copyOutput = await wasmModule.runWasmCompute(copyKernel, input, 3);
    assert.strictEqual(copyOutput[0], 2.0);
    assert.strictEqual(copyOutput[1], 3.0);
    assert.strictEqual(copyOutput[2], 4.0);

    const sqrtKernel = wasmModule.compileWasmKernel('sqrt');
    const sqrtInput = new Float32Array([4.0, 9.0, 16.0]);
    const sqrtOutput = await wasmModule.runWasmCompute(sqrtKernel, sqrtInput, 3);
    assert.strictEqual(sqrtOutput[0], 2.0);
    assert.strictEqual(sqrtOutput[1], 3.0);
    assert.strictEqual(sqrtOutput[2], 4.0);

    const absKernel = wasmModule.compileWasmFromCode('float abs_val = abs(x);');
    const negInput = new Float32Array([-5.0, -10.0, 15.0]);
    const absOutput = await wasmModule.runWasmCompute(absKernel, negInput, 3);
    assert.strictEqual(absOutput[0], 5.0);
    assert.strictEqual(absOutput[1], 10.0);
    assert.strictEqual(absOutput[2], 15.0);

    const negKernel = wasmModule.compileWasmKernel('neg');
    const negOutput = await wasmModule.runWasmCompute(negKernel, input, 3);
    assert.strictEqual(negOutput[0], -2.0);
    assert.strictEqual(negOutput[1], -3.0);
    assert.strictEqual(negOutput[2], -4.0);

    const ceilKernel = wasmModule.compileWasmKernel('ceil');
    const ceilInput = new Float32Array([1.2, 2.7, -3.4]);
    const ceilOutput = await wasmModule.runWasmCompute(ceilKernel, ceilInput, 3);
    assert.strictEqual(ceilOutput[0], 2.0);
    assert.strictEqual(ceilOutput[1], 3.0);
    assert.strictEqual(ceilOutput[2], -3.0);

    const floorKernel = wasmModule.compileWasmKernel('floor');
    const floorOutput = await wasmModule.runWasmCompute(floorKernel, ceilInput, 3);
    assert.strictEqual(floorOutput[0], 1.0);
    assert.strictEqual(floorOutput[1], 2.0);
    assert.strictEqual(floorOutput[2], -4.0);
});

test('runWasmCompute throws on invalid binary', async () => {
    const invalidBin = new Uint8Array([1, 2, 3, 4]);
    const input = new Float32Array([1.0]);
    await assert.rejects(
        () => wasmModule.runWasmCompute(invalidBin, input, 1),
        /Failed to instantiate WASM module/
    );
});

test('runWasmCompute throws if no compute exported', async () => {
    const originalInstantiate = WebAssembly.instantiate;
    WebAssembly.instantiate = async () => ({
        instance: {
            exports: {
                memory: new WebAssembly.Memory({ initial: 1 })
            }
        }
    });

    await assert.rejects(
        () => wasmModule.runWasmCompute(new Uint8Array(), new Float32Array(1), 1),
        /must export a 'compute' function/
    );
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

test('WasmTensor calculates multi-dimensional shapes and strides', () => {
    const tensor = new wasmModule.WasmTensor([2, 3, 4], 64);
    assert.deepStrictEqual(tensor.shape, [2, 3, 4]);
    assert.deepStrictEqual(tensor.strides, [12, 4, 1]);
    assert.strictEqual(tensor.numElements, 24);
    assert.strictEqual(tensor.byteLength, 96);
    assert.strictEqual(tensor.byteOffset, 64);
});

test('compileWasmGraph schedules multi-node IRGraph execution', () => {
    const graph = {
        inputs: ['x'],
        outputs: ['out'],
        nodes: [
            { id: 'x', op_type: 'Input', shape_metadata: [4] },
            { id: 'sq', op_type: 'Mul', inputs: ['x', 'x'], shape_metadata: [4] },
            { id: 'out', op_type: 'Add', inputs: ['sq', 'x'], shape_metadata: [4] }
        ]
    };

    const plan = wasmModule.compileWasmGraph(graph);
    assert.ok(plan);
    assert.strictEqual(plan.executionSteps.length, 2);
    assert.deepStrictEqual(plan.shapes['x'], [4]);
    assert.deepStrictEqual(plan.shapes['sq'], [4]);
    assert.deepStrictEqual(plan.shapes['out'], [4]);
    assert.ok(plan.totalMemoryBytes > 0);
});

test('runWasmGraph executes multi-node computation and extracts telemetry', async () => {
    const graph = {
        inputs: ['x'],
        outputs: ['out'],
        nodes: [
            { id: 'x', op_type: 'Input', shape_metadata: [3] },
            { id: 'out', op_type: 'Mul', inputs: ['x', 'x'], shape_metadata: [3] }
        ]
    };

    const plan = wasmModule.compileWasmGraph(graph);
    const inputs = { x: new Float32Array([2.0, 3.0, 4.0]) };
    const res = await wasmModule.runWasmGraph(plan, inputs);

    assert.ok(res.outputs['out']);
    assert.strictEqual(res.outputs['out'][0], 4.0);
    assert.strictEqual(res.outputs['out'][1], 9.0);
    assert.strictEqual(res.outputs['out'][2], 16.0);

    assert.ok(res.telemetry);
    assert.deepStrictEqual(res.telemetry.shapes['x'], [3]);
    assert.deepStrictEqual(res.telemetry.shapes['out'], [3]);
    assert.ok(res.telemetry.memory_bytes > 0);
});

test('runWasmBackward executes reverse-mode AD gradient graph', async () => {
    const bwdGraph = {
        inputs: ['primal_x', 'grad_out'],
        outputs: ['grad_x'],
        nodes: [
            { id: 'primal_x', op_type: 'Input', shape_metadata: [2] },
            { id: 'grad_out', op_type: 'Input', shape_metadata: [2] },
            { id: 'grad_x', op_type: 'Add', inputs: ['primal_x', 'primal_x'], shape_metadata: [2] }
        ]
    };

    const plan = wasmModule.compileWasmGraph(bwdGraph);
    const primals = { primal_x: new Float32Array([3.0, 5.0]) };
    const grads = { grad_out: new Float32Array([1.0, 1.0]) };

    const res = await wasmModule.runWasmBackward(plan, primals, grads);
    assert.ok(res.outputs['grad_x']);
    assert.strictEqual(res.outputs['grad_x'][0], 6.0);
    assert.strictEqual(res.outputs['grad_x'][1], 10.0);
});

test('encodeU32Leb128 encodes integers correctly', () => {
    assert.deepStrictEqual(wasmModule.encodeU32Leb128(0), [0]);
    assert.deepStrictEqual(wasmModule.encodeU32Leb128(1), [1]);
    assert.deepStrictEqual(wasmModule.encodeU32Leb128(127), [127]);
    assert.deepStrictEqual(wasmModule.encodeU32Leb128(128), [128, 1]);
    assert.deepStrictEqual(wasmModule.encodeU32Leb128(624485), [229, 142, 38]);
});

test('WasmMemoryArena allocates, aligns, grows, and manages memory safely', () => {
    const arena = new wasmModule.WasmMemoryArena(1, 4);
    assert.strictEqual(arena.offset, 0);

    const off1 = arena.allocate(32, 16);
    assert.strictEqual(off1, 0);
    assert.strictEqual(arena.offset, 32);

    const off2 = arena.allocate(16, 16);
    assert.strictEqual(off2, 32);
    assert.strictEqual(arena.offset, 48);

    // Test writing and reading floats
    arena.writeFloat32(off1, [1.5, 2.5, 3.5, 4.5]);
    const floats = arena.getFloat32Array(off1, 4);
    assert.strictEqual(floats[0], 1.5);
    assert.strictEqual(floats[1], 2.5);
    assert.strictEqual(floats[2], 3.5);
    assert.strictEqual(floats[3], 4.5);

    // Test bounds checking
    assert.throws(() => arena.validateBounds(-1, 16), /bounds violation/);
    assert.throws(() => arena.validateBounds(0, 100000000), /bounds violation/);

    // Test memory growth
    const largeOff = arena.allocate(100000, 16);
    assert.ok(largeOff >= 48);
    assert.ok(arena.memory.buffer.byteLength >= 100000);

    // Exceed max pages throws
    assert.throws(() => arena.allocate(1000000, 16), /exceeded max pages/);

    // Reset arena
    arena.reset();
    assert.strictEqual(arena.offset, 0);
});

test('WasmBinaryBuilder constructs executable custom binary module', async () => {
    const builder = new wasmModule.WasmBinaryBuilder();
    const typeIdx = builder.addType([0x7f, 0x7f], [0x7f]);
    builder.setMemory(1, 2);
    builder.addExport("add_ints", 0, 0);

    // compute: return a + b (local.get 0, local.get 1, i32.add)
    const body = [0x20, 0x00, 0x20, 0x01, 0x6a];
    builder.addFunction(typeIdx, [], body);

    const bin = builder.build();
    assert.ok(bin instanceof Uint8Array);

    const inst = (await WebAssembly.instantiate(bin, {})).instance;
    assert.strictEqual(typeof inst.exports.add_ints, 'function');
    assert.strictEqual(inst.exports.add_ints(15, 27), 42);
});

test('compileWasmMatmulKernel executes 2D matrix multiplication', async () => {
    const kernel = wasmModule.compileWasmMatmulKernel(2);
    const inst = (await WebAssembly.instantiate(kernel, {})).instance;
    const mem = new Float32Array(inst.exports.memory.buffer);

    // A = [[1, 2], [3, 4]] (2x2) at offset 0
    mem.set([1, 2, 3, 4], 0);
    // B = [[5, 6], [7, 8]] (2x2) at offset 16 (index 4)
    mem.set([5, 6, 7, 8], 4);
    // C at offset 32 (index 8)
    inst.exports.compute_matmul(0, 16, 32, 2, 2, 2);

    // Expected C = [[19, 22], [43, 50]]
    assert.strictEqual(mem[8], 19);
    assert.strictEqual(mem[9], 22);
    assert.strictEqual(mem[10], 43);
    assert.strictEqual(mem[11], 50);
});

test('compileWasmTransposeKernel transposes 2D tensors', async () => {
    const kernel = wasmModule.compileWasmTransposeKernel(1);
    const inst = (await WebAssembly.instantiate(kernel, {})).instance;
    const mem = new Float32Array(inst.exports.memory.buffer);

    // 2x3 matrix: [[1, 2, 3], [4, 5, 6]]
    mem.set([1, 2, 3, 4, 5, 6], 0);
    inst.exports.compute_transpose(0, 32, 2, 3);

    // Expected 3x2: [[1, 4], [2, 5], [3, 6]]
    assert.deepStrictEqual(Array.from(mem.slice(8, 14)), [1, 4, 2, 5, 3, 6]);
});

test('compileWasmSliceKernel slices strided linear tensors', async () => {
    const kernel = wasmModule.compileWasmSliceKernel(1);
    const inst = (await WebAssembly.instantiate(kernel, {})).instance;
    const mem = new Float32Array(inst.exports.memory.buffer);

    mem.set([10, 20, 30, 40, 50, 60], 0);
    // start=1, step=2, count=3 -> [20, 40, 60]
    inst.exports.compute_slice(0, 32, 1, 2, 3);
    assert.deepStrictEqual(Array.from(mem.slice(8, 11)), [20, 40, 60]);
});

test('compileWasmReduceKernel reduces tensors (sum, mean, min, max, prod)', async () => {
    const ops = ['sum', 'mean', 'min', 'max', 'prod'];
    const data = [2.0, 4.0, 6.0, 8.0];

    for (const op of ops) {
        const kernel = wasmModule.compileWasmReduceKernel(op, 1);
        const inst = (await WebAssembly.instantiate(kernel, {})).instance;
        const mem = new Float32Array(inst.exports.memory.buffer);
        mem.set(data, 0);
        inst.exports.compute_reduce(0, 32, 4);

        if (op === 'sum') assert.strictEqual(mem[8], 20.0);
        if (op === 'mean') assert.strictEqual(mem[8], 5.0);
        if (op === 'min') assert.strictEqual(mem[8], 2.0);
        if (op === 'max') assert.strictEqual(mem[8], 8.0);
        if (op === 'prod') assert.strictEqual(mem[8], 384.0);
    }
});

test('compileWasmGraph and runWasmGraph handle 2D matmul and reductions', async () => {
    const graph = {
        inputs: ['A', 'B'],
        outputs: ['C', 'T', 'S', 'R'],
        nodes: [
            { id: 'A', op_type: 'Input', shape_metadata: [2, 2] },
            { id: 'B', op_type: 'Input', shape_metadata: [2, 2] },
            { id: 'C', op_type: 'Matmul', inputs: ['A', 'B'], shape_metadata: [2, 2] },
            { id: 'T', op_type: 'Transpose', inputs: ['C'], shape_metadata: [2, 2] },
            { id: 'S', op_type: 'Slice', inputs: ['C'], shape_metadata: [2], attributes: { start: 0, step: 2 } },
            { id: 'R', op_type: 'ReduceSum', inputs: ['C'], shape_metadata: [1] }
        ]
    };

    const plan = wasmModule.compileWasmGraph(graph);
    assert.strictEqual(plan.executionSteps.length, 4);

    const inputs = {
        A: new Float32Array([1, 2, 3, 4]),
        B: new Float32Array([5, 6, 7, 8])
    };

    const res = await wasmModule.runWasmGraph(plan, inputs);
    assert.ok(res.outputs['C']);
    assert.deepStrictEqual(Array.from(res.outputs['C']), [19, 22, 43, 50]);

    assert.ok(res.outputs['T']);
    assert.deepStrictEqual(Array.from(res.outputs['T']), [19, 43, 22, 50]);

    assert.ok(res.outputs['S']);
    assert.deepStrictEqual(Array.from(res.outputs['S']), [19, 43]);

    assert.ok(res.outputs['R']);
    assert.strictEqual(res.outputs['R'][0], 134);
});
