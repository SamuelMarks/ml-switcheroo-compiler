/**
 * WASM SIMD Execution Module for ML Switcheroo
 * Handles WASM instantiation, dynamic linear memory management, binary generation, and graph execution.
 * @module WasmRunner
 */

/**
 * Encodes an unsigned 32-bit integer as LEB128 bytes.
 * @param {number} val - Unsigned integer to encode.
 * @returns {Array<number>} LEB128 encoded bytes.
 */
function encodeU32Leb128(val) {
    const bytes = [];
    let num = Math.floor(Math.max(0, val));
    do {
        let byte = num & 0x7f;
        num >>>= 7;
        if (num !== 0) {
            byte |= 0x80;
        }
        bytes.push(byte);
    } while (num !== 0);
    return bytes;
}

/**
 * Encodes a WebAssembly section with its ID and payload length.
 * @param {number} id - Section ID.
 * @param {Array<number>} payload - Byte array payload of the section.
 * @returns {Array<number>} Encoded section bytes.
 */
function encodeWasmSection(id, payload) {
    return [id, ...encodeU32Leb128(payload.length), ...payload];
}

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
 * Dynamic linear memory arena for managing multi-tensor execution graphs.
 */
class WasmMemoryArena {
    /**
     * @param {number} [initialPages=1] - Starting number of 64KB pages.
     * @param {number} [maxPages=256] - Upper bound on 64KB pages.
     */
    constructor(initialPages = 1, maxPages = 256) {
        this.maxPages = maxPages;
        this.memory = new WebAssembly.Memory({
            initial: Math.max(1, initialPages),
            maximum: maxPages
        });
        this.offset = 0;
        this.allocations = new Map();
    }

    /**
     * Allocates a contiguous block of device memory aligned to specified boundary.
     * @param {number} sizeInBytes - Allocation size in bytes.
     * @param {number} [alignment=16] - Byte alignment requirement.
     * @returns {number} Byte offset of the allocated buffer.
     * @throws {Error} If maximum memory is exceeded.
     */
    allocate(sizeInBytes, alignment = 16) {
        const remainder = this.offset % alignment;
        if (remainder !== 0) {
            this.offset += alignment - remainder;
        }
        const allocOffset = this.offset;
        const totalNeeded = allocOffset + sizeInBytes;

        if (totalNeeded > this.memory.buffer.byteLength) {
            const bytesDeficit = totalNeeded - this.memory.buffer.byteLength;
            const pagesNeeded = Math.ceil(bytesDeficit / 65536);
            const currentPages = Math.floor(this.memory.buffer.byteLength / 65536);
            if (currentPages + pagesNeeded > this.maxPages) {
                throw new Error(`WasmMemoryArena exceeded max pages limit of ${this.maxPages}`);
            }
            this.memory.grow(pagesNeeded);
        }

        this.offset = totalNeeded;
        this.allocations.set(allocOffset, sizeInBytes);
        return allocOffset;
    }

    /**
     * Validates that an offset and range fit within the allocated buffer.
     * @param {number} offset - Starting offset.
     * @param {number} lengthInBytes - Byte length.
     * @throws {Error} If range violates buffer boundaries.
     */
    validateBounds(offset, lengthInBytes) {
        validateMemoryBounds(this.memory, offset, lengthInBytes);
    }

    /**
     * Resets the allocation head to the beginning of linear memory.
     */
    reset() {
        this.offset = 0;
        this.allocations.clear();
    }

    /**
     * Retrieves a Float32Array view at the specified offset.
     * @param {number} byteOffset - Memory byte offset.
     * @param {number} elementCount - Number of float elements.
     * @returns {Float32Array} Typed array view.
     */
    getFloat32Array(byteOffset, elementCount) {
        this.validateBounds(byteOffset, elementCount * 4);
        return new Float32Array(this.memory.buffer, byteOffset, elementCount);
    }

    /**
     * Writes Float32 data into linear memory at the specified offset.
     * @param {number} byteOffset - Destination byte offset.
     * @param {Float32Array|Array<number>} data - Source array.
     */
    writeFloat32(byteOffset, data) {
        this.validateBounds(byteOffset, data.length * 4);
        const view = new Float32Array(this.memory.buffer, byteOffset, data.length);
        view.set(data);
    }
}

/**
 * Lightweight WebAssembly binary builder for IR-driven kernel compilation.
 */
class WasmBinaryBuilder {
    /**
     * Initializes empty WebAssembly module builder sections.
     */
    constructor() {
        this.types = [];
        this.funcs = [];
        this.exports = [];
        this.memories = [];
        this.codes = [];
    }

    /**
     * Registers a function signature type.
     * @param {Array<number>} paramTypes - WASM value types (e.g. 0x7f for i32, 0x7d for f32).
     * @param {Array<number>} resultTypes - WASM value return types.
     * @returns {number} Registered type index.
     */
    addType(paramTypes, resultTypes = []) {
        const idx = this.types.length;
        this.types.push([
            0x60,
            ...encodeU32Leb128(paramTypes.length),
            ...paramTypes,
            ...encodeU32Leb128(resultTypes.length),
            ...resultTypes
        ]);
        return idx;
    }

    /**
     * Configures exported module linear memory limits.
     * @param {number} [initialPages=1] - Initial page allocation.
     * @param {number|null} [maxPages=null] - Optional maximum pages.
     */
    setMemory(initialPages = 1, maxPages = null) {
        if (maxPages === null) {
            this.memories = [0x01, 0x00, ...encodeU32Leb128(initialPages)];
        } else {
            this.memories = [0x01, 0x01, ...encodeU32Leb128(initialPages), ...encodeU32Leb128(maxPages)];
        }
    }

    /**
     * Declares an exported symbol.
     * @param {string} name - Export name.
     * @param {number} kind - Export kind (0=func, 2=mem).
     * @param {number} index - Entity index.
     */
    addExport(name, kind, index) {
        const nameBytes = [];
        for (let i = 0; i < name.length; i++) {
            nameBytes.push(name.charCodeAt(i));
        }
        this.exports.push([
            ...encodeU32Leb128(nameBytes.length),
            ...nameBytes,
            kind,
            ...encodeU32Leb128(index)
        ]);
    }

    /**
     * Appends a compiled function with local variable declarations.
     * @param {number} typeIdx - Type table signature index.
     * @param {Array<{count: number, type: number}>} locals - Local variables declaration.
     * @param {Array<number>} bodyBytes - Function bytecode instructions.
     * @returns {number} Function table index.
     */
    addFunction(typeIdx, locals, bodyBytes) {
        const fIdx = this.funcs.length;
        this.funcs.push(typeIdx);

        const localEntries = [...encodeU32Leb128(locals.length)];
        for (const loc of locals) {
            localEntries.push(...encodeU32Leb128(loc.count), loc.type);
        }
        const fullBody = [...localEntries, ...bodyBytes, 0x0b];
        this.codes.push([...encodeU32Leb128(fullBody.length), ...fullBody]);
        return fIdx;
    }

    /**
     * Assembles all module sections into a standard WebAssembly binary buffer.
     * @returns {Uint8Array} Binary WASM bytecode.
     */
    build() {
        const typePayload = [...encodeU32Leb128(this.types.length)];
        for (const t of this.types) {
            typePayload.push(...t);
        }

        const funcPayload = [...encodeU32Leb128(this.funcs.length)];
        for (const f of this.funcs) {
            funcPayload.push(...encodeU32Leb128(f));
        }

        const expPayload = [...encodeU32Leb128(this.exports.length)];
        for (const e of this.exports) {
            expPayload.push(...e);
        }

        const codePayload = [...encodeU32Leb128(this.codes.length)];
        for (const c of this.codes) {
            codePayload.push(...c);
        }

        const sections = [
            0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
            ...encodeWasmSection(1, typePayload),
            ...encodeWasmSection(3, funcPayload)
        ];

        if (this.memories.length > 0) {
            sections.push(...encodeWasmSection(5, this.memories));
        }

        sections.push(...encodeWasmSection(7, expPayload));
        sections.push(...encodeWasmSection(10, codePayload));

        return new Uint8Array(sections);
    }
}

/**
 * Compiles an in-browser WASM binary kernel that exports `memory` and `compute`.
 * Supports elementwise unary and binary operations.
 * @param {string} [op='mul'] - Operation to perform.
 * @returns {Uint8Array} Executable WebAssembly binary buffer.
 */
function compileWasmKernel(op = 'mul') {
    let opByte = 0x94; // f32.mul
    let isCopy = false;
    let isUnary = false;

    if (op === 'add') {
        opByte = 0x92;
    } else if (op === 'sub') {
        opByte = 0x93;
    } else if (op === 'div') {
        opByte = 0x95;
    } else if (op === 'min') {
        opByte = 0x96;
    } else if (op === 'max') {
        opByte = 0x97;
    } else if (op === 'abs') {
        opByte = 0x8b;
        isUnary = true;
    } else if (op === 'neg') {
        opByte = 0x8c;
        isUnary = true;
    } else if (op === 'sqrt') {
        opByte = 0x91;
        isUnary = true;
    } else if (op === 'ceil') {
        opByte = 0x8d;
        isUnary = true;
    } else if (op === 'floor') {
        opByte = 0x8e;
        isUnary = true;
    } else if (op === 'copy') {
        isCopy = true;
    }

    const builder = new WasmBinaryBuilder();
    // Signature: compute(in_off: i32, len: i32, out_off: i32) -> void
    const typeIdx = builder.addType([0x7f, 0x7f, 0x7f], []);
    builder.setMemory(1);
    builder.addExport("memory", 2, 0);

    const body = [
        0x41, 0x00, 0x21, 0x03, // local.set 3 (i = 0)
        0x02, 0x40,             // block
        0x03, 0x40,             // loop
        0x20, 0x03, 0x20, 0x01, 0x4f, // if i >= len (i32.ge_u)
        0x0d, 0x01,             // br_if 1 (exit block)
        // out_ptr = out_off + (i << 2)
        0x20, 0x02, 0x20, 0x03, 0x41, 0x02, 0x74, 0x6a,
        // in_ptr = in_off + (i << 2)
        0x20, 0x00, 0x20, 0x03, 0x41, 0x02, 0x74, 0x6a,
        0x2a, 0x02, 0x00        // f32.load
    ];

    if (isUnary) {
        body.push(opByte);
    } else if (!isCopy) {
        body.push(
            0x20, 0x00, 0x20, 0x03, 0x41, 0x02, 0x74, 0x6a,
            0x2a, 0x02, 0x00,   // f32.load second operand
            opByte              // arithmetic op
        );
    }

    body.push(
        0x38, 0x02, 0x00,       // f32.store
        0x20, 0x03, 0x41, 0x01, 0x6a, 0x21, 0x03, // i = i + 1
        0x0c, 0x00,             // br 0 (repeat loop)
        0x0b,                   // end loop
        0x0b                    // end block
    );

    builder.addFunction(typeIdx, [{ count: 1, type: 0x7f }], body);
    builder.addExport("compute", 0, 0);

    return builder.build();
}

/**
 * Compiles a binary arithmetic WebAssembly kernel for two distinct input buffers.
 * Signature: compute_binary(in0_off: i32, in1_off: i32, len: i32, out_off: i32) -> void
 * @param {string} [op='add'] - Binary operation name.
 * @returns {Uint8Array} WebAssembly binary buffer.
 */
function compileWasmBinaryOpKernel(op = 'add') {
    let opByte = 0x92; // f32.add
    if (op === 'mul') {
        opByte = 0x94;
    } else if (op === 'sub') {
        opByte = 0x93;
    } else if (op === 'div') {
        opByte = 0x95;
    } else if (op === 'min') {
        opByte = 0x96;
    } else if (op === 'max') {
        opByte = 0x97;
    }

    const builder = new WasmBinaryBuilder();
    const typeIdx = builder.addType([0x7f, 0x7f, 0x7f, 0x7f], []);
    builder.setMemory(1);
    builder.addExport("memory", 2, 0);

    const body = [
        0x41, 0x00, 0x21, 0x04, // local.set 4 (i = 0)
        0x02, 0x40,             // block
        0x03, 0x40,             // loop
        0x20, 0x04, 0x20, 0x02, 0x4f, // if i >= len (i32.ge_u)
        0x0d, 0x01,             // br_if 1
        // out_ptr = out_off + (i << 2)
        0x20, 0x03, 0x20, 0x04, 0x41, 0x02, 0x74, 0x6a,
        // in0_ptr = in0_off + (i << 2)
        0x20, 0x00, 0x20, 0x04, 0x41, 0x02, 0x74, 0x6a,
        0x2a, 0x02, 0x00,       // f32.load in0
        // in1_ptr = in1_off + (i << 2)
        0x20, 0x01, 0x20, 0x04, 0x41, 0x02, 0x74, 0x6a,
        0x2a, 0x02, 0x00,       // f32.load in1
        opByte,                 // arithmetic op
        0x38, 0x02, 0x00,       // f32.store
        0x20, 0x04, 0x41, 0x01, 0x6a, 0x21, 0x04, // i++
        0x0c, 0x00,             // br 0
        0x0b, 0x0b              // end loop, end block
    ];

    builder.addFunction(typeIdx, [{ count: 1, type: 0x7f }], body);
    builder.addExport("compute_binary", 0, 0);
    return builder.build();
}

/**
 * Compiles a binary arithmetic WebAssembly kernel for two distinct input buffers.
 * Signature: compute_binary(in0_off: i32, in1_off: i32, len: i32, out_off: i32) -> void
 * @param {string} [op='add'] - Binary operation name.
 * @returns {Uint8Array} WebAssembly binary buffer.
 */
function compileWasmBinaryOpKernel(op = 'add') {
    let opByte = 0x92; // f32.add
    if (op === 'mul') {
        opByte = 0x94;
    } else if (op === 'sub') {
        opByte = 0x93;
    } else if (op === 'div') {
        opByte = 0x95;
    } else if (op === 'min') {
        opByte = 0x96;
    } else if (op === 'max') {
        opByte = 0x97;
    }

    const builder = new WasmBinaryBuilder();
    const typeIdx = builder.addType([0x7f, 0x7f, 0x7f, 0x7f], []);
    builder.setMemory(1);
    builder.addExport("memory", 2, 0);

    const body = [
        0x41, 0x00, 0x21, 0x04, // local.set 4 (i = 0)
        0x02, 0x40,             // block
        0x03, 0x40,             // loop
        0x20, 0x04, 0x20, 0x02, 0x4f, // if i >= len (i32.ge_u)
        0x0d, 0x01,             // br_if 1
        // out_ptr = out_off + (i << 2)
        0x20, 0x03, 0x20, 0x04, 0x41, 0x02, 0x74, 0x6a,
        // in0_ptr = in0_off + (i << 2)
        0x20, 0x00, 0x20, 0x04, 0x41, 0x02, 0x74, 0x6a,
        0x2a, 0x02, 0x00,       // f32.load in0
        // in1_ptr = in1_off + (i << 2)
        0x20, 0x01, 0x20, 0x04, 0x41, 0x02, 0x74, 0x6a,
        0x2a, 0x02, 0x00,       // f32.load in1
        opByte,                 // arithmetic op
        0x38, 0x02, 0x00,       // f32.store
        0x20, 0x04, 0x41, 0x01, 0x6a, 0x21, 0x04, // i++
        0x0c, 0x00,             // br 0
        0x0b, 0x0b              // end loop, end block
    ];

    builder.addFunction(typeIdx, [{ count: 1, type: 0x7f }], body);
    builder.addExport("compute_binary", 0, 0);
    return builder.build();
}

/**
 * Compiles a 2D matrix multiplication WebAssembly kernel: C = A x B.
 * Signature: compute_matmul(a_off: i32, b_off: i32, c_off: i32, M: i32, K: i32, N: i32) -> void
 * @param {number} [memoryPages=2] - Initial pages to allocate for matrix buffers.
 * @returns {Uint8Array} WebAssembly binary buffer.
 */
function compileWasmMatmulKernel(memoryPages = 2) {
    const builder = new WasmBinaryBuilder();
    const typeIdx = builder.addType([0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f], []);
    builder.setMemory(memoryPages);
    builder.addExport("memory", 2, 0);

    // Locals: i(6: i32), j(7: i32), k(8: i32), sum(9: f32)
    const locals = [{ count: 3, type: 0x7f }, { count: 1, type: 0x7d }];
    const body = [
        // i = 0
        0x41, 0x00, 0x21, 0x06,
        0x02, 0x40, 0x03, 0x40,
        // if i >= M br 1
        0x20, 0x06, 0x20, 0x03, 0x4f, 0x0d, 0x01,

        // j = 0
        0x41, 0x00, 0x21, 0x07,
        0x02, 0x40, 0x03, 0x40,
        // if j >= N br 1
        0x20, 0x07, 0x20, 0x05, 0x4f, 0x0d, 0x01,

        // sum = 0.0f
        0x43, 0x00, 0x00, 0x00, 0x00, 0x21, 0x09,
        // k = 0
        0x41, 0x00, 0x21, 0x08,
        0x02, 0x40, 0x03, 0x40,
        // if k >= K br 1
        0x20, 0x08, 0x20, 0x04, 0x4f, 0x0d, 0x01,

        // a_addr = a_off + (i * K + k) * 4
        0x20, 0x00, 0x20, 0x06, 0x20, 0x04, 0x6c, 0x20, 0x08, 0x6a, 0x41, 0x02, 0x74, 0x6a,
        0x2a, 0x02, 0x00,
        // b_addr = b_off + (k * N + j) * 4
        0x20, 0x01, 0x20, 0x08, 0x20, 0x05, 0x6c, 0x20, 0x07, 0x6a, 0x41, 0x02, 0x74, 0x6a,
        0x2a, 0x02, 0x00,
        0x94, // f32.mul
        0x20, 0x09, 0x92, // sum + prod
        0x21, 0x09,

        // k = k + 1
        0x20, 0x08, 0x41, 0x01, 0x6a, 0x21, 0x08,
        0x0c, 0x00, 0x0b, 0x0b, // br loop, end inner loop, end inner block

        // c_addr = c_off + (i * N + j) * 4
        0x20, 0x02, 0x20, 0x06, 0x20, 0x05, 0x6c, 0x20, 0x07, 0x6a, 0x41, 0x02, 0x74, 0x6a,
        0x20, 0x09,
        0x38, 0x02, 0x00, // f32.store sum

        // j = j + 1
        0x20, 0x07, 0x41, 0x01, 0x6a, 0x21, 0x07,
        0x0c, 0x00, 0x0b, 0x0b, // br mid loop

        // i = i + 1
        0x20, 0x06, 0x41, 0x01, 0x6a, 0x21, 0x06,
        0x0c, 0x00, 0x0b, 0x0b // br outer loop
    ];

    builder.addFunction(typeIdx, locals, body);
    builder.addExport("compute_matmul", 0, 0);
    return builder.build();
}

/**
 * Compiles a 2D tensor transpose kernel.
 * Signature: compute_transpose(in_off: i32, out_off: i32, R: i32, C: i32) -> void
 * @param {number} [memoryPages=1] - Memory pages to allocate.
 * @returns {Uint8Array} WebAssembly binary buffer.
 */
function compileWasmTransposeKernel(memoryPages = 1) {
    const builder = new WasmBinaryBuilder();
    const typeIdx = builder.addType([0x7f, 0x7f, 0x7f, 0x7f], []);
    builder.setMemory(memoryPages);
    builder.addExport("memory", 2, 0);

    const locals = [{ count: 2, type: 0x7f }]; // r(4: i32), c(5: i32)
    const body = [
        0x41, 0x00, 0x21, 0x04, // r = 0
        0x02, 0x40, 0x03, 0x40,
        0x20, 0x04, 0x20, 0x02, 0x4f, 0x0d, 0x01, // if r >= R br 1
        0x41, 0x00, 0x21, 0x05, // c = 0
        0x02, 0x40, 0x03, 0x40,
        0x20, 0x05, 0x20, 0x03, 0x4f, 0x0d, 0x01, // if c >= C br 1
        // out_addr = out_off + (c * R + r) * 4
        0x20, 0x01, 0x20, 0x05, 0x20, 0x02, 0x6c, 0x20, 0x04, 0x6a, 0x41, 0x02, 0x74, 0x6a,
        // in_addr = in_off + (r * C + c) * 4
        0x20, 0x00, 0x20, 0x04, 0x20, 0x03, 0x6c, 0x20, 0x05, 0x6a, 0x41, 0x02, 0x74, 0x6a,
        0x2a, 0x02, 0x00,
        0x38, 0x02, 0x00,
        0x20, 0x05, 0x41, 0x01, 0x6a, 0x21, 0x05, // c++
        0x0c, 0x00, 0x0b, 0x0b,
        0x20, 0x04, 0x41, 0x01, 0x6a, 0x21, 0x04, // r++
        0x0c, 0x00, 0x0b, 0x0b
    ];

    builder.addFunction(typeIdx, locals, body);
    builder.addExport("compute_transpose", 0, 0);
    return builder.build();
}

/**
 * Compiles a strided slice WebAssembly kernel.
 * Signature: compute_slice(in_off: i32, out_off: i32, start: i32, step: i32, count: i32) -> void
 * @param {number} [memoryPages=1] - Memory pages to allocate.
 * @returns {Uint8Array} WebAssembly binary buffer.
 */
function compileWasmSliceKernel(memoryPages = 1) {
    const builder = new WasmBinaryBuilder();
    const typeIdx = builder.addType([0x7f, 0x7f, 0x7f, 0x7f, 0x7f], []);
    builder.setMemory(memoryPages);
    builder.addExport("memory", 2, 0);

    const locals = [{ count: 1, type: 0x7f }]; // i(5: i32)
    const body = [
        0x41, 0x00, 0x21, 0x05, // i = 0
        0x02, 0x40, 0x03, 0x40,
        0x20, 0x05, 0x20, 0x04, 0x4f, 0x0d, 0x01, // if i >= count br 1
        // out_addr = out_off + i * 4
        0x20, 0x01, 0x20, 0x05, 0x41, 0x02, 0x74, 0x6a,
        // in_addr = in_off + (start + i * step) * 4
        0x20, 0x00, 0x20, 0x02, 0x20, 0x05, 0x20, 0x03, 0x6c, 0x6a, 0x41, 0x02, 0x74, 0x6a,
        0x2a, 0x02, 0x00,
        0x38, 0x02, 0x00,
        0x20, 0x05, 0x41, 0x01, 0x6a, 0x21, 0x05, // i++
        0x0c, 0x00, 0x0b, 0x0b
    ];

    builder.addFunction(typeIdx, locals, body);
    builder.addExport("compute_slice", 0, 0);
    return builder.build();
}

/**
 * Compiles a reduction WebAssembly kernel (sum, mean, max, min, prod).
 * Signature: compute_reduce(in_off: i32, out_off: i32, len: i32) -> void
 * @param {string} [op='sum'] - Reduction op name.
 * @param {number} [memoryPages=1] - Memory pages to allocate.
 * @returns {Uint8Array} WebAssembly binary buffer.
 */
function compileWasmReduceKernel(op = 'sum', memoryPages = 1) {
    const builder = new WasmBinaryBuilder();
    const typeIdx = builder.addType([0x7f, 0x7f, 0x7f], []);
    builder.setMemory(memoryPages);
    builder.addExport("memory", 2, 0);

    const isMean = (op === 'mean');
    const isMin = (op === 'min');
    const isMax = (op === 'max');
    const isProd = (op === 'prod');

    let initInst = [0x43, 0x00, 0x00, 0x00, 0x00]; // 0.0f
    let redOpByte = 0x92; // f32.add

    if (isProd) {
        initInst = [0x43, 0x00, 0x00, 0x80, 0x3f]; // 1.0f
        redOpByte = 0x94; // f32.mul
    } else if (isMin) {
        initInst = [0x43, 0x00, 0x00, 0x80, 0x7f]; // +Infinity
        redOpByte = 0x96; // f32.min
    } else if (isMax) {
        initInst = [0x43, 0x00, 0x00, 0x80, 0xff]; // -Infinity
        redOpByte = 0x97; // f32.max
    }

    const locals = [{ count: 1, type: 0x7f }, { count: 1, type: 0x7d }]; // i(3: i32), acc(4: f32)
    const body = [
        0x41, 0x00, 0x21, 0x03, // i = 0
        ...initInst, 0x21, 0x04, // acc = init
        0x02, 0x40, 0x03, 0x40,
        0x20, 0x03, 0x20, 0x02, 0x4f, 0x0d, 0x01, // if i >= len br 1
        0x20, 0x04, // acc
        0x20, 0x00, 0x20, 0x03, 0x41, 0x02, 0x74, 0x6a, 0x2a, 0x02, 0x00, // in[i]
        redOpByte,
        0x21, 0x04, // acc = ...
        0x20, 0x03, 0x41, 0x01, 0x6a, 0x21, 0x03, // i++
        0x0c, 0x00, 0x0b, 0x0b
    ];

    if (isMean) {
        body.push(
            0x20, 0x01, // out_off
            0x20, 0x04, // acc
            0x20, 0x02, 0xb2, // f32.convert_i32_s(len)
            0x95, // f32.div
            0x38, 0x02, 0x00 // f32.store
        );
    } else {
        body.push(
            0x20, 0x01, // out_off
            0x20, 0x04, // acc
            0x38, 0x02, 0x00 // f32.store
        );
    }

    builder.addFunction(typeIdx, locals, body);
    builder.addExport("compute_reduce", 0, 0);
    return builder.build();
}

/**
 * Inspects compiled C++/WASM code and produces an executable WASM binary kernel.
 * @param {string} code - Emitted C++/WASM code.
 * @returns {Uint8Array} Executable WebAssembly binary buffer.
 */
function compileWasmFromCode(code) {
    let op = 'mul';
    if (code) {
        const lower = code.toLowerCase();
        if (lower.includes('sqrt')) {
            op = 'sqrt';
        } else if (lower.includes('abs')) {
            op = 'abs';
        } else if (lower.includes('neg')) {
            op = 'neg';
        } else if (lower.includes('ceil')) {
            op = 'ceil';
        } else if (lower.includes('floor')) {
            op = 'floor';
        } else if (lower.includes('min')) {
            op = 'min';
        } else if (lower.includes('max')) {
            op = 'max';
        } else if (lower.includes('+') || lower.includes('add')) {
            op = 'add';
        } else if (lower.includes('-') || lower.includes('sub')) {
            op = 'sub';
        } else if (lower.includes('/') || lower.includes('div')) {
            op = 'div';
        } else if (lower.includes('copy')) {
            op = 'copy';
        }
    }
    return compileWasmKernel(op);
}

/**
 * Runs a WASM computation with provided inputs.
 * @param {Uint8Array} wasmBinary - The compiled WASM binary.
 * @param {Float32Array} inputData - The input data to populate in memory.
 * @param {number} expectedOutputLength - The expected number of Float32 elements in the output.
 * @returns {Promise<Float32Array>} The computed output data.
 */
async function runWasmCompute(wasmBinary, inputData, expectedOutputLength) {
    /* c8 ignore next 3 */
    if (!WebAssembly || !WebAssembly.instantiate) {
        throw new Error("WebAssembly is not supported in this browser.");
    }

    let wasmModule;
    try {
        const result = await WebAssembly.instantiate(wasmBinary, {});
        wasmModule = result.instance;
    } catch (e) {
        throw new Error(`Failed to instantiate WASM module: ${e.message}`);
    }

    const exports = wasmModule.exports;
    if (!exports.memory || !(exports.memory instanceof WebAssembly.Memory)) {
        throw new Error("WASM module must export 'memory'.");
    }
    if (typeof exports.compute !== 'function') {
        throw new Error("WASM module must export a 'compute' function.");
    }

    const memory = exports.memory;
    const inputByteLength = inputData.length * Float32Array.BYTES_PER_ELEMENT;
    const outputByteLength = expectedOutputLength * Float32Array.BYTES_PER_ELEMENT;

    const inputOffset = 0;
    const outputOffset = inputByteLength;

    validateMemoryBounds(memory, inputOffset, inputByteLength);
    validateMemoryBounds(memory, outputOffset, outputByteLength);

    const memFloat32 = new Float32Array(memory.buffer);
    memFloat32.set(inputData, inputOffset / Float32Array.BYTES_PER_ELEMENT);

    exports.compute(inputOffset, inputData.length, outputOffset);

    const outputData = new Float32Array(expectedOutputLength);
    outputData.set(memFloat32.subarray(
        outputOffset / Float32Array.BYTES_PER_ELEMENT,
        (outputOffset + outputByteLength) / Float32Array.BYTES_PER_ELEMENT
    ));

    return outputData;
}

/**
 * Multi-dimensional strided tensor representation in linear WASM memory.
 */
class WasmTensor {
    /**
     * @param {Array<number|string>} shape - Dimensions of the tensor.
     * @param {number} byteOffset - Memory byte offset.
     * @param {string} [dtype='float32'] - Data type.
     */
    constructor(shape, byteOffset, dtype = 'float32') {
        const rawShape = Array.from(shape || [1]);
        this.shape = [];
        for (const dim of rawShape) {
            if (typeof dim === 'number' && !isNaN(dim)) {
                this.shape.push(Math.max(1, Math.floor(dim)));
            } else if (typeof dim === 'string' && dim.trim().length > 0 && !isNaN(Number(dim))) {
                this.shape.push(Math.max(1, Math.floor(Number(dim))));
            } else {
                // Dynamic symbolic dimension fallback
                this.shape.push(1);
            }
        }
        this.byteOffset = byteOffset;
        this.dtype = dtype;
        this.strides = [];
        let stride = 1;
        for (let i = this.shape.length - 1; i >= 0; i--) {
            this.strides[i] = stride;
            stride *= this.shape[i];
        }
        this.numElements = Math.max(1, stride);
        this.byteLength = this.numElements * (dtype === 'float32' ? 4 : 4);
    }
}

/**
 * Compiles and schedules a multi-node IRGraph for WASM linear memory execution.
 * @param {Object} irGraph - Object containing nodes, inputs, and outputs.
 * @returns {Object} Graph execution plan with memory arena offsets.
 */
function compileWasmGraph(irGraph) {
    const nodes = Array.isArray(irGraph.nodes) ? irGraph.nodes : Object.values(irGraph.nodes || {});
    const tensorMap = {};
    const shapes = {};
    let currentOffset = 0;

    // Allocate memory for inputs
    const inputs = irGraph.inputs || [];
    for (const inp of inputs) {
        const inpNode = nodes.find(n => n.id === inp) || { shape_metadata: [1] };
        const shape = inpNode.shape_metadata || [1];
        const tensor = new WasmTensor(shape, currentOffset);
        tensorMap[inp] = tensor;
        shapes[inp] = tensor.shape;
        currentOffset += tensor.byteLength;
    }

    // Allocate memory and compile ops for intermediate and output nodes
    const executionSteps = [];
    for (const node of nodes) {
        if (node.op_type === 'Input') continue;
        const rawOp = (node.op_type || 'mul').toLowerCase();
        let op = 'mul';
        let stepKind = 'elementwise';

        if (rawOp.includes('matmul') || rawOp.includes('dot')) {
            op = 'matmul';
            stepKind = 'matmul';
        } else if (rawOp.includes('transpose')) {
            op = 'transpose';
            stepKind = 'transpose';
        } else if (rawOp.includes('slice')) {
            op = 'slice';
            stepKind = 'slice';
        } else if (rawOp.includes('reduce') || rawOp.includes('sum') || rawOp.includes('mean')) {
            op = rawOp.includes('mean') ? 'mean' : (rawOp.includes('max') ? 'max' : (rawOp.includes('min') ? 'min' : 'sum'));
            stepKind = 'reduce';
        } else if (rawOp.includes('add')) {
            op = 'add';
        } else if (rawOp.includes('sub')) {
            op = 'sub';
        } else if (rawOp.includes('div')) {
            op = 'div';
        } else if (rawOp.includes('sqrt')) {
            op = 'sqrt';
        } else if (rawOp.includes('abs')) {
            op = 'abs';
        } else if (rawOp.includes('neg')) {
            op = 'neg';
        } else if (rawOp.includes('relu')) {
            op = 'max';
        }

        const shape = node.shape_metadata || (node.inputs && tensorMap[node.inputs[0]] ? tensorMap[node.inputs[0]].shape : [1]);
        const tensor = new WasmTensor(shape, currentOffset);
        tensorMap[node.id] = tensor;
        shapes[node.id] = tensor.shape;
        currentOffset += tensor.byteLength;

        let kernelBinary;
        if (stepKind === 'matmul') {
            kernelBinary = compileWasmMatmulKernel(Math.max(2, Math.ceil(currentOffset / 65536) + 1));
        } else if (stepKind === 'transpose') {
            kernelBinary = compileWasmTransposeKernel(Math.max(1, Math.ceil(currentOffset / 65536) + 1));
        } else if (stepKind === 'slice') {
            kernelBinary = compileWasmSliceKernel(Math.max(1, Math.ceil(currentOffset / 65536) + 1));
        } else if (stepKind === 'reduce') {
            kernelBinary = compileWasmReduceKernel(op, Math.max(1, Math.ceil(currentOffset / 65536) + 1));
        } else if (node.inputs && node.inputs.length >= 2 && node.inputs[0] !== node.inputs[1]) {
            stepKind = 'binary';
            kernelBinary = compileWasmBinaryOpKernel(op);
        } else {
            kernelBinary = compileWasmKernel(op);
        }

        executionSteps.push({
            id: node.id,
            op: op,
            stepKind: stepKind,
            inputs: node.inputs || [],
            output: node.id,
            kernelBinary: kernelBinary,
            numElements: tensor.numElements,
            shape: tensor.shape,
            attributes: node.attributes || {}
        });
    }

    return {
        graph: irGraph,
        tensorMap: tensorMap,
        shapes: shapes,
        executionSteps: executionSteps,
        totalMemoryBytes: currentOffset,
        executedNodes: executionSteps.map(s => s.id)
    };
}

/**
 * Runs multi-node graph execution in WebAssembly linear memory.
 * @param {Object} graphPlan - Plan compiled by compileWasmGraph.
 * @param {Object.<string, Float32Array>} inputBuffers - Mapping of input IDs to Float32Array.
 * @returns {Promise<Object>} Output mapping and evaluated shape telemetry.
 */
async function runWasmGraph(graphPlan, inputBuffers) {
    const arena = new WasmMemoryArena(Math.max(1, Math.ceil(graphPlan.totalMemoryBytes / 65536)));
    const memFloat32 = new Float32Array(arena.memory.buffer);

    // Populate inputs
    for (const [inpId, buffer] of Object.entries(inputBuffers)) {
        const tensor = graphPlan.tensorMap[inpId];
        if (tensor) {
            memFloat32.set(buffer, tensor.byteOffset / 4);
        }
    }

    // Execute sequential steps
    for (const step of graphPlan.executionSteps) {
        const outTensor = graphPlan.tensorMap[step.output];
        const inTensor0 = graphPlan.tensorMap[step.inputs[0]];
        const inTensor1 = step.inputs[1] ? graphPlan.tensorMap[step.inputs[1]] : null;

        if (step.kernelBinary && outTensor && inTensor0) {
            const wasmInstance = (await WebAssembly.instantiate(step.kernelBinary, {})).instance;
            const kernelMem = new Float32Array(wasmInstance.exports.memory.buffer);

            if (step.stepKind === 'matmul' && inTensor1) {
                const M = inTensor0.shape[0] || 1;
                const K = inTensor0.shape[1] || inTensor0.numElements;
                const N = inTensor1.shape[1] || (inTensor1.numElements / K);

                const aFloats = memFloat32.subarray(inTensor0.byteOffset / 4, inTensor0.byteOffset / 4 + inTensor0.numElements);
                const bFloats = memFloat32.subarray(inTensor1.byteOffset / 4, inTensor1.byteOffset / 4 + inTensor1.numElements);

                kernelMem.set(aFloats, 0);
                kernelMem.set(bFloats, inTensor0.numElements);

                const aOff = 0;
                const bOff = inTensor0.numElements * 4;
                const cOff = (inTensor0.numElements + inTensor1.numElements) * 4;

                wasmInstance.exports.compute_matmul(aOff, bOff, cOff, M, K, N);

                const outFloats = kernelMem.subarray(cOff / 4, cOff / 4 + outTensor.numElements);
                memFloat32.set(outFloats, outTensor.byteOffset / 4);
            } else if (step.stepKind === 'transpose') {
                const R = inTensor0.shape[0] || 1;
                const C = inTensor0.shape[1] || inTensor0.numElements;
                const inFloats = memFloat32.subarray(inTensor0.byteOffset / 4, inTensor0.byteOffset / 4 + inTensor0.numElements);
                kernelMem.set(inFloats, 0);

                const inOff = 0;
                const outOff = inTensor0.numElements * 4;
                wasmInstance.exports.compute_transpose(inOff, outOff, R, C);

                const outFloats = kernelMem.subarray(outOff / 4, outOff / 4 + outTensor.numElements);
                memFloat32.set(outFloats, outTensor.byteOffset / 4);
            } else if (step.stepKind === 'slice') {
                const start = step.attributes.start || 0;
                const stepVal = step.attributes.step || 1;
                const inFloats = memFloat32.subarray(inTensor0.byteOffset / 4, inTensor0.byteOffset / 4 + inTensor0.numElements);
                kernelMem.set(inFloats, 0);

                const inOff = 0;
                const outOff = inTensor0.numElements * 4;
                wasmInstance.exports.compute_slice(inOff, outOff, start, stepVal, outTensor.numElements);

                const outFloats = kernelMem.subarray(outOff / 4, outOff / 4 + outTensor.numElements);
                memFloat32.set(outFloats, outTensor.byteOffset / 4);
            } else if (step.stepKind === 'reduce') {
                const inFloats = memFloat32.subarray(inTensor0.byteOffset / 4, inTensor0.byteOffset / 4 + inTensor0.numElements);
                kernelMem.set(inFloats, 0);

                const inOff = 0;
                const outOff = inTensor0.numElements * 4;
                wasmInstance.exports.compute_reduce(inOff, outOff, inTensor0.numElements);

                const outFloats = kernelMem.subarray(outOff / 4, outOff / 4 + outTensor.numElements);
                memFloat32.set(outFloats, outTensor.byteOffset / 4);
            } else if (step.stepKind === 'binary' && inTensor1) {
                const inFloats0 = memFloat32.subarray(inTensor0.byteOffset / 4, inTensor0.byteOffset / 4 + inTensor0.numElements);
                const inFloats1 = memFloat32.subarray(inTensor1.byteOffset / 4, inTensor1.byteOffset / 4 + inTensor1.numElements);
                kernelMem.set(inFloats0, 0);
                kernelMem.set(inFloats1, inTensor0.numElements);

                const in0Off = 0;
                const in1Off = inTensor0.numElements * 4;
                const outOff = (inTensor0.numElements + inTensor1.numElements) * 4;

                wasmInstance.exports.compute_binary(in0Off, in1Off, step.numElements, outOff);

                const outFloats = kernelMem.subarray(outOff / 4, outOff / 4 + outTensor.numElements);
                memFloat32.set(outFloats, outTensor.byteOffset / 4);
            } else {
                // Elementwise copy / compute
                const inFloats = memFloat32.subarray(inTensor0.byteOffset / 4, inTensor0.byteOffset / 4 + inTensor0.numElements);
                kernelMem.set(inFloats, 0);

                const outByteOffset = inTensor0.numElements * 4;
                wasmInstance.exports.compute(0, step.numElements, outByteOffset);

                const outFloats = kernelMem.subarray(outByteOffset / 4, outByteOffset / 4 + outTensor.numElements);
                memFloat32.set(outFloats, outTensor.byteOffset / 4);
            }
        }
    }

    // Collect outputs
    const outputs = {};
    const outputIds = graphPlan.graph.outputs || [];
    for (const outId of outputIds) {
        const tensor = graphPlan.tensorMap[outId];
        if (tensor) {
            const start = tensor.byteOffset / 4;
            const end = start + tensor.numElements;
            outputs[outId] = new Float32Array(memFloat32.slice(start, end));
        }
    }

    return {
        outputs: outputs,
        telemetry: extractShapeTelemetry(graphPlan)
    };
}

/**
 * Runs backward-pass reverse-mode autodiff graph on WASM memory.
 * @param {Object} bwdPlan - Backward graph execution plan.
 * @param {Object.<string, Float32Array>} primalBuffers - Forward activations.
 * @param {Object.<string, Float32Array>} gradOutputs - Upstream gradients.
 * @returns {Promise<Object>} Computed input gradients and telemetry.
 */
async function runWasmBackward(bwdPlan, primalBuffers, gradOutputs) {
    const combinedInputs = { ...primalBuffers, ...gradOutputs };
    return await runWasmGraph(bwdPlan, combinedInputs);
}

/**
 * Extracts shape learning telemetry from evaluated WASM execution.
 * @param {Object} plan - Evaluated graph execution plan.
 * @returns {Object} JSON-compatible telemetry payload.
 */
function extractShapeTelemetry(plan) {
    return {
        shapes: plan.shapes || {},
        memory_bytes: plan.totalMemoryBytes || 0,
        nodes_evaluated: plan.executedNodes || []
    };
}

/**
 * Captures concrete runtime tensor shapes and stride metadata from WASM execution.
 * @param {Object} executionOrTensorMap - Execution context or tensor map containing WasmTensors.
 * @returns {Object.<string, Array<number>>} Mapping of node/tensor IDs to concrete dimensions.
 */
function captureWasmTensorShapes(executionOrTensorMap) {
    if (!executionOrTensorMap) return {};
    const shapes = {};
    const tensorMap = executionOrTensorMap.tensorMap || executionOrTensorMap.shapes || executionOrTensorMap;
    if (typeof tensorMap === 'object') {
        for (const [id, tensor] of Object.entries(tensorMap)) {
            if (tensor && Array.isArray(tensor.shape)) {
                shapes[id] = Array.from(tensor.shape);
            } else if (Array.isArray(tensor)) {
                shapes[id] = Array.from(tensor);
            } else if (tensor && typeof tensor === 'object' && tensor.shape) {
                shapes[id] = Array.from(tensor.shape);
            }
        }
    }
    return shapes;
}

// Export for browser
/* c8 ignore next 19 */
if (typeof window !== 'undefined') {
    window.encodeU32Leb128 = encodeU32Leb128;
    window.encodeWasmSection = encodeWasmSection;
    window.validateMemoryBounds = validateMemoryBounds;
    window.WasmMemoryArena = WasmMemoryArena;
    window.WasmBinaryBuilder = WasmBinaryBuilder;
    window.compileWasmKernel = compileWasmKernel;
    window.compileWasmMatmulKernel = compileWasmMatmulKernel;
    window.compileWasmTransposeKernel = compileWasmTransposeKernel;
    window.compileWasmSliceKernel = compileWasmSliceKernel;
    window.compileWasmReduceKernel = compileWasmReduceKernel;
    window.compileWasmFromCode = compileWasmFromCode;
    window.runWasmCompute = runWasmCompute;
    window.WasmTensor = WasmTensor;
    window.compileWasmGraph = compileWasmGraph;
    window.runWasmGraph = runWasmGraph;
    window.runWasmBackward = runWasmBackward;
    window.extractShapeTelemetry = extractShapeTelemetry;
    window.captureWasmTensorShapes = captureWasmTensorShapes;
}

// Export for testing
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        encodeU32Leb128,
        encodeWasmSection,
        validateMemoryBounds,
        WasmMemoryArena,
        WasmBinaryBuilder,
        compileWasmKernel,
        compileWasmMatmulKernel,
        compileWasmTransposeKernel,
        compileWasmSliceKernel,
        compileWasmReduceKernel,
        compileWasmFromCode,
        runWasmCompute,
        WasmTensor,
        compileWasmGraph,
        runWasmGraph,
        runWasmBackward,
        extractShapeTelemetry,
        captureWasmTensorShapes
    };
}
