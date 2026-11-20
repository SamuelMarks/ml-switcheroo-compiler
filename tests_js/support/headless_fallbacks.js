'use strict';

/**
 * Deterministic zero-dependency headless WebGPU WGSL evaluator, minimal DOM/WCAG shim,
 * and loopback WebRTC PeerConnection for offline Node.js test execution.
 */

const GPUBufferUsage = {
    MAP_READ: 0x0001,
    MAP_WRITE: 0x0002,
    COPY_SRC: 0x0004,
    COPY_DST: 0x0008,
    INDEX: 0x0010,
    VERTEX: 0x0020,
    UNIFORM: 0x0040,
    STORAGE: 0x0080,
    INDIRECT: 0x0100,
    QUERY_RESOLVE: 0x0200,
};

const GPUMapMode = {
    READ: 0x0001,
    WRITE: 0x0002,
};

function installWebGPUGlobals() {
    if (!globalThis.GPUBufferUsage) globalThis.GPUBufferUsage = GPUBufferUsage;
    if (!globalThis.GPUMapMode) globalThis.GPUMapMode = GPUMapMode;
}

function parseWgslModule(wgslCode) {
    const bindings = [];
    const bindRe = /@group\(\s*(\d+)\s*\)\s*@binding\(\s*(\d+)\s*\)\s*var<storage[^>]*>\s*([A-Za-z_]\w*)\s*:\s*array<\s*(f32|i32|u32)\s*>\s*;/g;
    let bm;
    while ((bm = bindRe.exec(wgslCode)) !== null) {
        bindings.push({
            group: Number(bm[1]),
            binding: Number(bm[2]),
            name: bm[3],
            elemType: bm[4],
        });
    }

    const entryPoints = new Map();
    const fnHeaderRe = /@compute\s+@workgroup_size\(\s*(\d+)(?:\s*,\s*(\d+))?(?:\s*,\s*(\d+))?\s*\)\s*fn\s+([A-Za-z_]\w*)\s*\(([\s\S]*?)\)\s*\{/g;
    let fm;
    while ((fm = fnHeaderRe.exec(wgslCode)) !== null) {
        const wx = Number(fm[1] || 1);
        const wy = Number(fm[2] || 1);
        const wz = Number(fm[3] || 1);
        const fnName = fm[4];
        const paramsStr = fm[5];
        const bodyStart = fnHeaderRe.lastIndex;
        let depth = 1;
        let idx = bodyStart;
        while (idx < wgslCode.length && depth > 0) {
            const ch = wgslCode[idx];
            if (ch === '{') depth++;
            else if (ch === '}') depth--;
            idx++;
        }
        const rawBody = wgslCode.slice(bodyStart, idx - 1);

        let globalIdVar = 'global_id';
        let localIdVar = 'local_id';
        const gidMatch = paramsStr.match(/@builtin\(\s*global_invocation_id\s*\)\s*([A-Za-z_]\w*)/);
        if (gidMatch) globalIdVar = gidMatch[1];
        const lidMatch = paramsStr.match(/@builtin\(\s*local_invocation_id\s*\)\s*([A-Za-z_]\w*)/);
        if (lidMatch) localIdVar = lidMatch[1];

        let jsBody = rawBody
            .replace(/\/\/.*$/gm, '')
            .replace(/\b(let|var)\s+([A-Za-z_]\w*)\s*:\s*[A-Za-z0-9_<>]+\s*=/g, '$1 $2 =')
            .replace(/(\d+\.\d+|\d+\.)f\b/g, '$1')
            .replace(/\b(\d+)u\b/g, '$1')
            .replace(/\b(\d+)i\b/g, '$1')
            .replace(/\bf32\s*\(/g, 'Number(')
            .replace(/\bu32\s*\(/g, 'Math.trunc(')
            .replace(/\bi32\s*\(/g, 'Math.trunc(')
            .replace(/\bmax\s*\(/g, 'Math.max(')
            .replace(/\bmin\s*\(/g, 'Math.min(')
            .replace(/\babs\s*\(/g, 'Math.abs(')
            .replace(/\bsqrt\s*\(/g, 'Math.sqrt(')
            .replace(/\bexp\s*\(/g, 'Math.exp(')
            .replace(/\blog\s*\(/g, 'Math.log(')
            .replace(/\bsin\s*\(/g, 'Math.sin(')
            .replace(/\bcos\s*\(/g, 'Math.cos(')
            .replace(/\btanh\s*\(/g, 'Math.tanh(')
            .replace(/\b([A-Za-z_]\w*_remaining)\s*=\s*\1\s*\/\s*([^;]+);/g, '$1 = Math.trunc($1 / ($2));');

        const paramNames = [globalIdVar, localIdVar, ...bindings.map((b) => b.name)];
        const compiledFn = new Function(...paramNames, jsBody);
        entryPoints.set(fnName, { wx, wy, wz, compiledFn });
    }

    return { bindings, entryPoints };
}

function viewForElemType(arrayBuffer, elemType) {
    if (elemType === 'i32') return new Int32Array(arrayBuffer);
    if (elemType === 'u32') return new Uint32Array(arrayBuffer);
    return new Float32Array(arrayBuffer);
}

function createCpuWebGPUDevice() {
    installWebGPUGlobals();

    class CpuGPUBuffer {
        constructor(desc) {
            this.size = desc.size;
            this.usage = desc.usage;
            this._raw = new ArrayBuffer(Math.max(4, Math.ceil(desc.size / 4) * 4));
            this.destroyed = false;
        }
        async mapAsync() {}
        getMappedRange(offset = 0, size = this.size - offset) {
            return this._raw.slice(offset, offset + size);
        }
        unmap() {}
        destroy() {
            this.destroyed = true;
        }
    }

    return {
        features: new Set(),
        createBuffer(desc) {
            const buf = new CpuGPUBuffer(desc);
            if (desc.mappedAtCreation) {
                buf.getMappedRange = () => buf._raw;
            }
            return buf;
        },
        createShaderModule({ code }) {
            return parseWgslModule(code);
        },
        createComputePipeline({ compute }) {
            return {
                module: compute.module,
                entryPoint: compute.entryPoint || Array.from(compute.module.entryPoints.keys())[0],
                getBindGroupLayout() {
                    return {};
                },
            };
        },
        createBindGroup({ entries }) {
            const map = new Map();
            for (const entry of entries) {
                map.set(entry.binding, entry.resource.buffer);
            }
            return map;
        },
        createCommandEncoder() {
            const ops = [];
            return {
                beginComputePass() {
                    const passState = { pipeline: null, bindGroups: new Map(), dispatches: [] };
                    return {
                        setPipeline(p) {
                            passState.pipeline = p;
                        },
                        setBindGroup(groupIdx, bg) {
                            passState.bindGroups.set(groupIdx, bg);
                        },
                        dispatchWorkgroups(dx = 1, dy = 1, dz = 1) {
                            passState.dispatches.push([dx, dy, dz]);
                        },
                        end() {
                            ops.push({ kind: 'compute', passState });
                        },
                    };
                },
                copyBufferToBuffer(src, srcOffset, dst, dstOffset, copySize) {
                    ops.push({ kind: 'copy', src, srcOffset, dst, dstOffset, copySize });
                },
                finish() {
                    return ops;
                },
            };
        },
        queue: {
            writeBuffer(buffer, bufferOffset, data) {
                const bytes = ArrayBuffer.isView(data)
                    ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
                    : new Uint8Array(data);
                new Uint8Array(buffer._raw, bufferOffset, bytes.byteLength).set(bytes);
            },
            submit(commandBuffers) {
                for (const ops of commandBuffers) {
                    for (const op of ops) {
                        if (op.kind === 'compute') {
                            const { pipeline, bindGroups, dispatches } = op.passState;
                            const ep = pipeline.module.entryPoints.get(pipeline.entryPoint);
                            if (!ep) {
                                throw new Error(`WGSL entry point '${pipeline.entryPoint}' not found`);
                            }
                            const bg0 = bindGroups.get(0) || new Map();
                            const views = pipeline.module.bindings.map((b) => {
                                const buf = bg0.get(b.binding);
                                return buf ? viewForElemType(buf._raw, b.elemType) : viewForElemType(new ArrayBuffer(4), b.elemType);
                            });
                            for (const [dx, dy, dz] of dispatches) {
                                for (let gz = 0; gz < dz * ep.wz; gz++) {
                                    for (let gy = 0; gy < dy * ep.wy; gy++) {
                                        for (let gx = 0; gx < dx * ep.wx; gx++) {
                                            ep.compiledFn(
                                                { x: gx, y: gy, z: gz },
                                                { x: gx % ep.wx, y: gy % ep.wy, z: gz % ep.wz },
                                                ...views
                                            );
                                        }
                                    }
                                }
                            }
                        } else if (op.kind === 'copy') {
                            const srcView = new Uint8Array(op.src._raw, op.srcOffset, op.copySize);
                            new Uint8Array(op.dst._raw, op.dstOffset, op.copySize).set(srcView);
                        }
                    }
                }
            },
        },
        destroy() {},
    };
}

function createCpuWebGPUNavigator() {
    installWebGPUGlobals();
    return {
        gpu: {
            async requestAdapter() {
                return {
                    features: new Set(),
                    async requestDevice() {
                        return createCpuWebGPUDevice();
                    },
                };
            },
        },
    };
}

class MinimalEvent {
    constructor(type, init = {}) {
        this.type = type;
        this.bubbles = Boolean(init.bubbles);
        this.detail = init.detail || null;
        this.target = null;
    }
}

class MinimalElement {
    constructor(tagName, ownerDocument) {
        this.tagName = tagName.toUpperCase();
        this.ownerDocument = ownerDocument;
        this.attributes = new Map();
        this.children = [];
        this.parentNode = null;
        this.style = {};
        this._listeners = new Map();
        this._text = '';
        this.checked = false;
        this._value = undefined;
        this.dataset = new Proxy(
            {},
            {
                get: (_, prop) => {
                    const attr = 'data-' + String(prop).replace(/[A-Z]/g, (m) => '-' + m.toLowerCase());
                    return this.attributes.has(attr) ? this.attributes.get(attr) : undefined;
                },
                set: (_, prop, val) => {
                    const attr = 'data-' + String(prop).replace(/[A-Z]/g, (m) => '-' + m.toLowerCase());
                    this.setAttribute(attr, String(val));
                    return true;
                },
            }
        );
    }

    get id() {
        return this.attributes.get('id') || '';
    }
    set id(v) {
        this.setAttribute('id', String(v));
    }

    get value() {
        if (this._value !== undefined) return this._value;
        if (this.tagName === 'SELECT') {
            const sel = this.children.find((c) => c.tagName === 'OPTION' && c.attributes.has('selected'));
            const first = sel || this.children.find((c) => c.tagName === 'OPTION');
            return first ? first.getAttribute('value') || first.textContent : '';
        }
        return this.attributes.get('value') || '';
    }
    set value(v) {
        this._value = String(v);
    }

    get textContent() {
        if (this.children.length === 0) return this._text;
        return this._text + this.children.map((c) => c.textContent).join('');
    }
    set textContent(v) {
        this._text = String(v);
        this.children = [];
    }

    getAttribute(name) {
        const key = name.toLowerCase();
        return this.attributes.has(key) ? this.attributes.get(key) : null;
    }

    setAttribute(name, value) {
        const key = name.toLowerCase();
        const strVal = String(value);
        this.attributes.set(key, strVal);
        if (key === 'style') {
            for (const part of strVal.split(';')) {
                const [k, v] = part.split(':').map((s) => s && s.trim());
                if (k && v !== undefined) this.style[k] = v;
            }
        }
        if (key === 'checked') this.checked = true;
    }

    hasAttribute(name) {
        return this.attributes.has(name.toLowerCase());
    }

    appendChild(child) {
        child.parentNode = this;
        this.children.push(child);
        return child;
    }

    addEventListener(type, listener) {
        if (!this._listeners.has(type)) this._listeners.set(type, []);
        this._listeners.get(type).push(listener);
    }

    dispatchEvent(event) {
        event.target = this;
        const handlers = this._listeners.get(event.type) || [];
        for (const fn of handlers) {
            fn.call(this, event);
        }
        return true;
    }

    click() {
        this.dispatchEvent(new MinimalEvent('click'));
    }

    _matchesSelector(selector) {
        const sel = selector.trim();
        if (sel.startsWith('#')) return this.id === sel.slice(1);
        const attrMatch = sel.match(/^([a-zA-Z0-9_-]*)\[([a-zA-Z0-9_-]+)(?:="([^"]*)")?\]$/);
        if (attrMatch) {
            const [, tag, attr, expected] = attrMatch;
            if (tag && this.tagName !== tag.toUpperCase()) return false;
            if (!this.hasAttribute(attr)) return false;
            return expected === undefined || this.getAttribute(attr) === expected;
        }
        const classMatch = sel.match(/^([a-zA-Z0-9_-]*)\.([a-zA-Z0-9_-]+)$/);
        if (classMatch) {
            const [, tag, cls] = classMatch;
            if (tag && this.tagName !== tag.toUpperCase()) return false;
            const classes = (this.getAttribute('class') || '').split(/\s+/);
            return classes.includes(cls);
        }
        return this.tagName === sel.toUpperCase();
    }

    querySelectorAll(selector) {
        const results = [];
        const visit = (node) => {
            for (const child of node.children) {
                if (child._matchesSelector(selector)) results.push(child);
                visit(child);
            }
        };
        visit(this);
        return results;
    }

    querySelector(selector) {
        return this.querySelectorAll(selector)[0] || null;
    }
}

function parseMinimalHtml(html, doc) {
    const voidTags = new Set(['INPUT', 'BR', 'HR', 'IMG', 'META', 'LINK']);
    const clean = html.replace(/<!DOCTYPE[^>]*>/gi, '').replace(/<!--[\s\S]*?-->/g, '');
    const root = new MinimalElement('root', doc);
    const stack = [root];
    const tokenRe = /<\/?([a-zA-Z0-9_-]+)([^>]*)>|([^<]+)/g;
    let m;
    while ((m = tokenRe.exec(clean)) !== null) {
        const full = m[0];
        if (m[3] !== undefined) {
            const txt = m[3];
            if (txt.trim().length > 0 && stack.length > 0) {
                const top = stack[stack.length - 1];
                top._text += txt.trim();
            }
            continue;
        }
        const tag = m[1].toUpperCase();
        const attrText = m[2] || '';
        if (full.startsWith('</')) {
            for (let i = stack.length - 1; i > 0; i--) {
                if (stack[i].tagName === tag) {
                    stack.length = i;
                    break;
                }
            }
            continue;
        }
        const el = new MinimalElement(tag, doc);
        const selfClosing = attrText.trim().endsWith('/') || voidTags.has(tag);
        const attrRe = /([a-zA-Z_:][a-zA-Z0-9_.:-]*)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+)))?/g;
        let am;
        while ((am = attrRe.exec(attrText)) !== null) {
            if (am[1] === '/') continue;
            const val = am[2] !== undefined ? am[2] : am[3] !== undefined ? am[3] : am[4] !== undefined ? am[4] : '';
            el.setAttribute(am[1], val);
        }
        stack[stack.length - 1].appendChild(el);
        if (!selfClosing) stack.push(el);
    }

    let htmlEl = root.children.find((c) => c.tagName === 'HTML');
    if (!htmlEl) {
        htmlEl = new MinimalElement('HTML', doc);
        const bodyEl = new MinimalElement('BODY', doc);
        for (const c of root.children) bodyEl.appendChild(c);
        htmlEl.appendChild(bodyEl);
    } else if (!htmlEl.children.some((c) => c.tagName === 'BODY')) {
        const bodyEl = new MinimalElement('BODY', doc);
        htmlEl.appendChild(bodyEl);
    }
    return htmlEl;
}

class MinimalDocument {
    constructor(html = '<html><body></body></html>') {
        this.documentElement = parseMinimalHtml(html, this);
        this.body = this.documentElement.querySelector('body') || this.documentElement;
    }
    createElement(tag) {
        return new MinimalElement(tag, this);
    }
    getElementById(id) {
        return this.documentElement.querySelector(`#${id}`);
    }
    querySelector(selector) {
        if (this.documentElement._matchesSelector(selector)) return this.documentElement;
        return this.documentElement.querySelector(selector);
    }
    querySelectorAll(selector) {
        const out = [];
        if (this.documentElement._matchesSelector(selector)) out.push(this.documentElement);
        return out.concat(this.documentElement.querySelectorAll(selector));
    }
}

class MinimalJSDOM {
    constructor(html = '<html><body></body></html>') {
        const document = new MinimalDocument(html);
        this.window = {
            document,
            Event: MinimalEvent,
            CustomEvent: MinimalEvent,
            addEventListener() {},
        };
    }
}

async function runMinimalAxeAudit(rootEl) {
    const violations = [];
    const labels = new Set(
        rootEl
            .querySelectorAll('label')
            .map((l) => l.getAttribute('for'))
            .filter(Boolean)
    );
    for (const ctrl of [...rootEl.querySelectorAll('select'), ...rootEl.querySelectorAll('input')]) {
        let ancestor = ctrl.parentNode;
        let wrappedByLabel = false;
        while (ancestor) {
            if (ancestor.tagName === 'LABEL') {
                wrappedByLabel = true;
                break;
            }
            ancestor = ancestor.parentNode;
        }
        const hasLabel = Boolean(
            wrappedByLabel ||
                ctrl.getAttribute('aria-label') ||
                ctrl.getAttribute('aria-labelledby') ||
                (ctrl.id && labels.has(ctrl.id))
        );
        if (!hasLabel) {
            violations.push({ id: 'label', target: ctrl.id || ctrl.tagName });
        }
    }
    for (const btn of rootEl.querySelectorAll('button')) {
        const hasName = Boolean((btn.textContent && btn.textContent.trim()) || btn.getAttribute('aria-label'));
        if (!hasName) {
            violations.push({ id: 'button-name', target: btn.id || 'button' });
        }
    }
    return { violations };
}

const _peerRegistry = new Map();

class InMemoryDataChannel {
    constructor(label) {
        this.label = label;
        this._peerChannel = null;
        this._onOpen = null;
        this._onMessage = null;
        this._open = false;
    }
    onOpen(cb) {
        this._onOpen = cb;
        if (this._open) queueMicrotask(() => cb());
    }
    onMessage(cb) {
        this._onMessage = cb;
    }
    sendMessage(msg) {
        const peer = this._peerChannel;
        if (peer && peer._onMessage) {
            queueMicrotask(() => peer._onMessage(msg));
        }
    }
    close() {
        this._open = false;
    }
    _markOpen() {
        if (!this._open) {
            this._open = true;
            if (this._onOpen) queueMicrotask(() => this._onOpen());
        }
    }
}

class InMemoryPeerConnection {
    constructor(name) {
        this.name = name;
        this._onLocalDesc = null;
        this._onLocalCand = null;
        this._onDataChannel = null;
        this._pendingChannels = [];
        this._remotePeer = null;
        this._hasRemoteDesc = false;
        this._hasRemoteCand = false;
        _peerRegistry.set(name, this);
    }
    onLocalDescription(cb) {
        this._onLocalDesc = cb;
    }
    onLocalCandidate(cb) {
        this._onLocalCand = cb;
    }
    onDataChannel(cb) {
        this._onDataChannel = cb;
    }
    createDataChannel(label) {
        const dc = new InMemoryDataChannel(label);
        this._pendingChannels.push(dc);
        queueMicrotask(() => {
            if (this._onLocalDesc) {
                this._onLocalDesc(JSON.stringify({ peer: this.name, label }), 'offer');
            }
            if (this._onLocalCand) {
                this._onLocalCand(`candidate:1 1 UDP 2130706431 127.0.0.1 5000 typ host (${this.name})`, '0');
            }
        });
        return dc;
    }
    setRemoteDescription(sdp, type) {
        const parsed = JSON.parse(sdp);
        const remote = _peerRegistry.get(parsed.peer);
        this._remotePeer = remote;
        this._hasRemoteDesc = true;
        if (type === 'offer') {
            queueMicrotask(() => {
                if (this._onLocalDesc) {
                    this._onLocalDesc(JSON.stringify({ peer: this.name }), 'answer');
                }
                if (this._onLocalCand) {
                    this._onLocalCand(`candidate:2 1 UDP 2130706431 127.0.0.1 5001 typ host (${this.name})`, '0');
                }
            });
        }
        this._tryConnectChannels();
    }
    addRemoteCandidate() {
        this._hasRemoteCand = true;
        this._tryConnectChannels();
    }
    _tryConnectChannels() {
        const remote = this._remotePeer;
        if (!remote || !this._hasRemoteDesc || !remote._hasRemoteDesc || !this._hasRemoteCand || !remote._hasRemoteCand) {
            return;
        }
        while (this._pendingChannels.length > 0) {
            const dc1 = this._pendingChannels.shift();
            const dc2 = new InMemoryDataChannel(dc1.label);
            dc1._peerChannel = dc2;
            dc2._peerChannel = dc1;
            if (remote._onDataChannel) remote._onDataChannel(dc2);
            dc1._markOpen();
            dc2._markOpen();
        }
        while (remote._pendingChannels.length > 0) {
            const dc1 = remote._pendingChannels.shift();
            const dc2 = new InMemoryDataChannel(dc1.label);
            dc1._peerChannel = dc2;
            dc2._peerChannel = dc1;
            if (this._onDataChannel) this._onDataChannel(dc2);
            dc1._markOpen();
            dc2._markOpen();
        }
    }
    close() {
        _peerRegistry.delete(this.name);
    }
}

module.exports = {
    GPUBufferUsage,
    GPUMapMode,
    InMemoryPeerConnection,
    MinimalJSDOM,
    createCpuWebGPUDevice,
    createCpuWebGPUNavigator,
    installWebGPUGlobals,
    runMinimalAxeAudit,
};
