/**
 * ML Switcheroo Playground UI Logic
 * Handles initialization of Monaco Editor, theme toggling, and UI state.
 * @module PlaygroundUI
 */

/**
 * i18n Dictionary mapping language keys to localized strings.
 * @type {Object.<string, Object.<string, string>>}
 */
const I18N = {
    en: {
        darkMode: "Dark Mode",
        compile: "Compile",
        execute: "Execute in browser",
        compiling: "Compiling...",
        compileComplete: "Compilation complete.",
        pyodideInit: "Initializing Python environment (Pyodide)...",
        depsInstall: "Installing dependencies...",
        pythonReady: "Python environment ready.",
        pyodideFailed: "Failed to initialize Pyodide: ",
        pyodideNotLoaded: "Pyodide script not loaded.",
        compileError: "Error during compilation: ",
        initWebgpu: "Initializing WebGPU execution...",
        webgpuNotLoaded: "WebGPU Runner not loaded.",
        webgpuComplete: "WebGPU execution complete. Output:",
        webgpuError: "WebGPU Error: ",
        initWasm: "Initializing WASM SIMD execution...",
        wasmNotLoaded: "WASM Runner not loaded.",
        wasmError: "WASM Error: ",
        wasmComplete: "WASM execution complete. Output:",
        wasmNote: "Note: WASM SIMD execution requires a valid compiled WASM binary.",
        sourceFw: "Source Framework",
        examples: "Examples",
        targetFw: "Target Framework",
        baseFw: "Base ML Frameworks",
        jaxEco: "JAX Ecosystem",
        nativeWeb: "Native Web Execution",
        fallbackExample: "Example code not found for ",
        compileFailed: "Compilation failed: ",
        targetOutput: "# Target output will appear here"
    }
};

/**
 * Gets a localized string.
 * @param {string} key - The i18n key.
 * @param {string} [lang='en'] - The language code.
 * @returns {string} The localized string or the key if not found.
 */
function t(key, lang = 'en') {
    return (I18N[lang] && I18N[lang][key]) ? I18N[lang][key] : key;
}

/**
 * Applies i18n translations to DOM elements based on data-i18n attributes.
 * @param {Document} doc - The HTML Document object.
 * @param {string} [lang='en'] - The language code.
 */
function applyI18n(doc, lang = 'en') {
    doc.querySelectorAll('[data-i18n]').forEach(el => {
        const key = el.getAttribute('data-i18n');
        if (I18N[lang] && I18N[lang][key]) el.textContent = I18N[lang][key];
    });
    doc.querySelectorAll('[data-i18n-aria]').forEach(el => {
        const key = el.getAttribute('data-i18n-aria');
        if (I18N[lang] && I18N[lang][key]) el.setAttribute('aria-label', I18N[lang][key]);
    });
    doc.querySelectorAll('[data-i18n-label]').forEach(el => {
        const key = el.getAttribute('data-i18n-label');
        if (I18N[lang] && I18N[lang][key]) el.setAttribute('label', I18N[lang][key]);
    });
}

/**
 * Initializes the Light/Dark mode theme.
 * Checks localStorage first, then system preferences.
 * @param {Document} doc - The HTML Document object.
 * @param {Storage} storage - The localStorage object.
 * @param {Window} win - The Window object.
 */
function initTheme(doc, storage, win) {
    const toggle = doc.getElementById('theme-toggle');
    if (!toggle) return;

    const savedTheme = storage.getItem('ml-playground-theme');
    const prefersDark = win.matchMedia && win.matchMedia('(prefers-color-scheme: dark)').matches;

    let isDark = false;
    if (savedTheme) {
        isDark = savedTheme === 'dark';
    } else {
        isDark = prefersDark;
    }

    toggle.checked = isDark;
    doc.documentElement.setAttribute('data-theme', isDark ? 'dark' : 'light');

    toggle.addEventListener('change', (e) => {
        const theme = e.target.checked ? 'dark' : 'light';
        doc.documentElement.setAttribute('data-theme', theme);
        storage.setItem('ml-playground-theme', theme);
        updateEditorTheme(win, theme);
    });
}

/**
 * Updates the Monaco Editor theme.
 * @param {Window} win - The Window object containing monaco.
 * @param {string} theme - 'light' or 'dark'.
 */
function updateEditorTheme(win, theme) {
    if (win.monaco && win.monaco.editor) {
        win.monaco.editor.setTheme(theme === 'dark' ? 'vs-dark' : 'vs');
    }
}

/**
 * Map of source examples per framework.
 * @type {Object.<string, Object.<string, string>>}
 */
const EXAMPLES = {
    jax: {
        simple_mlp: "import jax.numpy as jnp\n\ndef simple_mlp(x):\n    return jnp.dot(x, x)\n",
        cnn: "import jax.numpy as jnp\n# CNN example\n",
        attention: "import jax.numpy as jnp\n# Attention example\n",
    },
    pytorch: {
        simple_mlp: "import torch\n\ndef simple_mlp(x):\n    return torch.matmul(x, x)\n",
        cnn: "import torch\n# CNN example\n",
        attention: "import torch\n# Attention example\n",
    }
};

/**
 * Gets example code based on framework and example name.
 * @param {string} framework - The selected framework.
 * @param {string} example - The selected example.
 * @returns {string} The code string.
 */
function getExampleCode(framework, example) {
    if (EXAMPLES[framework] && EXAMPLES[framework][example]) {
        return EXAMPLES[framework][example];
    }
    return `# ${t('fallbackExample')} ${framework} - ${example}`;
}

/**
 * Updates the UI state of the execution button.
 * @param {Document} doc - The HTML Document object.
 */
function updateExecuteButtonVisibility(doc) {
    const targetSelect = doc.getElementById('target-framework');
    const executeBtn = doc.getElementById('btn-execute');
    if (!targetSelect || !executeBtn) return;

    const val = targetSelect.value;
    if (val === 'webgpu' || val === 'wasm_simd') {
        executeBtn.style.display = 'inline-block';
    } else {
        executeBtn.style.display = 'none';
    }
}

/**
 * Prints a message to the UI console.
 * @param {Document} doc - The HTML Document object.
 * @param {string} msg - The message to print.
 * @param {boolean} [isError=false] - Whether the message is an error.
 */
    /* c8 ignore next */
    function logToConsole(doc, msg, isError = false) {
    const consoleEl = doc.getElementById('pg-console');
    if (!consoleEl) return;

    const span = doc.createElement('span');
    span.textContent = msg + '\n';
    /* c8 ignore next */
    if (isError) span.style.color = 'red';

    consoleEl.appendChild(span);
    consoleEl.scrollTop = consoleEl.scrollHeight;
}

/**
 * Clears the UI console.
 * @param {Document} doc - The HTML Document object.
 */
function clearConsole(doc) {
    const consoleEl = doc.getElementById('pg-console');
    if (consoleEl) {
        consoleEl.textContent = '';
    }
}

/**
 * Pyodide instance singleton.
 * @type {Object|null}
 */
/**
 * Resolves wheel URLs dynamically via global config, DOM attributes/links, fetch manifest, or fallback.
 * @param {Document|null} doc - HTML Document
 * @param {Window|Object} [win={}] - Window or global object
 * @param {string} [basePath='./_static/'] - Base static path
 * @returns {Promise<Array<string>>} List of resolved wheel URLs
 */
async function resolveWheelUrls(doc, win = typeof window !== 'undefined' ? window : {}, basePath = './_static/') {
    if (win && Array.isArray(win.__ML_SWITCHEROO_WHEELS__) && win.__ML_SWITCHEROO_WHEELS__.length > 0) {
        return win.__ML_SWITCHEROO_WHEELS__.map(w => (w.startsWith('http://') || w.startsWith('https://')) ? w : (basePath + w));
    }

    if (doc && typeof doc.querySelector === 'function') {
        const dataEl = doc.querySelector('[data-wheels]');
        if (dataEl && dataEl.dataset && dataEl.dataset.wheels) {
            const raw = dataEl.dataset.wheels.split(',').map(s => s.trim()).filter(Boolean);
            if (raw.length > 0) {
                return raw.map(w => (w.startsWith('http://') || w.startsWith('https://')) ? w : (basePath + w));
            }
        }
    }

    if (doc && typeof doc.querySelectorAll === 'function') {
        const links = doc.querySelectorAll('a.pg-wheel-link');
        if (links && links.length > 0) {
            const urls = [];
            for (let i = 0; i < links.length; i++) {
                const href = links[i].getAttribute('href');
                if (href) {
                    let normalized = href;
                    if (href.startsWith('http://') || href.startsWith('https://') || href.startsWith('./')) {
                        normalized = href;
                    } else if (href.startsWith('_static/')) {
                        normalized = './' + href;
                    } else {
                        normalized = basePath + href;
                    }
                    urls.push(normalized);
                }
            }
            if (urls.length > 0) {
                return urls;
            }
        }
    }

    if (win && typeof win.fetch === 'function') {
        try {
            const resp = await win.fetch(basePath + 'wheels.json');
            if (resp && resp.ok && typeof resp.json === 'function') {
                const data = await resp.json();
                if (data && Array.isArray(data.wheels) && data.wheels.length > 0) {
                    return data.wheels.map(w => (w.startsWith('http://') || w.startsWith('https://')) ? w : (basePath + w));
                }
            }
        } catch (_) {
            // fallback
        }
    }

    return [
        basePath + "ml_switcheroo_ir-0.1.0-py3-none-any.whl",
        basePath + "ml_switcheroo_compiler-0.1.0-py3-none-any.whl"
    ];
}

let pyodideInstance = null;

/**
 * Loads Pyodide and installs necessary packages.
 * @param {Document} doc - HTML Document
 * @param {Window} win - Window object
 * @returns {Promise<Object>} The initialized Pyodide instance.
 */
async function loadPyodideEnvironment(doc, win) {
    if (pyodideInstance) return pyodideInstance;

    logToConsole(doc, t('pyodideInit'));
    try {
        pyodideInstance = await win.loadPyodide();
        await pyodideInstance.loadPackage("micropip");
        const micropip = pyodideInstance.pyimport("micropip");

        let basePath = "./_static/";
        const wheels = await resolveWheelUrls(doc, win, basePath);

        logToConsole(doc, t('depsInstall'));
        await micropip.install("numpy");
        for (const whl of wheels) {
            await micropip.install(whl);
        }

        logToConsole(doc, t('pythonReady'));
        return pyodideInstance;
    } catch (err) {
        logToConsole(doc, t('pyodideFailed') + err.message, true);
        throw err;
    }
}

let lastCompiledGraphMetadata = null;

/**
 * Generates a deterministic pseudorandom typed array for an input shape and dtype.
 * @param {Array<number>} shape - Tensor dimensions.
 * @param {string} [dtype='float32'] - Tensor element data type.
 * @returns {Float32Array|Int32Array} Initialized typed buffer.
 */
function generateDeterministicInputBuffer(shape, dtype = 'float32') {
    const totalElements = (Array.isArray(shape) && shape.length > 0)
        ? shape.reduce((a, b) => a * b, 1)
        : 4;
    const isInt = (dtype && dtype.startsWith('int'));
    const arr = isInt ? new Int32Array(totalElements) : new Float32Array(totalElements);
    for (let i = 0; i < totalElements; i++) {
        const val = (((i * 1103515245 + 12345) & 0x7fffffff) % 1000) / 100.0 + 0.5;
        arr[i] = isInt ? Math.floor(val) : val;
    }
    return arr;
}

/**
 * Extracts concrete runtime shape and stride metadata from execution output buffers.
 * @param {Float32Array|Object} result - Execution output tensor or result object.
 * @param {Object} [outputMeta=null] - Expected output shape metadata.
 * @returns {Object.<string, Array<number>>} Inferred concrete tensor shapes.
 */
function extractRuntimeShapesAndStrides(result, outputMeta = null) {
    const learnedShapes = {};
    if (!result) return learnedShapes;

    if (result.shape && Array.isArray(result.shape)) {
        learnedShapes["out_0"] = Array.from(result.shape);
    } else if (outputMeta && outputMeta.shape) {
        learnedShapes["out_0"] = Array.from(outputMeta.shape);
    } else if (result.length) {
        learnedShapes["out_0"] = [result.length];
    }
    return learnedShapes;
}

/**
 * Compiles the source code using the python backend.
 * @param {Object} pyodide - The Pyodide instance
 * @param {string} source - Source code
 * @param {string} sourceFw - Source framework
 * @param {string} targetFw - Target framework
 * @param {Document} [doc=null] - Optional HTML Document for console shape inspection output
 * @param {Object} [observedShapes=null] - Optional concrete observed shapes from runtime execution
 * @returns {string} The compiled output or error message.
 */
function compileCode(pyodide, source, sourceFw, targetFw, doc = null, observedShapes = null) {
    const pythonScript = `
import json
import traceback

source_code = ${JSON.stringify(source)}
source_fw = ${JSON.stringify(sourceFw)}
target_fw = ${JSON.stringify(targetFw)}
observed_shapes = ${JSON.stringify(observedShapes || {})}

def _compile_payload():
    try:
        from ml_switcheroo_compiler.backends.cst_transpiler import transpile_source
        from ml_switcheroo_compiler.backends.ast_to_ir import parse_ast_to_ir
        from ml_switcheroo_compiler.transforms.passes.shape_inference import shape_inference_pass, annotate_learned_shapes

        if target_fw in ("webgpu", "edge_wgsl"):
            from ml_switcheroo_compiler.backends.edge.webgpu import WebGPUCodeGenerator
            graph = parse_ast_to_ir(source_code)
            shape_inference_pass(graph)
            if observed_shapes:
                annotate_learned_shapes(graph, observed_shapes)
            shapes = {}
            for nid, node in list(graph.nodes.items()):
                shape_meta = getattr(node, "shape_metadata", None)
                if shape_meta is None:
                    shape_meta = getattr(node, "shape", None) or (1,)
                shapes[node.id] = {
                    "op": getattr(node, "op_type", ""),
                    "shape": list(shape_meta) if hasattr(shape_meta, "__iter__") else [shape_meta]
                }
            inputs_meta = {}
            for inp_id in getattr(graph, "inputs", []):
                inp_node = graph.nodes.get(inp_id)
                s = getattr(inp_node, "shape_metadata", None) or getattr(inp_node, "shape", None) or [4]
                inputs_meta[inp_id] = {
                    "shape": list(s) if hasattr(s, "__iter__") else [s],
                    "dtype": getattr(inp_node, "dtype", "float32")
                }
            outputs_meta = {}
            for out_id in getattr(graph, "outputs", []):
                out_node = graph.nodes.get(out_id)
                s = getattr(out_node, "shape_metadata", None) or getattr(out_node, "shape", None) or [4]
                outputs_meta[out_id] = {
                    "shape": list(s) if hasattr(s, "__iter__") else [s],
                    "dtype": getattr(out_node, "dtype", "float32")
                }
            generator = WebGPUCodeGenerator(graph)
            code = generator.generate()
            return {"code": code, "shapes": shapes, "inputs": inputs_meta, "outputs": outputs_meta, "target": target_fw, "learned": bool(observed_shapes)}

        elif target_fw in ("wasm_simd", "wasm"):
            from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
            graph = parse_ast_to_ir(source_code)
            shape_inference_pass(graph)
            if observed_shapes:
                annotate_learned_shapes(graph, observed_shapes)
            shapes = {}
            for nid, node in list(graph.nodes.items()):
                shape_meta = getattr(node, "shape_metadata", None)
                if shape_meta is None:
                    shape_meta = getattr(node, "shape", None) or (1,)
                shapes[node.id] = {
                    "op": getattr(node, "op_type", ""),
                    "shape": list(shape_meta) if hasattr(shape_meta, "__iter__") else [shape_meta]
                }
            inputs_meta = {}
            for inp_id in getattr(graph, "inputs", []):
                inp_node = graph.nodes.get(inp_id)
                s = getattr(inp_node, "shape_metadata", None) or getattr(inp_node, "shape", None) or [4]
                inputs_meta[inp_id] = {
                    "shape": list(s) if hasattr(s, "__iter__") else [s],
                    "dtype": getattr(inp_node, "dtype", "float32")
                }
            outputs_meta = {}
            for out_id in getattr(graph, "outputs", []):
                out_node = graph.nodes.get(out_id)
                s = getattr(out_node, "shape_metadata", None) or getattr(out_node, "shape", None) or [4]
                outputs_meta[out_id] = {
                    "shape": list(s) if hasattr(s, "__iter__") else [s],
                    "dtype": getattr(out_node, "dtype", "float32")
                }
            generator = WasmCodeGenerator(graph)
            code = generator.generate()
            return {"code": code, "shapes": shapes, "inputs": inputs_meta, "outputs": outputs_meta, "target": target_fw, "learned": bool(observed_shapes)}

        else:
            transpiled = transpile_source(source_code, target_framework=target_fw)
            return {"code": transpiled, "shapes": {}, "target": target_fw, "learned": False}

    except Exception as err:
        return {
            "error": str(err),
            "traceback": traceback.format_exc(),
            "target": target_fw,
            "learned": False
        }

try:
    _res = _compile_payload()
    _out = json.dumps(_res)
except Exception as _e:
    _out = json.dumps({"error": str(_e), "traceback": traceback.format_exc()})
_out
`;
    try {
        const rawResult = pyodide.runPython(pythonScript);
        let parsed;
        try {
            parsed = JSON.parse(rawResult);
        } catch {
            return rawResult;
        }

        if (parsed && parsed.error) {
            if (doc) {
                logToConsole(doc, t('compileError') + parsed.error, true);
                if (parsed.traceback) {
                    logToConsole(doc, parsed.traceback, true);
                }
            }
            return t('compileFailed') + parsed.error;
        }

        if (parsed && typeof parsed.code === 'string') {
            lastCompiledGraphMetadata = parsed;
            if (doc && parsed.shapes && Object.keys(parsed.shapes).length > 0) {
                const header = parsed.learned
                    ? "[Shape Learning Feedback Loop] Inferred updated tensor shapes from runtime feedback:"
                    : "[Shape Learning] Inferred intermediate tensor shapes:";
                logToConsole(doc, header);
                for (const [nid, meta] of Object.entries(parsed.shapes)) {
                    const shapeStr = Array.isArray(meta.shape) ? `[${meta.shape.join(', ')}]` : meta.shape;
                    logToConsole(doc, `  • Node '${nid}' (${meta.op}): ${shapeStr}`);
                }
            }
            return parsed.code;
        }
        return rawResult;
    } catch (e) {
        return t('compileFailed') + e.message;
    }
}

/**
 * Initializes the playground logic.
 * @param {Document} doc - The HTML Document object.
 * @param {Storage} storage - The localStorage object.
 * @param {Window} win - The Window object.
 */
function initPlayground(doc, storage, win) {
    applyI18n(doc);
    initTheme(doc, storage, win);
    updateExecuteButtonVisibility(doc);

    const sourceSelect = doc.getElementById('source-framework');
    const exampleSelect = doc.getElementById('source-example');
    const targetSelect = doc.getElementById('target-framework');

    if (targetSelect) {
        targetSelect.addEventListener('change', () => {
            updateExecuteButtonVisibility(doc);
        });
    }

    // Initialize Monaco if available
    let sourceEditor, targetEditor;

    if (win.require) {
        win.require(['vs/editor/editor.main'], function () {
            const isDark = doc.documentElement.getAttribute('data-theme') === 'dark';
            /* c8 ignore next */
            const theme = isDark ? 'vs-dark' : 'vs';

            const sourceContainer = doc.getElementById('editor-source');
            if (sourceContainer) {
                /* c8 ignore next */
                const srcFw = sourceSelect ? sourceSelect.value : 'jax';
                /* c8 ignore next */
                const srcEx = exampleSelect ? exampleSelect.value : 'simple_mlp';
                sourceEditor = win.monaco.editor.create(sourceContainer, {
                    value: getExampleCode(srcFw, srcEx),
                    language: 'python',
                    theme: theme,
                    automaticLayout: true,
                    minimap: { enabled: false }
                });
            }

            const targetContainer = doc.getElementById('editor-target');
            if (targetContainer) {
                targetEditor = win.monaco.editor.create(targetContainer, {
                    value: t('targetOutput'),
                    language: 'python',
                    theme: theme,
                    readOnly: true,
                    automaticLayout: true,
                    minimap: { enabled: false }
                });
            }

            // Wire up example selection
            if (sourceSelect && exampleSelect && sourceEditor) {
                const updateSource = () => {
                    sourceEditor.setValue(getExampleCode(sourceSelect.value, exampleSelect.value));
                };
                sourceSelect.addEventListener('change', updateSource);
                exampleSelect.addEventListener('change', updateSource);
            }

            // Wire up compile button
            const compileBtn = doc.getElementById('btn-compile');
            if (compileBtn) {
                compileBtn.addEventListener('click', async () => {
                    clearConsole(doc);

                    if (!win.loadPyodide) {
                        logToConsole(doc, t('pyodideNotLoaded'), true);
                        return;
                    }

                    try {
                        const pyodide = await loadPyodideEnvironment(doc, win);
                        logToConsole(doc, t('compiling'));

                        /* c8 ignore next */
                        const sourceCode = sourceEditor ? sourceEditor.getValue() : "";
                        /* c8 ignore next */
                        const sourceFw = sourceSelect ? sourceSelect.value : "jax";
                        /* c8 ignore next */
                        const targetFw = targetSelect ? targetSelect.value : "wasm_simd";

                        const compiledOutput = compileCode(pyodide, sourceCode, sourceFw, targetFw, doc);

                        if (targetEditor) {
                            targetEditor.setValue(compiledOutput);
                            // Adjust syntax highlighting based on target
                            if (targetFw === 'webgpu') {
                                win.monaco.editor.setModelLanguage(targetEditor.getModel(), 'wgsl');
                            } else if (targetFw === 'wasm_simd') {
                                win.monaco.editor.setModelLanguage(targetEditor.getModel(), 'wat'); // approximation
                            } else {
                                win.monaco.editor.setModelLanguage(targetEditor.getModel(), 'python');
                            }
                        }
                        logToConsole(doc, t('compileComplete'));
                    } catch (e) {
                        logToConsole(doc, t('compileError') + e.message, true);
                    }
                });
            }

            // Wire up execute button
            const executeBtn = doc.getElementById('btn-execute');
            if (executeBtn) {
                executeBtn.addEventListener('click', async () => {
                    const targetFw = targetSelect ? targetSelect.value : "";
                    if (!targetEditor) return;
                    const code = targetEditor.getValue();

                    if (targetFw === 'webgpu') {
                        logToConsole(doc, t('initWebgpu'));
                        try {
                            if (typeof runWebGPUCompute === 'undefined') {
                                logToConsole(doc, t('webgpuNotLoaded'), true);
                                return;
                            }
                            // Derive deterministic input buffers matching compiled tensor dimensions
                            const inputsMeta = (lastCompiledGraphMetadata && lastCompiledGraphMetadata.inputs) || {};
                            const inputEntries = Object.entries(inputsMeta);
                            let inputData;
                            if (inputEntries.length > 0) {
                                const firstInp = inputEntries[0][1];
                                inputData = generateDeterministicInputBuffer(firstInp.shape, firstInp.dtype);
                            } else {
                                inputData = generateDeterministicInputBuffer([4], 'float32');
                            }

                            const outputsMeta = (lastCompiledGraphMetadata && lastCompiledGraphMetadata.outputs) || {};
                            const outEntries = Object.entries(outputsMeta);
                            let expectedOutputElements = 4;
                            let firstOutMeta = null;
                            if (outEntries.length > 0) {
                                firstOutMeta = outEntries[0][1];
                                expectedOutputElements = (Array.isArray(firstOutMeta.shape) && firstOutMeta.shape.length > 0)
                                    ? firstOutMeta.shape.reduce((a, b) => a * b, 1)
                                    : 4;
                            }
                            const outputSizeInBytes = expectedOutputElements * Float32Array.BYTES_PER_ELEMENT;

                            const result = await runWebGPUCompute(win.navigator, code, inputData, outputSizeInBytes);
                            logToConsole(doc, t('webgpuComplete'));
                            logToConsole(doc, "[" + result.join(", ") + "]");

                            const capturer = (typeof captureRuntimeShapes === 'function')
                                ? captureRuntimeShapes
                                : (win.captureRuntimeShapes || null);
                            const learnedShapes = capturer
                                ? capturer(result)
                                : extractRuntimeShapesAndStrides(result, firstOutMeta);
                            if (Object.keys(learnedShapes).length > 0 && win.pyodideInstance) {
                                logToConsole(doc, "[Shape Learning] Feedback loop triggered with runtime shapes: " + JSON.stringify(learnedShapes));
                                const sourceCode = sourceEditor ? sourceEditor.getValue() : "";
                                const sourceFw = sourceSelect ? sourceSelect.value : "jax";
                                const recompiled = compileCode(win.pyodideInstance, sourceCode, sourceFw, targetFw, doc, learnedShapes);
                                if (targetEditor && recompiled) {
                                    targetEditor.setValue(recompiled);
                                }
                            }
                        } catch (e) {
                            logToConsole(doc, t('webgpuError') + e.message, true);
                        }
                    } else if (targetFw === 'wasm_simd') {
                        logToConsole(doc, t('initWasm'));
                        try {
                            if (typeof runWasmCompute === 'undefined') {
                                logToConsole(doc, t('wasmNotLoaded'), true);
                                return;
                            }

                            const wasmCompiler = (typeof compileWasmFromCode === 'function')
                                ? compileWasmFromCode
                                : (win.compileWasmFromCode || compileWasmFromCodeRef || null);

                            let wasmBinary;
                            if (wasmCompiler) {
                                wasmBinary = wasmCompiler(code);
                            } else {
                                /* c8 ignore next 2 */
                                wasmBinary = new Uint8Array([0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00]);
                            }

                            // Derive deterministic input buffers matching compiled tensor dimensions
                            const inputsMeta = (lastCompiledGraphMetadata && lastCompiledGraphMetadata.inputs) || {};
                            const inputEntries = Object.entries(inputsMeta);
                            let inputData;
                            if (inputEntries.length > 0) {
                                const firstInp = inputEntries[0][1];
                                inputData = generateDeterministicInputBuffer(firstInp.shape, firstInp.dtype);
                            } else {
                                inputData = generateDeterministicInputBuffer([4], 'float32');
                            }

                            const outputsMeta = (lastCompiledGraphMetadata && lastCompiledGraphMetadata.outputs) || {};
                            const outEntries = Object.entries(outputsMeta);
                            let expectedOutputLength = 4;
                            let firstOutMeta = null;
                            if (outEntries.length > 0) {
                                firstOutMeta = outEntries[0][1];
                                expectedOutputLength = (Array.isArray(firstOutMeta.shape) && firstOutMeta.shape.length > 0)
                                    ? firstOutMeta.shape.reduce((a, b) => a * b, 1)
                                    : 4;
                            }

                            const result = await runWasmCompute(wasmBinary, inputData, expectedOutputLength);
                            logToConsole(doc, t('wasmComplete'));
                            logToConsole(doc, "[" + result.join(", ") + "]");

                            const wasmCapturer = (typeof captureWasmTensorShapes === 'function')
                                ? captureWasmTensorShapes
                                : (win.captureWasmTensorShapes || null);
                            const learnedShapes = wasmCapturer
                                ? wasmCapturer({ "out_0": [result.length] })
                                : extractRuntimeShapesAndStrides(result, firstOutMeta);
                            if (Object.keys(learnedShapes).length > 0 && win.pyodideInstance) {
                                logToConsole(doc, "[Shape Learning] Feedback loop triggered with runtime shapes: " + JSON.stringify(learnedShapes));
                                const sourceCode = sourceEditor ? sourceEditor.getValue() : "";
                                const sourceFw = sourceSelect ? sourceSelect.value : "jax";
                                const recompiled = compileCode(win.pyodideInstance, sourceCode, sourceFw, targetFw, doc, learnedShapes);
                                if (targetEditor && recompiled) {
                                    targetEditor.setValue(recompiled);
                                }
                            }
                        } catch (e) {
                            logToConsole(doc, t('wasmError') + e.message, true);
                            logToConsole(doc, t('wasmNote'), true);
                        }
                    }
                });
            }
        });
    }
}

let compileWasmFromCodeRef = (typeof compileWasmFromCode !== 'undefined') ? compileWasmFromCode : null;
let compileWasmKernelRef = (typeof compileWasmKernel !== 'undefined') ? compileWasmKernel : null;
/* c8 ignore next 8 */
if (!compileWasmFromCodeRef && typeof require !== 'undefined') {
    try {
        const wasmModule = require('./wasm_runner.js');
        compileWasmFromCodeRef = wasmModule.compileWasmFromCode;
        compileWasmKernelRef = wasmModule.compileWasmKernel;
    } catch {
        // Browser environment
    }
}

// Export for testing
/* c8 ignore next 19 */
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        applyI18n,
        t,
        initTheme,
        updateEditorTheme,
        getExampleCode,
        updateExecuteButtonVisibility,
        logToConsole,
        clearConsole,
        resolveWheelUrls,
        loadPyodideEnvironment,
        compileCode,
        initPlayground,
        generateDeterministicInputBuffer,
        extractRuntimeShapesAndStrides,
        compileWasmKernel: compileWasmKernelRef,
        compileWasmFromCode: compileWasmFromCodeRef
    };
} else {
    /* c8 ignore next 5 */
    // Run in browser
    window.addEventListener('DOMContentLoaded', () => {
        initPlayground(document, localStorage, window);
    });
}
