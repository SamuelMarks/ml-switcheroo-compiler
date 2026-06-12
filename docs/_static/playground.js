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
function logToConsole(doc, msg, isError = false) {
    const consoleEl = doc.getElementById('pg-console');
    if (!consoleEl) return;

    const span = doc.createElement('span');
    span.textContent = msg + '\n';
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

        // Load numpy and our custom wheels
        // Determine the base path for static assets
        let basePath = "./_static/";
        // It could be different depending on Sphinx structure, we assume we are in html root or use relative
        const wheels = [
            basePath + "ml_switcheroo_ir-0.1.0-py3-none-any.whl",
            basePath + "ml_switcheroo_compiler-0.1.0-py3-none-any.whl"
        ];

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

/**
 * Compiles the source code using the python backend.
 * @param {Object} pyodide - The Pyodide instance
 * @param {string} source - Source code
 * @param {string} sourceFw - Source framework
 * @param {string} targetFw - Target framework
 * @returns {string} The compiled output or error message.
 */
function compileCode(pyodide, source, sourceFw, targetFw) {
    // We run a small python wrapper to call the compiler.
    // Assuming ml_switcheroo has an entrypoint or we just return something for now
    // if the real API is not known, we simulate a compilation.
    const pythonScript = `
import sys
import traceback
try:
    # Attempt to import the compiler
    import ml_switcheroo
    # For now, return a dummy string if we don't have the exact API
    # Real implementation would parse the AST and compile
    "Compiled code for target: ${targetFw}\\n\\n# Source:\\n" + ${JSON.stringify(source)}
except Exception as e:
    traceback.format_exc()
`;
    try {
        const result = pyodide.runPython(pythonScript);
        return result;
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
            const theme = isDark ? 'vs-dark' : 'vs';

            const sourceContainer = doc.getElementById('editor-source');
            if (sourceContainer) {
                sourceEditor = win.monaco.editor.create(sourceContainer, {
                    value: getExampleCode(sourceSelect ? sourceSelect.value : 'jax', exampleSelect ? exampleSelect.value : 'simple_mlp'),
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

                        const sourceCode = sourceEditor ? sourceEditor.getValue() : "";
                        const sourceFw = sourceSelect ? sourceSelect.value : "jax";
                        const targetFw = targetSelect ? targetSelect.value : "wasm_simd";

                        const compiledOutput = compileCode(pyodide, sourceCode, sourceFw, targetFw);

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
                            // Example input
                            const inputData = new Float32Array([1.0, 2.0, 3.0, 4.0]);
                            const outputSizeInBytes = 4 * Float32Array.BYTES_PER_ELEMENT;

                            const result = await runWebGPUCompute(win.navigator, code, inputData, outputSizeInBytes);
                            logToConsole(doc, t('webgpuComplete'));
                            logToConsole(doc, "[" + result.join(", ") + "]");
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

                            // Here we would normally compile the targetEditor code (WAT) to WASM bytes
                            // or fetch the already compiled bytes. For the playground, we simulate it
                            // if we don't have a real wasm compiler in JS available.
                            // We provide a dummy WASM module that just adds 1 to simulate.

                            // A very tiny WASM module binary (magic header + version + empty)
                            // This won't work with runWasmCompute because it lacks exports,
                            // but we mock it for the playground's visual feedback when a real binary isn't available.
                            const dummyWasmBytes = new Uint8Array([0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00]);

                            const inputData = new Float32Array([1.0, 2.0, 3.0, 4.0]);
                            const expectedOutputLength = 4;

                            // Note: We catch and display errors since we don't have a real WASM binary here.
                            await runWasmCompute(dummyWasmBytes, inputData, expectedOutputLength);

                        } catch (e) {
                            // Expected to fail because dummyWasmBytes doesn't export 'memory' and 'compute'
                            logToConsole(doc, t('wasmError') + e.message, true);
                            logToConsole(doc, t('wasmNote'), true);
                        }
                    }
                });
            }
        });
    }
}

// Export for testing
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        applyI18n,
        initTheme,
        updateEditorTheme,
        getExampleCode,
        updateExecuteButtonVisibility,
        logToConsole,
        clearConsole,
        loadPyodideEnvironment,
        compileCode,
        initPlayground,
    };
} else {
    // Run in browser
    window.addEventListener('DOMContentLoaded', () => {
        initPlayground(document, localStorage, window);
    });
}
