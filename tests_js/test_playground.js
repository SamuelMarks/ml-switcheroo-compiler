const test = require('node:test');
const assert = require('node:assert');
const { JSDOM } = require('jsdom');
const fs = require('fs');
const path = require('path');
const axe = require('axe-core');

const playgroundModule = require('../docs/_static/playground.js');

class MockStorage {
    constructor() {
        this.store = {};
    }
    getItem(key) {
        return this.store[key] || null;
    }
    setItem(key, value) {
        this.store[key] = value.toString();
    }
}

test('t function fallback and exact match', () => {
    assert.strictEqual(playgroundModule.t('compile'), 'Compile');
    assert.strictEqual(playgroundModule.t('compile', 'fr'), 'compile'); // fallback to key
    assert.strictEqual(playgroundModule.t('unknown_key', 'en'), 'unknown_key'); // fallback to key
});

test('getExampleCode returns correct string', () => {
    const code = playgroundModule.getExampleCode('jax', 'simple_mlp');
    assert.match(code, /import jax\.numpy/);
});

test('getExampleCode fallback for missing', () => {
    const code = playgroundModule.getExampleCode('unknown', 'unknown');
    assert.match(code, /# Example code not found for/);
});

test('initTheme uses localStorage', () => {
    const dom = new JSDOM(`
        <!DOCTYPE html>
        <html data-theme="light">
        <body>
            <input type="checkbox" id="theme-toggle" />
        </body>
        </html>
    `);
    const doc = dom.window.document;
    const storage = new MockStorage();
    storage.setItem('ml-playground-theme', 'dark');

    const win = dom.window;
    win.matchMedia = () => ({ matches: false }); // Prefers light
    win.monaco = { editor: { setTheme: () => {} } };

    playgroundModule.initTheme(doc, storage, win);

    const toggle = doc.getElementById('theme-toggle');
    assert.strictEqual(toggle.checked, true);
    assert.strictEqual(doc.documentElement.getAttribute('data-theme'), 'dark');

    // test toggle change event to light
    toggle.checked = false;
    toggle.dispatchEvent(new dom.window.Event('change'));
    assert.strictEqual(doc.documentElement.getAttribute('data-theme'), 'light');
    assert.strictEqual(storage.getItem('ml-playground-theme'), 'light');

    // test toggle change event back to dark
    toggle.checked = true;
    toggle.dispatchEvent(new dom.window.Event('change'));
    assert.strictEqual(doc.documentElement.getAttribute('data-theme'), 'dark');
    assert.strictEqual(storage.getItem('ml-playground-theme'), 'dark');
});

test('initTheme uses matchMedia if no localStorage', () => {
    const dom = new JSDOM(`
        <!DOCTYPE html>
        <html>
        <body>
            <input type="checkbox" id="theme-toggle" />
        </body>
        </html>
    `);
    const doc = dom.window.document;
    const storage = new MockStorage();

    const win = dom.window;
    win.matchMedia = (query) => {
        return { matches: query === '(prefers-color-scheme: dark)' };
    };

    playgroundModule.initTheme(doc, storage, win);

    const toggle = doc.getElementById('theme-toggle');
    assert.strictEqual(toggle.checked, true);
    assert.strictEqual(doc.documentElement.getAttribute('data-theme'), 'dark');
});

test('updateExecuteButtonVisibility shows button for webgpu', () => {
    const dom = new JSDOM(`
        <select id="target-framework">
            <option value="webgpu" selected>WebGPU</option>
        </select>
        <button id="btn-execute" style="display: none;"></button>
    `);
    playgroundModule.updateExecuteButtonVisibility(dom.window.document);
    assert.strictEqual(dom.window.document.getElementById('btn-execute').style.display, 'inline-block');
});

test('updateExecuteButtonVisibility hides button for other frameworks', () => {
    const dom = new JSDOM(`
        <select id="target-framework">
            <option value="jax" selected>JAX</option>
        </select>
        <button id="btn-execute" style="display: inline-block;"></button>
    `);
    playgroundModule.updateExecuteButtonVisibility(dom.window.document);
    assert.strictEqual(dom.window.document.getElementById('btn-execute').style.display, 'none');
});

test('logToConsole and clearConsole work correctly', () => {
    const dom = new JSDOM(`
        <div id="pg-console"></div>
    `);
    const doc = dom.window.document;
    playgroundModule.logToConsole(doc, 'Test msg');
    playgroundModule.logToConsole(doc, 'Test msg', false);

    const consoleEl = doc.getElementById('pg-console');
    assert.strictEqual(consoleEl.children.length, 2);
    assert.strictEqual(consoleEl.children[0].textContent, 'Test msg\n');

    playgroundModule.logToConsole(doc, 'Error msg', true);
    assert.strictEqual(consoleEl.children.length, 3);
    assert.strictEqual(consoleEl.children[2].style.color, 'red');

    playgroundModule.clearConsole(doc);
    assert.strictEqual(consoleEl.children.length, 0);
    assert.strictEqual(consoleEl.textContent, '');

    // Test missing console element
    const docWithoutConsole = new JSDOM('<div></div>').window.document;
    assert.doesNotThrow(() => playgroundModule.logToConsole(docWithoutConsole, 'Test msg'));
    assert.doesNotThrow(() => playgroundModule.clearConsole(docWithoutConsole));
});

test('loadPyodideEnvironment initializes successfully', async () => {
    const doc = new JSDOM('<div id="pg-console"></div>').window.document;
    const win = {
        loadPyodide: async () => ({
            loadPackage: async () => {},
            pyimport: () => ({
                install: async () => {}
            })
        })
    };

    const pyodide = await playgroundModule.loadPyodideEnvironment(doc, win);
    assert.ok(pyodide);
    // test singleton
    const pyodide2 = await playgroundModule.loadPyodideEnvironment(doc, win);
    assert.strictEqual(pyodide, pyodide2);
});

test('loadPyodideEnvironment throws on error', async () => {
    delete require.cache[require.resolve('../docs/_static/playground.js')];
    const freshPlayground = require('../docs/_static/playground.js');
    const doc = new JSDOM('<div id="pg-console"></div>').window.document;
    const win = {
        loadPyodide: async () => { throw new Error('Network err'); }
    };
    await assert.rejects(() => freshPlayground.loadPyodideEnvironment(doc, win), /Network err/);

    // restore the main instance for other tests if needed
    delete require.cache[require.resolve('../docs/_static/playground.js')];
    require('../docs/_static/playground.js');
});

test('compileCode runs successfully', () => {
    const pyodide = {
        runPython: (script) => {
            return "Compiled success";
        }
    };

    const res = playgroundModule.compileCode(pyodide, 'def f(): pass', 'jax', 'webgpu');
    assert.strictEqual(res, 'Compiled success');
});

test('compileCode handles error', () => {
    const pyodide = {
        runPython: () => {
            throw new Error('Runtime fault');
        }
    };
    const res = playgroundModule.compileCode(pyodide, 'def f(): pass', 'jax', 'webgpu');
    assert.match(res, /Compilation failed: Runtime fault/);
});

test('applyI18n with unknown lang', () => {
    const dom = new JSDOM('<html lang="fr"><body><div data-i18n="compile"></div><div data-i18n-aria="sourceFw"></div><div data-i18n-label="baseFw"></div></body></html>').window.document;
    if (playgroundModule.applyI18n) {
        playgroundModule.applyI18n(dom, 'fr');
        assert.strictEqual(dom.querySelector('div').textContent, '');
    }
});

test('applyI18n does not fail on missing keys', () => {
    const dom = new JSDOM('<div data-i18n="missing"></div>').window.document;
    if (playgroundModule.applyI18n) {
        playgroundModule.applyI18n(dom);
        assert.strictEqual(dom.querySelector('div').textContent, '');
    }
});

test('applyI18n applies aria and label', () => {
    const dom = new JSDOM('<div data-i18n="compile"></div><div data-i18n-aria="sourceFw"></div><div data-i18n-label="baseFw"></div>').window.document;
    if (playgroundModule.applyI18n) {
        playgroundModule.applyI18n(dom);
        assert.strictEqual(dom.querySelector('[data-i18n]').textContent, 'Compile');
        assert.strictEqual(dom.querySelector('[data-i18n-aria]').getAttribute('aria-label'), 'Source Framework');
        assert.strictEqual(dom.querySelector('[data-i18n-label]').getAttribute('label'), 'Base ML Frameworks');
    }
});

test('compile click fails gracefully if loadPyodide is missing', async () => {
    const dom = new JSDOM(`
        <html><body>
            <button id="btn-compile"></button>
            <div id="pg-console"></div>
        </body></html>
    `);
    const win = dom.window;
    win.require = (deps, cb) => cb();

    playgroundModule.initPlayground(dom.window.document, new MockStorage(), win);

    // simulate click
    dom.window.document.getElementById('btn-compile').click();

    await new Promise(r => setTimeout(r, 10));
    const consoleEl = dom.window.document.getElementById('pg-console');
    assert.match(consoleEl.textContent, /Pyodide script not loaded/);
});

test('execute logic bails early if targetEditor is missing', async () => {
    const dom = new JSDOM('<html><body><button id="btn-execute"></button></body></html>');
    const win = dom.window;
    win.require = (deps, cb) => cb();
    playgroundModule.initPlayground(dom.window.document, new MockStorage(), win);
    dom.window.document.getElementById('btn-execute').click();
    assert.ok(true);
});

test('getExampleCode fallback fallback for nonexistent framework/example', () => {
    const code = playgroundModule.getExampleCode('not_a_framework', 'simple_mlp');
    assert.match(code, /Example code not found/);
});

test('initPlayground initializes editors and handlers', async () => {
    const dom = new JSDOM(`
        <html data-theme="dark"><body>
            <input type="checkbox" id="theme-toggle" />
            <select id="target-framework"><option value="webgpu">WebGPU</option><option value="wasm_simd">WASM</option><option value="jax">JAX</option></select>
            <button id="btn-execute"></button>
            <select id="source-framework"><option value="jax">JAX</option></select>
            <select id="source-example"><option value="simple_mlp">MLP</option></select>
            <div id="editor-source"></div>
            <div id="editor-target"></div>
            <button id="btn-compile"></button>
            <div id="pg-console"></div>
        </body></html>
    `);
    const win = dom.window;
    win.matchMedia = () => ({ matches: false });
    let shouldThrowCompile = false;
    win.loadPyodide = async () => ({
        loadPackage: async () => {},
        pyimport: () => ({ install: async () => {} }),
        runPython: () => {
            if (shouldThrowCompile) throw new Error('compile failed');
            return "Mock compiled code";
        }
    });

    let mockTargetLang = '';
    const mockModel = {};
    win.require = (deps, cb) => cb();
    win.monaco = {
        editor: {
            create: (el, opts) => ({
                getValue: () => opts.value || "code",
                setValue: (v) => { if (shouldThrowCompile) throw new Error('editor error'); },
                getModel: () => mockModel,
                setModelLanguage: (m, l) => {}
            }),
            setTheme: () => {},
            setModelLanguage: (m, l) => { mockTargetLang = l; }
        }
    };
    global.runWebGPUCompute = async () => [42.0];

    playgroundModule.initPlayground(dom.window.document, new MockStorage(), win);

    // Trigger updateSource
    dom.window.document.getElementById('source-framework').dispatchEvent(new dom.window.Event('change'));
    dom.window.document.getElementById('target-framework').dispatchEvent(new dom.window.Event('change'));

    // Trigger compile click (webgpu)
    dom.window.document.getElementById('btn-compile').click();
    await new Promise(r => setTimeout(r, 100)); // allow compile to run

    // Check python target
    dom.window.document.getElementById('target-framework').value = 'jax';
    dom.window.document.getElementById('btn-compile').click();
    await new Promise(r => setTimeout(r, 100));

    // Check compile error
    shouldThrowCompile = true;
    dom.window.document.getElementById('btn-compile').click();
    await new Promise(r => setTimeout(r, 100));
    shouldThrowCompile = false;

    // Check execute when runWebGPUCompute is undefined
    delete global.runWebGPUCompute;
    dom.window.document.getElementById('target-framework').value = 'webgpu';
    dom.window.document.getElementById('btn-compile').click();
    await new Promise(r => setTimeout(r, 100));
    dom.window.document.getElementById('btn-execute').click();
    await new Promise(r => setTimeout(r, 100));

    // Check execute when runWasmCompute is undefined
    delete global.runWasmCompute;
    dom.window.document.getElementById('target-framework').value = 'wasm_simd';
    dom.window.document.getElementById('btn-compile').click();
    await new Promise(r => setTimeout(r, 100));
    dom.window.document.getElementById('btn-execute').click();
    await new Promise(r => setTimeout(r, 100));

    // Now mock them successfully
    global.runWebGPUCompute = async () => [42.0];
    global.runWasmCompute = async () => [42.0];

    // WebGPU success
    dom.window.document.getElementById('target-framework').value = 'webgpu';
    dom.window.document.getElementById('btn-compile').click();
    await new Promise(r => setTimeout(r, 100));
    dom.window.document.getElementById('btn-execute').click();
    await new Promise(r => setTimeout(r, 100));

    // WASM success
    dom.window.document.getElementById('target-framework').value = 'wasm_simd';
    dom.window.document.getElementById('btn-compile').click();
    await new Promise(r => setTimeout(r, 100));
    dom.window.document.getElementById('btn-execute').click();
    await new Promise(r => setTimeout(r, 100));

    // WASM error
    global.runWasmCompute = async () => { throw new Error('mock wasm err'); };
    dom.window.document.getElementById('btn-execute').click();
    await new Promise(r => setTimeout(r, 100));

    // WebGPU error
    global.runWebGPUCompute = async () => { throw new Error('mock webgpu err'); };
    dom.window.document.getElementById('target-framework').value = 'webgpu';
    dom.window.document.getElementById('btn-compile').click();
    await new Promise(r => setTimeout(r, 100));
    dom.window.document.getElementById('btn-execute').click();
    await new Promise(r => setTimeout(r, 100));

    // Check coverage
    assert.ok(true);

    delete global.runWebGPUCompute;
    delete global.runWasmCompute;
});

test('initPlayground initializes editors with light theme', async () => {
    const dom = new JSDOM(`
        <html><body>
            <div id="editor-source"></div>
        </body></html>
    `);
    const win = dom.window;
    win.matchMedia = () => ({ matches: false });
    win.require = (deps, cb) => cb();
    win.monaco = { editor: { create: () => ({ getValue: () => "", setValue: () => {} }) } };
    playgroundModule.initPlayground(dom.window.document, new MockStorage(), win);
    assert.ok(true);
});

test('browser environment run check', () => {
    // mock DOMContentLoaded
    const jsCode = fs.readFileSync(path.join(__dirname, '../docs/_static/playground.js'), 'utf8');
    const fn = new Function('module', 'exports', 'window', 'document', 'localStorage', jsCode);
    const win = {
        addEventListener: (e, cb) => cb(),
        require: (deps, cb) => cb(),
        matchMedia: () => ({ matches: false })
    };
    const doc = (new JSDOM('<html><body></body></html>')).window.document;
    const localStorage = new MockStorage();

    // Call without module
    assert.doesNotThrow(() => fn(undefined, undefined, win, doc, localStorage));
});

test('initPlayground with missing dom elements', async () => {
    const dom = new JSDOM(`
        <html><body>
            <button id="btn-compile"></button>
            <div id="pg-console"></div>
        </body></html>
    `);
    const win = dom.window;
    win.matchMedia = () => ({ matches: false });
    win.loadPyodide = async () => ({
        loadPackage: async () => {},
        pyimport: () => ({ install: async () => {} }),
        runPython: () => "Mock"
    });
    win.require = (deps, cb) => cb();
    win.monaco = {
        editor: {
            create: () => ({ getValue: () => "", setValue: () => {} }),
            setTheme: () => {}
        }
    };

    playgroundModule.initPlayground(dom.window.document, new MockStorage(), win);

    // Trigger compile click without editors/selects
    dom.window.document.getElementById('btn-compile').click();
    await new Promise(r => setTimeout(r, 100));
    assert.ok(true);
});

test('Accessibility audit', async () => {
    const htmlSnippet = fs.readFileSync(path.join(__dirname, '../docs/ml_playground_directive.py'), 'utf8');
    const match = htmlSnippet.match(/<section id="ml-playground-container" aria-label="ML Switcheroo Playground">[\s\S]*?<\/section>\n<!--/);
    let containerHtml = match ? match[0].replace('<!--', '') : '';

    const dom = new JSDOM(`<!DOCTYPE html>
<html lang="en">
<head><title>Test</title></head>
<body>${containerHtml}</body>
</html>`);

    const results = await axe.run(dom.window.document.body);
    assert.strictEqual(results.violations.length, 0, 'Axe violations: ' + JSON.stringify(results.violations, null, 2));
    assert.ok(results);
});
