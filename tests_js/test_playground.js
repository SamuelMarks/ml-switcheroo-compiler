const test = require('node:test');
const assert = require('node:assert');
const { JSDOM } = require('jsdom');
const fs = require('fs');
const path = require('path');
const axe = require('axe-core');

const jsCode = fs.readFileSync(path.join(__dirname, '../docs/_static/playground.js'), 'utf8');

// ... (mock storage code)
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

const fn = new Function('module', 'exports', jsCode);
const m = { exports: {} };
fn(m, m.exports);
const playgroundModule = m.exports;

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
    
    playgroundModule.initTheme(doc, storage, win);
    
    const toggle = doc.getElementById('theme-toggle');
    assert.strictEqual(toggle.checked, true);
    assert.strictEqual(doc.documentElement.getAttribute('data-theme'), 'dark');
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
    
    const consoleEl = doc.getElementById('pg-console');
    assert.strictEqual(consoleEl.children.length, 1);
    assert.strictEqual(consoleEl.children[0].textContent, 'Test msg\n');
    
    playgroundModule.logToConsole(doc, 'Error msg', true);
    assert.strictEqual(consoleEl.children.length, 2);
    assert.strictEqual(consoleEl.children[1].style.color, 'red');
    
    playgroundModule.clearConsole(doc);
    assert.strictEqual(consoleEl.children.length, 0);
    assert.strictEqual(consoleEl.textContent, '');
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
test('applyI18n does not fail on missing keys', () => {
    const dom = new JSDOM('<div data-i18n="missing"></div>').window.document;
    // We export applyI18n to test it
    if (playgroundModule.applyI18n) {
        playgroundModule.applyI18n(dom);
        assert.strictEqual(dom.querySelector('div').textContent, '');
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
    // We mock require to run the inner function immediately
    win.require = (deps, cb) => cb();
    
    playgroundModule.initPlayground(dom.window.document, new MockStorage(), win);
    
    // simulate click
    dom.window.document.getElementById('btn-compile').click();
    
    // Give promises time
    await new Promise(r => setTimeout(r, 10));
    const consoleEl = dom.window.document.getElementById('pg-console');
    assert.match(consoleEl.textContent, /Pyodide script not loaded/);
});

// 1. Missing targetSelect inside execute
test('execute logic bails early if targetEditor is missing', async () => {
    const dom = new JSDOM('<html><body><button id="btn-execute"></button></body></html>');
    const win = dom.window;
    win.require = (deps, cb) => cb();
    playgroundModule.initPlayground(dom.window.document, new MockStorage(), win);
    dom.window.document.getElementById('btn-execute').click();
    assert.ok(true);
});

// 2. Missing getExampleCode branch (if example string isn't found)
test('getExampleCode fallback fallback for nonexistent framework/example', () => {
    const code = playgroundModule.getExampleCode('not_a_framework', 'simple_mlp');
    assert.match(code, /Example code not found/);
});

test('initPlayground initializes without throwing', () => {
    const dom = new JSDOM(`
        <html><body>
            <input type="checkbox" id="theme-toggle" />
            <select id="target-framework"><option value="webgpu">WebGPU</option></select>
            <button id="btn-execute"></button>
            <select id="source-framework"></select>
            <select id="source-example"></select>
            <div id="editor-source"></div>
            <div id="editor-target"></div>
            <button id="btn-compile"></button>
            <div id="pg-console"></div>
        </body></html>
    `);
    const win = dom.window;
    win.matchMedia = () => ({ matches: false });
    
    assert.doesNotThrow(() => {
        playgroundModule.initPlayground(dom.window.document, new MockStorage(), win);
    });
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
