# ML Switcheroo Web Frontend & API Docs Plan

## 1. Overview
The goal is to build an interactive Web Playground embedded within a unified, Sphinx-generated documentation site. This playground will allow users to write code in various ML frameworks, compile it using `ml-switcheroo-compiler` to different target frameworks, and natively execute WebGPU/WASM SIMD targets directly in the browser. The documentation will aggregate API references across the main compiler and its extensive `zero-*` ecosystem.

## 2. UI/UX Design (The Playground)
The playground will be cleanly integrated into the Sphinx theme (e.g., Furo or Sphinx Book Theme) using a wide, split-pane layout to maximize editor real estate.

### Layout Elements & Features
- [ ] **Global Controls:**
  - [ ] **Theme Toggle:** An accessible toggle switch for Light/Dark mode that respects system preferences (`prefers-color-scheme`) by default.
- [ ] **Left Pane (Source):**
  - [ ] **Header Controls:**
    - [ ] **Source Framework Dropdown:** Uses `<optgroup>` to organize options logically:
      - *Group: Base ML Frameworks* (TensorFlow, Keras, PyTorch, MLX)
      - *Group: JAX Ecosystem* (JAX, Flax NNX, Flax Linen)
    - [ ] **Examples Dropdown:** Pre-populated code snippets that auto-update based on the selected Source Framework (e.g., Simple MLP, CNN, Attention Block).
  - [ ] **Editor:** A Monaco Editor instance configured for Python.
    - [ ] Enable syntax highlighting.
    - [ ] Enable basic autocomplete for Python syntax.
    - [ ] Line numbers and minimap configured for readability.
- [ ] **Right Pane (Target/Output):**
  - [ ] **Header Controls:**
    - [ ] **Target Framework Dropdown:** Uses `<optgroup>` to organize options logically:
      - *Group: Base ML Frameworks* (TensorFlow, Keras, PyTorch, MLX)
      - *Group: JAX Ecosystem* (JAX, Flax NNX, Flax Linen)
      - *Group: Native Web Execution* (WebGPU, WASM SIMD)
    - [ ] **Action Buttons:**
      - [ ] `Compile`: Runs the ML Switcheroo compiler to generate the target code.
      - [ ] `Execute in browser`: **Dynamically visible** *only* when WebGPU or WASM SIMD is selected.
  - [ ] **Editors/Panels:**
    - [ ] **Output Editor:** A read-only Monaco Editor instance. Syntax highlighting dynamically switches based on the target (e.g., Python, C++, WGSL, WAT).
    - [ ] **Console Panel:** A terminal-like output view below the editor to show compilation logs, errors, or execution results (e.g., resulting tensors).

## 3. Technical Architecture & Constraints

### 3.1 Strict Frontend Constraints
- **Pure Vanilla Stack:** The frontend must be built using purely vanilla HTML, CSS, and JavaScript. **No frontend frameworks or build tools** (like React, Vue, Webpack, Vite, or Tailwind) are permitted.
- **Responsive CSS:** All CSS must be handwritten vanilla CSS. It must be fully responsive, gracefully collapsing the split-pane layout into a stacked layout for mobile and smaller viewports.
- **Light & Dark Mode:** The CSS must implement a robust theming system using CSS Custom Properties (variables). It must auto-detect system preferences via media queries (`@media (prefers-color-scheme: dark)`) and provide a manual override toggle that persists in `localStorage`.
- **Accessibility (a11y):** The entire playground must strictly adhere to WCAG 2.1 AA standards. This includes:
  - Full keyboard navigability (proper `tabindex` and focus states).
  - Correct ARIA attributes (e.g., `aria-live` for the compilation console, `aria-expanded` for dropdowns).
  - High contrast ratios in both light and dark modes.
  - Semantic HTML elements (`<main>`, `<nav>`, `<optgroup>`, `<section>`).
- **100% Test Coverage:** All JavaScript logic (UI state machines, Pyodide bridge, WebGPU runner, WASM runner) must be tested using a lightweight, dependency-free testing approach (e.g., Node's native test runner for logic, or a headless browser setup like Puppeteer for integration) to achieve 100% test coverage.
- **100% Doc Coverage:** Every JavaScript function, class, and module must have complete JSDoc coverage describing inputs, outputs, side effects, and state mutations.

### 3.2 Static Site & Integration
- **Generator:** Sphinx (`sphinx-build`).
- **Playground Injection:** Created via a custom Sphinx Directive (`.. ml-playground::`) that injects a `div` scaffold, which is then hydrated by Sphinx `_static/` JavaScript assets.
- **Frontend Assets:** Loaded locally via the `_static` directory. The Monaco Editor may be loaded via CDN (e.g., jsDelivr) to keep the repository light, but the core application logic is strictly vanilla JS.

### 3.3 Compilation Strategy
To compile Python ML code into other representations within a static HTML site, we will evaluate two viable paths:
- **Option A (Client-Side Pyodide - Recommended for static hosting):**
  - [ ] Run `ml-switcheroo-compiler` directly in the browser via Pyodide.
  - [ ] Compile the library and its pure-Python dependencies into a `.whl`.
  - [ ] JavaScript calls Python compilation functions via `pyodide.runPython` or exposed JS-to-Python bindings.
- **Option B (Server-Side Backend - Fallback):**
  - [ ] Host a lightweight FastAPI service.
  - [ ] The static HTML makes REST POST calls containing the source code and target parameters.
  - [ ] Receive the compiled output (or compilation errors) back as JSON.

### 3.4 In-Browser Execution Environments
When the user clicks "Execute in browser":

- **WebGPU Target:**
  - [ ] The compiler outputs valid WGSL (WebGPU Shading Language) and a JSON metadata payload detailing inputs/outputs.
  - [ ] The frontend JS verifies `navigator.gpu` is available.
  - [ ] Requests an adapter and device (`navigator.gpu.requestAdapter()`).
  - [ ] Allocates GPU buffers using `device.createBuffer()`.
  - [ ] Creates a compute pipeline (`device.createComputePipeline()`).
  - [ ] Dispatches the work via a command encoder.
  - [ ] Maps the output buffer (`buffer.mapAsync()`), reads back the `Float32Array`, and prints to the UI console.

- **WASM SIMD Target:**
  - [ ] The compiler outputs WebAssembly (WASM binary format) with SIMD instructions enabled (`-msimd128`).
  - [ ] The frontend fetches and instantiates the WASM module (`WebAssembly.instantiate()`).
  - [ ] Populates shared `WebAssembly.Memory` with input tensors.
  - [ ] Calls the exported WASM computation functions.
  - [ ] Reads the resulting memory buffer back and parses it into typed arrays for display in the UI console.

## 4. Multi-Project Sphinx Documentation Setup
The documentation site will serve as a monolithic portal for 13 distinct codebases.

### Path Configuration
In `docs/conf.py`, the path resolution must reach out to the sibling directories dynamically:
```python
import os
import sys

# All target directories relative to the current conf.py
projects = [
    "../ml-switcheroo-compiler", "../ml-switcheroo-ir", "../zero-chex",
    "../zero-flax", "../zero-grain", "../zero-jax", "../zero-keras",
    "../zero-mlx", "../zero-optax", "../zero-orbax", "../zero-pax",
    "../zero-pytorch", "../zero-tensorflow"
]

for p in projects:
    sys.path.insert(0, os.path.abspath(p))
```

### Doc Generation Tools
- [ ] **`sphinx.ext.autodoc` & `sphinx.ext.autosummary`**: Used to recursively scrape docstrings from all registered paths.
- [ ] **Sphinx extensions to add:** `sphinx_copybutton`, `sphinx_design`, `myst_parser`.
- [ ] **Table of Contents (TOC) Organization**:
  - **1. Playground & Interactive Sandbox** (The primary landing experience).
  - **2. Compiler Core**: `ml-switcheroo-compiler`, `ml-switcheroo-ir`.
  - **3. Frontend/Backend Ecosystem**: `zero-jax`, `zero-pytorch`, `zero-tf`, `zero-keras`, `zero-mlx`.
  - **4. High-Level Libraries**: `zero-optax`.

### Fast Development Workflow (Makefile)
Because scraping 13 distinct projects via `autosummary` will make iterative frontend development extremely slow, we require optimized `make` targets. The `conf.py` will read environment variables (e.g., `FAST_BUILD=1`) to conditionally bypass scraping sibling projects and only build the core pages and interactive frontend.

- [ ] `make docs`: Standard command. Scrapes all 13 projects, generates complete API documentation, and builds the playground.
- [ ] `make docs-fast`: Fast command. Disables heavy `autosummary` generation and builds *only* the web frontend/playground pages for rapid UI/UX iteration.
- [ ] `make serve_docs`: Builds the full docs and serves them locally (e.g., via `python -m http.server -d docs/_build/html`).
- [ ] `make serve_docs-fast`: Runs the fast frontend-only build and serves it locally.

## 5. Execution Phasing

### Phase 1: Multi-Repo Sphinx Plumbing
- [ ] Initialize the base `docs/` folder using `sphinx-quickstart`.
- [ ] Configure path resolution in `conf.py` to all `../zero-*` sibling directories.
- [ ] Create `.rst` or `.md` (via MyST) stub files for each project.
- [ ] Configure `autosummary` to generate the unified API documentation stubs.
- [ ] Setup the `FAST_BUILD` environment variable toggle in `conf.py` to optionally skip large project imports.
- [ ] Implement `make docs`, `make docs-fast`, `make serve_docs`, and `make serve_docs-fast` in the repository `Makefile`.
- [ ] Test the static builds (both full and fast) to ensure routing and static assets compile properly.

### Phase 2: Playground UI Scaffold & Testing Baseline
- [ ] Setup vanilla JS testing framework (e.g., native Node.js test runner) and JSDoc linting.
- [ ] Create a custom Sphinx Directive to output the playground HTML container.
- [ ] Embed Monaco Editor via CDN into the static assets.
- [ ] Build the **responsive vanilla CSS** to handle the split-pane layout and mobile stacking.
- [ ] Implement the UI state machine in strictly vanilla JS:
  - [ ] Setup Light/Dark mode toggle (respecting system prefs and saving to `localStorage`).
  - [ ] Setup Source Framework dropdown logic using semantic `<optgroup>` elements.
  - [ ] Wire the Examples dropdown to update the left editor.
  - [ ] Setup Target Framework dropdown logic using semantic `<optgroup>` elements.
  - [ ] Write logic to dynamically show/hide the "Execute in browser" button based on target selection.
- [ ] **Ensure 100% test coverage and 100% JSDoc coverage for all UI logic before proceeding.**
- [ ] **Conduct full accessibility (a11y) audit via axe-core or Lighthouse to ensure WCAG 2.1 AA compliance (keyboard nav, ARIA labels, contrast ratios).**

### Phase 3: Compiler Hookup
- [ ] Create a build script to package `ml-switcheroo-compiler` as a Pyodide-compatible wheel.
- [ ] Integrate Pyodide loading into the Sphinx static JS.
- [ ] Connect the "Compile" button click event to execute the Pyodide compiler function.
- [ ] Handle compilation errors gracefully and pipe them to the UI console.
- [ ] Update the right-hand editor with the successfully compiled output string.
- [ ] **Write tests ensuring the JS-to-Pyodide bridge is fully covered.**

### Phase 4: WebGPU Execution Engine
- [ ] Build a JS module `webgpu_runner.js`.
- [ ] Implement initialization check for WebGPU support.
- [ ] Implement data serialization/deserialization to move JS arrays into GPU buffers.
- [ ] Hook the "Execute in browser" button to run the `webgpu_runner` when WGSL is the target.
- [ ] Pipe the numeric outputs (or execution errors) to the UI console.
- [ ] **Ensure 100% test and JSDoc coverage for `webgpu_runner.js`.**

### Phase 5: WASM SIMD Execution Engine
- [ ] Build a JS module `wasm_runner.js`.
- [ ] Implement WASM instantiation logic and `WebAssembly.Memory` management.
- [ ] Hook the "Execute in browser" button to run the `wasm_runner` when WASM is the target.
- [ ] Ensure proper memory bounds checking and read-back for the final tensor results.
- [ ] Output the results to the UI console.
- [ ] **Ensure 100% test and JSDoc coverage for `wasm_runner.js`.**
