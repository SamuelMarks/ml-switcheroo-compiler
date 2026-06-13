"""Sphinx extension for embedding an interactive Machine Learning playground.

This module defines a custom Docutils directive that renders a split-pane code editor
and execution console. It allows users to write, compile, and run ML code (e.g., JAX,
PyTorch, TensorFlow) directly in the browser using Pyodide, WebGPU, and WASM.
"""

from docutils import nodes
from docutils.parsers.rst import Directive


class MLPlaygroundDirective(Directive):
    """A Docutils directive for rendering the ML Playground interface.

    This directive generates the raw HTML structure required to display the interactive
    playground, including framework selectors, code editors, and execution consoles.

    Attributes:
    has_content (bool): Indicates whether the directive content block is
        allowed. Defaults to True.
    """

    has_content = True

    def run(self) -> object:
        """Docstring."""
        # We output a section with an id that our JS will hydrate
        html = """
<section id="ml-playground-container" aria-label="ML Switcheroo Playground">
    <header class="pg-header">
        <label class="theme-switch" aria-label="Toggle Dark Mode">
            <input type="checkbox" id="theme-toggle">
            <span class="slider round"></span>
        </label>
        <span class="theme-label" data-i18n="darkMode">Dark Mode</span>
    </header>
    <div class="pg-split-pane">
        <section class="pg-left-pane" aria-label="Source Editor">
            <header class="pg-pane-header">
                <select id="source-framework"
                        aria-label="Source Framework"
                        data-i18n-aria="sourceFw"
                >
                    <optgroup label="Base ML Frameworks" data-i18n-label="baseFw">
                        <option value="tensorflow">TensorFlow</option>
                        <option value="keras">Keras</option>
                        <option value="pytorch">PyTorch</option>
                        <option value="mlx">MLX</option>
                    </optgroup>
                    <optgroup label="JAX Ecosystem" data-i18n-label="jaxEco">
                        <option value="jax" selected>JAX</option>
                        <option value="flax_nnx">Flax NNX</option>
                        <option value="flax_linen">Flax Linen</option>
                        <option value="pax">Pax</option>
                    </optgroup>
                </select>
                <select id="source-example" aria-label="Examples"
                    data-i18n-aria="examples"
                >
                    <option value="simple_mlp">Simple MLP</option>
                    <option value="cnn">CNN</option>
                    <option value="attention">Attention Block</option>
                </select>
            </header>
            <div id="editor-source" class="pg-editor"></div>
        </section>
        <section class="pg-right-pane" aria-label="Target Editor and Console">
            <header class="pg-pane-header">
                <select id="target-framework"
                        aria-label="Target Framework"
                        data-i18n-aria="targetFw"
                >
                    <optgroup label="Base ML Frameworks" data-i18n-label="baseFw">
                        <option value="tensorflow">TensorFlow</option>
                        <option value="keras">Keras</option>
                        <option value="pytorch">PyTorch</option>
                        <option value="mlx">MLX</option>
                    </optgroup>
                    <optgroup label="JAX Ecosystem" data-i18n-label="jaxEco">
                        <option value="jax">JAX</option>
                        <option value="flax_nnx">Flax NNX</option>
                        <option value="flax_linen">Flax Linen</option>
                        <option value="pax">Pax</option>
                    </optgroup>
                    <optgroup label="Native Web Execution" data-i18n-label="nativeWeb">
                        <option value="webgpu">WebGPU</option>
                        <option value="wasm_simd" selected>WASM SIMD</option>
                    </optgroup>
                </select>
                <button id="btn-compile" data-i18n="compile">Compile</button>
                <button id="btn-execute" style="display: none;" data-i18n="execute">
                    Execute in browser
                </button>
            </header>
            <div id="editor-target" class="pg-editor"></div>
            <div id="pg-console" class="pg-console" aria-live="polite"
                 role="region"
                 aria-label="Compilation Console"></div>
        </section>
    </div>
</section>
<!-- Pyodide Loader -->
<script src="https://cdn.jsdelivr.net/pyodide/v0.25.0/full/pyodide.js"></script>

<!-- Monaco Editor Loader -->
<script>
var require = { paths: { 'vs':
    'https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.45.0/min/vs' } };
</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/" +
                     "0.45.0/min/vs/loader.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/" +
                     "0.45.0/min/vs/editor/editor.main.nls.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/" +
                     "0.45.0/min/vs/editor/editor.main.js"></script>
"""

        return [nodes.raw("", html, format="html")]


def setup(app: object) -> object:
    """Initializes the Sphinx extension.

    Registers the `ml-playground` directive and associates the required
    CSS and JavaScript assets for the playground's interactive features.

    Args:
        app (object): The Sphinx application object.

    Returns:
        dict: A dictionary containing extension metadata, including the
            version and parallel read/write safety flags.
    """
    app.add_directive("ml-playground", MLPlaygroundDirective)

    # We will also add JS and CSS assets here
    app.add_css_file("playground.css")
    app.add_js_file("playground.js")
    app.add_js_file("webgpu_runner.js")
    app.add_js_file("wasm_runner.js")

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
