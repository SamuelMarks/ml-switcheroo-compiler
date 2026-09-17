"""Sphinx extension for embedding an interactive Machine Learning playground.

This module defines a custom Docutils directive that renders a split-pane code editor
and execution console. It allows users to write, compile, and run ML code (e.g., JAX,
PyTorch, TensorFlow) directly in the browser using Pyodide, WebGPU, and WASM.
"""

import glob
import json
import os
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    import sphinx.application  # type: ignore[import-untyped]

from docutils import nodes  # type: ignore[import-untyped]
from docutils.parsers.rst import Directive  # type: ignore[import-untyped]


def get_wheel_assets() -> list[str]:
    """Finds wheel (.whl) distribution files in the docs/_static directory.

    Returns:
        Sorted list of wheel filenames available in the _static directory.
    """
    static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "_static"))
    if not os.path.isdir(static_dir):
        return []
    whls = [os.path.basename(f) for f in glob.glob(os.path.join(static_dir, "*.whl"))]
    sorted_whls = sorted(whls)
    if sorted_whls:
        manifest_path = os.path.join(static_dir, "wheels.json")
        try:
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump({"wheels": sorted_whls}, f, indent=2)
        except OSError:
            pass
    return sorted_whls


class MLPlaygroundDirective(Directive):
    """A Docutils directive for rendering the ML Playground interface.

    This directive generates the raw HTML structure required to display the interactive
    playground, including framework selectors, code editors, and execution consoles.

    Attributes:
    has_content (bool): Indicates whether the directive content block is
        allowed. Defaults to True.
    """

    has_content: bool = True

    def run(self) -> list[nodes.Node]:
        """Execute standard operation.

        Returns:
            list[nodes.Node]: List of docutils nodes representing the raw HTML playground.
        """
        wheel_assets = get_wheel_assets()
        wheels_csv = ",".join(wheel_assets)
        wheel_links_html = "".join(f'\n            <a href="_static/{w}" class="pg-wheel-link" download>{w}</a>' for w in wheel_assets)

        # We output a section with an id that our JS will hydrate
        html = """
<section id="ml-playground-container" aria-label="ML Switcheroo Playground">
    <header class="pg-header">
        <label class="theme-switch" aria-label="Toggle Dark Mode">
            <input type="checkbox" id="theme-toggle">
            <span class="slider round"></span>
        </label>
        <span class="theme-label" data-i18n="darkMode">Dark Mode</span>
        <nav class="pg-wheel-links" aria-label="Wheel Assets" data-wheels="{wheels_attr}"><!-- WHEEL_LINKS_PLACEHOLDER -->
        </nav>
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
        rendered_html = html.replace("{wheels_attr}", wheels_csv).replace("<!-- WHEEL_LINKS_PLACEHOLDER -->", wheel_links_html)
        return [nodes.raw("", rendered_html, format="html")]


def setup(app: "sphinx.application.Sphinx") -> dict[str, Union[str, bool]]:
    """Initializes the Sphinx extension.

    Registers the `ml-playground` directive and associates the required
    CSS and JavaScript assets for the playground's interactive features.

    Args:
        app: The Sphinx application.

    Returns:
        A dictionary containing extension metadata, including the
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
