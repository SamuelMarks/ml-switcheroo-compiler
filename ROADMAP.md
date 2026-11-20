# ML Switcheroo Compiler Roadmap

This document outlines the architectural milestones completed and future enhancements planned for `ml-switcheroo-compiler`.

## Completed Milestones

### Array Execution Backends Expansion
* **Status:** **Completed**
* **Summary:** Implemented specialized open-source array and hardware backends for JIT compilation, Intel SYCL acceleration, ragged/nested structures, columnar zero-copy execution, and sparse algebra.
* **Impact:** Transparently supports:
  - **Numba (`numba`)**: Near-C CPU/GPU execution speeds via JIT-compiled `@njit` kernels lowered from Unified IR.
  - **Data Parallel NumPy (`dpnp`)**: Hardware-accelerated linear algebra across Intel CPUs, GPUs, and FPGAs using SYCL.
  - **Awkward Array (`awkward`)**: Native support for nested, variable-sized (jagged) data structures without padding.
  - **Apache Arrow Compute (`pyarrow_compute`)**: Columnar data-to-ML execution directly on Arrow buffers.
  - **Sparse (`sparse` / `sparse_coo`)**: Multidimensional coordinate (COO) sparse arrays for Graph Neural Networks (GNNs) and scientific workloads.

### Stubs & Incomplete Features Resolution
* **Status:** **Completed**
* **Summary:** Remediated all operator stubs, incomplete transformations, and fallback placeholders across the compiler.
* **Impact:** Achieved 100% line, branch, and docstring coverage, strict `mypy --strict` typing without `Any` or `cast`, snapshot-grounded backend parameter validation, and foreign graph ingestion (`ingest_torch_fx`, `ingest_jaxpr`, `ForeignCall`).

### ODL Operations Ecosystem Expansion
* **Status:** **Completed**
* **Summary:** Migrated over 4,700 operation definitions from the ODL metadata catalog into the `ml-switcheroo-compiler` ops registry with unified schema translation and code generators across dense, domain, hardware, and edge targets (`cupy`, `dask`, `numpy`, `llvm_cpp`, `edge_onnx`, `edge_stablehlo`, etc.).
* **Impact:** Comprehensive higher-order control flow (`cond`, `while_loop`, `scan`, `vmap`, `pmap`), linear algebra, signal/vision/audio primitives, and neural network operations across registered backends.

### Interactive Web Playground & Native Browser Execution
* **Status:** **Completed**
* **Summary:** Integrated an accessible (WCAG 2.1 AA), vanilla HTML/CSS/JS interactive playground (`.. ml-playground::`) into the Sphinx documentation site (`docs/`) backed by headless Node/browser verification (`tests_js/`).
* **Impact:** Enables client-side compilation via Pyodide alongside live in-browser execution on **WebGPU (`edge_wgsl` / `webgpu`)**, **WASM SIMD (`edge_wasm_simd`)**, and peer-to-peer **WebRTC** collective meshes.

### Tier-1 `ml-switcheroo-ir` Un-Vendoring & Zero-NumPy-Leak Isolation
* **Status:** **Completed**
* **Summary:** Removed vendored IR stubs in favor of direct Tier-1 `ml-switcheroo-ir` imports without runtime `model_config` mutation, and eliminated all global `numpy`/`scipy` dependencies outside `backends/numpy/`.
* **Impact:** Enforced strict single-backend third-party isolation (`scripts/lint_dependencies.py`, `scripts/lint_numpy_leak.py`, `scripts/lint_dynamic_numpy.py`) and guaranteed cross-repo `LogicalGraph`/`LogicalNode` class identity with zero framework lock-in.

### Haskell-Style Backend Typeclasses, Unified Dispatch & Connected `jit` / `eval_shape`
* **Status:** **Completed**
* **Summary:** Introduced generic capability protocols and dictionary-passing dispatch (`CoreEagerTypeclass`, `CapabilityKind.CORE_EAGER` vs `DOMAIN_EAGER`, `CodeEmitterTypeclass`, `AOTCompilerTypeclass`) verified statically via `scripts/verify_typeclasses.py`; unified eager/tracing op routing in `ops/dispatcher.py`; hardened `Config`, `TracingState`, and `callback_registry` with `ContextVar` isolation and `O(1)` weak live-tensor registries; and wired end-to-end `jit`, `disable_jit`, and zero-allocation symbolic `eval_shape`.
* **Impact:** Eliminated untyped reflection and `gc.get_objects()` heap scans in `backward()`, guaranteed dense eager prelude parity across all 8 `CORE_EAGER` backends (`numpy`, `pure_python`, `pytorch`, `jax`, `tensorflow`, `keras`, `mlx`, `dask`), and enabled composable PyTree `jit`/`grad` execution.

## Proposed Future Enhancements

*(All currently planned core compiler architectures, capability verifiers, and execution backends have been integrated. Future milestones will be tracked here as new ecosystem targets are proposed.)*
