# ML Switcheroo Compiler Roadmap

This document outlines the future vision, architectural enhancements, and ecosystem expansions planned for the `ml-switcheroo-compiler`.

## Completed Milestones

### Array Execution Backends Expansion
* **Status:** **Completed**
* **Summary:** Successfully implemented several open-source, NumPy-like implementations for specialized hardware acceleration, sparse computation, and complex data-structure support.
* **Impact:** The ecosystem now transparently supports:
  - **Numba (`numba`)**: Near-C CPU execution speeds via JIT-compiling Unified IR into pure Python loops.
  - **Data Parallel NumPy (`dpnp`)**: Hardware-accelerated linear algebra across Intel CPUs, GPUs, and FPGAs using SYCL.
  - **Awkward Array (`awkward`)**: Seamless support for nested, variable-sized (jagged) data structures without requiring padding.
  - **Apache Arrow Compute (`pyarrow.compute`)**: Bridge between Data Engineering and ML, executing graphs directly against columnar data with zero-copy overhead.
  - **Sparse (`sparse`)**: Native support for multidimensional sparse arrays (COO layout), critical for Graph Neural Networks (GNNs).

### Stubs & Incomplete Features Resolution
* **Status:** **Completed**
* **Summary:** Remediated all identified stubs, incomplete features, and fallback placeholders across the compiler.
* **Impact:** Achieved 100% test and docstring coverage, strict typing, upstream grounding, and architectural purity. Mathematical operator stubs and shape inference mechanisms were fully resolved, and foreign graph ingestion (e.g., Torch FX) was significantly expanded.

### ODL Operations Ecosystem Expansion
* **Status:** **Completed**
* **Summary:** Successfully migrated over 4,700 operation definitions from the ODL metadata dataset into the `ml-switcheroo-compiler` ops registry. This involved building translation layers for the unified schema, implementing generators across `cupy`, `dask`, `numpy`, `llvm_cpp`, `edge_onnx`, and `edge_stablehlo`, and drastically expanding the natively supported capabilities of the middle-end execution engine.
* **Impact:** The ecosystem now features comprehensive control flow, higher-order operations (Map, Scan, Cond, WhileLoop), and an exhaustive set of linear algebra and neural network components across all Tier 1 backends.

## Proposed Future Enhancements

*(Currently, all major proposed architectures and backends have been successfully integrated. Future enhancements will be documented here as they are planned.)*
