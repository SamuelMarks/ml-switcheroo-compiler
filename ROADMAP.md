# ML Switcheroo Compiler Roadmap

This document outlines the future vision, architectural enhancements, and ecosystem expansions planned for the `ml-switcheroo-compiler`.

## Proposed Array Execution Backends

To further our goal of being a universally flexible execution engine, the following open-source, NumPy-like implementations should be strongly considered for future backend development. Implementing these would allow the `zero-*` ecosystem to transparently inherit specialized hardware acceleration, sparse computation, and complex data-structure support.

### 1. [Numba](https://numba.pydata.org/) (`numba`)
* **Why Consider:** While Numba is primarily a JIT compiler rather than a standalone array container library, it possesses deep, native support for NumPy arrays via its `@njit` and `@vectorize` decorators.
* **Architectural Fit:** A `NumbaGenerator` could translate our Unified IR into pure Python loops wrapped in Numba decorators, JIT-compiling them into optimized LLVM IR / machine code. This would provide near-C execution speeds on CPUs for workloads where standard NumPy overhead dominates (e.g., highly iterative dynamic control flow).

### 2. [Data Parallel NumPy](https://github.com/IntelPython/dpnp) (`dpnp`)
* **Why Consider:** Developed by Intel as part of the oneAPI specification, DPnp is a drop-in replacement for the NumPy API specifically engineered to accelerate execution across Intel CPUs, GPUs, and FPGAs using SYCL.
* **Architectural Fit:** For users locked into the Intel hardware ecosystem (such as supercomputing clusters without NVIDIA GPUs), `dpnp` provides an immediate path to hardware-accelerated linear algebra. The implementation would mirror the current `numpy` backend, requiring minimal abstraction overhead while unlocking immense performance gains on specialized silicon.

### 3. [Awkward Array](https://awkward-array.org/) (`awkward`)
* **Why Consider:** Awkward Array is a library designed for nested, variable-sized (jagged) data structures (like arrays of lists or JSON-like data). It provides a NumPy-like idiom for operating on non-rectangular arrays.
* **Architectural Fit:** Standard tensor compilers struggle significantly with ragged tensors. By abstracting Awkward Array as a backend, `ml-switcheroo-compiler` could seamlessly support High Energy Physics (HEP) datasets or deeply nested NLP token sequences without requiring users to pad their inputs to uniform dimensions, drastically saving memory.

### 4. [Apache Arrow Compute](https://arrow.apache.org/) (`pyarrow.compute`)
* **Why Consider:** Apache Arrow is the industry standard for in-memory columnar data. Its `pyarrow.compute` module provides a suite of vectorized array operations.
* **Architectural Fit:** Adding an Arrow backend would bridge the gap between Data Engineering and Machine Learning. If `zero-*` users are reading massive Parquet files or fetching from column-oriented databases, an Arrow backend would allow the compiler to execute ML graphs directly against the columnar data with zero-copy overhead, entirely avoiding the costly serialization step to NumPy or PyTorch formats.

### 5. [Sparse](https://sparse.pydata.org/) (`sparse`)
* **Why Consider:** The `sparse` library implements multidimensional sparse arrays conforming to the `numpy.ndarray` interface, primarily using the Coordinate List (COO) layout.
* **Architectural Fit:** Major ML frameworks often treat sparse operations as a second-class citizen. Implementing a dedicated `SparseGenerator` backend would allow the compiler to natively route specific ops (like sparse-dense matrix multiplication) to an engine explicitly built for them. This is critical for Graph Neural Networks (GNNs) or models with massive, highly sparse attention masks.

### 6. [Bohrium](https://github.com/bh107/bohrium) (`bohrium`)
* **Why Consider:** Bohrium provides a drop-in replacement for NumPy but uses a lazy evaluation strategy to construct a computation graph before executing it on multi-core CPUs or GPU clusters via a runtime written in C/OpenCL.
* **Architectural Fit:** Like Dask, Bohrium handles distributed computation, but it aims for a more seamless, single-machine multi-core experience. Implementing Bohrium would give users an alternative scaling strategy that doesn't require the explicit chunking and cluster management overhead required by Dask, making scaling out simpler for intermediate users.
