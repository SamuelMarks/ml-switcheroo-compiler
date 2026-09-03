# zero-* Engine Core

> **Note:** This repository serves as the core execution engine and abstract representation framework for the `zero-*` ecosystem.

# [ml-switcheroo-compiler](https://github.com/SamuelMarks/ml-switcheroo-compiler)

[![License](https://img.shields.io/badge/license-Apache--2.0%20OR%20MIT-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/SamuelMarks/ml-switcheroo-compiler/actions/workflows/ci.yml/badge.svg)](https://github.com/SamuelMarks/ml-switcheroo-compiler/actions)
[![Test Coverage](https://img.shields.io/badge/test_coverage-100.0%25-green.svg)](#)
[![Doc Coverage](https://img.shields.io/badge/doc_coverage-100%25-brightgreen.svg)](#)

The `ml-switcheroo-compiler` is the universal hub and core execution engine for the ML Switcheroo ecosystem. It provides a robust intermediate representation (IR) and compilation pipeline to seamlessly translate machine learning models between major Python frameworks and compile them directly for highly optimized edge execution. Crucially, this architecture empowers developers to run precise forward and backward passes directly in the browser for exact shape learning, and to empirically benchmark any ML syntax across different execution backends.

## Major Project Goals

1. **In-Browser Shape Learning & Transpilation:** Enhance the original [ML Switcheroo](https://samuelmarks.github.io/ml-switcheroo) project (focused on ML framework and SASS/RDNA transpilation) with the ability to learn precise tensor shapes by executing actual forward and backward passes directly in the browser.
2. **Cross-Backend Benchmarking:** Benchmark different execution backends independently of the frontend API (e.g., testing TensorFlow syntax running on the MLX backend). This allows us to empirically test hardware-specific performance claims—such as whether PyTorch is truly better for GPUs, JAX for TPUs, or MLX for Apple Silicon.

## Architectural Vision

The compiler resolves the impedance mismatch between different machine learning paradigms, operating as a strictly decoupled, purely functional computational hub (Tier 2) with two primary targets:

1. **Source-to-Source (AST-to-AST) Transpilation:** Seamlessly convert ML logic between frameworks like PyTorch, Keras, JAX, and MLX. This includes state lifting/lowering, explicit broadcasting rules, and mapping ecosystem-specific quirks using our universal Intermediate Representation (IR). This decoupling is the foundation for our **cross-backend benchmarking**—allowing you to write TensorFlow syntax but execute it on MLX to test hardware claims.
2. **Direct-to-Edge Compilation:** Bypass Python deployment entirely by lowering the Unified IR down to highly optimized browser and edge executables powered by **WebGPU** and **WASM SIMD**. This target drives our **in-browser shape learning** goals, allowing the engine to execute precise forward and backward passes live in the client without a backend server.

**Strict Decoupling Rule ("No Math in Frontends"):** The `ml-switcheroo-compiler` repository is exclusively responsible for all math, Automatic Differentiation (AD), and transformations. Frontend repositories (like `zero-pytorch` or `zero-jax`) contain NO math implementations; they are purely Tier 3/4 lightweight API shells that route inputs and lift object-oriented state into this compiler. Likewise, this compiler strictly forbids any framework-specific API mimicry.

Please refer to [`ARCHITECTURE.md`](ARCHITECTURE.md) for an in-depth dive into the compiler's architecture, including its intermediate representation, execution engine modes, and transformation pipeline.

## Compilation Pipeline

```mermaid
flowchart TD
    subgraph Frontends ["zero-* Frontends (API Shells)"]
        direction LR
        PT[PyTorch API]
        JX[JAX API]
        KR[Keras API]
        MLX_F[MLX API]
    end

    subgraph Compiler ["ml-switcheroo-compiler"]
        TR[Tracer & AD Engine]
        IR[Unified IR]
        PM[Middle-End Optimizations]

        TR -->|Captures Graph| IR
        IR -->|Target-Agnostic Passes| PM
    end

    subgraph Backends ["Emitters (Execution Targets)"]
        subgraph S2S ["AST-to-AST Transpilation"]
            PY[Python Frameworks]
        end
        subgraph Edge ["Direct-to-Edge Compilation"]
            WG[WebGPU / WGSL]
            WA[WASM SIMD]
        end
    end

    PT & JX & KR & MLX_F -->|Proxy Tensors| TR
    PM -->|Optimized IR| PY
    PM -->|Optimized IR| WG
    PM -->|Optimized IR| WA
```

- **Unified IR:** A strict, framework-agnostic intermediate representation defining precise shape semantics (learned via live forward/backward passes), mathematical primitives, control flow, and state management.
- **Middle-End Optimization:** Executes high-level passes (state transformation, type promotion) and low-level passes (buffer allocation, kernel fusion, loop tiling) before code generation.
- **Python Emission:** Emits idiomatic source code for the target Python framework (e.g., PyTorch `nn.Module`s, JAX Pytrees, Keras subclassed models, MLX classes).
- **Web & Edge Emission:** Translates computation graphs into WGSL shaders for WebGPU parallel compute and C++/Rust for WASM SIMD (v128) CPU compute.


## Advanced Transformations and Distributed Support

The engine supports a comprehensive suite of advanced optimizations and parity features across all backends:
- **Compiler Optimizations:** Built-in Dead Code Elimination (DCE), Common Subexpression Elimination (CSE), Constant Folding, Operator Fusion, Loop Unrolling, Memory Planning, and Scheduling logic via the `PassManager`.
- **Automatic Differentiation:** Full support for `jvp` (Forward-Mode), `vjp` / `grad` (Reverse-Mode), and `hvp` (Higher-Order Derivatives), accompanied by memory-efficient checkpointing/rematerialization and custom gradient hooks.
- **Hardware Targets:** Support for LLVM/C++ fallbacks, WebAssembly (WASM), WebGPU WGSL, ONNX, and StableHLO native exports.
- **Distributed Parity:** Collectives (`AllReduce`, `AllGather`, `AllToAll`, `ReduceScatter`), SPMD annotations, and pipeline parallelism primitives.

## Core Execution Modes

To provide a standard developer experience, the engine supports two distinct execution paradigms:

- **Eager Mode (Debug / Interactive):** Immediate-execution path where mathematical operations are evaluated eagerly, backed by NumPy or pure Python for accurate, host-level execution without compilation overhead. This is ideal for extracting exact shapes and executing live in-browser shape inference.
- **Graph Mode (Compiled):** A tracing and parsing execution path that constructs the Unified IR. The resulting computation graph is then routed through the optimization middle-end to the selected deployment backend, enabling apples-to-apples cross-backend benchmarking.

## Ecosystem Dependency Graph

```mermaid
graph TD
    subgraph "Verification Tier (zero-zoo)"
        ZZ[zero-zoo / The Model Zoo]
    end

    subgraph "API-Compatible Shells"
        ZJ[zero-jax]
        ZF[zero-flax]
        ZP[zero-pytorch]
        ZK[zero-keras]
        ZT[zero-tensorflow]
        ZM[zero-mlx]
        ZPX[zero-pax]
    end

    subgraph "Compilation Core"
        COMP[ml-switcheroo-compiler]
    end
    style COMP fill:#ff9900,color:#fff,stroke:#333,stroke-width:4px

    subgraph "Internal Backends"
        NUMPY[numpy]
        JAX_B[jax]
        MLX_B[mlx]
        CUPY[cupy]
        DASK[dask]
        TORCH_B[torch]
        KERAS_B[keras]
        TF_B[tensorflow]
        EDGE_WGPU[webgpu/wgsl]
        EDGE_WASM[wasm]
        EDGE_WEBGL[webgl]
        LLVM[llvm_cpp]
        ONNX[onnx]
        STABLEHLO[stablehlo]
    end

    ZZ -.->|Validates Float Equivalence| ZJ
    ZZ -.->|Validates Float Equivalence| ZF
    ZZ -.->|Validates Float Equivalence| ZP
    ZZ -.->|Validates Float Equivalence| ZK
    ZZ -.->|Validates Float Equivalence| ZT
    ZZ -.->|Validates Float Equivalence| ZM
    ZZ -.->|Validates Float Equivalence| ZPX

    ZJ --> COMP
    ZF --> ZJ
    ZP --> COMP
    ZK --> COMP
    ZT --> ZK
    ZM --> COMP
    ZPX --> ZJ

    COMP --> NUMPY
    COMP --> JAX_B
    COMP --> MLX_B
    COMP --> CUPY
    COMP --> DASK
    COMP --> TORCH_B
    COMP --> KERAS_B
    COMP --> TF_B
    COMP --> EDGE_WGPU
    COMP --> EDGE_WASM
    COMP --> EDGE_WEBGL
    COMP --> LLVM
    COMP --> ONNX
    COMP --> STABLEHLO
```

## Internal Backends

The `ml-switcheroo-compiler` serves as the unifying engine for the `zero-*` ecosystem. While the frontends provide the API interfaces, the actual execution is delegated to one of several internal execution backends. You can specifically choose between the following backends depending on your platform and performance requirements. This decoupled backend architecture is the very foundation for our benchmarking goals—allowing you to run the exact same PyTorch or Keras code across MLX, JAX, and CuPy to verify hardware-specific performance claims:

- **`numpy`**: Reference eager execution CPU backend.
- **`jax`**: High-performance compiler and array library backend.
- **`mlx`**: Apple Silicon optimized array framework backend.
- **`cupy`**: GPU-accelerated array computing backend.
- **`dask`**: Specialized distributed or alternative backend.
- **`torch`**: Native PyTorch execution backend.
- **`keras`**: Keras execution backend.
- **`tensorflow`**: TensorFlow execution backend.
- **`edge (webgpu/wgsl)`**: In-browser parallel GPU compute backend.
- **`edge (wasm)`**: In-browser and edge CPU compute backend (WASM SIMD).
- **`edge (webgl)`**: Legacy in-browser GPU backend.
- **`edge (onnx / stablehlo)`**: Export targets for standardized ML exchange.
- **`llvm_cpp`**: C++ fallback and LLVM execution backend.

---

## Related Projects

| Name | Description | CI Shields |
|---|---|---|
| [`ml-framework-snapshots`](https://github.com/SamuelMarks/ml-framework-snapshots) | Static API extraction and schema formalization for major ML frameworks. | [![CI](https://github.com/SamuelMarks/ml-framework-snapshots/actions/workflows/ci.yml/badge.svg)](https://github.com/SamuelMarks/ml-framework-snapshots/actions/workflows/ci.yml) |
| [`ml-switcheroo-ir`](https://github.com/SamuelMarks/ml-switcheroo-ir) | The core dependency-free IR for the ml-switcheroo model translation ecosystem. | [![CI](https://github.com/SamuelMarks/ml-switcheroo-ir/actions/workflows/ci.yml/badge.svg)](https://github.com/SamuelMarks/ml-switcheroo-ir/actions/workflows/ci.yml) |
| [`zero-chex`](https://github.com/SamuelMarks/zero-chex) | Chex is a library of utilities for helping to write reliable JAX code. | [![CI](https://github.com/SamuelMarks/zero-chex/actions/workflows/ci.yml/badge.svg)](https://github.com/SamuelMarks/zero-chex/actions/workflows/ci.yml) |
| [`zero-flax`](https://github.com/SamuelMarks/zero-flax) | Flax is a neural network library for JAX that is designed for flexibility. | [![CI](https://github.com/SamuelMarks/zero-flax/actions/workflows/ci.yml/badge.svg)](https://github.com/SamuelMarks/zero-flax/actions/workflows/ci.yml) |
| [`zero-grain`](https://github.com/SamuelMarks/zero-grain) | Library for reading and processing ML training data. | [![CI](https://github.com/SamuelMarks/zero-grain/actions/workflows/ci.yml/badge.svg)](https://github.com/SamuelMarks/zero-grain/actions/workflows/ci.yml) |
| [`zero-jax`](https://github.com/SamuelMarks/zero-jax) | Composable transformations of Python+NumPy programs: differentiate, vectorize, JIT to GPU/TPU, and more | [![CI](https://github.com/SamuelMarks/zero-jax/actions/workflows/ci.yml/badge.svg)](https://github.com/SamuelMarks/zero-jax/actions/workflows/ci.yml) |
| [`zero-keras`](https://github.com/SamuelMarks/zero-keras) | Deep Learning for humans | [![CI](https://github.com/SamuelMarks/zero-keras/actions/workflows/ci.yml/badge.svg)](https://github.com/SamuelMarks/zero-keras/actions/workflows/ci.yml) |
| [`zero-mlx`](https://github.com/SamuelMarks/zero-mlx) | MLX: An array framework for Apple silicon | [![CI](https://github.com/SamuelMarks/zero-mlx/actions/workflows/ci.yml/badge.svg)](https://github.com/SamuelMarks/zero-mlx/actions/workflows/ci.yml) |
| [`zero-optax`](https://github.com/SamuelMarks/zero-optax) | Optax is a gradient processing and optimization library for JAX. | [![CI](https://github.com/SamuelMarks/zero-optax/actions/workflows/ci.yml/badge.svg)](https://github.com/SamuelMarks/zero-optax/actions/workflows/ci.yml) |
| [`zero-orbax`](https://github.com/SamuelMarks/zero-orbax) | Orbax provides common checkpointing and persistence utilities for JAX users | [![CI](https://github.com/SamuelMarks/zero-orbax/actions/workflows/ci.yml/badge.svg)](https://github.com/SamuelMarks/zero-orbax/actions/workflows/ci.yml) |
| [`zero-pax`](https://github.com/SamuelMarks/zero-pax) | Pax is a Jax-based machine learning framework for training large scale models. Pax allows for advanced and fully configurable experimentation and parallelization, and has demonstrated industry leading model flop utilization rates. | [![CI](https://github.com/SamuelMarks/zero-pax/actions/workflows/ci.yml/badge.svg)](https://github.com/SamuelMarks/zero-pax/actions/workflows/ci.yml) |
| [`zero-pytorch`](https://github.com/SamuelMarks/zero-pytorch) | Tensors and Dynamic neural networks in Python with strong GPU acceleration | [![CI](https://github.com/SamuelMarks/zero-pytorch/actions/workflows/ci.yml/badge.svg)](https://github.com/SamuelMarks/zero-pytorch/actions/workflows/ci.yml) |
| [`zero-tensorflow`](https://github.com/SamuelMarks/zero-tensorflow) | An Open Source Machine Learning Framework for Everyone | [![CI](https://github.com/SamuelMarks/zero-tensorflow/actions/workflows/ci.yml/badge.svg)](https://github.com/SamuelMarks/zero-tensorflow/actions/workflows/ci.yml) |

---

## License

Licensed under either of

- Apache License, Version 2.0 (LICENSE-APACHE or https://www.apache.org/licenses/LICENSE-2.0)
- MIT license (LICENSE-MIT or https://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
