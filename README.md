# [ml-switcheroo-compiler](https://github.com/SamuelMarks/ml-switcheroo-compiler)

[![License](https://img.shields.io/badge/license-Apache--2.0%20OR%20MIT-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/SamuelMarks/ml-switcheroo-compiler/actions/workflows/ci.yml/badge.svg)](https://github.com/SamuelMarks/ml-switcheroo-compiler/actions)
[![Test Coverage](https://img.shields.io/badge/test_coverage-93.8%25-green.svg)](#)
[![Doc Coverage](https://img.shields.io/badge/doc_coverage-100%25-brightgreen.svg)](#)

The `ml-switcheroo-compiler` is the universal hub and core execution engine for the ML Switcheroo ecosystem. It provides a robust intermediate representation (IR) and compilation pipeline to seamlessly translate machine learning models between major Python frameworks and compile them directly for highly optimized edge execution.

## Architectural Vision

The compiler resolves the impedance mismatch between different machine learning paradigms, operating as a centralized hub with two primary targets:

1. **Source-to-Source (AST-to-AST) Transpilation:** Seamlessly convert ML logic between frameworks like PyTorch, Keras, JAX, and MLX. This includes state lifting/lowering, explicit broadcasting rules, and mapping ecosystem-specific quirks.
2. **Direct-to-Edge Compilation:** Bypass Python deployment entirely by lowering the Unified IR down to highly optimized browser and edge executables powered by **WebGPU** and **WASM SIMD**.

Please refer to [`ARCHITECTURE.md`](ARCHITECTURE.md) for an in-depth dive into the compiler's architecture, including its intermediate representation, execution engine modes, and transformation pipeline.
For the detailed implementation roadmap and task tracking, see the [`ML_SWITCHEROO_COMPILER_BUILD_PLAN.md`](ML_SWITCHEROO_COMPILER_BUILD_PLAN.md).

## Compilation Pipeline

```mermaid
flowchart LR
    subgraph Frontends ["zero-* Frontends (API Shells)"]
        PT[PyTorch API]
        JX[JAX API]
        KR[Keras API]
    end

    subgraph Compiler ["ml-switcheroo-compiler"]
        TR[Tracer & AD Engine] --> IR[Unified IR]
        IR --> PM[Middle-End Optimizations]
    end

    subgraph Backends ["Emitters (Execution Targets)"]
        PY[Python Source]
        WG[WebGPU Shaders]
        WA[WASM SIMD]
    end

    Frontends -->|Proxy Tensors| Compiler
    PM -->|Optimized Graph| Backends
```

- **Unified IR:** A strict, framework-agnostic intermediate representation defining shape semantics, mathematical primitives, control flow, and state management.
- **Middle-End Optimization:** Executes high-level passes (state transformation, type promotion) and low-level passes (buffer allocation, kernel fusion, loop tiling) before code generation.
- **Python Emission:** Emits idiomatic source code for the target Python framework (e.g., PyTorch `nn.Module`s, JAX Pytrees, Keras subclassed models, MLX classes).
- **Web & Edge Emission:** Translates computation graphs into WGSL shaders for WebGPU parallel compute and C++/Rust for WASM SIMD (v128) CPU compute.

## Core Execution Modes

To provide a standard developer experience, the engine supports two distinct execution paradigms:

- **Eager Mode (Debug / Interactive):** Immediate-execution path where mathematical operations are evaluated eagerly, backed by NumPy or pure Python for accurate, host-level execution without compilation overhead.
- **Graph Mode (Compiled):** A tracing and parsing execution path that constructs the Unified IR. The resulting computation graph is then routed through the optimization middle-end to the selected deployment backend.

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