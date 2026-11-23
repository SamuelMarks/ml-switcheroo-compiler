# Ecosystem

This section covers the base ML framework ecosystem packages. The ecosystem relies on API-compatible implementations (frontends) that utilize a shared compiler backend (`ml-switcheroo-compiler`).

## API Compatible Frontends

The following frontend repositories provide API-compatible shells that mimic major ML frameworks and utilities while delegating all execution to the compiler backend:

- `zero-chex`
- `zero-flax`
- `zero-grain`
- `zero-jax`
- `zero-keras`
- `zero-mlx`
- `zero-optax`
- `zero-orbax`
- `zero-pax`
- `zero-pytorch`
- `zero-tensorflow`

## Compiler Engine & Internal Backends

All frontends map their operations to `ml-switcheroo-compiler`. This core engine provides isolated internal Python and array backends, allowing you to explicitly choose your underlying execution environment:

- `numpy`
- `pure_python`
- `pytorch` (`torch`)
- `jax`
- `mlx`
- `keras`
- `tensorflow`
- `dask`
- `cupy`
- `numba`
- `sparse` (`sparse_coo`)
- `awkward`
- `dpnp`
- `pyarrow_compute`

## Hardware & Edge Code Generation Targets

The compiler pipeline provides native code emission and wire-format serializers translating intermediate representation logic for specific execution endpoints:

- **CPU / LLVM (`llvm_cpp`)**: Emit vectorized C++17 native blocks
- **CUDA / PTX (`cuda`)**: Target NVIDIA GPUs directly
- **ROCm (`rocm`)**: AMD HIP GPU code emission
- **Metal (`metal`)**: Apple Silicon Metal Shading Language definitions
- **WebGPU / WGSL (`webgpu` / `edge_wgsl`)**: Export for native in-browser GPU execution
- **WebGL 2.0 (`edge_webgl`)**: Fallback browser fragment-shader compute target
- **WASM SIMD (`edge_wasm` / `edge_wasm_simd`)**: WebAssembly C++ SIMD (`v128`) targets
- **ONNX & StableHLO (`edge_onnx` / `edge_stablehlo`)**: Standardized graph and MLIR export payloads
