# Ecosystem

This section covers the base ML framework ecosystem packages. The ecosystem relies on API-compatible implementations (frontends) that utilize a shared compiler backend (`ml-switcheroo-compiler`).

## API Compatible Frontends

The following frontend repositories provide API-compatible shells that mimic major ML frameworks but delegate all execution to the compiler backend:

- `zero-flax`
- `zero-jax`
- `zero-keras`
- `zero-mlx`
- `zero-pax`
- `zero-pytorch`
- `zero-tensorflow`

## Compiler Engine & Internal Backends

All frontends map their operations to the `ml-switcheroo-compiler`. This core engine has multiple internal backends, allowing you to explicitly choose your underlying execution environment:

- `numpy`
- `jax`
- `mlx`
- `cupy`
- `dusk`
- `torch`
- `keras`
