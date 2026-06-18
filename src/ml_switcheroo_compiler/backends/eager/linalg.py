"""Linalg utilities."""

from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry


@global_eager_registry.register("PowerIteration")
def _power_iteration(
    backend_module: object, w: object, num_iters: int = 1, u: object = None
) -> tuple[object, object, object]:
    linalg = getattr(backend_module, "linalg", None)
    if linalg is None:
        raise NotImplementedError("Backend module missing linalg submodule.")

    # We assume w is an array. We need to implement eager logic using backend_module.
    # backend_module is typically np, jnp, mlx.core, etc.
    shape = w.shape
    dtype = w.dtype

    if u is None:
        u = backend_module.ones(shape[:-2] + (shape[-2], 1), dtype=dtype)

    for _ in range(num_iters):
        w_t = backend_module.swapaxes(w, -1, -2)
        v = backend_module.matmul(w_t, u)
        v_norm = linalg.norm(v, axis=-2, keepdims=True) + 1e-12
        v = v / v_norm

        u = backend_module.matmul(w, v)
        u_norm = linalg.norm(u, axis=-2, keepdims=True) + 1e-12
        u = u / u_norm

    sigma = backend_module.matmul(backend_module.swapaxes(u, -1, -2), backend_module.matmul(w, v))
    return (
        backend_module.squeeze(v, -1),
        backend_module.squeeze(u, -1),
        backend_module.squeeze(backend_module.squeeze(sigma, -1), -1),
    )
