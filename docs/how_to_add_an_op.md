# How to Add an Op

To add a new operation to the `ml-switcheroo-compiler`, you need to:

1. **Create the Op Definition**:
   Add a new class inheriting from `OpDef` in `src/ml_switcheroo_compiler/ops/`. Make sure you place it in the appropriate subdirectory (e.g., `binary/`, `unary/`, `linalg/`).

2. **Implement Required Methods**:
   You must implement `infer_shape`, `infer_dtype`, and `numpy_eval`. If it supports gradients, implement `vjp` and `jvp`.

3. **Register the Op**:
   Ensure the Op is discoverable by `get_op` by adding it to the respective `__init__.py` file in `src/ml_switcheroo_compiler/ops/`.

4. **Add Tests**:
   Create a test in `tests/ops/`. Ensure 100% test coverage for your new operation.
