# How to Add a Backend

The compiler uses a pluggable registry for backends. To add a new backend (e.g., `my_backend`):

1. **Create the Backend Module**:
   Create a new file `src/ml_switcheroo_compiler/backends/my_backend.py`.

2. **Implement Code Generator**:
   Create a class that inherits from `BaseGenerator` and use the `@register_backend` decorator.

   ```python
   from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
   from ml_switcheroo_compiler.backends.registry import register_backend

   @register_backend("my_backend")
   class MyBackendCodeGenerator(BaseGenerator):
       def _dispatch_op_template(self, op_instance, *args, **kwargs) -> str:
           return op_instance.emit_my_backend(*args, **kwargs)
   ```

3. **Expose the Backend**:
   Import your backend in `src/ml_switcheroo_compiler/backends/__init__.py`.

4. **Update Op Emit Methods**:
   Update existing `OpDef` classes in `src/ml_switcheroo_compiler/ops/` to support `emit_my_backend` or handle fallback properly.
