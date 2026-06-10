# Stubs and Incomplete Features

| Status | File / Component | Incomplete Feature / Stub | Details |
| :---: | :--- | :--- | :--- |
| [x] | **`src/ml_switcheroo/backends/cst_transpiler.py`** | `leave_ImportFrom`, `leave_Call`, `validate_diff`, `type_infer_dry_run` | These methods contain dummy implementations or bare `pass` stubs. |
| [x] | **`src/ml_switcheroo/backends/jax.py`** | `JAXCodeGenerator.generate` | Missing the core logic; contains an empty loop meant to emit `jax.numpy.*` and `jax.lax.*` equivalents. |
| [x] | **`src/ml_switcheroo/backends/keras.py`** | `KerasCodeGenerator.generate` | Missing the core logic; contains an empty loop meant to map IR nodes to `keras.layers.*`. |
| [x] | **`src/ml_switcheroo/backends/mlx.py`** | `MLXCodeGenerator.generate` | Missing the core logic; contains an empty loop meant to emit operations inside the generated `__call__` method. |
| [x] | **`src/ml_switcheroo/backends/python_generator.py`** | `PythonCodeGenerator.generate` | Missing input handling (`# Handle inputs through args`) and complete operation mapping (`# Dummy code folding/op mapping`). |
| [x] | **`src/ml_switcheroo/backends/pytorch.py`** | `PyTorchCodeGenerator.generate` | Missing parameter registration in `__init__` and topologically sorted `torch.*` ops mapping in `forward`. |
| [x] | **`src/ml_switcheroo/backends/tensorflow.py`** | `TensorFlowCodeGenerator.generate` | Missing the core logic; contains an empty loop meant to map math to `tf.raw_ops.*`. |
| [x] | **`src/ml_switcheroo/diagnostics.py`** | Multiple diagnostic tools | `NumericalAnomalyDetector.check` is a `pass` stub. `debug_shapes`, `estimate_flops`, and `memory_profiler` return dummy string/integer implementations. |
| [x] | **`src/ml_switcheroo/interpreter.py`** | General Operator Support | The fallback loop `raise NotImplementedError(...)` is triggered for any operation that isn't `Transpose`, `Relu`, `Greater`, `Where`, or `Expand`. |
| [x] | **`src/ml_switcheroo/optimization.py`** | `constant_folding` | Explicitly marked in the docstring as `"Currently a simplified placeholder for pure-Python fallback evaluation"`. |
| [x] | **`src/ml_switcheroo/plugins_legacy.py`** | Legacy Reimplementation Framework | The entire file consists of **37 empty stub classes** (e.g., `AttentionPacking`, `BatchNormPlugin`, `StateContainer`, `AutoFSDPWrapper`, etc.) waiting for logic. |
