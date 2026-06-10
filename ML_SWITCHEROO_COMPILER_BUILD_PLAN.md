# ML Switcheroo Compiler: Exhaustive Build & Implementation Plan

This document serves as the canonical, exhaustive blueprint and task backlog for the `ml-switcheroo-compiler`. It strictly dictates the architecture, components, and milestones required for full implementation. 

**Crucial Architecture Note regarding `zero-*` repositories:**
The existing `zero-*` codebases (e.g., `zero-pytorch`, `zero-jax`, `zero-keras`) will **NOT** be deleted or fully absorbed into this compiler. Instead, they will be **retained as independent, lightweight frontend API shells**. 
Every `zero-*` repository will add `ml-switcheroo-compiler` as its core backend dependency. All mathematical implementations, array allocations, and computation graph logic currently residing in the `zero-*` repos will be ripped out and replaced with delegations to the compiler. The `zero-*` repos will purely handle framework-specific API routing, argument parsing (handling kwargs like `dim` vs `axis`), and syntactic sugar.

---

## 1. Architectural Vision & API Boundaries

### Core Architecture & Dependency Flow
- [x] Define the overarching Architecture diagram: `zero-*` Frontend -> `switcheroo` Eager/Tracer -> `switcheroo` IR -> `switcheroo` PassManager -> `switcheroo` Backend Emitters.
- [x] Establish strict dependency separation: `zero-*` repos depend on `ml-switcheroo-compiler`; the compiler **must never** depend on any `zero-*` repo.
- [x] Specify exact directory structure in the compiler repo: `src/ml_switcheroo/core`, `src/ml_switcheroo/ops`, `src/ml_switcheroo/ir`, `src/ml_switcheroo/transforms`, `src/ml_switcheroo/backends`.

### The Universal Tensor Interface
- [x] Define the `switcheroo.Tensor` base class (the unified backend array).
  - *Mapping:* `zero_torch.tensor.Tensor` will hold a `switcheroo.Tensor` as its `.data` payload. `zero_jax.numpy.ndarray` will subclass or wrap `switcheroo.Tensor`.
- [x] Implement `switcheroo.Tensor.shape` property.
- [x] Implement `switcheroo.Tensor.dtype` property.
- [x] Implement `switcheroo.Tensor.device` property.
- [x] Implement `switcheroo.Tensor.requires_grad` property.
- [x] Implement cross-framework device management API (CPU, GPU, WebGPU).

### Configuration & State Management
- [x] Create `switcheroo.config` singleton.
- [x] Implement `switcheroo.config.eager_mode` (boolean toggle).
- [x] Implement `switcheroo.config.default_float_dtype` (Float32, Float16).
- [x] Implement `switcheroo.config.default_int_dtype` (Int64, Int32).
- [x] Implement `switcheroo.config.default_device`.
- [x] Implement scoped context manager: `switcheroo.ConfigContext`.
- [x] Map configuration to environment variables (e.g., `SWITCHEROO_EAGER_MODE=1`).
  - *Mapping:* The `zero-*` frameworks will query this config to determine if they should execute eagerly or trace an IR graph.

### Error Handling Hierarchy
- [x] Define `SwitcherooError` base class.
- [x] Define `TracingError` (raised when control flow breaks proxy tensor tracing).
- [x] Define `CompilationError` (raised during IR generation or pass failure).
- [x] Define `ShapeMismatchError` (raised during static shape inference).
- [x] Define `DTypePromotionError` (raised on invalid automatic type casting).
- [x] Define `BackendNotSupportedError` (raised when an edge target lacks an op).
- [x] Define `UnimplementedMathError` (raised if NumPy/SciPy fallback is missing in Eager Mode).

---

## 2. Exhaustive Universal Math & Primitives Library
The core ops library (`switcheroo.ops`) must subsume all math logic. The `zero-*` repositories will implement their native APIs (e.g., `zero_torch.sin`) by simply returning `switcheroo.ops.sin(input)`.

### Constants & Creation Ops
- [ ] `switcheroo.ops.zeros`
- [ ] `switcheroo.ops.ones`
- [ ] `switcheroo.ops.full`
- [ ] `switcheroo.ops.zeros_like`
- [ ] `switcheroo.ops.ones_like`
- [ ] `switcheroo.ops.full_like`
- [ ] `switcheroo.ops.arange`
- [ ] `switcheroo.ops.linspace`
- [ ] `switcheroo.ops.eye`
- [ ] `switcheroo.ops.identity`
- [ ] `switcheroo.ops.diag`
- [ ] `switcheroo.ops.empty` (uninitialized memory)

### Random Operations (Functional & Stateful)
- [ ] `switcheroo.random.PRNGKey` (pure functional state management, mapping to JAX).
- [ ] `switcheroo.random.split`
- [ ] `switcheroo.random.fold_in`
- [ ] `switcheroo.random.uniform`
- [ ] `switcheroo.random.normal`
- [ ] `switcheroo.random.bernoulli`
- [ ] `switcheroo.random.truncated_normal`
- [ ] `switcheroo.random.randint`
- [ ] `switcheroo.random.seed` (global stateful fallback for PyTorch/Keras mapping).

### Unary Operations
- [ ] `switcheroo.ops.abs`
- [ ] `switcheroo.ops.acos`
- [ ] `switcheroo.ops.acosh`
- [ ] `switcheroo.ops.asin`
- [ ] `switcheroo.ops.asinh`
- [ ] `switcheroo.ops.atan`
- [ ] `switcheroo.ops.atan2`
- [ ] `switcheroo.ops.atanh`
- [ ] `switcheroo.ops.bitwise_not`
- [ ] `switcheroo.ops.cbrt`
- [ ] `switcheroo.ops.ceil`
- [ ] `switcheroo.ops.conj`
- [ ] `switcheroo.ops.cos`
- [ ] `switcheroo.ops.cosh`
- [ ] `switcheroo.ops.deg2rad`
- [ ] `switcheroo.ops.digamma`
- [ ] `switcheroo.ops.erf`
- [ ] `switcheroo.ops.erfc`
- [ ] `switcheroo.ops.erfinv`
- [ ] `switcheroo.ops.exp`
- [ ] `switcheroo.ops.exp2`
- [ ] `switcheroo.ops.expm1`
- [ ] `switcheroo.ops.fix`
- [ ] `switcheroo.ops.floor`
- [ ] `switcheroo.ops.frexp`
- [ ] `switcheroo.ops.imag`
- [ ] `switcheroo.ops.isfinite`
- [ ] `switcheroo.ops.isinf`
- [ ] `switcheroo.ops.isnan`
- [ ] `switcheroo.ops.lgamma`
- [ ] `switcheroo.ops.log`
- [ ] `switcheroo.ops.log10`
- [ ] `switcheroo.ops.log1p`
- [ ] `switcheroo.ops.log2`
- [ ] `switcheroo.ops.logical_not`
- [ ] `switcheroo.ops.negative`
- [ ] `switcheroo.ops.positive`
- [ ] `switcheroo.ops.rad2deg`
- [ ] `switcheroo.ops.real`
- [ ] `switcheroo.ops.reciprocal`
- [ ] `switcheroo.ops.round`
- [ ] `switcheroo.ops.rsqrt`
- [ ] `switcheroo.ops.sign`
- [ ] `switcheroo.ops.sin`
- [ ] `switcheroo.ops.sinc`
- [ ] `switcheroo.ops.sinh`
- [ ] `switcheroo.ops.sqrt`
- [ ] `switcheroo.ops.square`
- [ ] `switcheroo.ops.tan`
- [ ] `switcheroo.ops.tanh`
- [ ] `switcheroo.ops.trunc`
- [ ] `switcheroo.ops.cast` (dtype conversion)
- [ ] `switcheroo.ops.bitcast`

### Binary Operations
- [ ] `switcheroo.ops.add`
- [ ] `switcheroo.ops.bitwise_and`
- [ ] `switcheroo.ops.bitwise_or`
- [ ] `switcheroo.ops.bitwise_xor`
- [ ] `switcheroo.ops.copysign`
- [ ] `switcheroo.ops.divide` (true division)
- [ ] `switcheroo.ops.divmod`
- [ ] `switcheroo.ops.equal`
- [ ] `switcheroo.ops.float_power`
- [ ] `switcheroo.ops.floor_divide`
- [ ] `switcheroo.ops.fmax`
- [ ] `switcheroo.ops.fmin`
- [ ] `switcheroo.ops.fmod`
- [ ] `switcheroo.ops.gcd`
- [ ] `switcheroo.ops.greater`
- [ ] `switcheroo.ops.greater_equal`
- [ ] `switcheroo.ops.heaviside`
- [ ] `switcheroo.ops.hypot`
- [ ] `switcheroo.ops.lcm`
- [ ] `switcheroo.ops.ldexp`
- [ ] `switcheroo.ops.left_shift`
- [ ] `switcheroo.ops.less`
- [ ] `switcheroo.ops.less_equal`
- [ ] `switcheroo.ops.logaddexp`
- [ ] `switcheroo.ops.logaddexp2`
- [ ] `switcheroo.ops.logical_and`
- [ ] `switcheroo.ops.logical_or`
- [ ] `switcheroo.ops.logical_xor`
- [ ] `switcheroo.ops.maximum`
- [ ] `switcheroo.ops.minimum`
- [ ] `switcheroo.ops.mod`
- [ ] `switcheroo.ops.multiply`
- [ ] `switcheroo.ops.nextafter`
- [ ] `switcheroo.ops.not_equal`
- [ ] `switcheroo.ops.power`
- [ ] `switcheroo.ops.remainder`
- [ ] `switcheroo.ops.right_shift`
- [ ] `switcheroo.ops.subtract`
- [ ] `switcheroo.ops.allclose`
- [ ] `switcheroo.ops.isclose`

### Reductions
- [ ] `switcheroo.ops.sum`
- [ ] `switcheroo.ops.prod`
- [ ] `switcheroo.ops.mean`
- [ ] `switcheroo.ops.variance`
- [ ] `switcheroo.ops.std`
- [ ] `switcheroo.ops.max`
- [ ] `switcheroo.ops.min`
- [ ] `switcheroo.ops.argmax`
- [ ] `switcheroo.ops.argmin`
- [ ] `switcheroo.ops.all`
- [ ] `switcheroo.ops.any`
- [ ] `switcheroo.ops.logsumexp`
- [ ] `switcheroo.ops.count_nonzero`
- [ ] `switcheroo.ops.norm` (L1, L2, Lp norms)
- [ ] *Implementation note:* All reductions must rigorously support `axis` (tuple of ints or int) and `keepdims` (bool).

### Linear Algebra
- [ ] `switcheroo.ops.matmul` (vector-matrix, matrix-matrix, batched)
- [ ] `switcheroo.ops.dot`
- [ ] `switcheroo.ops.tensordot`
- [ ] `switcheroo.ops.vdot`
- [ ] `switcheroo.ops.inner`
- [ ] `switcheroo.ops.outer`
- [ ] `switcheroo.ops.einsum`
- [ ] `switcheroo.ops.cholesky`
- [ ] `switcheroo.ops.svd`
- [ ] `switcheroo.ops.qr`
- [ ] `switcheroo.ops.inv`
- [ ] `switcheroo.ops.pinv`
- [ ] `switcheroo.ops.det`
- [ ] `switcheroo.ops.slogdet`
- [ ] `switcheroo.ops.eigh`
- [ ] `switcheroo.ops.eigvalsh`
- [ ] `switcheroo.ops.matrix_power`

### Neural Network Primitives
- [ ] `switcheroo.nn.conv1d`
- [ ] `switcheroo.nn.conv2d`
- [ ] `switcheroo.nn.conv3d`
- [ ] `switcheroo.nn.conv_transpose1d`
- [ ] `switcheroo.nn.conv_transpose2d`
- [ ] `switcheroo.nn.conv_transpose3d`
- [ ] `switcheroo.nn.max_pool1d`
- [ ] `switcheroo.nn.max_pool2d`
- [ ] `switcheroo.nn.max_pool3d`
- [ ] `switcheroo.nn.avg_pool1d`
- [ ] `switcheroo.nn.avg_pool2d`
- [ ] `switcheroo.nn.avg_pool3d`
- [ ] `switcheroo.nn.adaptive_avg_pool2d`
- [ ] `switcheroo.nn.fractional_max_pool2d`
- [ ] `switcheroo.nn.layer_norm`
- [ ] `switcheroo.nn.batch_norm`
- [ ] `switcheroo.nn.group_norm`
- [ ] `switcheroo.nn.rms_norm`
- [ ] `switcheroo.nn.instance_norm`
- [ ] `switcheroo.nn.dropout`
- [ ] `switcheroo.nn.alpha_dropout`
- [ ] `switcheroo.nn.feature_alpha_dropout`
- [ ] `switcheroo.nn.spatial_dropout`
- [ ] `switcheroo.nn.embedding`
- [ ] `switcheroo.nn.pad` (constant, reflect, replicate, circular modes)
- [ ] `switcheroo.nn.upsample_bilinear`
- [ ] `switcheroo.nn.upsample_nearest`
- [ ] `switcheroo.nn.scaled_dot_product_attention`
- [ ] `switcheroo.nn.rnn_cell`
- [ ] `switcheroo.nn.lstm_cell`
- [ ] `switcheroo.nn.gru_cell`

### Activations
- [ ] `switcheroo.nn.relu`
- [ ] `switcheroo.nn.leaky_relu`
- [ ] `switcheroo.nn.gelu`
- [ ] `switcheroo.nn.swish` (silu)
- [ ] `switcheroo.nn.sigmoid`
- [ ] `switcheroo.nn.tanh`
- [ ] `switcheroo.nn.softplus`
- [ ] `switcheroo.nn.elu`
- [ ] `switcheroo.nn.selu`
- [ ] `switcheroo.nn.celu`
- [ ] `switcheroo.nn.glu`
- [ ] `switcheroo.nn.mish`
- [ ] `switcheroo.nn.hardswish`
- [ ] `switcheroo.nn.softmax`
- [ ] `switcheroo.nn.log_softmax`

### Shape, Memory, & Movement Ops
- [ ] `switcheroo.ops.reshape`
- [ ] `switcheroo.ops.flatten`
- [ ] `switcheroo.ops.squeeze`
- [ ] `switcheroo.ops.unsqueeze`
- [ ] `switcheroo.ops.expand`
- [ ] `switcheroo.ops.broadcast_to`
- [ ] `switcheroo.ops.transpose`
- [ ] `switcheroo.ops.permute`
- [ ] `switcheroo.ops.swapaxes`
- [ ] `switcheroo.ops.moveaxis`
- [ ] `switcheroo.ops.roll`
- [ ] `switcheroo.ops.slice`
- [ ] `switcheroo.ops.dynamic_slice`
- [ ] `switcheroo.ops.update_slice`
- [ ] `switcheroo.ops.strided_slice`
- [ ] `switcheroo.ops.concatenate`
- [ ] `switcheroo.ops.stack`
- [ ] `switcheroo.ops.split`
- [ ] `switcheroo.ops.unstack`
- [ ] `switcheroo.ops.tile`
- [ ] `switcheroo.ops.repeat`
- [ ] `switcheroo.ops.gather`
- [ ] `switcheroo.ops.gather_nd`
- [ ] `switcheroo.ops.scatter`
- [ ] `switcheroo.ops.scatter_nd`
- [ ] `switcheroo.ops.scatter_add`
- [ ] `switcheroo.ops.take`
- [ ] `switcheroo.ops.where`
- [ ] `switcheroo.ops.triu`
- [ ] `switcheroo.ops.tril`
- [ ] `switcheroo.ops.meshgrid`

### Standardized Semantics & Type Promotion
- [ ] Implement strict NumPy-compliant broadcasting resolver (right-to-left alignment).
- [ ] Implement comprehensive Type Promotion rules matrix:
  - [ ] Int vs Float promotion logic.
  - [ ] Precision upcasting (e.g., Float16 + Float32 -> Float32).
  - [ ] Complex number upcasting.
- [ ] Define standardized NaN-propagation rules.
- [ ] Define standardized `axis` normalization (handling negative indices natively).

---

## 3. Core Execution Engine Modes

### Eager Mode Engine (NumPy/SciPy Dispatch)
- [ ] Implement `switcheroo.numpy_backend`.
- [ ] Map all 100+ `switcheroo.ops` to their `numpy.*` or `scipy.*` equivalents.
- [ ] Implement zero-copy `numpy.ndarray` wrapper for the `switcheroo.Tensor.data` payload.
- [ ] Implement explicit fallback errors (`UnimplementedMathError`) for ops with no direct NumPy equivalent (e.g., specialized scaled dot product attention without an explicit loop).
- [ ] Ensure `in-place` mutations are correctly applied in eager mode while enforcing pure function semantics during tracing.

### Graph Tracing Engine (Proxy Tensors)
- [ ] Implement `switcheroo.tracing.ProxyTensor`.
- [ ] Overload exactly all Python magic methods on `ProxyTensor` (`__add__`, `__sub__`, `__mul__`, `__matmul__`, `__truediv__`, `__floordiv__`, `__mod__`, `__pow__`, `__and__`, `__or__`, `__xor__`, `__lshift__`, `__rshift__`, `__neg__`, `__pos__`, `__abs__`, `__invert__`, `__getitem__`).
- [ ] Implement `GraphContext` (Thread-Local Storage) to record the execution tape.
- [ ] Implement stack frame capture during proxy execution to retain source-code line numbers for debugging.
- [ ] Detect and block cycles/infinite recursion during graph generation.

### Automatic Differentiation (AD)
- [ ] Build VJP (Vector-Jacobian Product) registry.
- [ ] Build JVP (Jacobian-Vector Product) registry.
- [ ] Implement forward-mode AutoDiff tape.
- [ ] Implement reverse-mode AutoDiff tape.
- [ ] Implement `switcheroo.grad` primitive.
- [ ] Implement `switcheroo.custom_vjp` decorator for custom gradient injection.
  - *Mapping:* `zero_torch.autograd.backward` and `zero_jax.grad` will directly invoke the compiler's AD tape.

### Higher-Order Control Flow Primitives
- [ ] Implement `switcheroo.control_flow.cond(pred, true_fn, false_fn)`.
- [ ] Implement `switcheroo.control_flow.while_loop(cond, body, init_val)`.
- [ ] Implement `switcheroo.control_flow.scan(f, init, xs)`.
- [ ] Implement `switcheroo.control_flow.vmap` (vectorizing map - batching dimension injector).
- [ ] Implement `switcheroo.control_flow.pmap` (parallel map).
  - *Mapping:* Replaces `zero_jax.lax.cond`, `zero_jax.lax.scan`, `zero_jax.vmap`.

---

## 4. Unified Intermediate Representation (IR) Schema

### Base Data Structures
- [ ] Define `IRGraph` (contains Inputs, Outputs, and a collection of `IRNode`s).
- [ ] Define `IRNode` (fields: `id`, `opcode`, `inputs`, `outputs`, `attributes`, `metadata`).
- [ ] Define `IRBlock` (represents nested scopes for control flow like `cond` and `while_loop`).
- [ ] Define `TensorSpec` (fields: `shape`, `dtype`, `sparsity`).

### Type & Shape System
- [x] Define Enum `DType` (`Float64`, `Float32`, `Float16`, `BFloat16`, `Complex64`, `Complex128`, `Int64`, `Int32`, `Int16`, `Int8`, `UInt8`, `Bool`).
- [x] Define Enum `QuantDType` (`QInt8`, `QUInt8`, `QInt4` with scale/zero-point).
- [ ] Implement `ShapeTracker` for calculating exact output shapes given input `TensorSpec`s.
- [ ] Implement `SymInt` (Symbolic Integer) to trace graphs with dynamic dimensions (e.g., `batch_size`).
- [ ] Implement Symbolic Expression Solver to validate shape consistency mathematically without concrete numbers.

### State Mutation & Aliasing Representation
- [ ] Define `ReadVariable` node.
- [ ] Define `AssignVariable` node.
- [ ] Define `ScatterUpdate` node.
- [ ] Implement functional-purity constraints (mutations must return an updated state edge in the DAG).
  - *Mapping:* `zero_pytorch.nn.Parameter` assignments generate `AssignVariable` nodes, bridging PyTorch's OOP state to the IR.

---

## 5. Exhaustive Middle-End Transformations (Pass Manager)

### Infrastructure
- [ ] Implement `PassManager`.
- [ ] Implement DAG Topological Sorter.
- [ ] Implement `FixpointIterator` (run passes until DAG hash stops changing).
- [ ] Implement pre-pass and post-pass IR validators (shape-checking, cycle-checking).

### High-Level Canonicalization Passes
- [ ] `StateLiftingPass`: Hoist `ReadVariable`/`AssignVariable` nodes out of the graph, making them explicit inputs/outputs. (Crucial for converting PyTorch state to JAX functional pure inputs).
- [ ] `StateLoweringPass`: Convert functional inputs/outputs back to `AssignVariable`.
- [ ] `AxisTranslationPass`: Inject `Transpose` nodes globally to convert `NCHW` to `NHWC` or vice versa depending on backend requirements.
- [ ] `DTypePromotionPass`: Traverse the AST and explicitly insert `Cast` nodes based on promotion rules, removing implicit ambiguity for Edge compilers.
- [ ] `BroadcastExplicitizerPass`: Replace implicit tensor broadcasting with concrete `BroadcastTo` nodes.

### Target-Agnostic Optimization Passes
- [ ] `ConstantFoldingPass`: Pre-evaluate purely deterministic mathematical sub-graphs.
- [ ] `DeadCodeEliminationPass` (DCE): Prune nodes whose outputs do not trace to graph outputs or state updates.
- [ ] `CommonSubexpressionEliminationPass` (CSE): Identify duplicate sub-graphs and route them to a single node.
- [ ] `AlgebraicSimplificationPass`: 
  - `x * 0 -> 0`
  - `x + 0 -> x`
  - `x * 1 -> x`
  - `x / 1 -> x`
  - `x - x -> 0`
- [ ] `MixedPrecisionPass`: Identify MatMuls/Convs and auto-cast FP32 inputs to FP16/BF16 if safe.
- [ ] `BatchNormFoldingPass`: Pre-calculate and fuse `BatchNorm` weights directly into preceding `Conv2D` kernels.
- [ ] `ReshapeSimplificationPass`: Collapse adjacent `Reshape` or `Flatten` nodes into a single node.
- [ ] `TransposeCancellationPass`: Remove adjacent `Transpose` nodes that reverse each other.

### Low-Level / Edge Compilation Passes
- [ ] `ElementwiseKernelFusionPass`: Identify chains of element-wise ops (e.g., `Add` -> `BatchNorm` -> `ReLU`) and fuse them into a single `FusedElementwise` node for custom shader generation.
- [ ] `AttentionFusionPass`: Pattern-match distinct matmuls/softmax into `ScaledDotProductAttention` for backend optimization.
- [ ] `BufferAllocationPass`: Calculate exact byte offsets and sizes in a linear memory arena for WebGPU/WASM targets.
- [ ] `MemoryReusePass` (Liveness Analysis): Allow non-overlapping intermediate tensors to reuse the same byte offsets in the memory arena.
- [ ] `LoopUnrollingPass`: Unroll small constant loops to prepare for vectorization.

---

## 6. Python Source-to-Source Emission Backends
Generates raw Python code from the compiled IR. Used when cross-compiling from one framework to another.

### Base Infrastructure
- [ ] Implement `PythonCodeGenerator` class.
- [ ] Implement deterministic variable naming (`tensor_0`, `tensor_1`).
- [ ] Implement automatic scope and indentation tracking.
- [ ] Output formatting with standard Python typing hints.

### Target: PyTorch (`target='pytorch'`)
- [ ] Emit `torch.nn.Module` class definition.
- [ ] Emit `__init__` method, mapping IR variables to `self.register_parameter` and `self.register_buffer`.
- [ ] Emit `forward` method containing topologically sorted `torch.*` ops.
- [ ] Avoid dynamic loops to guarantee `torch.compile` compatibility.

### Target: JAX / Flax (`target='jax'`)
- [ ] Emit a pure Python function `apply_model(params, x)`.
- [ ] Format `params` as a standard JAX PyTree / nested dictionary.
- [ ] Emit `jax.numpy.*` and `jax.lax.*` mathematical equivalents.
- [ ] Emit control flow as `jax.lax.cond`, `jax.lax.while_loop`.
- [ ] Alternatively, emit `flax.linen.Module` representations.

### Target: Keras (`target='keras'`)
- [ ] Emit Keras Functional API script.
- [ ] Map IR nodes directly to `keras.layers.*`.
- [ ] Map unsupported nodes to Keras 3 `keras.ops.*` lambdas.
- [ ] Emit `keras.Model(inputs=..., outputs=...)` finalizer.

### Target: MLX (`target='mlx'`)
- [ ] Emit `mlx.nn.Module` classes.
- [ ] Map parameters to class attributes.
- [ ] Map math to `mlx.core.*`.
- [ ] Emit `__call__` method.

### Target: TensorFlow (`target='tensorflow'`)
- [ ] Emit `@tf.function` wrapped graphs.
- [ ] Emit `tf.raw_ops.*` math ops.
- [ ] Support dumping strictly into the Protobuf `SavedModel` layout.

---

## 7. Edge, Web & Native Backend Emitters

### Target: WebGPU (`target='webgpu'`)
- [ ] Develop WGSL Jinja2 templates for Element-wise operations.
- [ ] Develop WGSL Jinja2 templates for Tiled MatMul operations (using workgroup shared memory).
- [ ] Develop WGSL Jinja2 templates for `Conv2D` (im2col + matmul).
- [ ] Generate `@binding` and `@group` WGSL buffer mappings automatically based on Liveness Analysis.
- [ ] Emit JS/TS Orchestrator script: `navigator.gpu.requestDevice()`.
- [ ] Emit JS/TS Orchestrator script: Manage `GPUBuffer` upload/download arrays.
- [ ] Emit JS/TS Orchestrator script: Compute workgroup dispatch sizes based on `TensorSpec`.

### Target: WebGL 2.0 Fallback (`target='webgl'`)
- [ ] Generate GLSL fragment shaders mapping 2D texture coordinates back to tensor coordinates.
- [ ] Emit JS/TS Orchestrator script: Pack 4 Float32s into RGBA32F `WebGLTexture` formats.
- [ ] Emit JS/TS Orchestrator script: Setup `WebGLFramebuffer` attachment rendering loop.

### Target: WASM SIMD (`target='wasm'`)
- [ ] Emit C++ source mapping IR to `<wasm_simd128.h>` standard headers.
- [ ] Map math nodes to `wasm_f32x4_*` hardware intrinsics.
- [ ] Emit memory allocation macros (`__attribute__((aligned(16)))`).
- [ ] Integrate Emscripten (`emcc`) build scripts in a subprocess with `-O3 -msimd128`.
- [ ] Emit JS/TS Wrapper script: `WebAssembly.instantiateStreaming` configuration.
- [ ] Emit JS/TS Wrapper script: Read/Write from `WebAssembly.Memory`.

### Target: ONNX (`target='onnx'`)
- [ ] Map IR schemas to standardized ONNX Opset versions.
- [ ] Emit pure ONNX Protobuf payload.
- [ ] Support serialization of ONNX QLinear (Quantization) specs.

---

## 8. DX, Diagnostics & Error Handling

### Traceback & Source Mapping
- [ ] Store `inspect.getframeinfo` on every `ProxyTensor` generation.
- [ ] Implement `TracebackReconstructor`: When compiler crashes on an IR pass, format the stack trace to point to the line in the `zero-*` frontend code where the operation was called, hiding the `ml-switcheroo-compiler` internal stack trace.

### Profiling & Dry-Run
- [ ] Implement `switcheroo.debug_shapes(model, input_shape)`: Run static shape inference and print a markdown table of tensor flow.
- [ ] Implement analytical FLOPs counter based on the static shape inference.
- [ ] Implement Memory Profiler: Output theoretical peak memory usage based on Liveness analysis.
- [ ] Implement Numerical Anomaly Detector: In Eager mode, warn if a node transitions from finite numbers to NaN/Inf.

### Visualization
- [ ] Implement `to_graphviz()`: Export DAG to `.dot`.
- [ ] Implement `to_html()`: Export interactive D3.js visualizer.

---

## 9. Serialization, Testing & Packaging

### Serialization Formats
- [ ] Define `IRGraph` Protobuf `.proto` spec.
- [ ] Implement JSON serialization for snapshot testing IR passes.
- [ ] Implement FlatBuffers serialization for zero-copy JS/TS loading.

### Testing & Equivalence Validation
- [ ] Implement Parameterized Equivalence CI:
  - *Mapping constraint:* The test suites currently in `zero_pytorch/tests/` and `zero_jax/tests/` will be run exactly as they are. However, a pytest fixture will globally inject `ml-switcheroo-compiler` to run the math. The tests must assert `np.allclose(atol=1e-5)` comparing the compiler output against true native PyTorch/JAX execution.
- [ ] Test cross-platform emission: Compile an IR graph to Python PyTorch, execute it, compile the same graph to JAX, execute it, and assert outputs match.
- [ ] Automate headless browser testing (Puppeteer) to execute compiled WebGPU/WASM graphs and compare outputs to native Python.

### Packaging
- [ ] `pyproject.toml` setup for PyPI publishing (`ml-switcheroo-compiler`).
- [ ] Setup strict constraints: Ensure compiler has minimal dependencies (`numpy` mainly), avoiding any hard dependencies on PyTorch, JAX, or TensorFlow.
- [ ] Generate `package.json` for JS/TS wrapper libraries and publish to NPM (`@switcheroo/web-runtime`).
- [ ] Configure GitHub Actions matrix CI for Linux, macOS, and Windows.