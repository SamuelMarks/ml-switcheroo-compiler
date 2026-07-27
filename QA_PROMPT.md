**QA CHECK: STRICT ARCHITECTURAL SEPARATION & ECOSYSTEM HIERARCHY**

**WARNING: You are editing `ml-switcheroo-compiler`. This is Tier 2 of the ML ecosystem. Before outputting any code modifications, you MUST verify your proposed changes against the following exhaustive constraints.**

### 1. FRONTEND INDEPENDENCE (The "No API Shell" Rule)
This repository is the shared execution engine, trace mechanism, and middle-end. It is completely blind to which frontend invoked it.
* **MUST NOT** contain framework-specific syntax mimicry or utilities (e.g., PyTorch's `nn.Module` logic, Keras's `model.fit()`, JAX's `PRNGKey` management, or frontend-specific exceptions).
* **MUST NOT** contain argument-parsing aliases designed to bridge frontend APIs (e.g., handling `dim` vs `axis`, or `keepdim` vs `keepdims`).
* **MUST NOT** include any feature, op, or pass that is only logically useful to a single ML frontend.
* **WHERE IT BELONGS:** All API routing, syntactic sugar, framework-specific state mutation tracking, and argument normalization belongs strictly in the respective `../zero-*` repository (e.g., `zero-pytorch`, `zero-jax`, `zero-keras`).

### 2. BACKEND DECOUPLING (The "Registry Only" Rule)
The tracing engine, IR graph builder (`core/`, `tracing/`, `ir/`), and middle-end optimization passes (`transforms/`) must remain entirely abstract and symbolic.
* **MUST NOT** import or depend on target execution libraries such as `import torch`, `import jax`, `import mlx`, `import cupy`, `import dusk`, or `import keras` outside of their designated backend emitter directories.
* **EXCEPTION:** `numpy` is explicitly permitted globally as the foundational fallback for reference eager-mode evaluations, host-level shape broadcasting, and standard library operations.
* **WHERE IT BELONGS:** Any logic that maps IR ops to specific framework runtime logic or AST generation MUST be strictly confined to `ml_switcheroo_compiler/backends/` and must be dynamically registered using `@register_backend("name")` via the `BackendRegistry`.

### 3. DEPENDENCY HIERARCHY (The "No Upward Calls" Rule)
The ecosystem enforces a strict, cycle-free Directed Acyclic Graph (DAG) for dependencies.
* **ALLOWED IMPORTS:** This repository may only import from itself, the Python Standard Library, `numpy`, and Tier 1 libraries (`ml-switcheroo-ir`).
* **FORBIDDEN IMPORTS:** You **MUST NOT** import from any `zero-*` repository (Tier 3/4) or `zero-zoo` (Tier 5). Circular dependencies will immediately break the compilation pipeline.

### 4. THE N-TO-M UNIVERSAL UTILITY RULE
To justify adding a new operation to `ml_switcheroo_compiler/ops/` or a new pass to `transforms/`, it must be universally translatable.
* It must be reducible to the Unified IR (`ml-switcheroo-ir`).
* It must be capable of being emitted by at least two or more backends.

### 5. MATHEMATICAL & AUTODIFF CENTRALIZATION (The "No Math in Frontends" Rule)
This repository (`ml-switcheroo-compiler`) is the EXCLUSIVE home for all actual mathematical computations, Automatic Differentiation (AD), forward passes, backward passes, and Jacobian-vector products (JVPs/VJPs).
* **MUST NOT:** A `zero-*` frontend must NEVER implement custom math, gradient tracking, or raw array manipulation. They are purely API routing shells.
* **WHERE IT BELONGS:** If a user requests adding a new mathematical operation, a custom gradient formula, or AD logic, it MUST be implemented here in the compiler (e.g., inside `ml_switcheroo_compiler/ops/` or `ml_switcheroo_compiler/vjp/`). The `zero-*` repo will then simply delegate to this implementation.

---
**ACTION REQUIRED ON VIOLATION:**
If the user's prompt requests a change that violates *any* of the rules above, you MUST:
1. **REFUSE** to make the change in the requested repository if it breaks the boundaries.
2. **INSTRUCT** the user on which subsystem or repository the logic actually belongs in (e.g., "This syntactic sugar must be implemented in `zero-pytorch`", or "This custom VJP logic belongs in `ml-switcheroo-compiler`, not `zero-jax`").
