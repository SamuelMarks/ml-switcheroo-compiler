**QA CHECK: STRICT ARCHITECTURAL SEPARATION & ECOSYSTEM HIERARCHY**

**WARNING: You are editing `ml-switcheroo-compiler`. This is Tier 2 of the ML ecosystem. Before outputting any code modifications, you MUST verify your proposed changes against the following exhaustive constraints.**

### 1. FRONTEND INDEPENDENCE (The "No API Shell" Rule)
This repository is the shared execution engine, trace mechanism, and middle-end. Its core, tracing, IR, ops, and transformation layers are completely blind to which frontend invoked them.
* **MUST NOT** contain framework-specific syntax mimicry or runtime API shells in `core/`, `tracing/`, `ir/`, `ops/`, `grad/`, `nn/`, or `transforms/` (e.g., PyTorch's `nn.Module` runtime wrapper, Keras's `model.fit()`, or frontend-specific exception hierarchies).
* **MUST NOT** contain argument-parsing aliases designed to bridge frontend runtime APIs in core math/ops layers (e.g., handling `dim` vs `axis`, or `keepdim` vs `keepdims`). *(Note: Static source-to-source syntax/argument rewrite rules belong exclusively inside `ml_switcheroo_compiler/backends/` CST transpiler and target emitters.)*
* **MUST NOT** include any core feature, op, or middle-end pass that is only logically useful to a single ML frontend.
* **WHERE IT BELONGS:** All frontend API routing, syntactic sugar, framework-specific object wrappers, and runtime argument normalization belong strictly in the respective `../zero-*` repository (e.g., `zero-pytorch`, `zero-jax`, `zero-keras`).

### 2. BACKEND DECOUPLING & ZERO NUMPY LEAKS (The "Registry & Single-Backend Isolation" Rule)
The core tensor types, tracing engine, IR utilities (`core/`, `tracing/`, `ir/`, `ops/`), and middle-end optimization passes (`transforms/`) must remain entirely abstract, symbolic, or backend-dispatched.
* **MUST NOT** import or depend on target execution libraries (`torch`, `jax`, `mlx`, `cupy`, `dask`, `tensorflow`, `keras`, `numpy`, `scipy`, `numba`, `sparse`, `onnx`, `Metal`, `dpnp`, `awkward`, `pyarrow`) outside of their designated `src/ml_switcheroo_compiler/backends/<target>/` directory.
* **NO GLOBAL NUMPY FALLBACK:** `numpy` and `scipy` are **strictly forbidden** outside `src/ml_switcheroo_compiler/backends/numpy/` (enforced by `scripts/lint_numpy_leak.py` and `scripts/lint_dynamic_numpy.py`). Do NOT import `numpy` globally or call `BackendRegistry.get("numpy")` as a hidden fallback in other backends or core modules.
* **SINGLE-BACKEND ISOLATION:** Every `backends/<target>/` directory may import at most one external backend package root (`scipy` is permitted alongside `numpy` solely inside `backends/numpy/`), enforced by `scripts/lint_dependencies.py`.
* **WHERE IT BELONGS:** Any logic mapping IR or eager ops to a specific framework runtime or code emitter MUST live in `ml_switcheroo_compiler/backends/<target>/`, register dynamically via `@register_backend("name")` in `BackendRegistry`, and conform to static capability protocols in `ml_switcheroo_compiler/backends/typeclasses.py` (`scripts/verify_typeclasses.py`).

### 3. DEPENDENCY HIERARCHY (The "No Upward Calls" Rule)
The ecosystem enforces a strict, cycle-free Directed Acyclic Graph (DAG) for dependencies (`scripts/lint_dependencies.py`).
* **ALLOWED GLOBAL IMPORTS:** Outside isolated `backends/<target>/` directories and `tests/`, this repository may ONLY import from itself (`ml_switcheroo_compiler`), the Python Standard Library, Tier 1 `ml-switcheroo-ir` (`ml_switcheroo_ir`), and approved lightweight infrastructure packages (`pydantic`, `pyyaml`/`yaml`, `libcst`, `h5py`, `pillow`/`PIL`, `cdd`).
* **NO VENDORING TIER 1 IR:** Shared IR graph/node/type schemas (`LogicalGraph`, `LogicalNode`, `TensorShape`, `TensorSpec`, `PartitionSpec`, etc.) MUST be imported directly from `ml_switcheroo_ir` — never vendored into `src/ml_switcheroo_compiler/ir/` or monkey-patched at runtime.
* **FORBIDDEN IMPORTS:** You **MUST NOT** import from any `zero-*` repository (Tier 3/4) or `zero-zoo` (Tier 5). Circular or upward dependencies will immediately break the compilation pipeline.

### 4. THE N-TO-M UNIVERSAL UTILITY RULE
To justify adding a new operation to `ml_switcheroo_compiler/ops/` or a new pass to `transforms/`, it must be universally translatable (`scripts/validate_rules.py`).
* It must be reducible to the Unified IR (`ml-switcheroo-ir`).
* It must be capable of being emitted or executed by at least two or more backends without cross-backend fallback.

### 5. MATHEMATICAL & AUTODIFF CENTRALIZATION (The "No Math in Frontends" Rule)
This repository (`ml-switcheroo-compiler`) is the EXCLUSIVE home for all actual mathematical computations, Automatic Differentiation (AD), `jit` compilation, forward passes, backward passes, and Jacobian-vector / vector-Jacobian products (JVPs/VJPs).
* **MUST NOT:** A `zero-*` frontend must NEVER implement custom math, gradient tracking, or raw array manipulation. They are purely API routing shells.
* **WHERE IT BELONGS:** If a user requests adding a new mathematical operation, a custom gradient formula, or AD logic, it MUST be implemented here in the compiler (inside `ml_switcheroo_compiler/ops/`, `ml_switcheroo_compiler/grad/`, or `ml_switcheroo_compiler/transforms/autodiff_rules/`). The `zero-*` repo will then simply delegate to this implementation.

### 6. STRICT STATIC TYPING, ANTI-REFLECTION & SNAPSHOT GROUNDING
All code in `src/ml_switcheroo_compiler/`, `scripts/`, and `tests/` must pass strict AST and schema verifiers:
* **NO `Any`, `cast`, OR DYNAMIC REFLECTION:** `typing.Any` (`scripts/lint_any.py --max-count 0`), `typing.cast`, disguised casts, and dynamic attribute reflection (`getattr`, `setattr`, `hasattr`, `delattr`, `vars`, `globals`, `locals` via `scripts/lint_typing.py`) are strictly forbidden; use explicit `typing.Protocol` / `TypeGuard` narrowing under `mypy --strict`.
* **SNAPSHOT GROUNDING & QUALITY GATES:** Backend mappings and operator parameters must match upstream `ml-ecosystem-snapshots` with zero hallucinated kwargs (`scripts/validate_backend_mappings.py`, `scripts/validate_parameters.py`), tests must contain substantive assertions without bare `pass` (`scripts/lint_pass.py --check-zero-assertions`), and all modules/classes/functions must maintain 100% line, branch, and granular AST docstring coverage (`scripts/check_docs.py`).

---
**ACTION REQUIRED ON VIOLATION:**
If the user's prompt requests a change that violates *any* of the rules above, you MUST:
1. **REFUSE** to make the change in the requested repository if it breaks the boundaries.
2. **INSTRUCT** the user on which subsystem or repository the logic actually belongs in (e.g., "This syntactic sugar must be implemented in `zero-pytorch`", or "This custom VJP logic belongs in `ml_switcheroo_compiler.grad`, not `zero-jax`").
