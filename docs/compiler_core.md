# Compiler Core

This section documents the core compiler and IR.

```{autosummary}
:toctree: _autosummary

ml_switcheroo_compiler
```

## Compiler Optimizations

The pass manager evaluates a strict optimization topology:
- Constant Folding
- Dead Code Elimination (DCE)
- Common Subexpression Elimination (CSE)
- Operator Fusion (Conv+Relu, Softmax+CrossEntropy)
- Loop Unrolling (evaluating static trip counts)
- Memory Planning (buffer alias analysis)
- Scheduling (compute bounds depth analysis)
- Distributed Parity (SPMD, Collectives, Pipeline Parallelism)

## Automatic Differentiation

The engine fully models backpropagation logic on-the-fly:
- `jvp` (Forward-Mode) tape generation.
- `vjp` (Reverse-Mode) graph derivation.
- `hvp` (Higher-Order) derivatives via forward-over-reverse.
