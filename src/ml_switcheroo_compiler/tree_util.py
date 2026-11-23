"""PyTree utilities, open PyTreeNode typeclass instances, and Foldable/Traversable combinators."""

from __future__ import annotations

import builtins
import dataclasses
import functools
import typing
from collections.abc import Callable, Iterable, Iterator, Sequence

from ml_switcheroo_ir.typeclasses import (
    BOOL_ALL_MONOID,
    BOOL_ANY_MONOID,
    Eq,
    Err,
    HashableWitness,
    Invalid,
    Monoid,
    Ok,
    Pretty,
    Result,
    Semigroup,
    Valid,
    Validation,
    traverse_result,
    traverse_validation,
)

from ml_switcheroo_compiler.core.shape import OpValue, is_variadic_callable

T = typing.TypeVar("T")
_EqT = typing.TypeVar("_EqT")
_M = typing.TypeVar("_M")
_E = typing.TypeVar("_E")
_NodeT = typing.TypeVar("_NodeT", bound=OpValue)

AuxData = typing.Union[str, int, tuple[typing.Union[str, int], ...], None]  # noqa: UP007


class TreeDef:
    """Provide a structural definition of a flattened PyTree container hierarchy."""

    def __init__(
        self,
        node_type: type,
        children_defs: list[TreeDef] | None = None,
        keys: list[str | int] | None = None,
        aux_data: AuxData = None,
    ) -> None:
        """Initialize a ``TreeDef`` node descriptor.

        Args:
            node_type (type): Container class for internal nodes, or ``type(None)`` for leaf nodes.
            children_defs (list[TreeDef] | None): Structural descriptors of immediate child subtrees.
            keys (list[str | int] | None): Sorted dictionary keys when ``node_type is dict``.
            aux_data (AuxData): Static auxiliary metadata for custom registered ``PyTreeNodeTypeclass`` instances.
        """
        self.node_type = node_type
        self.children_defs = children_defs or []
        self.keys = keys
        self.aux_data = aux_data

    def __repr__(self) -> str:
        """Return deterministic string representation of this ``TreeDef``.

        Returns:
            str: String representation of the ``TreeDef``.
        """
        if self.aux_data is not None:
            return f"TreeDef({self.node_type.__name__}, {self.children_defs}, aux={self.aux_data!r})"
        return f"TreeDef({self.node_type.__name__}, {self.children_defs})"

    def __hash__(self) -> int:
        """Compute a key- and auxiliary-metadata-sensitive hash of this ``TreeDef``.

        Returns:
            int: Structural hash integer.
        """
        keys_tup: tuple[str | int, ...] | None = None if self.keys is None else tuple(self.keys)
        return hash((self.node_type, tuple(self.children_defs), keys_tup, self.aux_data))

    def __eq__(self, other: _EqT) -> bool:
        """Check structural equality with another ``TreeDef``.

        Args:
            other (_EqT): Candidate value to compare with.

        Returns:
            bool: ``True`` if structurally equal, ``False`` otherwise.
        """
        if not isinstance(other, TreeDef):
            return False
        return (
            self.node_type == other.node_type
            and self.children_defs == other.children_defs
            and self.keys == other.keys
            and self.aux_data == other.aux_data
        )


TREE_DEF_EQ: Eq[TreeDef] = Eq(eq=lambda a, b: a == b)
TREE_DEF_HASH: HashableWitness[TreeDef] = HashableWitness(
    eq_witness=TREE_DEF_EQ,
    hash_value=hash,
)
TREE_DEF_PRETTY: Pretty[TreeDef] = Pretty(render=repr)


@dataclasses.dataclass(frozen=True)
class PyTreeNodeTypeclass(typing.Generic[_NodeT]):
    """Explicit Haskell-style typeclass dictionary witnessing that ``_NodeT`` is a ``PyTree`` container."""

    node_type: type[_NodeT]
    flatten_fn: Callable[[_NodeT], tuple[Sequence[OpValue], AuxData]]
    unflatten_fn: Callable[[AuxData, Sequence[OpValue]], _NodeT]


@dataclasses.dataclass(frozen=True)
class _ErasedPyTreeEntry:
    """Internal type-safe erased adapter invoked during dynamic PyTree traversal."""

    flatten_erased: Callable[[OpValue], tuple[Sequence[OpValue], AuxData]]
    unflatten_erased: Callable[[AuxData, Sequence[OpValue]], OpValue]


_PYTREE_REGISTRY: dict[type, _ErasedPyTreeEntry] = {}
_BUILTINS_INITIALIZED: list[bool] = [False]


def register_pytree_node(
    node_cls: type[_NodeT],
    flatten_fn: Callable[[_NodeT], tuple[Sequence[OpValue], AuxData]],
    unflatten_fn: Callable[[AuxData, Sequence[OpValue]], _NodeT],
) -> PyTreeNodeTypeclass[_NodeT]:
    """Register ``node_cls`` in the open ``PyTreeNodeTypeclass`` instance table.

    Args:
        node_cls (type[_NodeT]): Custom container class to register as a traversable PyTree node.
        flatten_fn (Callable[[_NodeT], tuple[Sequence[OpValue], AuxData]]): Extracts child subtrees and static aux data.
        unflatten_fn (Callable[[AuxData, Sequence[OpValue]], _NodeT]): Reconstructs ``node_cls`` from aux and children.

    Returns:
        PyTreeNodeTypeclass[_NodeT]: Explicit typeclass witness dictionary for ``node_cls``.
    """

    def _flatten_erased(val: OpValue) -> tuple[Sequence[OpValue], AuxData]:
        """Validate ``isinstance(val, node_cls)`` and invoke ``flatten_fn``.

        Args:
            val (OpValue): Runtime PyTree node value.

        Returns:
            tuple[Sequence[OpValue], AuxData]: Child sequence and static auxiliary metadata.

        Raises:
            TypeError: If ``val`` is not an instance of ``node_cls``.
        """
        if not isinstance(val, node_cls):
            raise TypeError(f"Expected instance of {node_cls.__name__}, got {type(val).__name__}")
        return flatten_fn(val)

    def _unflatten_erased(aux: AuxData, children: Sequence[OpValue]) -> OpValue:
        """Reconstruct ``node_cls`` from ``aux`` and reconstructed ``children``.

        Args:
            aux (AuxData): Static auxiliary metadata stored in ``TreeDef.aux_data``.
            children (Sequence[OpValue]): Reconstructed immediate child subtrees.

        Returns:
            OpValue: Reconstructed container instance.
        """
        return unflatten_fn(aux, children)

    _PYTREE_REGISTRY[node_cls] = _ErasedPyTreeEntry(
        flatten_erased=_flatten_erased,
        unflatten_erased=_unflatten_erased,
    )
    return PyTreeNodeTypeclass(
        node_type=node_cls,
        flatten_fn=flatten_fn,
        unflatten_fn=unflatten_fn,
    )


def is_registered_pytree_node(node_cls: type) -> bool:
    """Return whether ``node_cls`` has an explicit ``PyTreeNodeTypeclass`` instance registered.

    Args:
        node_cls (type): Candidate container type.

    Returns:
        bool: ``True`` if registered in the custom PyTree instance table.
    """
    _ensure_builtin_pytree_instances()
    return node_cls in _PYTREE_REGISTRY


def _ensure_builtin_pytree_instances() -> None:
    """Lazily register built-in compiler tensor container types (``SparseTensorCOO/CSR``, ``RaggedTensor``)."""
    if _BUILTINS_INITIALIZED[0]:
        return
    _BUILTINS_INITIALIZED[0] = True

    from ml_switcheroo_compiler.core.ragged_tensor import RaggedTensor
    from ml_switcheroo_compiler.core.sparse_tensor import SparseTensorCOO, SparseTensorCSR
    from ml_switcheroo_compiler.core.tensor import ensure_tensor

    def _flatten_coo(sp: SparseTensorCOO) -> tuple[Sequence[OpValue], AuxData]:
        """Flatten ``SparseTensorCOO`` into ``(indices, values)`` children and ``dense_shape`` aux data.

        Args:
            sp (SparseTensorCOO): COO sparse tensor container.

        Returns:
            tuple[Sequence[OpValue], AuxData]: Child tensors and dense shape tuple.
        """
        return (sp.indices, sp.values), tuple(sp.dense_shape)

    def _unflatten_coo(aux: AuxData, children: Sequence[OpValue]) -> SparseTensorCOO:
        """Reconstruct ``SparseTensorCOO`` from ``dense_shape`` aux data and ``(indices, values)``.

        Args:
            aux (AuxData): Dense shape integer tuple.
            children (Sequence[OpValue]): Two-element sequence ``(indices, values)``.

        Returns:
            SparseTensorCOO: Reconstructed COO sparse tensor.
        """
        dense_shape: tuple[int, ...] = tuple(int(d) for d in aux) if isinstance(aux, tuple) else ()
        return SparseTensorCOO(ensure_tensor(children[0]), ensure_tensor(children[1]), dense_shape)

    def _flatten_csr(sp: SparseTensorCSR) -> tuple[Sequence[OpValue], AuxData]:
        """Flatten ``SparseTensorCSR`` into ``(row_pointers, column_indices, values)`` and ``dense_shape``.

        Args:
            sp (SparseTensorCSR): CSR sparse tensor container.

        Returns:
            tuple[Sequence[OpValue], AuxData]: Child tensors and dense shape tuple.
        """
        return (sp.row_pointers, sp.column_indices, sp.values), tuple(sp.dense_shape)

    def _unflatten_csr(aux: AuxData, children: Sequence[OpValue]) -> SparseTensorCSR:
        """Reconstruct ``SparseTensorCSR`` from ``dense_shape`` aux data and CSR component tensors.

        Args:
            aux (AuxData): Dense 2D shape integer tuple.
            children (Sequence[OpValue]): Three-element sequence ``(row_pointers, column_indices, values)``.

        Returns:
            SparseTensorCSR: Reconstructed CSR sparse tensor.
        """
        dense_shape: tuple[int, ...] = tuple(int(d) for d in aux) if isinstance(aux, tuple) else ()
        return SparseTensorCSR(
            ensure_tensor(children[0]),
            ensure_tensor(children[1]),
            ensure_tensor(children[2]),
            dense_shape,
        )

    def _flatten_ragged(rt: RaggedTensor) -> tuple[Sequence[OpValue], AuxData]:
        """Flatten ``RaggedTensor`` into ``(values, row_splits)`` child tensors.

        Args:
            rt (RaggedTensor): Ragged tensor container.

        Returns:
            tuple[Sequence[OpValue], AuxData]: Pair ``(values, row_splits)`` and ``None`` aux data.
        """
        return (rt.values, rt.row_splits), None

    def _unflatten_ragged(_aux: AuxData, children: Sequence[OpValue]) -> RaggedTensor:
        """Reconstruct ``RaggedTensor`` from ``(values, row_splits)`` child tensors.

        Args:
            _aux (AuxData): Unused auxiliary metadata (``None``).
            children (Sequence[OpValue]): Two-element sequence ``(values, row_splits)``.

        Returns:
            RaggedTensor: Reconstructed ragged tensor.
        """
        return RaggedTensor(ensure_tensor(children[0]), ensure_tensor(children[1]))

    register_pytree_node(SparseTensorCOO, _flatten_coo, _unflatten_coo)
    register_pytree_node(SparseTensorCSR, _flatten_csr, _unflatten_csr)
    register_pytree_node(RaggedTensor, _flatten_ragged, _unflatten_ragged)


PyTree = typing.Union[OpValue, TreeDef]  # noqa: UP007


def tree_flatten(tree: PyTree) -> tuple[list[OpValue], TreeDef]:
    """Flatten a PyTree into a list of leaves and an auxiliary ``TreeDef``.

    Args:
        tree (PyTree): The tree to flatten.

    Returns:
        tuple[list[OpValue], TreeDef]: Flattened leaf list and structural ``TreeDef``.
    """
    _ensure_builtin_pytree_instances()
    node_cls = type(tree)
    custom_entry = _PYTREE_REGISTRY.get(node_cls)
    if custom_entry is not None and not isinstance(tree, TreeDef):
        children, aux = custom_entry.flatten_erased(tree)
        leaves: list[OpValue] = []
        children_defs: list[TreeDef] = []
        for child in children:
            child_leaves, child_def = tree_flatten(child)
            leaves.extend(child_leaves)
            children_defs.append(child_def)
        return leaves, TreeDef(node_cls, children_defs, None, aux)
    if isinstance(tree, dict):
        keys = sorted(tree.keys(), key=lambda k: (type(k).__name__, str(k)))
        leaves = []
        children_defs = []
        for k in keys:
            child_leaves, child_def = tree_flatten(tree[k])
            leaves.extend(child_leaves)
            children_defs.append(child_def)
        return leaves, TreeDef(dict, children_defs, keys)
    if isinstance(tree, (list, tuple)):
        leaves = []
        children_defs = []
        for child in tree:
            child_leaves, child_def = tree_flatten(child)
            leaves.extend(child_leaves)
            children_defs.append(child_def)
        return leaves, TreeDef(type(tree), children_defs)
    assert not isinstance(tree, TreeDef)
    return [tree], TreeDef(type(None))


def _unflatten_leaf(leaves_it: Iterator[OpValue]) -> OpValue:
    """Unflatten a single leaf node from ``leaves_it``.

    Args:
        leaves_it (Iterator[OpValue]): Iterator of leaves.

    Returns:
        OpValue: Next leaf value.

    Raises:
        ValueError: If there are too few leaves remaining.
    """
    try:
        return next(leaves_it)
    except StopIteration as err:
        msg = "Too few leaves for treedef"
        raise ValueError(msg) from err


def _unflatten_dict(t_def: TreeDef, leaves_it: Iterator[OpValue]) -> dict[str | int, OpValue]:
    """Unflatten a ``dict`` node.

    Args:
        t_def (TreeDef): The ``TreeDef`` for this dictionary node.
        leaves_it (Iterator[OpValue]): Iterator of leaves.

    Returns:
        dict[str | int, OpValue]: Reconstructed dictionary.

    Raises:
        ValueError: If the dictionary ``TreeDef`` has no keys.
    """
    if t_def.keys is None:
        msg_0 = "Dict treedef must have keys"
        raise ValueError(msg_0)
    return {k: _unflatten_node(c_def, leaves_it) for k, c_def in zip(t_def.keys, t_def.children_defs)}


def _unflatten_sequence(t_def: TreeDef, leaves_it: Iterator[OpValue]) -> OpValue:
    """Unflatten a ``list``, ``tuple``, or ``NamedTuple`` node.

    Args:
        t_def (TreeDef): The ``TreeDef`` for this sequence node.
        leaves_it (Iterator[OpValue]): Iterator of leaves.

    Returns:
        OpValue: Reconstructed sequence container.
    """
    children = [_unflatten_node(c_def, leaves_it) for c_def in t_def.children_defs]
    if t_def.node_type is list:
        return list(children)
    if t_def.node_type is tuple:
        return tuple(children)
    named_res: OpValue = t_def.node_type(*children)
    return named_res


def _unflatten_node(t_def: TreeDef, leaves_it: Iterator[OpValue]) -> OpValue:
    """Unflatten a single node based on its ``TreeDef``.

    Args:
        t_def (TreeDef): The ``TreeDef`` for this node.
        leaves_it (Iterator[OpValue]): Iterator of leaves.

    Returns:
        OpValue: Reconstructed node or leaf.

    Raises:
        ValueError: If ``t_def.node_type`` is unsupported.
    """
    if t_def.node_type is type(None):
        return _unflatten_leaf(leaves_it)
    custom_entry = _PYTREE_REGISTRY.get(t_def.node_type)
    if custom_entry is not None:
        children = [_unflatten_node(c_def, leaves_it) for c_def in t_def.children_defs]
        return custom_entry.unflatten_erased(t_def.aux_data, children)
    if t_def.node_type is dict:
        return _unflatten_dict(t_def, leaves_it)
    if t_def.node_type in (list, tuple) or issubclass(t_def.node_type, tuple):
        return _unflatten_sequence(t_def, leaves_it)

    msg = f"Unsupported treedef node_type: {t_def.node_type}"
    raise ValueError(msg)


def tree_unflatten(treedef: TreeDef, leaves: Iterable[OpValue]) -> OpValue:
    """Reconstruct a PyTree from a ``TreeDef`` and an iterable of leaves.

    Args:
        treedef (TreeDef): Structural ``TreeDef`` to reconstruct.
        leaves (Iterable[OpValue]): Sequence or iterator of leaf values.

    Returns:
        OpValue: Reconstructed PyTree.

    Raises:
        ValueError: If too few or too many leaves are supplied.
    """
    _ensure_builtin_pytree_instances()
    leaves_it = iter(leaves)
    res = _unflatten_node(treedef, leaves_it)

    for _ in leaves_it:
        msg = "Too many leaves for treedef"
        raise ValueError(msg)

    return res


def tree_map(f: OpValue, tree: PyTree, *rest: PyTree) -> OpValue:
    """Map a function over the leaves of one or more isomorphic PyTrees (Haskell ``Functor`` / ``Applicative``).

    Args:
        f (OpValue): Leaf transformation callable.
        tree (PyTree): Primary PyTree.
        *rest (PyTree): Additional PyTrees sharing identical ``TreeDef`` structure.

    Returns:
        OpValue: New PyTree with ``f`` applied across corresponding leaves.

    Raises:
        ValueError: If the supplied trees do not share the exact same ``TreeDef``.
    """
    leaves, treedef = tree_flatten(tree)
    rest_leaves = []
    for r in rest:
        r_leaves, r_treedef = tree_flatten(r)
        if r_treedef != treedef:
            msg = "All trees must have the same structure"
            raise ValueError(msg)
        rest_leaves.append(r_leaves)
    _callable_fn_1: OpValue = f
    assert is_variadic_callable(_callable_fn_1)
    mapped_leaves = [_callable_fn_1(leaf, *[rl[i] for rl in rest_leaves]) for i, leaf in enumerate(leaves)]
    return tree_unflatten(treedef, mapped_leaves)


def tree_map_result(f: OpValue, tree: PyTree, *rest: PyTree) -> Result[ValueError, OpValue]:
    """Total ``Result``-returning ``Functor`` / ``Applicative`` leaf map without raising on structure mismatch.

    Args:
        f (OpValue): Leaf transformation callable.
        tree (PyTree): Primary PyTree.
        *rest (PyTree): Additional PyTrees expected to share identical ``TreeDef`` structure.

    Returns:
        Result[ValueError, OpValue]: ``Ok(mapped_tree)`` on matching structure or ``Err(ValueError)``.
    """
    try:
        return Ok(tree_map(f, tree, *rest))
    except ValueError as exc:
        return Err(exc)


def tree_leaves(tree: PyTree) -> list[OpValue]:
    """Extract the flat leaf list of a PyTree.

    Args:
        tree (PyTree): Input PyTree.

    Returns:
        list[OpValue]: Ordered list of leaves.
    """
    leaves, _ = tree_flatten(tree)
    return leaves


def tree_structure(tree: PyTree) -> TreeDef:
    """Extract the ``TreeDef`` structure of a PyTree.

    Args:
        tree (PyTree): Input PyTree.

    Returns:
        TreeDef: Structural ``TreeDef``.
    """
    _, treedef = tree_flatten(tree)
    return treedef


def tree_fold_map(
    monoid: Monoid[_M],
    fn: Callable[[OpValue], _M],
    tree: PyTree,
) -> _M:
    """Map each leaf of ``tree`` into ``monoid`` and combine left-to-right (Haskell ``Foldable.foldMap``).

    Args:
        monoid (Monoid[_M]): Target monoid witness providing ``empty`` and associative ``combine``.
        fn (Callable[[OpValue], _M]): Leaf projection into the monoid carrier ``_M``.
        tree (PyTree): Input PyTree to fold.

    Returns:
        _M: Combined monoidal summary across all leaves.
    """
    acc: _M = monoid.empty()
    for leaf in tree_leaves(tree):
        acc = monoid.semigroup.combine(acc, fn(leaf))
    return acc


def tree_all(tree: PyTree) -> bool:
    """Check whether all leaves of ``tree`` are truthy via ``BOOL_ALL_MONOID``.

    Args:
        tree (PyTree): Input PyTree to inspect.

    Returns:
        bool: ``True`` if every leaf is truthy.
    """
    return builtins.bool(tree_fold_map(BOOL_ALL_MONOID, builtins.bool, tree))


def tree_any(tree: PyTree) -> bool:
    """Check whether at least one leaf of ``tree`` is truthy via ``BOOL_ANY_MONOID``.

    Args:
        tree (PyTree): Input PyTree to inspect.

    Returns:
        bool: ``True`` if any leaf is truthy.
    """
    return builtins.bool(tree_fold_map(BOOL_ANY_MONOID, builtins.bool, tree))


def tree_all_by(pred: Callable[[OpValue], bool], tree: PyTree) -> bool:
    """Check whether ``pred(leaf)`` holds for all leaves of ``tree`` via ``BOOL_ALL_MONOID``.

    Args:
        pred (Callable[[OpValue], bool]): Leaf predicate function.
        tree (PyTree): Input PyTree to inspect.

    Returns:
        bool: ``True`` if ``pred`` returns ``True`` for every leaf.
    """
    return tree_fold_map(BOOL_ALL_MONOID, pred, tree)


def tree_any_by(pred: Callable[[OpValue], bool], tree: PyTree) -> bool:
    """Check whether ``pred(leaf)`` holds for at least one leaf of ``tree`` via ``BOOL_ANY_MONOID``.

    Args:
        pred (Callable[[OpValue], bool]): Leaf predicate function.
        tree (PyTree): Input PyTree to inspect.

    Returns:
        bool: ``True`` if ``pred`` returns ``True`` for any leaf.
    """
    return tree_fold_map(BOOL_ANY_MONOID, pred, tree)


def tree_traverse_result(
    fn: Callable[[OpValue], Result[_E, OpValue]],
    tree: PyTree,
) -> Result[_E, OpValue]:
    """Effectfully traverse ``tree`` in ``Result``, short-circuiting on the first ``Err`` (Haskell ``Traversable``).

    Args:
        fn (Callable[[OpValue], Result[_E, OpValue]]): Fallible leaf transformation returning ``Result``.
        tree (PyTree): Input PyTree to traverse.

    Returns:
        Result[_E, OpValue]: ``Ok(reconstructed_tree)`` if all leaves succeed, else first ``Err``.
    """
    leaves, treedef = tree_flatten(tree)
    traversed = traverse_result(leaves, fn)
    if isinstance(traversed, Err):
        return traversed
    return Ok(tree_unflatten(treedef, traversed.value))


def tree_traverse_validation(
    sg: Semigroup[_E],
    fn: Callable[[OpValue], Validation[_E, OpValue]],
    tree: PyTree,
) -> Validation[_E, OpValue]:
    """Applicatively traverse ``tree`` in ``Validation``, accumulating all leaf errors via ``sg``.

    Args:
        sg (Semigroup[_E]): Error accumulation semigroup.
        fn (Callable[[OpValue], Validation[_E, OpValue]]): Validating leaf function.
        tree (PyTree): Input PyTree to validate and transform.

    Returns:
        Validation[_E, OpValue]: ``Valid(reconstructed_tree)`` or ``Invalid`` with all accumulated errors.
    """
    leaves, treedef = tree_flatten(tree)
    traversed = traverse_validation(sg, leaves, fn)
    if isinstance(traversed, Invalid):
        return traversed
    return Valid(tree_unflatten(treedef, traversed.value))


def tree_reduce(
    f: Callable[[OpValue, OpValue], OpValue],
    tree: PyTree,
    initializer: OpValue = None,
) -> OpValue:
    """Reduce a PyTree by applying a binary function across its leaves.

    Args:
        f (Callable[[OpValue, OpValue], OpValue]): Binary reduction function.
        tree (PyTree): Input PyTree to reduce.
        initializer (OpValue): Optional initial accumulator value.

    Returns:
        OpValue: Reduced value across all leaves.
    """
    leaves = tree_leaves(tree)

    if initializer is None:
        return functools.reduce(f, leaves)

    acc: OpValue = initializer
    for leaf in leaves:
        acc = f(acc, leaf)
    return acc


def _count_leaves(t_def: TreeDef) -> int:
    """Count total leaves represented by ``t_def``.

    Args:
        t_def (TreeDef): Structural ``TreeDef``.

    Returns:
        int: Total number of leaf positions.
    """
    if t_def.node_type is type(None):
        return 1
    return sum(_count_leaves(c) for c in t_def.children_defs)


def tree_transpose(
    outer_treedef: TreeDef,
    inner_treedef: TreeDef,
    pytree_to_transpose: PyTree,
) -> OpValue:
    """Transpose a nested PyTree of PyTrees from ``outer[inner]`` to ``inner[outer]``.

    Args:
        outer_treedef (TreeDef): Expected structure of the outer tree.
        inner_treedef (TreeDef): Expected structure of the inner trees.
        pytree_to_transpose (PyTree): Nested PyTree to transpose.

    Returns:
        OpValue: Transposed nested PyTree.

    Raises:
        ValueError: If the leaf count does not match ``outer_size * inner_size``.
    """
    leaves, _ = tree_flatten(pytree_to_transpose)

    outer_size = _count_leaves(outer_treedef)
    inner_size = _count_leaves(inner_treedef)

    if len(leaves) != outer_size * inner_size:
        raise ValueError("Tree size mismatch in tree_transpose")

    transposed_leaves = []
    for j in range(inner_size):
        inner_leaf_components = [leaves[i * inner_size + j] for i in range(outer_size)]
        transposed_leaves.append(tree_unflatten(outer_treedef, inner_leaf_components))

    return tree_unflatten(inner_treedef, transposed_leaves)
