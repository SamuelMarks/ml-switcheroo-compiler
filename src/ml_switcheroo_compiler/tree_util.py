"""PyTree utilities for structural manipulations."""

from __future__ import annotations

import builtins
import functools
from collections.abc import Iterable, Iterator
from typing import Callable, TypeVar, Union, cast

from ml_switcheroo_compiler.core.shape import TensorOrScalar

T = TypeVar("T")
PyTree = Union[TensorOrScalar, "TreeDef"]


class TreeDef:
    """Provide a definition of a tree structure."""

    def __init__(
        self,
        node_type: type,
        children_defs: list[TreeDef] | None = None,
        keys: list[str | int] | None = None,
    ) -> None:
        """Initialize.

        Args:
            node_type (type): The type of the node.
            children_defs (list[TreeDef] | None): The children definitions.
            keys (list[str | int] | None): The keys for dictionary nodes.
        """
        self.node_type = node_type
        self.children_defs = children_defs or []
        self.keys = keys

    def __repr__(self) -> str:
        """Return string representation.

        Returns:
            str: The string representation of the TreeDef.
        """
        return f"TreeDef({self.node_type.__name__}, {self.children_defs})"

    def __hash__(self) -> int:
        """Compute hash of the `TreeDef` structure.

        Returns:
            int: Structural hash integer.
        """
        return hash(str(self.node_type) + str(self.children_defs))

    def __eq__(self, other: object) -> bool:
        """Check equality with another TreeDef.

        Args:
            other (TensorOrScalar | TreeDef): The other object to compare with.

        Returns:
            bool: True if equal, False otherwise.
        """
        if not isinstance(other, TreeDef):
            return False
        return self.node_type == other.node_type and self.children_defs == other.children_defs and self.keys == other.keys

    _eq = __eq__


def tree_flatten(tree: PyTree) -> tuple[list[TensorOrScalar], TreeDef]:
    """Flatten a PyTree into a list of leaves and an auxiliary treedef.

    Args:
        tree (PyTree): The tree to flatten.

    Returns:
        tuple[list[TensorOrScalar], TreeDef]: The flattened leaves and the tree definition.
    """
    from typing import cast

    if isinstance(tree, dict):
        keys = sorted(tree.keys())
        leaves: list[TensorOrScalar] = []
        children_defs: list[TreeDef] = []
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
    return [cast(TensorOrScalar, tree)], TreeDef(type(None))


def _unflatten_leaf(leaves_it: Iterator[TensorOrScalar]) -> TensorOrScalar:
    """Unflatten a leaf node.

    Args:
        leaves_it (Iterator[TensorOrScalar]): Iterator of leaves.

    Returns:
        TensorOrScalar: The leaf.

    Raises:
        ValueError: If there are too few leaves.
    """
    try:
        return next(leaves_it)
    except StopIteration:
        msg = "Too few leaves for treedef"
        raise ValueError(msg) from None


def _unflatten_dict(t_def: TreeDef, leaves_it: Iterator[TensorOrScalar]) -> dict[str | int, TensorOrScalar]:
    """Unflatten a dict node.

    Args:
        t_def (TreeDef): The TreeDef for this node.
        leaves_it (Iterator[TensorOrScalar]): Iterator of leaves.

    Returns:
        dict[str | int, TensorOrScalar]: The unflattened dict.

    Raises:
        ValueError: If the dictionary TreeDef has no keys.
    """
    if t_def.keys is None:
        msg_0 = "Dict treedef must have keys"
        raise ValueError(msg_0)
    return {k: _unflatten_node(c_def, leaves_it) for k, c_def in zip(t_def.keys, t_def.children_defs)}


def _unflatten_sequence(t_def: TreeDef, leaves_it: Iterator[TensorOrScalar]) -> TensorOrScalar:
    """Unflatten a list or tuple node.

    Args:
        t_def (TreeDef): The TreeDef for this node.
        leaves_it (Iterator[TensorOrScalar]): Iterator of leaves.

    Returns:
        TensorOrScalar: The unflattened sequence.
    """
    children = [_unflatten_node(c_def, leaves_it) for c_def in t_def.children_defs]
    res: TensorOrScalar = t_def.node_type(children)
    return res


def _unflatten_node(t_def: TreeDef, leaves_it: Iterator[TensorOrScalar]) -> TensorOrScalar:
    """Unflatten a single node based on its TreeDef.

    Args:
        t_def (TreeDef): The TreeDef for this node.
        leaves_it (Iterator[TensorOrScalar]): Iterator of leaves.

    Returns:
        TensorOrScalar: The unflattened node.

    Raises:
        ValueError: If the node_type is unsupported.
    """
    if t_def.node_type is type(None):
        return _unflatten_leaf(leaves_it)
    if t_def.node_type is dict:
        return _unflatten_dict(t_def, leaves_it)
    if t_def.node_type in (list, tuple):
        return _unflatten_sequence(t_def, leaves_it)

    msg = f"Unsupported treedef node_type: {t_def.node_type}"
    raise ValueError(msg)


def tree_unflatten(treedef: TreeDef, leaves: Iterable[TensorOrScalar]) -> TensorOrScalar:
    """Reconstruct a PyTree from a treedef and a list of leaves.

    Args:
        treedef (TreeDef): The TreeDef to use for reconstruction.
        leaves (Iterable[TensorOrScalar]): The list of leaves to unflatten.

    Returns:
        TensorOrScalar: The reconstructed PyTree.

    Raises:
        ValueError: If there are too many leaves.
    """
    leaves_it = iter(leaves)
    res = _unflatten_node(treedef, leaves_it)

    try:
        next(leaves_it)
        msg = "Too many leaves for treedef"
        raise ValueError(msg)
    except StopIteration:
        _ = None

    return res


def tree_map(f: Callable[..., TensorOrScalar], tree: PyTree, *rest: PyTree) -> TensorOrScalar:
    """Map a function over the leaves of a PyTree.

    Args:
        f (Callable[..., TensorOrScalar]): The function to apply.
        tree (PyTree): The primary PyTree.
        *rest (PyTree): Additional PyTrees of the same structure.

    Returns:
        TensorOrScalar: A new PyTree with the function applied to its leaves.

    Raises:
        ValueError: If the trees do not have the same structure.
    """
    leaves, treedef = tree_flatten(tree)
    rest_leaves = []
    for r in rest:
        r_leaves, r_treedef = tree_flatten(r)
        if r_treedef != treedef:
            msg = "All trees must have the same structure"
            raise ValueError(msg)
        rest_leaves.append(r_leaves)

    mapped_leaves = [f(leaf, *[rl[i] for rl in rest_leaves]) for i, leaf in enumerate(leaves)]
    return tree_unflatten(treedef, mapped_leaves)


def tree_leaves(tree: PyTree) -> list[TensorOrScalar]:
    """Get the leaves of a PyTree.

    Args:
        tree (PyTree): The tree to extract leaves from.

    Returns:
        list[TensorOrScalar]: A list of leaves.
    """
    leaves, _ = tree_flatten(tree)

    return leaves


def tree_structure(tree: PyTree) -> TreeDef:
    """Get the structure of a PyTree.

    Args:
        tree (PyTree): The tree to extract structure from.

    Returns:
        TreeDef: The tree structure.
    """
    _, treedef = tree_flatten(tree)

    return treedef


def tree_all(tree: PyTree) -> bool:
    """Check if all leaves of a PyTree are truthy.

    Args:
        tree (PyTree): The tree to check.

    Returns:
        bool: True if all leaves are truthy.
    """
    return builtins.all(tree_leaves(tree))


def tree_reduce(
    f: Callable[[TensorOrScalar, TensorOrScalar], TensorOrScalar],
    tree: PyTree,
    initializer: TensorOrScalar = None,
) -> TensorOrScalar:
    """Reduce a PyTree by applying a function over its leaves.

    Args:
        f (Callable[[TensorOrScalar, TensorOrScalar], TensorOrScalar]): The reduction function.
        tree (PyTree): The tree to reduce.
        initializer (TensorOrScalar): Optional initial value.

    Returns:
        TensorOrScalar: The reduced value.
    """
    leaves = tree_leaves(tree)

    if initializer is None:
        return functools.reduce(f, leaves)

    return cast(TensorOrScalar, getattr(functools, "reduce")(f, leaves, initializer))


def _count_leaves(t_def: TreeDef) -> int:
    """Count leaves in a treedef.

    Args:
        t_def (TreeDef): The TreeDef to count leaves for.

    Returns:
        int: The number of leaves.
    """
    if t_def.node_type is type(None):
        return 1
    return sum(_count_leaves(c) for c in t_def.children_defs)


def tree_transpose(
    outer_treedef: TreeDef,
    inner_treedef: TreeDef,
    pytree_to_transpose: PyTree,
) -> TensorOrScalar:
    """Transpose a PyTree of PyTrees.

    Args:
        outer_treedef (TreeDef): The expected structure of the outer tree.
        inner_treedef (TreeDef): The expected structure of the inner trees.
        pytree_to_transpose (PyTree): The tree to transpose.

    Returns:
        TensorOrScalar: The transposed tree.

    Raises:
        ValueError: If there is a tree size mismatch.
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
