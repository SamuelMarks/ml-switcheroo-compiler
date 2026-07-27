"""PyTree utilities for structural manipulations."""

from __future__ import annotations

import builtins
import functools
from collections.abc import Iterator
from typing import Any, Callable


class TreeDef:
    """A definition of a tree structure."""

    def __init__(
        self,
        node_type: type,
        children_defs: list[TreeDef] | None = None,
        keys: list[Any] | None = None,
    ) -> None:
        """Initialize.

        Args:
            node_type (type): The node_type parameter for the operation.
            children_defs (list[TreeDef] | None): The children_defs parameter for the operation.
            keys (list[Any] | None): The keys parameter for the operation.
        """
        self.node_type = node_type
        self.children_defs = children_defs or []
        self.keys = keys

    def __repr__(self) -> str:
        """Repr.

        Returns:
            str: The evaluated output resulting from this operation.
        """
        return f"TreeDef({self.node_type.__name__}, {self.children_defs})"

    def __hash__(self) -> int:
        """Execute __hash__.

        Returns:
        Any: The result.
        """
        return hash(str(self.node_type) + str(self.children_defs))

    def __eq__(self, other: object) -> bool:
        """Equality.

        Args:
            other (object): The other parameter for the operation.

        Returns:
            bool: A boolean indicating the result of the check.
        """
        if not isinstance(other, TreeDef):
            return False
        return self.node_type == other.node_type and self.children_defs == other.children_defs and self.keys == other.keys


def tree_flatten(tree: object) -> tuple[list[object], TreeDef]:
    """Flattens a PyTree into a list of leaves and an auxiliary treedef.

    Args:
        tree (object): The tree parameter for the operation.

    Returns:
        tuple[list[object], TreeDef]: The evaluated output resulting from this operation.
    """
    if isinstance(tree, dict):
        keys = sorted(tree.keys())
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
    return [tree], TreeDef(type(None))


def _unflatten_leaf(leaves_it: Iterator[object]) -> object:
    """Unflatten a leaf node.

    Args:
        leaves_it (Iterator[object]): Iterator of leaves.

    Returns:
        object: The leaf.
    """
    try:
        return next(leaves_it)
    except StopIteration:
        msg = "Too few leaves for treedef"
        raise ValueError(msg) from None


def _unflatten_dict(t_def: TreeDef, leaves_it: Iterator[object]) -> object:
    """Unflatten a dict node.

    Args:
        t_def (TreeDef): The TreeDef for this node.
        leaves_it (Iterator[object]): Iterator of leaves.

    Returns:
        object: The unflattened dict.
    """
    if t_def.keys is None:
        msg_0 = "Dict treedef must have keys"
        raise ValueError(msg_0)
    return {k: _unflatten_node(c_def, leaves_it) for k, c_def in zip(t_def.keys, t_def.children_defs)}


def _unflatten_sequence(t_def: TreeDef, leaves_it: Iterator[object]) -> object:
    """Unflatten a list or tuple node.

    Args:
        t_def (TreeDef): The TreeDef for this node.
        leaves_it (Iterator[object]): Iterator of leaves.

    Returns:
        object: The unflattened sequence.
    """
    children = [_unflatten_node(c_def, leaves_it) for c_def in t_def.children_defs]
    return t_def.node_type(children)


def _unflatten_node(t_def: TreeDef, leaves_it: Iterator[object]) -> object:
    """Unflatten a single node based on its TreeDef."""
    if t_def.node_type is type(None):
        return _unflatten_leaf(leaves_it)
    if t_def.node_type is dict:
        return _unflatten_dict(t_def, leaves_it)
    if t_def.node_type in (list, tuple):
        return _unflatten_sequence(t_def, leaves_it)

    msg = f"Unsupported treedef node_type: {t_def.node_type}"
    raise ValueError(msg)


def tree_unflatten(treedef: TreeDef, leaves: list[object]) -> object:
    """Reconstructs a PyTree from a treedef and a list of leaves.

    Args:
        treedef (TreeDef): The treedef parameter for the operation.
        leaves (list[object]): The leaves parameter for the operation.

    Returns:
        object: The evaluated output resulting from this operation.
    """
    leaves_it = iter(leaves)
    res = _unflatten_node(treedef, leaves_it)

    try:
        next(leaves_it)
        msg = "Too many leaves for treedef"
        raise ValueError(msg)
    except StopIteration:
        pass

    return res


def tree_map(f: Callable, tree: object, *rest: object) -> object:
    """Maps a function over the leaves of a PyTree.

    Args:
        f (Callable): The f parameter for the operation.
        tree (object): The tree parameter for the operation.
        *rest: Additional arguments.

    Returns:
        object: The evaluated output resulting from this operation.
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


def tree_leaves(tree: object) -> list[object]:
    """Gets the leaves of a PyTree.

    Args:
        tree: The tree to extract leaves from.

    Returns:
        list[object]: A list of leaves.
    """
    leaves, _ = tree_flatten(tree)

    return leaves


def tree_structure(tree: object) -> TreeDef:
    """Gets the structure of a PyTree.

    Args:
        tree: The tree to extract structure from.



    Returns:
        TreeDef: The tree structure.

    """
    _, treedef = tree_flatten(tree)

    return treedef


def tree_all(tree: object) -> bool:
    """Checks if all leaves of a PyTree are truthy.

    Args:
        tree: The tree to check.



    Returns:
        bool: True if all leaves are truthy.

    """
    return builtins.all(tree_leaves(tree))


def tree_reduce(f: Callable, tree: object, initializer: object = None) -> object:
    """Reduces a PyTree by applying a function over its leaves.

    Args:
        f: The reduction function.

        tree: The tree to reduce.

        initializer: Optional initial value.



    Returns:
        Any: The reduced value.

    """
    leaves = tree_leaves(tree)

    if initializer is None:
        return functools.reduce(f, leaves)

    return functools.reduce(f, leaves, initializer)


def _count_leaves(t_def: TreeDef) -> int:
    """Count leaves in a treedef."""
    if t_def.node_type is type(None):
        return 1
    return sum(_count_leaves(c) for c in t_def.children_defs)


def tree_transpose(
    outer_treedef: TreeDef,
    inner_treedef: TreeDef,
    pytree_to_transpose: object,
) -> object:
    """Transposes a PyTree of PyTrees.

    Args:
        outer_treedef: The expected structure of the outer tree.
        inner_treedef: The expected structure of the inner trees.
        pytree_to_transpose: The tree to transpose.

    Returns:
        object: The transposed tree.
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
