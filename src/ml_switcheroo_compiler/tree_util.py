"""PyTree utilities for structural manipulations."""

from __future__ import annotations

# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
import builtins
import functools
from collections.abc import Iterator
from typing import Any, Callable


class TreeDef:
    """Provide a definition of a tree structure."""

    def __init__(
        self,
        node_type: type,
        children_defs: list[TreeDef] | None = None,
        keys: list[Any] | None = None,
    ) -> None:
        """Initialize.

        Args:
            node_type (type): The type of the node.
            children_defs (list[TreeDef] | None): The children definitions.
            keys (list[Any] | None): The keys for dictionary nodes.
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
        """Evaluate __hash__ operation.

        Returns:
        int: Result.
        """
        return hash(str(self.node_type) + str(self.children_defs))

    def __eq__(self, other: Any) -> bool:
        """Check equality with another TreeDef.

        Args:
            other (object): The other object to compare with.

        Returns:
            bool: True if equal, False otherwise.
        """
        if not isinstance(other, TreeDef):
            return False
        return self.node_type == other.node_type and self.children_defs == other.children_defs and self.keys == other.keys


def tree_flatten(tree: Any) -> tuple[list[Any], TreeDef]:
    """Flatten a PyTree into a list of leaves and an auxiliary treedef.

    Args:
        tree (object): The tree to flatten.

    Returns:
        tuple[list[Any], TreeDef]: The flattened leaves and the tree definition.
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


def _unflatten_leaf(leaves_it: Iterator[Any]) -> Any:
    """Unflatten a leaf node.

    Args:
        leaves_it (Iterator[Any]): Iterator of leaves.

    Returns: Any: The leaf.

    Raises:
        ValueError: If there are too few leaves.
    """
    try:
        return next(leaves_it)
    except StopIteration:
        msg = "Too few leaves for treedef"
        raise ValueError(msg) from None


def _unflatten_dict(t_def: TreeDef, leaves_it: Iterator[Any]) -> Any:
    """Unflatten a dict node.

    Args:
        t_def (TreeDef): The TreeDef for this node.
        leaves_it (Iterator[Any]): Iterator of leaves.

    Returns: Any: The unflattened dict.

    Raises:
        ValueError: If the dictionary TreeDef has no keys.
    """
    if t_def.keys is None:
        msg_0 = "Dict treedef must have keys"
        raise ValueError(msg_0)
    return {k: _unflatten_node(c_def, leaves_it) for k, c_def in zip(t_def.keys, t_def.children_defs)}


def _unflatten_sequence(t_def: TreeDef, leaves_it: Iterator[Any]) -> Any:
    """Unflatten a list or tuple node.

    Args:
        t_def (TreeDef): The TreeDef for this node.
        leaves_it (Iterator[Any]): Iterator of leaves.

    Returns: Any: The unflattened sequence.
    """
    children = [_unflatten_node(c_def, leaves_it) for c_def in t_def.children_defs]
    return t_def.node_type(children)


def _unflatten_node(t_def: TreeDef, leaves_it: Iterator[Any]) -> Any:
    """Unflatten a single node based on its TreeDef.

    Args:
        t_def (TreeDef): The TreeDef for this node.
        leaves_it (Iterator[Any]): Iterator of leaves.

    Returns: Any: The unflattened node.

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


def tree_unflatten(treedef: TreeDef, leaves: list[Any]) -> Any:
    """Reconstruct a PyTree from a treedef and a list of leaves.

    Args:
        treedef (TreeDef): The TreeDef to use for reconstruction.
        leaves (list[Any]): The list of leaves to unflatten.

    Returns: Any: The reconstructed PyTree.

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


def tree_map(f: Callable[..., Any], tree: Any, *rest: Any) -> Any:
    """Map a function over the leaves of a PyTree.

    Args:
        f (Callable): The function to apply.
        tree (object): The primary PyTree.
        *rest (object): Additional PyTrees of the same structure.

    Returns: Any: A new PyTree with the function applied to its leaves.

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


def tree_leaves(tree: Any) -> list[Any]:
    """Get the leaves of a PyTree.

    Args:
        tree (object): The tree to extract leaves from.

    Returns:
        list[Any]: A list of leaves.
    """
    leaves, _ = tree_flatten(tree)

    return leaves


def tree_structure(tree: Any) -> TreeDef:
    """Get the structure of a PyTree.

    Args:
        tree (object): The tree to extract structure from.

    Returns:
        TreeDef: The tree structure.
    """
    _, treedef = tree_flatten(tree)

    return treedef


def tree_all(tree: Any) -> bool:
    """Check if all leaves of a PyTree are truthy.

    Args:
        tree (object): The tree to check.

    Returns:
        bool: True if all leaves are truthy.
    """
    return builtins.all(tree_leaves(tree))


def tree_reduce(f: Callable[..., Any], tree: Any, initializer: Any = None) -> Any:
    """Reduce a PyTree by applying a function over its leaves.

    Args:
        f (Callable): The reduction function.
        tree (object): The tree to reduce.
        initializer (object): Optional initial value.

    Returns: Any: The reduced value.
    """
    leaves = tree_leaves(tree)

    if initializer is None:
        return functools.reduce(f, leaves)

    return functools.reduce(f, leaves, initializer)


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
    pytree_to_transpose: Any,
) -> Any:
    """Transpose a PyTree of PyTrees.

    Args:
        outer_treedef (TreeDef): The expected structure of the outer tree.
        inner_treedef (TreeDef): The expected structure of the inner trees.
        pytree_to_transpose (object): The tree to transpose.

    Returns: Any: The transposed tree.

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
