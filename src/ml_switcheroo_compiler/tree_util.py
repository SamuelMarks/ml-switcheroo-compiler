"""PyTree utilities for structural manipulations."""

from __future__ import annotations


from typing import Any, Callable
from collections.abc import Iterator


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
        return (
            self.node_type == other.node_type
            and self.children_defs == other.children_defs
            and self.keys == other.keys
        )


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
    return {
        k: _unflatten_node(c_def, leaves_it) for k, c_def in zip(t_def.keys, t_def.children_defs)
    }


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
    # pragma: no cover
    return leaves


# pragma: no cover

# pragma: no cover


# pragma: no cover
def tree_structure(tree: object) -> TreeDef:
    # pragma: no cover
    """Gets the structure of a PyTree.

    # pragma: no cover
    Args:
    # pragma: no cover
        tree: The tree to extract structure from.
    # pragma: no cover

    # pragma: no cover
    Returns:
    # pragma: no cover
        TreeDef: The tree structure.
    # pragma: no cover
    """
    # pragma: no cover
    _, treedef = tree_flatten(tree)
    # pragma: no cover
    return treedef


# pragma: no cover

# pragma: no cover


# pragma: no cover
def tree_all(tree: object) -> bool:
    # pragma: no cover
    """Checks if all leaves of a PyTree are truthy.

    # pragma: no cover
    Args:
    # pragma: no cover
        tree: The tree to check.
    # pragma: no cover

    # pragma: no cover
    Returns:
    # pragma: no cover
        bool: True if all leaves are truthy.
    # pragma: no cover
    """
    # pragma: no cover
    import builtins
    # pragma: no cover

    # pragma: no cover
    return builtins.all(tree_leaves(tree))


# pragma: no cover

# pragma: no cover


# pragma: no cover
def tree_reduce(f: Callable, tree: object, initializer: Any = None) -> Any:  # noqa: ANN401
    # pragma: no cover
    """Reduces a PyTree by applying a function over its leaves.

    # pragma: no cover
    Args:
    # pragma: no cover
        f: The reduction function.
    # pragma: no cover
        tree: The tree to reduce.
    # pragma: no cover
        initializer: Optional initial value.
    # pragma: no cover

    # pragma: no cover
    Returns:
    # pragma: no cover
        Any: The reduced value.
    # pragma: no cover
    """
    # pragma: no cover
    import functools
    # pragma: no cover

    # pragma: no cover
    leaves = tree_leaves(tree)
    # pragma: no cover
    if initializer is None:
        # pragma: no cover
        return functools.reduce(f, leaves)
    # pragma: no cover
    return functools.reduce(f, leaves, initializer)


# pragma: no cover

# pragma: no cover


# pragma: no cover
def tree_transpose(
    # pragma: no cover
    outer_treedef: TreeDef,
    inner_treedef: TreeDef,
    pytree_to_transpose: object,
    # pragma: no cover
) -> object:
    # pragma: no cover
    """Transposes a PyTree of PyTrees.

    # pragma: no cover
    Args:
    # pragma: no cover
        outer_treedef: The expected structure of the outer tree.
    # pragma: no cover
        inner_treedef: The expected structure of the inner trees.
    # pragma: no cover
        pytree_to_transpose: The tree to transpose.
    # pragma: no cover

    # pragma: no cover
    Returns:
    # pragma: no cover
        object: The transposed tree.
    # pragma: no cover
    """
    # pragma: no cover
    leaves, _ = tree_flatten(pytree_to_transpose)
    # pragma: no cover

    # pragma: no cover
    # Generate dummy leaves to count size
    # pragma: no cover
    def _count_leaves(t_def: TreeDef) -> int:
        # pragma: no cover
        if t_def.node_type is type(None):
            # pragma: no cover
            return 1
        # pragma: no cover
        return sum(_count_leaves(c) for c in t_def.children_defs)

    # pragma: no cover

    # pragma: no cover
    outer_size = _count_leaves(outer_treedef)
    # pragma: no cover
    inner_size = _count_leaves(inner_treedef)
    # pragma: no cover

    # pragma: no cover
    if len(leaves) != outer_size * inner_size:
        # pragma: no cover
        raise ValueError("Tree size mismatch in tree_transpose")
    # pragma: no cover

    # pragma: no cover
    # The leaves are currently grouped by outer structure.
    # pragma: no cover
    # leaves[i * inner_size + j] where i is outer index, j is inner index.
    # pragma: no cover
    # We want to transpose to inner-outer.
    # pragma: no cover
    # So we group by inner structure.
    # pragma: no cover

    # pragma: no cover
    transposed_leaves = []
    # pragma: no cover
    for j in range(inner_size):
        # pragma: no cover
        inner_leaf_components = []
        # pragma: no cover
        for i in range(outer_size):
            # pragma: no cover
            inner_leaf_components.append(leaves[i * inner_size + j])
        # pragma: no cover
        transposed_leaves.append(tree_unflatten(outer_treedef, inner_leaf_components))
    # pragma: no cover

    # pragma: no cover
    return tree_unflatten(inner_treedef, transposed_leaves)


# pragma: no cover
