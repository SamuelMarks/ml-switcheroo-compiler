"""PyTree utilities for structural manipulations."""

from __future__ import annotations

from typing import Any, Callable


class TreeDef:
    """A definition of a tree structure."""

    def __init__(
        self,
        node_type: type,
        children_defs: list[TreeDef] | None = None,
        keys: list[Any] | None = None,
    ) -> None:
        """Initialize."""
        self.node_type = node_type
        self.children_defs = children_defs or []
        self.keys = keys

    def __repr__(self) -> str:
        """Repr."""
        return f"TreeDef({self.node_type.__name__}, {self.children_defs})"

    def __eq__(self, other: object) -> bool:
        """Equality."""
        if not isinstance(other, TreeDef):
            return False
        return (
            self.node_type == other.node_type
            and self.children_defs == other.children_defs
            and self.keys == other.keys
        )


def tree_flatten(tree: object) -> tuple[list[object], TreeDef]:
    """Flattens a PyTree into a list of leaves and an auxiliary treedef."""
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


def tree_unflatten(treedef: TreeDef, leaves: list[object]) -> object:
    """Reconstructs a PyTree from a treedef and a list of leaves."""
    leaves_it = iter(leaves)

    def _unflatten(t_def: TreeDef) -> object:
        if t_def.node_type is type(None):
            try:
                return next(leaves_it)
            except StopIteration:
                msg = "Too few leaves for treedef"
                raise ValueError(msg) from None
        elif t_def.node_type is dict:
            if t_def.keys is None:
                msg_0 = "Dict treedef must have keys"
                raise ValueError(msg_0)
            return {k: _unflatten(c_def) for k, c_def in zip(t_def.keys, t_def.children_defs)}
        elif t_def.node_type in (list, tuple):
            children = [_unflatten(c_def) for c_def in t_def.children_defs]
            return t_def.node_type(children)
        else:
            msg = f"Unsupported treedef node_type: {t_def.node_type}"
            raise ValueError(msg)

    res = _unflatten(treedef)
    try:
        next(leaves_it)
        msg = "Too many leaves for treedef"
        raise ValueError(msg)
    except StopIteration:
        pass
    return res


def tree_map(f: Callable, tree: object, *rest: object) -> object:
    """Maps a function over the leaves of a PyTree."""
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
