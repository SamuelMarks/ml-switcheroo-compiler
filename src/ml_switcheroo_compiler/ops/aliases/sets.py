"""Aliases for sets."""

from ml_switcheroo_compiler.ops.base import get_op


from .common import create_eager_alias

intersect1d = create_eager_alias("intersect1d")


isin = create_eager_alias("isin")


setdiff1d = get_op("Setdiff1d")()
setxor1d = get_op("Setxor1d")()
union1d = get_op("Union1d")()
unique_all = get_op("UniqueAll")()
unique_counts = get_op("UniqueCounts")()
unique_inverse = get_op("UniqueInverse")()
unique_values = get_op("UniqueValues")()

unique = create_eager_alias("unique")
