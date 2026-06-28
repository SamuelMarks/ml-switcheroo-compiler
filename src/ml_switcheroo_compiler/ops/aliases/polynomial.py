"""Aliases for polynomial."""

from ml_switcheroo_compiler.ops.base import get_op

poly = get_op("Poly")()
polyadd = get_op("Polyadd")()
polyder = get_op("Polyder")()
polydiv = get_op("Polydiv")()
polyfit = get_op("Polyfit")()
polyint = get_op("Polyint")()
polymul = get_op("Polymul")()
polysub = get_op("Polysub")()
polyval = get_op("Polyval")()
