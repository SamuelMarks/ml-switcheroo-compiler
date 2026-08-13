import os
import struct

from ml_switcheroo_compiler.backends.edge.stablehlo import StableHLOCodeGenerator
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def test_export_mlirbc(tmp_path):
    g = IRGraph()
    n1 = IRNode(id="n1", op_type="Input")
    g.nodes = {"n1": n1}

    gen = StableHLOCodeGenerator(g)
    out_file = os.path.join(tmp_path, "model.mlirbc")

    gen.export_mlirbc(out_file)

    assert os.path.exists(out_file)
    with open(out_file, "rb") as f:
        magic = f.read(4)
        assert magic == b"ML\xefR"  # 4D 4C EF 52

        version = struct.unpack("<B", f.read(1))[0]
        assert version == 1

        length = struct.unpack("<I", f.read(4))[0]
        payload = f.read(length)

        mlir_text = payload.decode("utf-8")
        assert "module" in mlir_text
