"""Unit tests for composable YAML/bytes/proven IR Codecs, FrozenTensorArchive Codecs, and batch endpoint validation."""

from __future__ import annotations

import json
import os
from pathlib import Path

from ml_switcheroo_ir import Err, Invalid, IRGraph, IRNode, Ok, Valid

from ml_switcheroo_compiler.backends.snapshot_grounding import (
    SnapshotGroundingEngine,
    load_backend_snapshot_targets,
    validate_endpoints_batch_checked,
)
from ml_switcheroo_compiler.serialization import (
    COMPILER_IR_BYTES_CODEC,
    COMPILER_IR_YAML_CODEC,
    COMPILER_PROVEN_ACYCLIC_JSON_CODEC,
    FROZEN_TENSOR_ARCHIVE_CODEC,
    FROZEN_TENSOR_ARCHIVE_EQ,
    TENSOR_MAP_BYTES_CODEC,
    TENSOR_MAP_JSON_CODEC,
    UTF8_BYTES_STR_CODEC,
    FrozenTensorArchive,
)
from ml_switcheroo_compiler.transforms import LOGICAL_GRAPH_EQ


def test_utf8_and_tensor_archive_codecs() -> None:
    """Verify UTF8_BYTES_STR_CODEC, TENSOR_MAP_{JSON,BYTES}_CODEC, and FROZEN_TENSOR_ARCHIVE_CODEC."""
    enc_utf8 = UTF8_BYTES_STR_CODEC.encode("hello-λ")
    dec_utf8 = UTF8_BYTES_STR_CODEC.decode(enc_utf8)
    assert isinstance(dec_utf8, Ok) and dec_utf8.value == "hello-λ"

    bad_utf8 = UTF8_BYTES_STR_CODEC.decode(b"\xff\xfe")
    assert isinstance(bad_utf8, Err)

    weights = {"w2": (1.0, 2.5), "w1": (-0.5, 0.0, 3.0)}
    archive = FrozenTensorArchive.from_dict(weights)
    assert FROZEN_TENSOR_ARCHIVE_EQ.eq(archive, FrozenTensorArchive.from_dict(weights))
    assert archive.to_dict() == weights

    wire_bytes = FROZEN_TENSOR_ARCHIVE_CODEC.encode(archive)
    roundtripped = FROZEN_TENSOR_ARCHIVE_CODEC.decode(wire_bytes)
    assert isinstance(roundtripped, Ok) and FROZEN_TENSOR_ARCHIVE_EQ.eq(roundtripped.value, archive)

    map_rt = TENSOR_MAP_BYTES_CODEC.decode(TENSOR_MAP_BYTES_CODEC.encode(weights))
    assert isinstance(map_rt, Ok) and map_rt.value == weights

    # Validation error branches in _decode_tensor_map_json
    assert isinstance(TENSOR_MAP_JSON_CODEC.decode("{bad_json"), Err)
    assert isinstance(TENSOR_MAP_JSON_CODEC.decode("[1, 2, 3]"), Err)
    assert isinstance(TENSOR_MAP_JSON_CODEC.decode('{"w": "not_a_list"}'), Err)
    assert isinstance(TENSOR_MAP_JSON_CODEC.decode('{"w": [true]}'), Err)
    assert isinstance(TENSOR_MAP_JSON_CODEC.decode('{"w": [NaN]}'), Err)


def test_ir_yaml_bytes_and_proven_acyclic_codecs() -> None:
    """Verify COMPILER_IR_YAML_CODEC, COMPILER_IR_BYTES_CODEC, and COMPILER_PROVEN_ACYCLIC_JSON_CODEC."""
    dag = IRGraph(
        name="codec_dag",
        inputs=["x"],
        outputs=["y"],
        nodes=[
            IRNode(id="x", op_type="Input", inputs=[], shape_metadata=(2, 3)),
            IRNode(id="y", op_type="Relu", inputs=["x"], shape_metadata=(2, 3)),
        ],
    )
    yaml_rt = COMPILER_IR_YAML_CODEC.decode(COMPILER_IR_YAML_CODEC.encode(dag))
    assert isinstance(yaml_rt, Ok) and LOGICAL_GRAPH_EQ.eq(yaml_rt.value, dag)

    bytes_rt = COMPILER_IR_BYTES_CODEC.decode(COMPILER_IR_BYTES_CODEC.encode(dag))
    assert isinstance(bytes_rt, Ok) and LOGICAL_GRAPH_EQ.eq(bytes_rt.value, dag)

    raw_dag_json = COMPILER_IR_BYTES_CODEC.encode(dag).decode("utf-8")
    proven_rt_in = COMPILER_PROVEN_ACYCLIC_JSON_CODEC.decode(raw_dag_json)
    assert isinstance(proven_rt_in, Ok)
    re_encoded = COMPILER_PROVEN_ACYCLIC_JSON_CODEC.encode(proven_rt_in.value)
    proven_rt_out = COMPILER_PROVEN_ACYCLIC_JSON_CODEC.decode(re_encoded)
    assert isinstance(proven_rt_out, Ok) and [n.id for n in proven_rt_out.value.ordered_nodes] == ["x", "y"]

    cyclic = IRGraph(
        name="cyclic_codec",
        nodes=[
            IRNode(id="a", op_type="Relu", inputs=["b"]),
            IRNode(id="b", op_type="Relu", inputs=["a"]),
        ],
    )
    cyclic_json = COMPILER_IR_BYTES_CODEC.encode(cyclic).decode("utf-8")
    proven_err = COMPILER_PROVEN_ACYCLIC_JSON_CODEC.decode(cyclic_json)
    assert isinstance(proven_err, Err) and proven_err.error.path == ("nodes",)


def test_snapshot_grounding_validate_endpoints_batch_checked(tmp_path: Path) -> None:
    """Verify validate_endpoints_batch_checked accumulates multiple invalid endpoints and Levenshtein hints.

    Args:
        tmp_path (Path): Temporary pytest directory fixture.
    """
    snap_dir = tmp_path / "snapshots"
    snap_dir.mkdir()
    snap_file = snap_dir / "torch_vtest.json"
    snap_file.write_text(
        json.dumps(
            {
                "categories": {
                    "math": [
                        {"name": "add", "api_path": "torch.add", "params": []},
                        {"name": "matmul", "api_path": "torch.matmul", "params": []},
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    cfg_file = tmp_path / "targets.yaml"
    cfg_file.write_text(
        os.linesep.join(
            [
                "targets:",
                "  pytorch:",
                "    framework: torch",
                "    snapshot_glob: 'torch_*.json'",
                "    canonical_roots: ['torch']",
                "    status: 'available'",
            ]
        ),
        encoding="utf-8",
    )
    engine = SnapshotGroundingEngine(
        snapshot_dir=str(snap_dir),
        config=load_backend_snapshot_targets(str(cfg_file)),
    )

    ok_batch = engine.validate_endpoints_batch_checked("pytorch", ["torch.add", "torch.matmul"])
    assert isinstance(ok_batch, Valid) and ok_batch.value == ("torch.add", "torch.matmul")

    bad_batch = engine.validate_endpoints_batch_checked(
        "pytorch",
        ["torch.add", "torch.adx", "torch.completely_unrelated_symbol_name"],
    )
    assert isinstance(bad_batch, Invalid)
    assert len(bad_batch.errors) == 2
    assert "did you mean 'torch.add'?" in bad_batch.errors[0]
    assert "completely_unrelated_symbol_name" in bad_batch.errors[1]

    fn_batch = validate_endpoints_batch_checked("pytorch", ["torch.add"], engine=engine)
    assert isinstance(fn_batch, Valid) and fn_batch.value == ("torch.add",)
    default_batch = validate_endpoints_batch_checked("numpy", ["numpy.add"])
    assert isinstance(default_batch, Valid) and default_batch.value == ("numpy.add",)
