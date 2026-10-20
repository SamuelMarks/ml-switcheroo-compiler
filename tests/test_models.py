"""Pydantic models for test manifests."""

from __future__ import annotations

from pydantic import BaseModel


class AllOpsManifest(BaseModel):
    """Manifest of all expected operation names."""

    all_ops: list[str]
