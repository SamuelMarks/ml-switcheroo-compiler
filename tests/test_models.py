"""Pydantic models for test manifests."""

from pydantic import BaseModel


class AllOpsManifest(BaseModel):
    """Manifest of all expected operation names."""

    all_ops: list[str]
