"""Pydantic models for pipeline topologies."""

from pydantic import BaseModel, RootModel


class MicrobatchSplittingConfig(BaseModel):
    """Configuration for splitting microbatches."""

    num_microbatches: int
    strategy: str


class MeshMappingConfig(BaseModel):
    """Configuration for mapping stages to devices."""

    devices_per_stage: int


class StageCommunicationConfig(BaseModel):
    """Configuration for communication between pipeline stages."""

    protocol: str


class DependencyConfig(BaseModel):
    """Configuration for synchronization dependencies."""

    source_stage: str
    target_stage: str
    offset_mb: int


class TopologyConfig(BaseModel):
    """Configuration for a specific pipeline topology."""

    microbatch_splitting: MicrobatchSplittingConfig
    mesh_mapping: MeshMappingConfig
    stage_communication: StageCommunicationConfig
    dependencies: list[DependencyConfig] = []


class PipelineTopologiesConfig(RootModel[dict[str, TopologyConfig]]):
    """Configuration for all pipeline topologies."""

    root: dict[str, TopologyConfig]

    def dict(self, *args, **kwargs):
        """Return dict representation."""
        return super().model_dump(*args, **kwargs)

    def items(self):
        """Return items from the underlying dictionary."""
        return self.root.items()

    def get(self, key: str, default=None):
        """Get topology config by key."""
        return self.root.get(key, default)
