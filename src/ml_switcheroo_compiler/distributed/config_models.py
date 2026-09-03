"""Pydantic models for pipeline topologies."""

from typing import Optional

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


class SchedulePhaseConfig(BaseModel):
    """Configuration for a schedule phase."""

    type: str
    operations: list[str]
    count_expression: str  # e.g. "num_stages - 1" or "num_microbatches - num_stages + 1"


class ScheduleConfig(BaseModel):
    """Configuration for a pipeline schedule."""

    phases: list[SchedulePhaseConfig]


class TopologyConfig(BaseModel):
    """Configuration for a specific pipeline topology."""

    microbatch_splitting: MicrobatchSplittingConfig
    mesh_mapping: MeshMappingConfig
    stage_communication: StageCommunicationConfig
    dependencies: list[DependencyConfig] = []
    schedule: Optional[ScheduleConfig] = None


class PipelineTopologiesConfig(RootModel[dict[str, TopologyConfig]]):
    """Configuration for all pipeline topologies."""

    root: dict[str, TopologyConfig]

    def dict(self, *args: object, **kwargs: object) -> dict:
        """Return dict representation."""
        return super().model_dump(*args, **kwargs)

    def items(self):
        """Return items from the underlying dictionary."""
        return self.root.items()

    def get(self, key: str, default: object = None) -> Optional[TopologyConfig]:
        """Get topology config by key."""
        return self.root.get(key, default)
