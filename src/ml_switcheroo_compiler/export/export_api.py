"""Export API."""

from typing import Callable, Optional


class ExportArchive:
    """ExportArchive is used to write SavedModel artifacts (e.g. for TF Serving)."""

    def __init__(self) -> None:
        """Initialize."""
        self.trackables: dict[int, object] = {}
        self.endpoints: dict[str, Callable[..., object]] = {}

    def track(self, resource: object) -> None:
        """Track a resource.

        Args:
            resource: Resource to track.
        """
        self.trackables[id(resource)] = resource

    def add_endpoint(self, name: str, fn: Callable[..., object], **kwargs: object) -> None:
        """Add an endpoint.

        Args:
            name: Name of endpoint.
            fn: Function endpoint.
            kwargs: Kwargs.
        """
        self.endpoints[name] = fn

    def write_out(self, filepath: str, options: Optional[object] = None) -> None:
        """Write the archive to a directory.

        Args:
            filepath: Target path.
            options: Save options.
        """
        import os

        os.makedirs(filepath, exist_ok=True)
        # We would typically save the SavedModel protobuf here
        with open(os.path.join(filepath, "saved_model.pb"), "wb") as f:
            f.write(b"")

    def add_variable_collection(self, name: str, variables: object) -> None:
        """Add a variable collection.

        Args:
            name: Collection name.
            variables: Variables.
        """
        if not hasattr(self, "collections"):
            self.collections = {}
        self.collections[name] = variables
