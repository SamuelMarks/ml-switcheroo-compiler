"""Export API."""

from typing import Callable, Optional  # pragma: no cover


class ExportArchive:  # pragma: no cover
    """ExportArchive is used to write SavedModel artifacts (e.g. for TF Serving)."""

    def __init__(self) -> None:  # pragma: no cover
        """Initialize."""
        self.trackables: dict[int, object] = {}  # pragma: no cover
        self.endpoints: dict[str, Callable[..., object]] = {}  # pragma: no cover

    def track(self, resource: object) -> None:  # pragma: no cover
        """Track a resource.

        Args:
            resource: Resource to track.
        """
        self.trackables[id(resource)] = resource  # pragma: no cover

    def add_endpoint(self, name: str, fn: Callable[..., object], **kwargs: object) -> None:  # pragma: no cover
        """Add an endpoint.

        Args:
            name: Name of endpoint.
            fn: Function endpoint.
            kwargs: Kwargs.
        """
        self.endpoints[name] = fn  # pragma: no cover

    def write_out(self, filepath: str, options: Optional[object] = None) -> None:  # pragma: no cover
        """Write the archive to a directory.

        Args:
            filepath: Target path.
            options: Save options.
        """
        pass  # pragma: no cover

    def add_variable_collection(self, name: str, variables: object) -> None:  # pragma: no cover
        """Add a variable collection.

        Args:
            name: Collection name.
            variables: Variables.
        """
        pass  # pragma: no cover
