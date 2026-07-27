"""NPZ format serialization."""

from ml_switcheroo_compiler.serialization.formats.base import WeightLoader
from ml_switcheroo_compiler.serialization.utils import parse_npz


class NpzWeightFormat(WeightLoader):
    """NPZ weight format handler."""

    def load(self, filepath: str) -> dict:
        """Load npz weights."""
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        if hasattr(backend, "load_npz"):
            try:
                return backend.load_npz(filepath)
            except NotImplementedError:
                pass

        return parse_npz(filepath)
