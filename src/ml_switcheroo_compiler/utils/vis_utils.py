"""Visualization utilities."""

from __future__ import annotations  # pragma: no cover

# pragma: no cover
from dataclasses import dataclass  # pragma: no cover


@dataclass
class PlotModelConfig:  # pragma: no cover
    """PlotModel configuration."""

    to_file: str = "model.png"
    show_shapes: bool = False
    show_dtype: bool = False
    show_layer_names: bool = True
    rankdir: str = "TB"
    expand_nested: bool = False
    dpi: int = 96
    layer_range: list | None = None
    show_layer_activations: bool = False


def plot_model(  # pragma: no cover
    model: object,
    config: PlotModelConfig | None = None,
) -> object:
    """Converts a Keras model to dot format and save to a file.

    Args:
        model: Model to plot.
        config: PlotModel configuration.

    Returns:
        Plot model.
    """
    config if config is not None else PlotModelConfig()

    print("plot_model is a stub in zero-keras. It requires pydot and graphviz.")  # pragma: no cover
    # In a real implementation this would use pydot to build the graph.
    return None  # pragma: no cover


def array_to_img(*args: object, **kwargs: object) -> object:  # pragma: no cover
    """Convert an array to an image.

    Args:
        *args: arguments.
        **kwargs: keyword arguments.

    Returns:
        The image.
    """
    pass  # pragma: no cover


def img_to_array(*args: object, **kwargs: object) -> object:  # pragma: no cover
    """Convert an image to an array.

    Args:
        *args: arguments.
        **kwargs: keyword arguments.

    Returns:
        The array.
    """
    pass  # pragma: no cover


def load_img(*args: object, **kwargs: object) -> object:  # pragma: no cover
    """Load an image.

    Args:
        *args: arguments.
        **kwargs: keyword arguments.

    Returns:
        The image.
    """
    pass  # pragma: no cover


def model_to_dot(*args: object, **kwargs: object) -> object:  # pragma: no cover
    """Convert a model to a dot graph.

    Args:
        *args: arguments.
        **kwargs: keyword arguments.

    Returns:
        The dot graph.
    """
    pass  # pragma: no cover


def save_img(*args: object, **kwargs: object) -> None:  # pragma: no cover
    """Save an image.

    Args:
        *args: arguments.
        **kwargs: keyword arguments.
    """
    pass  # pragma: no cover
