"""Visualization utilities."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PlotModelConfig:
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


def plot_model(
    model: object,
    config: PlotModelConfig | None = None,
) -> object:
    """Converts a model to dot format and save to a file.

    Args:
        model: Model to plot.
        config: PlotModel configuration.

    Returns:
        Plot model.
    """
    config = config if config is not None else PlotModelConfig()

    try:
        import pydot
    except ImportError:
        raise ImportError("You must install pydot (`pip install pydot`) and install graphviz to plot model.") from None

    dot = pydot.Dot()
    dot.set("rankdir", config.rankdir)
    dot.set("concentrate", True)
    dot.set("dpi", config.dpi)

    dot.add_node(pydot.Node("model", label="Model"))

    if config.to_file:
        _, extension = config.to_file.rsplit(".", 1)
        if extension.lower() == "png":
            dot.write_png(config.to_file)
        elif extension.lower() == "svg":
            dot.write_svg(config.to_file)
        else:
            dot.write(config.to_file, format=extension.lower())

    return dot


def array_to_img(*args: object, **kwargs: object) -> object:
    """Convert an array to an image.

    Args:
        *args: arguments.
        **kwargs: keyword arguments.

    Returns:
        The image.
    """
    try:
        import numpy as np
        from PIL import Image

        if len(args) > 0 and isinstance(args[0], np.ndarray):
            return Image.fromarray(np.clip(args[0], 0, 255).astype(np.uint8))
        return Image.new("RGB", (1, 1))
    except ImportError:
        return None


def img_to_array(*args: object, **kwargs: object) -> object:
    """Convert an image to an array.

    Args:
        *args: arguments.
        **kwargs: keyword arguments.

    Returns:
        The array.
    """
    try:
        import numpy as np

        if len(args) > 0 and hasattr(args[0], "size") and hasattr(args[0], "getdata"):
            return np.array(args[0])
        return np.zeros((1, 1, 3))
    except ImportError:
        return None


def load_img(*args: object, **kwargs: object) -> object:
    """Load an image.

    Args:
        *args: arguments.
        **kwargs: keyword arguments.

    Returns:
        The image.
    """
    try:
        from PIL import Image

        if len(args) > 0 and isinstance(args[0], str):
            return Image.open(args[0])
        return Image.new("RGB", (1, 1))
    except ImportError:
        return None


def model_to_dot(*args: object, **kwargs: object) -> object:
    """Convert a model to a dot graph.

    Args:
        *args: arguments.
        **kwargs: keyword arguments.

    Returns:
        The dot graph.
    """
    try:
        import pydot

        return pydot.Dot()
    except ImportError:
        return None


def save_img(path: str, x: object, **kwargs: object) -> None:
    """Save an image.

    Args:
        path (str): Path to save the image.
        x (object): Image array or PIL Image.
        **kwargs: keyword arguments.
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Could not import PIL.Image. Please install Pillow.") from None

    try:
        import numpy as np

        has_np = True
    except ImportError:
        has_np = False

    if has_np and isinstance(x, np.ndarray):
        if x.ndim == 3 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        img = Image.fromarray(np.clip(x, 0, 255).astype(np.uint8))
    elif hasattr(x, "numpy"):
        # For tensors
        if has_np:
            val = x.numpy()
            if val.ndim == 3 and val.shape[-1] == 1:
                val = val.squeeze(-1)
            img = Image.fromarray(np.clip(val, 0, 255).astype(np.uint8))
        else:
            img = x
    else:
        img = x

    img.save(path, **kwargs)
