import numpy as np
from matplotlib.colors import ListedColormap
import importlib.resources as resources


def get_colormap(name: str = "default"):
    """
    Returns the custom LabToolbox colormap.

    Parameters
    ----------
    name : str, optional
        Currently only 'default' is supported.

    Returns
    -------
    matplotlib.colors.ListedColormap
        The colormap object.
    """

    if name != "default":
        raise ValueError(f"Unknown colormap '{name}'")

    with resources.files(__package__).joinpath("_colormap.npy").open("rb") as f:
        col = np.load(f)

    return ListedColormap(col)