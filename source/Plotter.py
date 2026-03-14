import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.interpolate import griddata


def PlotSolution(nodes, u, resolution=200, title="Solution", cmap="viridis",
                  savepath="solution.png", show=False):
    """
    Interpolate a scattered solution vector onto a regular grid and plot it
    as a 3D surface.

    Parameters
    ----------
    nodes      : (N, 2) array of node coordinates
    u          : (N,)   solution vector
    resolution : int, number of grid points per axis
    title      : str, plot title
    cmap       : str, matplotlib colormap
    savepath   : str or None, if given saves figure to this path
    show       : bool, whether to call plt.show()
    """
    x, y = nodes[:, 0], nodes[:, 1]

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    gx, gy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )

    # Interpolate scattered data onto grid (cubic, fall back to linear)
    grid_u = griddata((x, y), u, (gx, gy), method="cubic")

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(gx, gy, grid_u, cmap=cmap, linewidth=0, antialiased=True)
    fig.colorbar(surf, ax=ax, shrink=0.5, label="u")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u")

    if savepath is not None:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax
