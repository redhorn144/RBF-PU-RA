import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.animation import FuncAnimation, PillowWriter
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

def AnimateSolution(nodes, snapshots, times, resolution=150, cmap="viridis",
                    savepath="advection.gif", fps=20):
    """
    Interpolate a sequence of scattered solution snapshots onto a regular grid
    and save an animated GIF.

    Parameters
    ----------
    nodes     : (N, 2) array of node coordinates
    snapshots : list of (N,) solution vectors, one per frame
    times     : list of floats, simulation time for each snapshot
    resolution: int, grid points per axis for interpolation
    cmap      : str, matplotlib colormap
    savepath  : str, output path for the GIF
    fps       : int, frames per second
    """
    x, y = nodes[:, 0], nodes[:, 1]
    gx, gy = np.meshgrid(np.linspace(x.min(), x.max(), resolution),
                         np.linspace(y.min(), y.max(), resolution))

    fig, ax = plt.subplots(figsize=(6, 5))
    grid0 = griddata((x, y), snapshots[0], (gx, gy), method="cubic")
    im = ax.imshow(grid0, origin="lower", extent=[x.min(), x.max(), y.min(), y.max()],
                   vmin=np.nanmin(grid0), vmax=np.nanmax(grid0), cmap=cmap, animated=True)
    fig.colorbar(im, ax=ax, label="u")
    title = ax.set_title(f"t = {times[0]:.3f}  peak = {np.nanmax(grid0):.3f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    def update(frame):
        grid = griddata((x, y), snapshots[frame], (gx, gy), method="cubic")
        im.set_data(grid)
        vmin_f, vmax_f = np.nanmin(grid), np.nanmax(grid)
        im.set_clim(vmin_f, vmax_f)
        title.set_text(f"t = {times[frame]:.3f}  peak = {vmax_f:.3f}")
        return im, title

    ani = FuncAnimation(fig, update, frames=len(snapshots), interval=1000 // fps, blit=False)
    ani.save(savepath, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Animation saved to {savepath}")


def SaveField(nodes, u, filename):
    """
    Save the solution field to a text file.

    Parameters
    ----------
    nodes : (N, 2) array of node coordinates
    u     : (N,)   solution vector
    filename : str, path to save the file
    """
    data = np.hstack((nodes, u[:, None]))
    np.savetxt(filename, data, header="x y u", comments="")
    