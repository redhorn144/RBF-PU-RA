import numpy as np
from rbf.pde.nodes import poisson_disc_nodes, min_energy_nodes


def _star_polygon(cx, cy, R_out, R_in, n_points):
    angles_out = np.pi / 2 + np.arange(n_points) * 2 * np.pi / n_points
    angles_in  = angles_out + np.pi / n_points

    verts = []
    for ao, ai in zip(angles_out, angles_in):
        verts.append([cx + R_out * np.cos(ao), cy + R_out * np.sin(ao)])
        verts.append([cx + R_in  * np.cos(ai), cy + R_in  * np.sin(ai)])
    vert = np.array(verts)

    n = len(vert)
    edges = np.array([[i, (i + 1) % n] for i in range(n)])
    return vert, edges


def MinEnergyStarDomain(N, cx=0.5, cy=0.5, R_out=0.45, R_in=0.18, n_points=5):
    """
    5-pointed star domain centred at (cx, cy), filled with N minimum-energy
    nodes. Unlike PoissonStarDomain, the user controls the global node count
    directly via N (matches MinEnergySquareOne semantics).
    """
    vert, edges = _star_polygon(cx, cy, R_out, R_in, n_points)
    nodes, groups, normals = min_energy_nodes(N, (vert, edges))

    interior = groups['interior']
    boundary = groups['boundary:all']
    nodes   = np.vstack([nodes[interior],   nodes[boundary]])
    normals = np.vstack([normals[interior], normals[boundary]])

    n_int = len(interior)
    groups['interior']     = np.arange(n_int)
    groups['boundary:all'] = np.arange(n_int, len(nodes))

    return nodes, normals, groups, vert


def PoissonStarDomain(r, cx=0.5, cy=0.5, R_out=0.45, R_in=0.18, n_points=5,
                      refine_center=True, refinement_ratio=3.0):
    """
    5-pointed star domain centred at (cx, cy).

    Parameters
    ----------
    r                : base node spacing (used at the tips when refine_center=True)
    cx, cy           : centre of the star
    R_out            : outer tip radius
    R_in             : inner cusp radius
    n_points         : number of star points
    refine_center    : if True, use variable spacing denser at the center
    refinement_ratio : r_center = r / refinement_ratio
    """
    angles_out = np.pi / 2 + np.arange(n_points) * 2 * np.pi / n_points
    angles_in  = angles_out + np.pi / n_points

    verts = []
    for ao, ai in zip(angles_out, angles_in):
        verts.append([cx + R_out * np.cos(ao), cy + R_out * np.sin(ao)])
        verts.append([cx + R_in  * np.cos(ai), cy + R_in  * np.sin(ai)])
    vert = np.array(verts)

    n = len(vert)
    edges = np.array([[i, (i + 1) % n] for i in range(n)])

    if refine_center:
        r_min = r / refinement_ratio   # dense spacing at center
        r_max = r                       # coarse spacing at tips
        def spacing(x):
            dist = np.sqrt((x[:, 0] - cx)**2 + (x[:, 1] - cy)**2)
            t = np.clip(dist / R_out, 0.0, 1.0)
            return r_min + (r_max - r_min) * t
        spacing_arg = spacing
    else:
        spacing_arg = r

    nodes, groups, normals = poisson_disc_nodes(spacing_arg, (vert, edges))

    # Reorder: interior first, then boundary (matches SquareDomain convention)
    interior = groups['interior']
    boundary = groups['boundary:all']
    nodes   = np.vstack([nodes[interior],   nodes[boundary]])
    normals = np.vstack([normals[interior], normals[boundary]])

    n_int = len(interior)
    groups['interior']     = np.arange(n_int)
    groups['boundary:all'] = np.arange(n_int, len(nodes))

    return nodes, normals, groups, vert
