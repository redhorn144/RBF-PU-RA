import numpy as np
from .BaseHelpers import GenPhi

#In the LS-RBF-PUM method, nodes may be identically distributed across multiple patches.
#Because of this, and the fact that with LS nodes may ie outside the computational domain
#as well not needing to exist in the same place where overlapp occurs we need only generate 
#1 interpolation matrix 

def GenPatchNodes(n, r, d, layout = 'vogel'):
    if layout == 'polar_gll':
        return PolarGLLNodes(n, d, r)
    elif layout == 'vogel':
        return VogelPoints(n, d, r)
    else:
        raise ValueError("Unsupported node layout: choose 'polar_gll' or 'vogel'.")


#------------------------------------------------------------------------------------
# Node layout options for the patches
#------------------------------------------------------------------------------------

def PolarGLLNodes(n, d, r):
    if d == 2:
        return PolarGLLDisk(n, r)
    elif d == 3:
        return PolarGLLSphere(n, r)
    else:
        raise ValueError("Polar GLL nodes only implemented for d=2 or d=3.")

def PolarGLLDisk(n, r):
    # Generate GLL nodes in 1D
    gll_1d = np.cos(np.pi * np.arange(n) / (n - 1))
    # Create polar grid
    r_nodes = r * (gll_1d + 1) / 2  # Scale to [0, r]
    theta_nodes = np.linspace(0, 2 * np.pi, n, endpoint=False)
    R, Theta = np.meshgrid(r_nodes, theta_nodes)
    x = R * np.cos(Theta)
    y = R * np.sin(Theta)
    return np.column_stack((x.flatten(), y.flatten()))

def PolarGLLSphere(n, r):
    # Generate GLL nodes in 1D
    gll_1d = np.cos(np.pi * np.arange(n) / (n - 1))
    # Create spherical grid
    phi_nodes = np.arccos(gll_1d)  # Polar angle
    theta_nodes = np.linspace(0, 2 * np.pi, n, endpoint=False)  # Azimuthal angle
    Phi, Theta = np.meshgrid(phi_nodes, theta_nodes)
    x = r * np.sin(Phi) * np.cos(Theta)
    y = r * np.sin(Phi) * np.sin(Theta)
    z = r * np.cos(Phi)
    return np.column_stack((x.flatten(), y.flatten(), z.flatten()))

def VogelPoints(N, d, r):
    if d == 2:
        x, y = vogel_disk(N)
        return r * np.column_stack((x, y))
    elif d == 3:
        x, y, z = vogel_sphere(N)
        return r * np.column_stack((x, y, z))
    else:
        raise ValueError("Vogel points only implemented for d=2 or d=3.")

def vogel_disk(n):
    golden_angle = np.pi * (3 - np.sqrt(5))
    i = np.arange(n)
    r = np.sqrt(i / n)
    theta = i * golden_angle
    return r * np.cos(theta), r * np.sin(theta)

def vogel_sphere(n):
    golden_angle = np.pi * (3 - np.sqrt(5))
    i = np.arange(n)
    z = 1 - 2 * i / (n - 1)
    r = np.sqrt(1 - z**2)
    theta = i * golden_angle
    return r * np.cos(theta), r * np.sin(theta), z