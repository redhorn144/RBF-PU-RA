from source.PatchNodes import VogelPoints
from source.PatchNodes import PolarGLLNodes
import numpy as np
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------------

# Test the Vogel point generation
N = 100
d = 2
r = 1.0
points = VogelPoints(N, d, r)
plt.figure(figsize=(6, 6))
plt.scatter(points[:, 0], points[:, 1], s=10, color='blue')
circle = plt.Circle((0, 0), r, fill=False, color='red', linestyle='--')
plt.gca().add_patch(circle)
plt.axis('equal')
plt.savefig('figures/vogel_points.png')

#------------------------------------------------------------------------------------
n = 16
d = 2
r = 1.0
gll_points = PolarGLLNodes(n, d, r)
plt.figure(figsize=(6, 6))
plt.scatter(gll_points[:, 0], gll_points[:, 1], s=10, color='green')
circle = plt.Circle((0, 0), r, fill=False, color='red', linestyle='--')
plt.gca().add_patch(circle)
plt.axis('equal')
plt.savefig('figures/polar_gll_points.png')