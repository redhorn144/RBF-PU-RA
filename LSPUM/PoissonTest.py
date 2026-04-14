from mpi4py import MPI
import numpy as np
from source.PatchTiling import BoxGridTiling2D
from nodes.SquareDomain import PoissonSquareOne, MinEnergySquareOne
from source.LSSetup import Setup
from source.Operators import GenLap

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    print("Generating nodes...")
    eval_nodes, normals, groups = PoissonSquareOne(r=0.02)
    # eval_nodes, normals, groups = MinEnergySquareOne(N=1000)

    print("Tiling domain...")
    centers, r = BoxGridTiling2D(eval_nodes, groups, n_interp=30, eval_density=1000, oversample_factor=2.0, overlap=0.5)