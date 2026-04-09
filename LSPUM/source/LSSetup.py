import numpy as np
from mpi4py import MPI
from .PatchNodes import GenPatchNodes
from .BaseHelpers import GenMatrices
from .RAHelpers import GenEr

#------------------------------------------------------------------------------------
#
#
#------------------------------------------------------------------------------------

def SetupLS(comm, eval_nodes, npp, node_type = 'vogel', eval_eps = 0.0):
    rank = comm.Get_rank()

    if rank == 0:

