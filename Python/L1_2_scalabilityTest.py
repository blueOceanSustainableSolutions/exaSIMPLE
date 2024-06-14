import re
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas
import petsc4py
from petsc4py import PETSc
from mpi4py import MPI

from petscWrappers import solveWithCustomPC
import niceFigures as nF
import petscMatrixInterface
import dataInterface

# ===
# Set up and stuff.
petsc4py.init(sys.argv)

OptDB = PETSc.Options()

comm = MPI.COMM_WORLD
world_size = comm.Get_size()
rank = comm.Get_rank()
comm.barrier()

# TODO

# ===
# Post-processing.
if comm.Get_rank() == 0:
    
    plt.show()
