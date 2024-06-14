import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.sparse import coo_matrix 
from scipy.sparse.linalg import spsolve
import niceFigures as nF
import importMatrix
import sys
import petsc4py
from petsc4py import PETSc
from mpi4py import MPI

# Select the case to load the data for.
case = "../data_3_Weymouth2020/data_Nc_32_2D-static_1"

# Read the data.
A = importMatrix.readPetscMatrix(case+"_A.dat")
b = importMatrix.readPetscVector(case+"_b.dat")

#A = importMatrix.readPetscMatrix("../data_2_verificationSuiteCases/case_00_ist_curved_grid_ist_curved_201/a_mat_massTransport_outerloop_100.dat")
#b = importMatrix.readPetscVector("../data_2_verificationSuiteCases/case_00_ist_curved_grid_ist_curved_201/b_vec_massTransport_outerloop_100.dat")

A *= -1. # because Weymouth assumes negative definite mat
b *= -1.

# Set up and stuff.
petsc4py.init(sys.argv)

OptDB = PETSc.Options()

comm = MPI.COMM_WORLD
world_size = comm.Get_size()
rank = comm.Get_rank()

comm.barrier()
start_time = MPI.Wtime()
  
# Convert Matrix from COO format to PETSc format
A_petsc = importMatrix.convert_coo_to_petsc(A, comm)

# View matrix from command line using -view_mat option 
A_petsc.viewFromOptions('-view_mat')

start_time_1 = MPI.Wtime()
        
# Create KSP object
ksp = PETSc.KSP()
ksp.create(comm=A_petsc.getComm())
#ksp.setType(PETSc.KSP.Type.CG)        # preconditioned CG as solver
ksp.setType(PETSc.KSP.Type.RICHARDSON) # preconditioner as solver
ksp.setOperators(A_petsc, A_petsc)

# null space check
nullsp = PETSc.NullSpace()
nullsp.create(constant=True,vectors=(),comm=A_petsc.getComm())
if nullsp.test(A_petsc):
    print("matrix A has constant null space")
    A_petsc.setNullSpace(nullsp)

# Solution control.
ksp.setIterationNumber(100000)
ksp.setTolerances(rtol=1e-6, atol=1e-6)

# Convergence history plot
ksp.setConvergenceHistory(ksp.getIterationNumber())

# Create Algebraic Multi-Grid preconditioner
pc = ksp.getPC()
pc.setType(PETSc.PC.Type.GAMG)
ksp.setUp()

# Finish setting up solver (can override options set above)
ksp.setFromOptions()

# Create solution and right hand side vector
x_petsc, b_petsc = A_petsc.createVecs()
b_petsc.setValues(range(len(b)),b)
b_petsc.assemblyBegin()
b_petsc.assemblyEnd()

# Solve
start_time_2 = MPI.Wtime() 
ksp.solve(b_petsc, x_petsc)
end_time = MPI.Wtime()

# Compute residuals.
residuals_petsc = ksp.buildResidual()
l1_norm_petsc = residuals_petsc.norm(norm_type=PETSc.NormType.NORM_1)
linf_norm_petsc = residuals_petsc.norm(norm_type=PETSc.NormType.NORM_INFINITY)

# Extract convergence history  
res_petsc = ksp.getConvergenceHistory()

# Extract solution.
x_petsc = comm.gather(x_petsc.getArray(), root=0)

# Print the norms
if rank == 0:
    x_petsc = np.concatenate(x_petsc, axis=0)
    print("PETSc Residuals")
    print("L1 Norm:", l1_norm_petsc)
    print("Linf Norm:", linf_norm_petsc)

    print("Time assemble + solve: " + str(end_time-start_time_1))
    print("Time solve: " + str(end_time-start_time_2))

ksp.destroy()
A_petsc.destroy()
pc.destroy()

end_time = MPI.Wtime()

print("Average result time " + str(rank) + ": " + str(end_time-start_time))
