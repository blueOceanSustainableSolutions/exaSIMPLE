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

# ===
# Solution routine.
def solveWithCustomPC(A_petsc, b, precond=None, rtol=1e-3, atol=1e-3, verbose=False):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if verbose and (rank == 0):
        print("Solving using", precond)
    
    start_time_1 = MPI.Wtime()

    # Create KSP object
    ksp = PETSc.KSP()
    ksp.create(comm=A_petsc.getComm())
    ksp.setOperators(A_petsc, A_petsc)
    ksp.setType(PETSc.KSP.Type.CG)

    # Solution control.
    ksp.setIterationNumber(100)
    ksp.setTolerances(rtol=rtol, atol=atol)

    # convergence history plot
    ksp.setConvergenceHistory(ksp.getIterationNumber())

    if precond is None:
        # Create a built-in preconditioner
        #pc = PETSc.PC().create()
        #pc.setOperators(A_petsc)
        pc = ksp.getPC()
        pc.setType(PETSc.PC.Type.JACOBI)
        #ksp.setPC(pc)
    else:
        # Create own preconditioner
        #pc = ksp.pc
        pc = ksp.getPC()
        pc.setType(pc.Type.PYTHON)
        pc.setPythonContext(precond())
        ksp.setFromOptions()

    # Create solution and right hand side vector
    x_petsc, b_petsc = A_petsc.createVecs()
    b_petsc.setValues(range(len(b)), b)
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
    if (rank == 0) and verbose:
        x_petsc = np.concatenate(x_petsc, axis=0)
        print("  PETSc Residuals")
        print("    L1 Norm:", l1_norm_petsc)
        print("    Linf Norm:", linf_norm_petsc)
        print("  Time assemble + solve: " + str(end_time-start_time_1))
        print("  Time solve: " + str(end_time-start_time_2))
        print("")

    ksp.destroy()
    pc.destroy()

    return x_petsc, res_petsc, l1_norm_petsc, linf_norm_petsc, (end_time-start_time_2)

# ===
# Candidate preconditioners.

# This attempts to reproduce example 15 from PETSc tutorials:
class JacobiPC:
    # Setup the internal data. In this case, we access the matrix diagonal.
    def setUp(self, pc):
        Amat, Pmat= pc.getOperators()
        self.Diag = Pmat.getDiagonal()

    # Apply the preconditioner
    def apply(self, pc, x, y):
        y.pointwiseDivide(x, self.Diag)

# This attempts to reproduce the approach from the paper of Weymouth (2020):
class TunedJacobiPC:
    def setUp(self, pc):
        # Coefficients of the tuned Jacobi, taken from Gabe's repo.
        self.p = [-0.104447, -0.00238399, 0.00841367, -0.158046, -0.115103]
        
        # Functions for computing the diagonal and off-diagonal coefficients (Eq. 6 from Weymouth (2020)).
        def Dm(D):
            return 1. + self.p[0] + D*(self.p[1] + D*self.p[2])

        def Lm(L):
            return L*(self.p[3]*(L-2) + self.p[4]*(L-1))
        
        # Get the comms.
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
    
        # Get the input matrix.
        self.Amat, self.Pmat = pc.getOperators()
        
        # Comapute the scale factor.
        rows = self.Pmat.getOwnershipRange()
        factor = -1e16
        for i in range(rows[0], rows[1]):
            # Retrieve the values for this row.
            row_indices, row_values = self.Pmat.getRow(i)
            # Get the index of the diagonal element and set it to a very negative value as
            # it should be ignored when choosing the scaling factor.
            i_diagonal = np.where(row_indices == i)[0][0]
            row_values[i_diagonal] = -1.0e16
            # Find the max value.
            max_row = max(row_values)
            factor = max(max_row, factor)
        self.scale = comm.allreduce(factor, MPI.MAX)
        
        # Gather the diagonal. This is necessary for scaling the off-diagonal terms.
        diag = self.Amat.getDiagonal()
        # Gather the local numpy arrays on the root process.
        local_diag = diag.getArray()
        gathered_diag = comm.gather(local_diag, root=0)
        # Concatenate the gathered arrays at the root process
        if rank == 0:
            diag = np.concatenate(gathered_diag)
        else:
            diag = np.empty(self.Amat.getSize()[0], dtype=np.float64)
        # Broadcast the concatenated array to all processes
        comm.Bcast(diag, root=0)

        # Fill-in the values for the preconditioner matrix.
        self.Atilde = self.Pmat.duplicate(copy=False) 
        for i in range(rows[0], rows[1]):
            
            # Retrieve the values for this row.
            row_indices, row_values = self.Amat.getRow(i)
            i_diagonal = np.where(row_indices == i)[0][0]
            
            for j in row_indices:
                if j == row_indices[i_diagonal]:
                    # The diagonal term.
                    if abs(diag[i]) < 1e-8:
                        invD = 0.
                    else:
                        invD = 1./diag[i]
                    self.Atilde.setValues(i, i, invD*Dm(self.Amat[i, i]/self.scale))

                else:
                    # Lower-diagonal terms.
                    if abs(diag[i] + diag[j]) < 2e-8:
                        invD = 0.
                    else:
                        invD = 2. / (diag[i] + diag[j])
                    self.Atilde.setValues(i, j, invD*Lm(self.Amat[i, j]/self.scale))

        self.Atilde.assemblyBegin()
        self.Atilde.assemblyEnd()
        
        # Output for verification.
        #viewer = PETSc.Viewer().createASCII('Atilde_matrix.dat', mode='w')
        #self.Atilde.view(viewer)
        #viewer.destroy()

    def apply(self, pc, x, y):
        self.Atilde.mult(x, y)

# ===
# Set up and stuff.
petsc4py.init(sys.argv)

OptDB = PETSc.Options()

comm = MPI.COMM_WORLD
world_size = comm.Get_size()
rank = comm.Get_rank()
comm.barrier()

# ===
# Select the case to load the data for.

# Data from the Weymouth (2020) paper.
#tol = 1e-3
#case = "data_Nc_32_2D-sphere_1"
#A = importMatrix.readPetscMatrix(os.path.join("../data_3_Weymouth2020", case+"_A.dat"))
#b = importMatrix.readPetscVector(os.path.join("../data_3_Weymouth2020", case+"_b.dat"))

# ReFRESCO data.
#case = "case_00_ist_curved_grid_ist_curved_201"; tol = 1e-8  # No big difference between tuned and pure Jacobi
#case = "case_01_ist_plate_grid_ist_plate_801"; tol = 1e-8  # No big difference between tuned and pure Jacobi
#case = "case_02_Poiseuille_ogrid_rect-veryfine"; tol = 1e-8  # Tuned performs worse than pure Jacobi
#case = "case_03_Taylor_vortex_hgrid_h-box_128x128"; tol = 1e-8  # Tuned better than pure
case = "case_04_Taylor_vortex_ogrid_cube-veryfine"; tol = 1e-8  # Tuned better than pure
# Read the matricers and multiply both sides by -1 to match Gabe's covnention (for now)
A = importMatrix.readPetscMatrix(os.path.join("../data_2_verificationSuiteCases", case, "a_mat_massTransport_outerloop_1.dat"))
b = importMatrix.readPetscVector(os.path.join("../data_2_verificationSuiteCases", case, "b_vec_massTransport_outerloop_1.dat"))
A *= -1.
b *= -1.

# Convert Matrix from COO format to PETSc format
A_petsc = importMatrix.convert_coo_to_petsc(A, comm)

# View matrix from command line using -view_mat option 
A_petsc.viewFromOptions('-view_mat')

# Solve using different preconditioners.
keys = ["x", "res", "L1", "Linf", "time"]
results = {
    "PETSc Jacobi": dict(zip(keys, solveWithCustomPC(A_petsc, b, rtol=tol, atol=tol))),
    "Pure Jacobi": dict(zip(keys, solveWithCustomPC(A_petsc, b, precond=JacobiPC, rtol=tol, atol=tol))),
    "Tuned Jacobi": dict(zip(keys, solveWithCustomPC(A_petsc, b, precond=TunedJacobiPC, verbose=True, rtol=tol, atol=tol))),
}

# ===
# Post-processing.
if comm.Get_rank() == 0:
    print("Solution time in seconds:")
    for pc in results:
        print("  ", pc, results[pc]["time"])
        
    # Plot PETSc solver residuals 
    fig, ax = nF.niceFig("Iteration", "Residual")
    plt.yscale("log")
    for pc in results:
        if "PETSc" in pc:
            ls = "k.-"; lw = 2; alpha = 1.
        else:
            ls = "-"; lw = 2; alpha = 1.
        ax.plot(range(len(results[pc]["res"])), results[pc]["res"], ls, lw=lw, alpha=alpha, label=pc)
    ax.legend()
    #plt.savefig("../Figures_2_tunedJacobiTests/residuals_{}.png".format(case), dpi=200, bbox_inches="tight")
    
    plt.show()
