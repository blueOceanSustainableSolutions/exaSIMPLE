import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.sparse import coo_matrix 
from scipy.sparse.linalg import spsolve
import sys
import petsc4py
from petsc4py import PETSc
from mpi4py import MPI

import niceFigures as nF
import petscMatrixInterface
from petscWrappers import solveWithCustomPC

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
#A = petscMatrixInterface.readPetscMatrix(os.path.join("../data_3_Weymouth2020", case+"_A.dat"))
#b = petscMatrixInterface.readPetscVector(os.path.join("../data_3_Weymouth2020", case+"_b.dat"))

# ReFRESCO data.
#case = "case_00_ist_curved_grid_ist_curved_201"; tol = 1e-8  # No big difference between tuned and pure Jacobi
#case = "case_01_ist_plate_grid_ist_plate_801"; tol = 1e-8  # No big difference between tuned and pure Jacobi
#case = "case_02_Poiseuille_ogrid_rect-veryfine"; tol = 1e-8  # Tuned performs worse than pure Jacobi
#case = "case_03_Taylor_vortex_hgrid_h-box_128x128"; tol = 1e-8  # Tuned better than pure
case = "case_04_Taylor_vortex_ogrid_cube-veryfine"; tol = 1e-8  # Tuned better than pure
# Read the matricers and multiply both sides by -1 to match Gabe's covnention (for now)
A = petscMatrixInterface.readPetscMatrix(os.path.join("../data_2_verificationSuiteCases", case, "a_mat_massTransport_outerloop_1.dat"))
b = petscMatrixInterface.readPetscVector(os.path.join("../data_2_verificationSuiteCases", case, "b_vec_massTransport_outerloop_1.dat"))
A *= -1.
b *= -1.

# Convert Matrix from COO format to PETSc format
A_petsc = petscMatrixInterface.convert_coo_to_petsc(A, comm)
b_petsc, _ = A_petsc.createVecs()
b_petsc.setValues(range(len(b)), b)
b_petsc.assemblyBegin()
b_petsc.assemblyEnd()

# View matrix from command line using -view_mat option 
A_petsc.viewFromOptions('-view_mat')

# Solve using different preconditioners.
results = {
    "PETSc Jacobi": solveWithCustomPC(A_petsc, b_petsc, rtol=tol, atol=tol),
    "Pure Jacobi": solveWithCustomPC(A_petsc, b_petsc, precond=JacobiPC, rtol=tol, atol=tol),
    "Tuned Jacobi": solveWithCustomPC(A_petsc, b_petsc, precond=TunedJacobiPC, verbose=True, rtol=tol, atol=tol),
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
