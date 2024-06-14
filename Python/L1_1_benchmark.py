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

# ===
# Select the case to load the data for.
tol = 1e-2
maxIter = 100

caseName = "LDCF"
gridSize = "1.0x1.0x1.0_2D"
#caseName = "TaylorVortex"
#gridSize = "2.0x2.0x1.0_2D"
#caseName = "Poiseuille"
#gridSize = "2.0x1.0x1.0_2D"

gridNames = ["Structured", "Triangle_Delaunay", "TriangleQuad_AdvancingFront"]

# Solver and preconditioner combinations to try.
solverPermutations = [
#    ("gmres", "jacobi"),
    ("gmres", "bjacobi"),
#    ("gmres", "pbjacobi"),
    #("gmres", "sor"),
#    ("gmres", "ilu"),
    #("gmres", "kaczmarz"),
#    ("gmres", "icc"),
    
#    ("cg", "jacobi"),
    ("cg", "bjacobi"),
#    ("cg", "pbjacobi"),
    #("cg", "sor"),
#    ("cg", "ilu"),
    #("cg", "kaczmarz"),
#    ("cg", "icc"),

#    ("minres", "bjacobi"),
#    ("lsqr", "bjacobi"),
#    ("SYMMLQ", "bjacobi"),
]

results_coarse, results_fine = {}, {}
for iGrid, gridName in enumerate(gridNames):
    if rank == 0:
        print("Doing", gridName)
    
    # Read the data.
    gridId = "boxGrid_{}_{}".format(gridSize, gridName)
    matrices, grids = dataInterface.readCaseMatrices(caseName, gridId)

    # Subset to only the coarsest and finest grids for benchmarking
    grids = grids.loc[grids.index.values[[0, -1]], :]
    matrices = [matrices[0], matrices[-1]]

    # Solve using different preconditioners.
    results_coarse[gridName], results_fine[gridName] = {}, {}
    for perm in solverPermutations:
        sol, pc = perm
        key = "{}+{}".format(sol.upper(), pc.upper())
        if rank == 0:
            print("  ", key)
        
        results_coarse[gridName][key] = solveWithCustomPC(matrices[0]["A"], matrices[0]["b"], 
                solver=sol, precond=pc, rtol=tol, atol=1e-12, maxIter=maxIter, log=True)
        results_fine[gridName][key] = solveWithCustomPC(matrices[1]["A"], matrices[1]["b"], 
                solver=sol, precond=pc, rtol=tol, atol=1e-12, maxIter=maxIter, log=True)

# ===
# Post-processing.
if rank == 0:
    fig, axes = nF.niceFig("Iteration", "Residual", ncols=len(gridNames), figsize=(14, 6))
    
    print(results_fine["Structured"]["CG+BJACOBI"]["log"]["breakdown"])

    for iGrid, gridName in enumerate(gridNames):
        ax = axes[iGrid]
        ax.set_ylim((tol, 1))
        
        ax.set_yscale("log")
        ax.set_xscale("log")
        colours = plt.cm.nipy_spectral(np.linspace(0.05, 0.95, len(results_coarse[gridName])))
        lns = []
        for i, k in enumerate(results_coarse[gridName].keys()):
            ax.plot(range(len(results_coarse[gridName][k]["res"])),
                results_coarse[gridName][k]["res"]/results_coarse[gridName][k]["res"][0],
                ls="--", lw=2, c=colours[i])
            lns += ax.plot(range(len(results_fine[gridName][k]["res"])),
                results_fine[gridName][k]["res"]/results_fine[gridName][k]["res"][0],
                ls="-", lw=2, c=colours[i], label=k)

    for i in range(1, len(axes)):
        axes[i].set_ylabel("")
        axes[i].set_xlim((1, maxIter))
        axes[i].yaxis.set_ticklabels([])
        
    leg = fig.legend(lns, [l.get_label() for l in lns], loc="upper right", bbox_to_anchor=(0.99, 0.94))
    ax.add_artist(leg)
    
    xlim = ax.get_xlim()
    ax.set_xlim(xlim)
    lns = ax.plot([xlim[1]*10, xlim[1]*10+1], [1e-1, 1e-1], "k--", lw=2)
    lns += ax.plot([xlim[1]*10, xlim[1]*10+1], [1e-1, 1e-1], "k-", lw=2)
    fig.legend(handles=lns, labels=["Coarse", "Fine"], loc='lower right', ncol=2, bbox_to_anchor=(0.99, 0.92))

    #plt.subplots_adjust(top=0.98, bottom=0.11, left=0.075, right=0.880, hspace=0.185, wspace=0.195)
    plt.subplots_adjust(top=0.98, bottom=0.11, left=0.075, right=0.765, hspace=0.185, wspace=0.14)
    
    plt.savefig("../Figures_3_benchmark/benchmark_residuals_{:s}_Np_{:d}.png".format(caseName, comm.Get_size()), 
        dpi=200, bbox_inches="tight")
    
    plt.show()
