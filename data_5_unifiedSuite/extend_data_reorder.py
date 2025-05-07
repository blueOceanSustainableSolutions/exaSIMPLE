# -*- coding: utf-8 -*-
"""
@author: ALidtke
"""

import numpy as np
import re
import os
import sys
import pandas
import shutil
import argparse
from itertools import permutations
import petsc4py
from mpi4py import MPI
from petsc4py import PETSc

import resources

# ===
# Select source data and main controls.

dataDir = "../neural-sparse-linear-solvers-master/datasets/data_artur/processedData"
A_coeff_noise_mag = 0.05
overwrite = False
maxIter = int(1e4)
reorder = False
atol = 1e-24
rtol = 1e-24
solver = "gmres"
preconditioner = "jacobi"

outDir = "extendedData_RHS_coeffs_take2"
cases = ['LDCF', 'channel', 'convDiff', 'TaylorVortex', 'plate', 'Poiseuille']
maxSize = 100e3
nPermutations = 20  # 3680 cases * 20 permutations = 73,600 matrices

#outDir = "extendedData_Poiseuille_1kCellsMax_RHS_coeffs_take2"
#cases = ['Poiseuille']
#maxSize = 1e3
#nPermutations = 1000  # 60 matrices * 1000 permutations = 60,000 matrices

# ===

# Type-safe gathering of parallel vectors.
def gatherVec(vec):
    if isinstance(vec, PETSc.Vec):
        local_values = vec.getArray()
    elif isinstance(vec, np.ndarray):
        local_values = vec
    else:
        raise ValueError("Input must be a PETSc Vec or a NumPy array.")

    v_gathered = comm.gather(local_values, root=0)
    if rank == 0:
        v = np.concatenate(v_gathered, axis=0)
    else:
        v = None

    return v

# ===

# Parse kwargs for subsetting the summary when running in parallel on a shared node.
parser = argparse.ArgumentParser()
parser.add_argument("--instance_id", type=int, nargs=1, default=0, help="Parallel rank of this instance")
parser.add_argument("--n_tasks", type=int, nargs=1, default=1, help="Total no. tasks being applied to the source data")
args = parser.parse_args()

# Set up PETSc.
petsc4py.init(sys.argv)
OptDB = PETSc.Options()
comm = MPI.COMM_WORLD
world_size = comm.Get_size()
rank = comm.Get_rank()
comm.barrier()

# Create the output directory
if (rank == 0):
    if os.path.exists(outDir) and overwrite:
        shutil.rmtree(outDir)
    os.makedirs(outDir, exist_ok=True)

# Read the initial data summary.
summary = pandas.read_csv(os.path.join(dataDir, "../summary.csv"))

# Subset
subset = []
for case in cases:
    df = summary[(summary["case"] == case) & (summary["nCells"] < maxSize)].copy()
    df["iCase"] = df.index
    subset.append(df)
summary = pandas.concat(subset)
if rank == 0:
    print("Kept {:d} cases after subsetting".format(summary.shape[0]))

# Split further for parallel running.
# TODO should check if n_tasks is given without instance_id, which would cause
# the code to always work on the same batch of data. Leave for now.
try:
    n_tasks = args.n_tasks[0]
    instance_id = args.instance_id[0]
except TypeError:
    n_tasks = args.n_tasks
    instance_id = args.instance_id
split = np.array_split(summary.index, n_tasks)
keep = split[instance_id]
iMatrix = -1
for i in range(instance_id):
    iMatrix += len(split[i])*nPermutations
summary = summary.loc[keep, :].reset_index(drop=True)
if rank == 0:
    print("Kept {:d} cases after parallel division".format(summary.shape[0]))

# Loop over all cases.
matCounter = -1
summary_new = []
for iSrc in summary.index:
    iCase = summary.loc[iSrc, "iCase"]

    # Read the raw data.
    fname = os.path.join(dataDir, summary.loc[iSrc, "fname"])
    A_indices, A_values, b_vector, x_vector = np.load(fname).values()

    # Compute RMS of the RHS vector and coefficients of A.
    rms_A = np.sqrt(np.sum((A_values - np.mean(A_values))**2.) / len(A_values))
    rms_b = np.sqrt(np.sum((b_vector-np.mean(b_vector))**2.) / len(b_vector))

    # Create a PETSc matrix to get the parallel decomposition.
    A_petsc = resources.create_A(A_indices, A_values, comm)
    row_start, row_end = A_petsc.getOwnershipRange()

#    b_petsc = resources.create_vec(b_vector, comm)
#    results_0 = resources.solveWithCustomPC(
#            A_petsc, b_petsc, comm, solver=solver, precond=preconditioner, rtol=rtol, atol=atol,
#            maxIter=maxIter, log=False, verbose=False)

    A_petsc.destroy()

    # Generate permutations for reordering the elements.
    if (nPermutations > 1) and reorder:
        permutations = set()
        while len(permutations) < nPermutations:
            perm = tuple(np.random.permutation(range(row_end - row_start)))
            if perm not in permutations:
                permutations.add(perm)
        permutations = list([list(p) for p in permutations])
    else:
        permutations = [list(range(row_end - row_start))]*nPermutations

    for iPerm in range(len(permutations)):
        permutations[iPerm] = np.array(list(range(row_start, row_end, 1)))[permutations[iPerm]].astype(np.int32)

    # Make N similar problems.
    for iPerm, perm in enumerate(permutations):
        matCounter += 1
        iMatrix += 1

        # Output file. Check if need to redo.
        base_name = f"data_{iMatrix:d}.npz"
        npz_path = os.path.join(outDir, base_name)
        if os.path.exists(npz_path) and not overwrite:
            continue

        # Create random values for the RHS vector and add noise to the coefficients.
        A = A_values + np.random.normal(size=len(A_values), scale=rms_A*A_coeff_noise_mag)
        b = np.random.normal(size=len(b_vector), scale=rms_b)
        A_petsc = resources.create_A(A_indices, A, comm)
        b_petsc = resources.create_vec(b, comm)

#        results_1 = resources.solveWithCustomPC(
#                A_petsc, b_petsc, comm, solver=solver, precond=preconditioner, rtol=rtol, atol=atol,
#                maxIter=maxIter, log=False, verbose=False)

        # Reorder.
        # Create a PETSc index object.
        isrow = PETSc.IS().createGeneral(perm)
        # Keep the same ordering for columns.
        iscol = isrow.duplicate()
        # Apply the ordering to the matrix
        A_petsc = A_petsc.permute(isrow, iscol)
        # Apply the ordering to the RHS vector
        b_perm = b_petsc.copy()
        b_perm.setValues(range(row_start, row_end, 1), b_petsc.getValues(perm))
        b_perm.assemble()
        b_petsc.destroy()
        b_petsc = b_perm

        # Retrieve the parallel ordering.
        irow, icol, coeffs = [], [], []
        for i in range(row_start, row_end, 1):
            cols, vals = A_petsc.getRow(i)
            irow = np.append(irow, np.ones_like(cols)*i)
            icol = np.append(icol, cols)
            coeffs = np.append(coeffs, vals)

        # Gather.
        b = gatherVec(b_petsc)
        irow = gatherVec(irow)
        icol = gatherVec(icol)
        coeffs = gatherVec(coeffs)
        indices = np.vstack([irow, icol])

        # Solve to get the solution vector.
        results = resources.solveWithCustomPC(
                A_petsc, b_petsc, comm, solver=solver, precond=preconditioner, rtol=rtol, atol=atol,
                maxIter=maxIter, log=False, verbose=False)
        x = results["x"]

        # Clean up.
#        A_petsc.destroy()
#        b_petsc.destroy()

#        print("---")
#        print("rel res={:1e}, L1={:1e}, L2{:1e}, Linf={:1e}, niter={:d}".format(
#                    results_0["relative_res"], results_0["L1"], results_0["L2"], results_0["Linf"], results_0["niter"]))
#        print("rel res={:1e}, L1={:1e}, L2{:1e}, Linf={:1e}, niter={:d}".format(
#                    results_1["relative_res"], results_1["L1"], results_1["L2"], results_1["Linf"], results_1["niter"]))
#        print("rel res={:1e}, L1={:1e}, L2{:1e}, Linf={:1e}, niter={:d}".format(
#                    results["relative_res"], results["L1"], results["L2"], results["Linf"], results["niter"]))
#        break

        # Save the matrix for future reuse and also save the results to a file for future processing.
        if rank == 0:
            print("  Finished matrix {:d} / {:d}:".format(matCounter+1, summary.shape[0]*nPermutations),
                "rel res={:1e}, L1={:1e}, L2={:1e}, Linf={:1e}, niter={:d}".format(
                    results["relative_res"], results["L1"], results["L2"], results["Linf"], results["niter"]))

            summary_new.append(summary.loc[iSrc, :].copy())
            for k in ["L1", "L2", "Linf", "niter"]:
                summary_new[-1][k] = results[k]

            np.savez(npz_path, A_indices=indices, A_values=coeffs, b=b, x=x)

            resultsFile = os.path.join(outDir, "result_expanded_{:d}_case_{:d}_perm_{:d}_nproc_{:d}.json".format(iMatrix, iCase, iPerm, comm.Get_size()))
            resources.resultsToFile(results, resultsFile)

# Get a meaningful summary
if rank == 0:
    summary_new = pandas.DataFrame(summary_new)
    if n_tasks > 1:
        summary_new.to_csv(os.path.join(outDir, f"summary_{instance_id:d}.csv"), index=False)
    else:
        summary_new.to_csv(os.path.join(outDir, "summary.csv"), index=False)

    print("Max L2 residual: {:.1e}".format(summary_new["L2"].max()))
