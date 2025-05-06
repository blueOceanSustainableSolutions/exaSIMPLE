# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 13:46:39 2023

@author: ALidtke
"""

import numpy as np
import re
import os
import sys
import pandas
import petsc4py
from petsc4py import PETSc
from scipy.sparse import csr_matrix

dataDir = "./calcs"
outDir = "./processedData"
maxSize = 100e3


def read_petsc_matrix_binary(filename):
    viewer = PETSc.Viewer().createBinary(filename, mode='r')
    matrix = PETSc.Mat().load(viewer)
    csr_data = matrix.getValuesCSR()
    indptr, indices, data = csr_data
    csr = csr_matrix((data, indices, indptr), shape=matrix.getSize())
    return csr


def read_petsc_vector_binary(filename):
    viewer = PETSc.Viewer().createBinary(filename, mode='r')
    vec = PETSc.Vec().load(viewer)
    return vec.getArray()


# Set up and stuff.
petsc4py.init(sys.argv)

OptDB = PETSc.Options()

# Read the grid stats.
grids = pandas.read_csv("gridStats.csv")

# List the cases.
cases = os.listdir(dataDir)

# Loop 'em
dataSummary = []
iOutfile = -1
for case in cases:
    iGrid = int(case.split("_")[-1])
    gridName = grids.loc[iGrid, "grid"]
    caseName = case.split("_")[1]

    outits = np.sort([int(re.findall("outit_[0-9]+", f)[0].split("_")[1]) for f in os.listdir(os.path.join("./calcs", case))
        if f.startswith("mat_massTransport_outit") and f.endswith(".info")])

    nproc = [int(re.findall("np_[0-9]+", f)[0].split("_")[1]) for f in os.listdir(os.path.join("./calcs", case))
        if f.startswith("mat_massTransport_outit") and f.endswith(".info")][0]

    for iOuterLoop in outits:
        # Read the PETSc data that comes out of ReFRESCO.
        f_A = os.path.join(dataDir, case, f"mat_massTransport_outit_{iOuterLoop:d}_np_{nproc:d}.dat")
        f_b = os.path.join(dataDir, case, f"vec_massTransport_b_outit_{iOuterLoop:d}_np_{nproc:d}.dat")
        f_x = os.path.join(dataDir, case, f"vec_massTransport_x_outit_{iOuterLoop:d}_np_{nproc:d}.dat")
        A = read_petsc_matrix_binary(f_A)
        b = read_petsc_vector_binary(f_b)
        x = read_petsc_vector_binary(f_x)

        # Extract components
        A_indices = np.vstack((A.tocoo().row, A.tocoo().col)).astype(np.int32)
        A_values = A.tocoo().data.astype(np.float64)
        b_vector = b.astype(np.float64)
        x_vector = x.astype(np.float64)

        # Check if the matrix isn't too big.
        if len(b_vector) > maxSize:
            continue

        # Save components in .npz file
        iOutfile += 1
        base_name = f"data_{iOutfile:d}_{caseName}_{gridName}_outerloop_{iOuterLoop}.npz"
        npz_path = os.path.join(outDir, base_name)
        np.savez(npz_path, A_indices=A_indices, A_values=A_values, b=b_vector, x=x_vector)
        print(f"Processed and saved: {npz_path}")

        # Store the summary.
        dataSummary.append({
            "case": caseName,
            "grid": gridName,
            "outerLoop": iOuterLoop,
            "nCells": grids.loc[iGrid, "nCells"],
            "nProc": nproc,
            "matrixSize": len(b_vector),
            "fname": base_name,
        })

pandas.DataFrame(dataSummary).to_csv(os.path.join(outDir, "summary.csv"), index=False)
