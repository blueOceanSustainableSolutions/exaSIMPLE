import re
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas

import niceFigures as nF
import petscMatrixInterface
import petsc4py
from petsc4py import PETSc
from mpi4py import MPI


def readCaseMatrices(caseName, gridName, outerIter=1, dataDir="../data_5_unifiedSuite"):
    # Read the grids and find cases
    grids = pandas.read_csv(os.path.join(dataDir, "gridStats.csv"))
    grids["nProc"] = 0
    grids["case"] = "none"

    # Subset cases.
    df = grids[grids["grid"].str.contains(gridName+"_")]

    # Read data for each case.
    matrices = []
    for iGrid in df.sort_values("nCells").index:
        case = "case_{}_grid_{:d}".format(caseName, iGrid)
        df.loc[iGrid, "case"] = case
        
        with open(os.path.join("../data_5_unifiedSuite/calcs", case, "job.sh"), "r") as infile:
            s = infile.read()
        nProc = int(re.findall("mpirun -np [0-9]+", s)[0].split()[-1])
        df.loc[iGrid, "nProc"] = nProc

        A_petsc = petscMatrixInterface.readMat(
            os.path.join("../data_5_unifiedSuite/calcs", case, "mat_massTransport_outit_{:d}_np_{:d}.dat".format(outerIter, nProc)))
        b_petsc = petscMatrixInterface.readVec(
            os.path.join("../data_5_unifiedSuite/calcs", case, "vec_massTransport_b_outit_{:d}_np_{:d}.dat".format(outerIter, nProc)))
        x_petsc = petscMatrixInterface.readVec(
            os.path.join("../data_5_unifiedSuite/calcs", case, "vec_massTransport_b_outit_{:d}_np_{:d}.dat".format(outerIter, nProc)))
        matrices.append({"A": A_petsc, "b": b_petsc, "x_ref": x_petsc})

    return matrices, df
