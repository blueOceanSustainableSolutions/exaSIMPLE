# -*- coding: utf-8 -*-

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas
import os
import re
import sys
import petsc4py
from scipy.sparse import csr_matrix, coo_matrix
from petsc4py import PETSc
from mpi4py import MPI

font = {"family": "serif",
        "weight": "normal",
        "size": 16}

matplotlib.rc("font", **font)
matplotlib.rcParams["figure.figsize"] = (9, 6)

# Read the grids and find cases
grids = pandas.read_csv("gridStats.csv")

cases = os.listdir("./calcs")

# Read residuals for each case.
residuals = {}
caseNames = []
for case in cases:
    try:
        residuals[case] = pandas.read_csv(os.path.join("./calcs", case, "residuals.dat"), skiprows=2, header=None,
            sep="\s+", names=["TotalIter", "L2_VelocityX", "L2_VelocityY", "L2_VelocityZ", "L2_Pressure",
            "Linf_VelocityX", "Linf_VelocityY", "Linf_VelocityZ", "Linf_Pressure"])

        if len(residuals[case]) < 50:
            print("Only {:d} outer loops for {}".format(len(residuals[case]), case))

        caseName, iGrid = case.replace("case_", "").split("_grid_")
        if caseName not in caseNames:
            caseNames.append(caseName)

    except FileNotFoundError:
        print("No residuals for", case)

# Extract max residual per case and grid.
maxResArr = np.zeros((len(caseNames), len(grids)))
for case in cases:
    caseName, iGrid = case.replace("case_", "").split("_grid_")
    iGrid = int(iGrid)
    iCase = caseNames.index(caseName)
    maxResArr[iCase, iGrid] = np.log10(residuals[case]["Linf_Pressure"].values[-1])
maxResArr = np.ma.MaskedArray(maxResArr, np.abs(maxResArr) < 1e-3)

# Plot max residuals.
fig, ax = plt.subplots(figsize=(12, 5))
plt.subplots_adjust(top=1.0, bottom=0.14, left=0.135, right=0.945)
cs = ax.imshow(maxResArr, cmap="plasma", interpolation='nearest', origin='lower',
    extent=[0.5, len(grids)+0.5, 0.5, len(caseNames)*10+0.5], vmin=-6, vmax=-1)
cbar = fig.colorbar(cs, orientation='horizontal', aspect=40, fraction=0.05, pad=0.2)
cbar.set_label('log$_{10}$(L$_\infty$ pressure residual)')
ax.yaxis.set_ticks(range(5, 5+10*len(caseNames), 10))
ax.set_yticklabels(caseNames)
ax.set_xlabel("Grid")

# Read all the matrices for one of the cases.
case = cases[0]

petsc4py.init(sys.argv)

def readMat(filename):
    viewer = PETSc.Viewer().createBinary(filename, mode='r', comm=PETSc.COMM_WORLD)
    A_petsc = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
    A_petsc.setType(PETSc.Mat.Type.AIJ)
    A_petsc.setFromOptions()
    A_petsc.load(viewer)
    I, J, V = A_petsc.getValuesCSR()
    return csr_matrix((V, J, I)).tocoo()

def readVec(filename):
    viewer = PETSc.Viewer().createBinary(filename, mode='r', comm=PETSc.COMM_WORLD)
    v_petsc = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
    v_petsc.setFromOptions()
    v_petsc.load(viewer)
    return np.array(v_petsc)

# Read each type of matrix for every case to be sure it worked.
for case in cases:
    with open(os.path.join("./calcs", case, "job.sh"), "r") as infile:
        s = infile.read()
    nProc = int(re.findall("mpirun -np [0-9]+", s)[0].split()[-1])

    A_p = readMat(os.path.join("./calcs", case, "mat_massTransport_outit_1_np_{:d}.dat".format(nProc)))
    A_u = readMat(os.path.join("./calcs", case, "mat_momentumTransport_outit_1_np_{:d}.dat".format(nProc)))
    A_ds = readMat(os.path.join("./calcs", case, "mat_Divergence_Miller_scaling_outit_1_np_{:d}.dat".format(nProc)))
    A_d = readMat(os.path.join("./calcs", case, "mat_Divergence_outit_0_np_{:d}.dat".format(nProc)))
    A_g = readMat(os.path.join("./calcs", case, "mat_Gradient_outit_0_np_{:d}.dat".format(nProc)))

    v_p_x = readVec(os.path.join("./calcs", case, "vec_massTransport_x_outit_1_np_{:d}.dat".format(nProc)))
    v_p_b = readVec(os.path.join("./calcs", case, "vec_massTransport_b_outit_1_np_{:d}.dat".format(nProc)))
    v_u_x = readVec(os.path.join("./calcs", case, "vec_momentumTransport_x_outit_1_np_{:d}.dat".format(nProc)))
    v_u_b = readVec(os.path.join("./calcs", case, "vec_momentumTransport_b_outit_1_np_{:d}.dat".format(nProc)))

    res_p = np.max(A_p.dot(v_p_x) - v_p_b)
    res_u = np.max(A_p.dot(v_u_x) - v_u_b)

plt.show()
