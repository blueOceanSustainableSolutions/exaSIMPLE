import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

font = {"family": "serif",
        "weight": "normal",
        "size": 15}

matplotlib.rc("font", **font)
matplotlib.rcParams["figure.figsize"] = (8, 6)
matplotlib.rcParams['figure.dpi'] = 75

# ===
# Nice figures
def makeNiceAxes(ax, xlab=None, ylab=None):
    ax.tick_params(axis='both', reset=False, which='both', length=5, width=2)
    ax.tick_params(axis='y', direction='out', which="both")
    ax.tick_params(axis='x', direction='out', which="both")
    for spine in ['top', 'right','bottom','left']:
        ax.spines[spine].set_linewidth(2)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

def niceFig(xlab=None, ylab=None, figsize=None, nrows=1, ncols=1):
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize)
    if (nrows == 1) and (ncols == 1):
        makeNiceAxes(ax, xlab, ylab)
    else:
        for axx in ax:
            makeNiceAxes(axx, xlab, ylab)
    return fig, ax

def addColourBar(fig, cs, cbarLabel, pos=[0.85, .25, 0.03, 0.5], orientation="vertical"):
    position = fig.add_axes(pos)
    cbar = fig.colorbar(cs, cax=position, orientation=orientation)
    cbar.set_label(cbarLabel)
    return cbar

# ===
# PETSC-formatted matrix import.
def readPetscMatrix(filename):
    # Read and upack the compact row storage format.
    with open(filename, 'rt', encoding='ascii') as infile:
        fString = infile.readlines()
    A_vals = []
    A_rows = []
    A_cols = []
    for line in fString:
        if line.startswith("row"):
            iRow = int(line.split(":")[0].split()[1])
            vals = [v.strip("()").split(",") for v in re.findall("\([0-9]+,.*?\)", line)]
            for v in vals:
                A_rows.append(iRow); A_cols.append(int(v[0])); A_vals.append(float(v[1]))
    # Convert to a sparse matrix.
    return coo_matrix((A_vals, (A_rows, A_cols)))

def readPetscVector(filename):
    with open(filename, 'rt', encoding='ascii') as infile:
        fString = infile.readlines()
    b = []
    for line in fString:
        try:
            b.append(float(line.strip()))
        except ValueError:
            pass
    return np.array(b)

saveFigs = True
# case = "../subcase_0_box/baseCase/data_channel_grid_0_pointwise_structured_np_1"
# case = "../subcase_0_box/baseCase/data_channel_grid_1_pointwise_tri_np_1"
# case = "../subcase_0_box/baseCase/data_channel_grid_2_pointwise_triAndQuad_np_1"
# case = "../subcase_0_box/baseCase/data_channel_grid_3_pointwise_triOrdered_np_1"
case = "../data_0_simple2Dgrids/data_convDiff_grid_3_pointwise_triOrdered_np_1"

# Read the data.
A = readPetscMatrix(os.path.join(case, "a_mat_ascii_massTransport.dat"))
b = readPetscVector(os.path.join(case, "b_vec_ascii_massTransport.dat"))
try:
    x_refresco = readPetscVector(os.path.join(case, "x_vec_ascii_massTransport.dat"))
except FileNotFoundError:
    x_refresco = None

# Solve!
x = spsolve(A.tocsr(), b)

# Compute residuals.
residuals = b - A.dot(x)
l1_norm = np.linalg.norm(residuals, ord=1)
linf_norm = np.linalg.norm(residuals, ord=np.inf)

# Print the norms
print("L1 Norm:", l1_norm)
print("Linf Norm:", linf_norm)

# Plot the matrix.
fig, ax = niceFig("Column", "Row")
ax.invert_yaxis()
ax.set_aspect("equal")
cs = ax.scatter(A.col, A.row, c=A.data)
addColourBar(fig, cs, "Coefficient value")
if saveFigs:
    plt.savefig("../Figures_0_initialTests/exampleMatrix_{}.png".format(case.split("/")[-1].replace("data_", "")),
        bbox_inches="tight", dpi=200)

# Print max difference between ReFRESCO and scipy.
if x_refresco is not None:
    x_diff = np.log10(np.abs(x-x_refresco))
    print("Max log10 of difference between ReFRESCO and scipy is", x_diff.max())

    # Plot values.
    fig, ax = niceFig("scipy", "ReFRESCO")
    ax.set_title("Max log$_{{10}}$(difference) = {:.1f}".format(x_diff.max()))
    ax.set_aspect("equal")
    cs = ax.scatter(x, x_refresco, c=x_diff, cmap=plt.cm.bwr)
    addColourBar(fig, cs, "log$_{{10}}$ of difference")
    if saveFigs:
        plt.savefig("../Figures_0_initialTests/exampleSolution_{}.png".format(case.split("/")[-1].replace("data_", "")),
            bbox_inches="tight", dpi=200)

plt.show()
