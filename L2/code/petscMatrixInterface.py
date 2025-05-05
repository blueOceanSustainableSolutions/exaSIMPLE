import re
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix 

#JMuralha - Import PETSc  and mpi4py
import sys
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
OptDB = PETSc.Options()

# ===
# PETSC-formatted matrix import (ASCII).
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

# JMuralha - read PETSc matrix in a usable format to be used using petsc4py
def convert_coo_to_petsc(A, comm):
  petsc_mat = PETSc.Mat()
  petsc_mat.create(comm=comm)
  petsc_mat.setSizes((A.tocsr().shape[1],A.tocsr().shape[1]))
  petsc_mat.setType(PETSc.Mat.Type.AIJ)   
  rstart, rend = petsc_mat.getOwnershipRange()
  petsc_mat.createAIJ(size=(A.tocsr().shape), csr=(A.tocsr().indptr[rstart:rend+1] - A.tocsr().indptr[rstart], 
                                                    A.tocsr().indices[A.tocsr().indptr[rstart]:A.tocsr().indptr[rend]], 
                                                    A.tocsr().data[A.tocsr().indptr[rstart]:A.tocsr().indptr[rend]]))

  petsc_mat.assemblyBegin()
  petsc_mat.assemblyEnd()
   
  return petsc_mat

def readPetscMatrix_binary(filename, comm):
  petsc_mat = PETSc.Mat()
  petsc_mat.create(comm=comm)
  petsc_mat.setType(PETSc.Mat.Type.AIJ)   
  
  #viewer = PETSc.Viewer().createBinary('teste.dat', mode=PETSc.Viewer.Mode.READ, comm=comm)
  viewer = PETSc.Viewer().createMPIIO(filename, mode=PETSc.Viewer.Mode.READ, comm=comm)
  petsc_mat.load(viewer)

  return petsc_mat
  
# ===  
# Final and tested matrix and vector import - PETSC binary format.

def readMat(filename, toScipy=False, nrows=None, ncolumns=None):
    #Specify matrix size
    viewer = PETSc.Viewer().createBinary(filename, mode='r', comm=PETSc.COMM_WORLD)
    A_petsc = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
    A_petsc.setType(PETSc.Mat.Type.AIJ)
    if nrows!=None and ncolumns!=None:
         A_petsc.setSizes(((nrows, PETSc.DETERMINE), (ncolumns, PETSc.DETERMINE)))    
    A_petsc.setFromOptions()
    A_petsc.load(viewer)
    A_petsc.assemblyBegin()
    A_petsc.assemblyEnd() 
    viewer.destroy()
    if toScipy:
        I, J, V = A_petsc.getValuesCSR()
        return csr_matrix((V, J, I)).tocoo()
    else:
        return A_petsc

def readVec(filename, toNumpy=False, nrows=None):
    viewer = PETSc.Viewer().createBinary(filename, mode='r', comm=PETSc.COMM_WORLD)
    v_petsc = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
    if nrows!=None:
         v_petsc.setSizes((nrows, PETSc.DETERMINE))
    v_petsc.setFromOptions()
    v_petsc.load(viewer)
    v_petsc.assemblyBegin()
    v_petsc.assemblyEnd()     
    viewer.destroy()
    if toNumpy:
        return np.array(v_petsc)
    else:
        return v_petsc


def writeVec(filename, x):
    viewer = PETSc.Viewer().createBinary(filename, mode='w', comm=PETSc.COMM_WORLD)
    viewer(x)#.view(viewer)
    viewer.destroy()

def writeASCII(filename, x):
    viewer = PETSc.Viewer().createASCII(filename, mode='w', comm=PETSc.COMM_WORLD)
    viewer(x)#.view(viewer)
    viewer.destroy()
