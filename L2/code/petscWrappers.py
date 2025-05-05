import re
import os
import io
import json
import numpy as np
import sys
import pandas

import petscMatrixInterface

import petsc4py
from petsc4py import PETSc
from mpi4py import MPI

def parsePetscLog(log_contents):
    """ Retrieve relevant data from the output of PETSc.Log.view function.
    """
    
    time_total = float(re.findall("Time\\s+\(sec\):.*", log_contents)[0].split()[2])
    flops_total = int(float(re.findall("Flops:.*", log_contents)[0].split()[-1]))

    dataLabels = [
        ("Event", str),
        ("Count_max", int),
        ("Count_ratio", float),
        ("Time_max", float),
        ("Time_ratio", float),
        ("Flop_max", float),
        ("Flop_ratio", float),
        ("Mess", float),
        ("AvgLen", float),
        ("Reduct", float),
        ("Global_%T", int),
        ("Global_%F", int),
        ("Global_%M", int),
        ("Global_%L", int),
        ("Global_%R", int),
        ("Stage_%T", int),
        ("Stage_%F", int),
        ("Stage_%M", int),
        ("Stage_%L", int),
        ("Stage_%R", int),
        ("Mflop/s", int)
    ]
    breakdown = [l.split() for l in re.findall("[a-zA-Z]+\s+[0-9]+\s[0-9]+\.[0-9]+\s.*", log_contents)]
    breakdown = pandas.DataFrame(data=breakdown, columns=[d[0] for d in dataLabels])
    breakdown = breakdown.astype(dict(dataLabels))

    #return dict(zip(["time", "flops", "breakdown"], [time_total, flops_total, breakdown]))
    return time_total, flops_total, breakdown

#=============================================================
# Function to create KSP object
# Recives the MPI communication and PETSc solver settings
# Returns the KSP object
#=============================================================
def defineKSP(comm = PETSc.COMM_WORLD, solver="gmres", precond="bjacobi", verbose=False):
    if verbose and (rank == 0):
        print("Solving using", solver, "and", precond)

    # Create KSP object
    ksp = PETSc.KSP()
    ksp.create(comm=comm)

    # Set solver type
    ksp.setType(solver)

    # Set preconditioner
    if type(precond) is str:
        # Create a built-in preconditioner
        pc = ksp.getPC()
        pc.setType(precond)
    else:
        # Create own preconditioner
        pc = ksp.getPC()
        pc.setType(pc.Type.PYTHON)
        pc.setPythonContext(precond())
       
    # Option to override previous setting from command line arguments
    ksp.setFromOptions()

    return ksp


#=============================================================
# Function to create solve the system of equations Ax=B
# Recives the KSP object, and the matrix A and vector B
# Returns the result dictionary
# Adapt from Artur Lidtke function solveWithCustomPC
#=============================================================
def solvePETSc(ksp, A_petsc, b_petsc, comm, maxIter=100, rtol=1e-3, atol=1e-12, log=False, verbose=False):

    # MPI operations
    rank = comm.Get_rank()

    # Set operator for KSP object
    ksp.setOperators(A_petsc, A_petsc)

    # Set tolerances for solution control
    ksp.setTolerances(rtol=rtol, atol=atol, max_it=maxIter)
    ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED) #ReFRESCO uses unpreconditioner residuals
    # convergence history plot
    ksp.setConvergenceHistory(ksp.getIterationNumber())

    # Check for null space
    nullsp = PETSc.NullSpace()
    nullsp.create(constant=True, vectors=(), comm=A_petsc.getComm())
    if nullsp.test(A_petsc):
        if (rank == 0) and verbose:
            print("  Matrix A has constant null space")
        A_petsc.setNullSpace(nullsp)

    # Create solution and right hand side vector
    x_petsc, _ = A_petsc.createVecs()

    # Start logging.
    if log:
        PETSc.Log.begin()

    # Solve
    ksp.solve(b_petsc, x_petsc)

    start_time = MPI.Wtime() 
    
    # Get the log output.
    log_contents = [None]*3
    if log:
        logfile = "/tmp/petsc_log.dat"
        viewer = PETSc.Viewer().createASCII(logfile)
        PETSc.Log.view(viewer)
        viewer.destroy()
        if (rank == 0):
            with open(logfile, 'r') as infile:
                log_contents = parsePetscLog(infile.read())
            os.remove(logfile)

    end_time = MPI.Wtime()
    
    # Compute residuals.
    residuals_petsc = ksp.buildResidual()
    l1_norm_petsc = residuals_petsc.norm(norm_type=PETSc.NormType.NORM_1)
    linf_norm_petsc = residuals_petsc.norm(norm_type=PETSc.NormType.NORM_INFINITY)

    # Extract convergence history  
    res_petsc = ksp.getConvergenceHistory()

    # Extract solution.
    x = comm.gather(x_petsc.getArray(), root=0)
    if (rank == 0):
        x = np.concatenate(x, axis=0)
        
    # Print the norms
    if (rank == 0) and verbose:
        print("  PETSc Residuals")
        print("    L1 Norm:", l1_norm_petsc)
        print("    Linf Norm:", linf_norm_petsc)
        print("  Time assemble + solve: " + str(end_time-start_time_1))
        print("  Time solve: " + str(end_time-start_time_2))
        print("")

    return dict(zip(
        ["x", "x_petsc", "res", "L1", "Linf", "time", "log_time", "log_flops", "log_breakdown"],
        [x, x_petsc, res_petsc, l1_norm_petsc, linf_norm_petsc, (end_time-start_time)] + list(log_contents)
    ))

def solveWithCustomPC(A_petsc, b_petsc, solver="gmres", precond="bjacobi",
        maxIter=100, rtol=1e-3, atol=1e-3, log=True, verbose=False):
    """ Soves the Ax=b problem using PETSc and performs post-processing.
    
    The preconditioner can be a string for selection of built-in classes or derived
    PC object for use of own implementations.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if verbose and (rank == 0):
        print("Solving using", solver, "and", precond)
    
    start_time_1 = MPI.Wtime()

    # Create KSP object
    ksp = PETSc.KSP()
    ksp.create(comm=A_petsc.getComm())
    ksp.setOperators(A_petsc, A_petsc)
    ksp.setType(solver)
    
    # null space check
    nullsp = PETSc.NullSpace()
    nullsp.create(constant=True, vectors=(), comm=A_petsc.getComm())
    if nullsp.test(A_petsc):
        if (rank == 0) and verbose:
            print("  Matrix A has constant null space")
        A_petsc.setNullSpace(nullsp)

    # Solution control.
    ksp.setIterationNumber(maxIter)
    ksp.setTolerances(rtol=rtol, atol=atol)
    ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED) #ReFRESCO uses unpreconditioner residuals

    # convergence history plot
    ksp.setConvergenceHistory(ksp.getIterationNumber())

    if type(precond) is str:
        # Create a built-in preconditioner
        pc = ksp.getPC()
        pc.setType(precond)
    else:
        # Create own preconditioner
        pc = ksp.getPC()
        pc.setType(pc.Type.PYTHON)
        pc.setPythonContext(precond())
        
    ksp.setFromOptions()

    # Create solution and right hand side vector
    x_petsc, _ = A_petsc.createVecs()

    start_time_2 = MPI.Wtime() 
    
    # Start logging.
    if log:
        PETSc.Log.begin()

    # Solve
    ksp.solve(b_petsc, x_petsc)
    
    # Get the log output.
    log_contents = [None]*3
    if log:
        logfile = "/tmp/petsc_log.dat"
        viewer = PETSc.Viewer().createASCII(logfile)
        PETSc.Log.view(viewer)
        viewer.destroy()
        if (rank == 0):
            with open(logfile, 'r') as infile:
                log_contents = parsePetscLog(infile.read())
            os.remove(logfile)

    end_time = MPI.Wtime()
    
    # Compute residuals.
    residuals_petsc = ksp.buildResidual()
    l1_norm_petsc = residuals_petsc.norm(norm_type=PETSc.NormType.NORM_1)
    linf_norm_petsc = residuals_petsc.norm(norm_type=PETSc.NormType.NORM_INFINITY)

    # Extract convergence history  
    res_petsc = ksp.getConvergenceHistory()

    # Extract solution.
    x = comm.gather(x_petsc.getArray(), root=0)
    if (rank == 0):
        x = np.concatenate(x, axis=0)
        
    # Print the norms
    if (rank == 0) and verbose:
        print("  PETSc Residuals")
        print("    L1 Norm:", l1_norm_petsc)
        print("    Linf Norm:", linf_norm_petsc)
        print("  Time assemble + solve: " + str(end_time-start_time_1))
        print("  Time solve: " + str(end_time-start_time_2))
        print("")

    ksp.destroy()
    pc.destroy()

    return dict(zip(
        ["x", "x_petsc", "res", "L1", "Linf", "time", "log_time", "log_flops", "log_breakdown"],
        [x, x_petsc, res_petsc, l1_norm_petsc, linf_norm_petsc, (end_time-start_time_2)] + list(log_contents)
    ))

# =============================================================================
# Function to invert a matrix passed in PETSc format
# Creates two dense matrices: 
#  -matrix B which is the indentity matrix.
#  -matrix X which is the inverse of input matrix.
# Directly solves the system A X = B to obtain the inverse of A
# =============================================================================
def invertMatrixPETSc(A_petsc):

    # Create Dense matrices B and X with the same size as A_petsc
    matrix_size = A_petsc.getSize()
    B = PETSc.Mat() 
    B.create(comm=comm)
    B.createDense(matrix_size)
    diag = B.createVecs("right")
    diagonal=np.ones(matrix_size[0])
    diag.setValues(range(len(diagonal)),diagonal)
    B.setDiagonal(diag, None)
    B.assemblyBegin()
    B.assemblyEnd()

    X = PETSc.Mat() 
    X.create(comm=comm)
    X.createDense(matrix_size)
    X.assemblyBegin()
    X.assemblyEnd()

    rows, columns = A_petsc.getOrdering(PETSc.Mat.OrderingType.NATURAL)
    A_petsc.factorLU(rows,columns)
    
    A_petsc.matSolve(B,X)

    B.destroy()
  
    return X

# =============================================================================
# Function to approximatly invert a matrix passed in PETSc format
# Creates two dense matrices: 
#  -matrix B which is the indentity matrix.
#  -matrix X which is the inverse of input matrix.
# Uses a direct LU (superlu_dist) solver for system A X = I to obtain the inverse of A
# HIGH MEMORY FOOTPRINT
# =============================================================================
def approx_invertMatrixPETSc(A_petsc, ksp):

    # Create Dense matrices B and X with the same size as A_petsc
    matrix_size = A_petsc.getSizes()
    B = PETSc.Mat() 
    B.create(comm=A_petsc.getComm())
    B.createDense(matrix_size)
    diag = B.createVecs("right")
    diagonal=np.ones(matrix_size[0][1])
    diag.setValues(range(len(diagonal)),diagonal)
    B.setDiagonal(diag, None)
    B.assemblyBegin()
    B.assemblyEnd() 

    X_approx = PETSc.Mat() 
    X_approx.create(comm=A_petsc.getComm())
    X_approx.createDense(matrix_size)
    X_approx.assemblyBegin()
    X_approx.assemblyEnd()

    # Solve - petsc
    ksp.setOperators(A_petsc, A_petsc)
    ksp.setTolerances(rtol=1e-13, atol=1e-20, divtol=1e20, max_it=10000) #number of iterations
    ksp.setOptionsPrefix("teste_")
    ksp.setFromOptions()

    #Solve
    ksp.matSolve(B, X_approx)
      
    B.destroy()
   
    return X_approx

# =============================================================================
# Function to create the three block diagonal momentum matrix
# Receives the momentum matrix extracted from ReFRESCO and copies it in a block matrix
# =============================================================================
def blockDiagonalMatrix(diagBlock, ncells, comm):
  
    mat_size = diagBlock.getSizes()
    m = 3*ncells # local number of rows
    N = 3*mat_size[0][1] # Global number of collumns
    diagMat_size = ((m,PETSc.DETERMINE),(m,N))
    petsc_mat = PETSc.Mat()
    petsc_mat.createAIJ(diagMat_size, comm=comm)
    petsc_mat.viewFromOptions('-view_mat')

    rows, rowe = diagBlock.getOwnershipRange()
    rows_b, rowe_b = petsc_mat.getOwnershipRange()

    for i in range(rows, rowe):
        #Extract Matrix Values and indices
        diag_c, diag_value = diagBlock.getRow(i)
        for j in diag_c:
            petsc_mat.setValue((rows_b-rows)+i,(rows_b-rows)+j,diagBlock[i,j])  
            petsc_mat.setValue((rows_b-rows)+i+ncells,(rows_b-rows)+j+ncells,diagBlock[i,j])
            petsc_mat.setValue((rows_b-rows)+i+2*ncells,(rows_b-rows)+j+2*ncells, diagBlock[i,j])

    petsc_mat.assemblyBegin()
    petsc_mat.assemblyEnd()

    return petsc_mat

# =============================================================================
# Function to create calculate the polynomial preconditioner
# Receives the full momentum matrix, the polinomial weigths
# and returns the approximate inverse
# =============================================================================
def approxInv(Q, weigths=None, optimization=False):
    diagQ_vec = Q.getDiagonal()
    diagQ_vec.reciprocal()

    diagQ_inv = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
    diagQ_inv.setType(PETSc.Mat.Type.AIJ)
    diagQ_inv.setSizes(((Q.getLocalSize()[0], PETSc.DETERMINE), (Q.getLocalSize()[0], PETSc.DETERMINE))) 
    diagQ_inv.setDiagonal(diagQ_vec)
    diagQ_inv.assemblyBegin()
    diagQ_inv.assemblyEnd()
    Q_scaled = diagQ_inv.matMult(Q)
   
    I = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
    I.setType(PETSc.Mat.Type.AIJ)
    I.setSizes(((Q.getLocalSize()[0], PETSc.DETERMINE), (Q.getLocalSize()[0], PETSc.DETERMINE))) 
    I.setDiagonal(diagQ_vec/diagQ_vec)
    I.assemblyBegin()
    I.assemblyEnd()
    
    aux = I - Q_scaled
    aux1 = I + weigths[0]*aux
    #loop between 1 and range(weights)
    for p in range(1,len(weigths)):
        #print(p)
        aux = aux.matMult(aux)
        aux1.axpy(weigths[p],aux)

    Mp = aux1.matMult(diagQ_inv)

    if optimization:
    	Result = (I - Mp.matMult(Q)).norm(PETSc.NormType.NORM_FROBENIUS)
    else:
    	Result = None

    aux.destroy()
    aux1.destroy()
    I.destroy()
    diagQ_vec.destroy()
    Q_scaled.destroy()
    diagQ_inv.destroy()

    return Mp, Result


