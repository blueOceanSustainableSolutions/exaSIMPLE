import re
import os
#import io
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas

import niceFigures as nF
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

    return dict(zip(["time", "flops", "breakdown"], [time_total, flops_total, breakdown]))


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
    ksp.setType(solver.lower())
    
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

    # convergence history plot
    ksp.setConvergenceHistory(ksp.getIterationNumber())

    if type(precond) is str:
        # Create a built-in preconditioner
        pc = ksp.getPC()
        pc.setType(precond.lower())
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
    log_contents = None
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

    return dict(zip(
        ["x", "res", "L1", "Linf", "time", "log"],
        [x_petsc, res_petsc, l1_norm_petsc, linf_norm_petsc, (end_time-start_time_2), log_contents]
    ))
