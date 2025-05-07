# =============================================================================
# exaSIMPLE Project
# Script to interact with ReFRESCO
#   - Automatic read matrices files present in a directory
#   - Creates and solves the pressure correction equation
#   - Writes the pressure correction and velocity vectors 
# Contributors: João Muralha
# Maintainer: João Muralha
# =============================================================================

import os
#DEBUG
#import psutil
#process = psutil.Process()
#DEBUG
import time
import numpy as np
#import matplotlib.pyplot as plt
#import niceFigures as nF
from filelock import FileLock
import sys
sys.path.insert(0, '/projects/F202414686CPCAA3/exaSIMPLE/Python')

import petscWrappers
import petscMatrixInterface
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc

from mpi4py import MPI
comm = MPI.COMM_WORLD
world_size = comm.Get_size()
rank = comm.Get_rank()

#mem_process = process.memory_info().rss / 1024**3
#mem_allprocess=comm.gather(mem_process,root=0)
#if rank==0: print('Mem Start rank: ',rank,'=',process.memory_info().rss / 1024**3  )

import xml.etree.ElementTree as ET #READ CONTROLS


# =============================================================================
# READ CONTROLS
# =============================================================================
if comm.Get_rank() == 0:
  tree = ET.parse("controls.xml")
  root = tree.getroot()
  
  developer_control = root.find("developer")
  if int(developer_control.find("savePETScMatrixDataEveryNouterLoops").text) <= 0:
      print("ReFRESCO is not saving PETSc Matrics")
      exit()
  
  outerLoop_control = root.find("outerLoop")
  max_outer_loops = int(outerLoop_control.find("maxIteration").text)
  
  #needs to be generalized for all type of names in equations  
  pressure_equations = root.find("equations")
  for child in pressure_equations:
      if child.attrib["name"] == "pres":
          #print(child.attrib)
          pressure_solver  = child.find("EQPressure").find("solver")
          pressure_petsc   = pressure_solver.find("PETSC")
          pressure_PETSc_solver = pressure_petsc.find("solver").text.lower()
          pressure_PETSc_pc     = pressure_petsc.find("preconditioner").text.lower()
          pressure_maxIter      = int(child.find("EQPressure").find("maxIteration").text)
          pressure_rtol      = float(child.find("EQPressure").find("convergenceTolerance").text)
  
else:
  max_outer_loops = 0
  pressure_PETSc_solver = None
  pressure_PETSc_pc = None
  pressure_maxIter =None
  pressure_rtol = None

max_outer_loops = comm.bcast(max_outer_loops, root=0) 
pressure_PETSc_solver = comm.bcast(pressure_PETSc_solver, root=0)
pressure_PETSc_pc = comm.bcast(pressure_PETSc_pc, root=0)
pressure_maxIter = comm.bcast(pressure_maxIter, root=0)
pressure_rtol = comm.bcast(pressure_rtol, root=0)

exit_python = False

#Create PETSc.ksp object
ksp = petscWrappers.defineKSP(comm = comm, solver=pressure_PETSc_solver, precond=pressure_PETSc_pc)

#Create PETSc.ksp object for direct solver use
if sys.argv[1] == "exactInverse":
    ksp_dSolver = petscWrappers.defineKSP(comm = comm, solver="preonly", precond="lu")

# =============================================================================
# READ FILES
# =============================================================================

# Gradient and Divergence matrix are constant (for non deforming or moving grids)
in_path = "./PETSc_Files_Python/"
out_path = "./PETSc_Files_ReFRESCO/"

status_in_fileName = os.path.join(in_path, "status.io")
status_in_fileName_lock = os.path.join(in_path, "status.io.lock")
stop_file_python = os.path.join("stop_python.io")

#Read status of ReFRESCO files
#only master core
#reads a file with the number os cells in each core, 
#this is done to guarantee that the matrices created by petsc4py are of the same local dimensions
if comm.Get_rank() == 0:
    print("Waiting for Status File")
    while not os.path.exists(status_in_fileName):
        pass
    print("Status File Found")  
    with open(status_in_fileName, 'r') as f:
        while os.path.exists(status_in_fileName_lock):
            pass
        Lines = f.readlines()
        status_io = Lines[0].strip('\n')
        ncells = [int(num) for i in Lines[1:] for num in i.split()]
        while status_io != "TRUE":
            print("Waiting for ReFRESCO files in outerloop: {:d}".format(0))
            time.sleep(5)
else:
  status_io = None
  ncells   = None

status_io = comm.bcast(status_io, root=0)
ncells = comm.scatter(ncells, root=0)

filename_D = os.path.join(in_path,"Divergence_outerloop_0.dat")
filename_G = os.path.join(in_path, "Gradient_outerloop_0.dat")

if comm.Get_rank() == 0:
    while not os.path.exists(filename_D):
        print("Waiting for ReFRESCO Divergence matrix")
    while not os.path.exists(filename_G):
        print("Waiting for ReFRESCO Gradient matrix")

D = petscMatrixInterface.readMat(filename_D, nrows=ncells,   ncolumns=3*ncells)
G = petscMatrixInterface.readMat(filename_G, nrows=3*ncells, ncolumns=ncells)

#Write file header for innerloop information
python_counter = "linSolverInfo_python.dat"
if comm.Get_rank() == 0:
    with open(python_counter, 'w') as filecounter:
        filecounter.write("Title=\"Number of used Iteration for the pressure correction equation\" \n")
        filecounter.write("Variables=\"OuterLoop\"  \"Pressure\" \n")   

#Read files that depend on the outerloop
for outerLoop in range(1,max_outer_loops + 1):

    #only master core
    if comm.Get_rank() == 0:
        with open(status_in_fileName, 'r') as f:
            Lines = f.readlines()
            status_io = Lines[0].strip('\n')        
            while status_io != "TRUE":
                print("Waiting for ReFRESCO files in outerloop: {:d}".format(outerLoop))
                #time.sleep(5)

        #Delete python status.io file so it does not execute itself
        os.remove(status_in_fileName)

    filename_D_exasimple = os.path.join(in_path,"Divergence_exaSIMPLE_outerloop_{:d}.dat".format(outerLoop))
    filename_Q_diag      = os.path.join(in_path,"a_mat_momentumTransport_outerloop_{:d}.dat".format(outerLoop))
    filename_M           = os.path.join(in_path,"a_mat_massTransport_outerloop_{:d}.dat".format(outerLoop))
    filename_b           = os.path.join(in_path,"b_vec_massTransport_outerloop_{:d}.dat".format(outerLoop))
    filename_diagScale   = os.path.join(in_path,"DiagScaling_outerloop_{:d}.dat".format(outerLoop))

    if comm.Get_rank() == 0:
        while not os.path.exists(filename_D_exasimple):
            pass
            #print("Waiting for ReFRESCO ExaSIMPLE Divergence matrix {:d}".format(outerLoop))
        print("Reading ExaSIMPLE Divergence matrix {:d}".format(outerLoop))
        while not os.path.exists(filename_Q_diag):
            pass 
            #print("Waiting for ReFRESCO Momentum matrix {:d}".format(outerLoop))
        print("Reading Momentum matrix {:d}".format(outerLoop))
        while not os.path.exists(filename_M):
            pass
            #print("Waiting for ReFRESCO 1/dQ L p\' matrix {:d}".format(outerLoop))
        print("Reading 1/dQ L p\' matrix {:d}".format(outerLoop))
        while not os.path.exists(filename_b):
            pass
            #print("Waiting for ReFRESCO mass imbalance vector {:d}".format(outerLoop))
        print("Reading mass imbalance vector {:d}".format(outerLoop))
        while not os.path.exists(filename_diagScale):
            pass
            #print("Waiting for ReFRESCO diagonal scaling {:d}".format(outerLoop))
        print("Reading diagonal scaling vector {:d}".format(outerLoop))

    D_exasimple = petscMatrixInterface.readMat(filename_D_exasimple, nrows=ncells, ncolumns=3*ncells)
    Q_diag = petscMatrixInterface.readMat(filename_Q_diag, nrows=ncells, ncolumns=ncells)
    M = petscMatrixInterface.readMat(filename_M, nrows=ncells, ncolumns=ncells)
    b = petscMatrixInterface.readVec(filename_b, nrows=ncells)
    diagScale = petscMatrixInterface.readVec(filename_diagScale, nrows=3*ncells)

# =============================================================================
# PETSc operations
# =============================================================================   
    #Invert Q matrix
    if sys.argv[1] == "exactInverse":
        #Calculates the inverse of the momentum matrix Q
        Q_inv_diag = petscWrappers.invertMatrixPETSc(Q_diag)
        #Q_inv_diag = petscWrappers.approx_invertMatrixPETSc(Q_diag, ksp_dSolver)

    elif sys.argv[1] == "approxInverse":
        #Calculates the approximate inverse of the momentum matrix Q
        Q_inv_diag, Result = petscWrappers.approxInv(Q_diag,np.ones(2))
    elif sys.argv[1] == "approxInverse_w":
        #Calculates the approximate inverse of the momentum matrix Q
        Q_inv_diag, Result = petscWrappers.approxInv(Q_diag,np.array([0.749663, 0.386763]))
    elif sys.argv[1] == "diagInverse":        
        #Calculates the approximate inverse of the momentum matrix Q
        Q_inv_diag, Result = petscWrappers.approxInv(Q_diag,np.zeros(1))
    else:
        print("Wrong Command Line Input")
        sys.exit()

    #Creates the blockDiagonal Matrix Q
    Q_inv = petscWrappers.blockDiagonalMatrix(Q_inv_diag, ncells)

    if sys.argv[1] == "diagInverse":
       FullMassSystem = M
    else:
       #Creates the extra matrix to add to mass transport matrix extracted from ReFRESCO
       #Level 2 Matrix -> -D Q^-1 G + (1/dQ D) G
       G_copy=G.copy()
       G_copy.diagonalScale(L=diagScale)
       aux_1 = -D.matMatMult(Q_inv,G)
       aux_2 = D_exasimple.matMult(G_copy)
       aux_3 = aux_1 + aux_2

       #Assembles the Full Pressure Correction Equation Linear System
       # -D Q^-1 G + (1/dQ D) G - 1/dQ L
       FullMassSystem = aux_3 + M

    #mem_process = process.memory_info().rss / 1024**3
    #mem_allprocess=comm.gather(mem_process,root=0)
    #if rank==0: print('Mem Before Results: ',mem_allprocess  )

    #Solves the linear system of equations
    results = {"result" : petscWrappers.solvePETSc(ksp, FullMassSystem, b, comm=comm, maxIter=pressure_maxIter, rtol=pressure_rtol)}    

    #Pressure Correction Vector
    pp_c = results["result"]["x_petsc"]
    pp_c_fileName = os.path.join(out_path, "pp_c_outerloop_{:d}.dat".format(outerLoop))
    petscMatrixInterface.writeVec(pp_c_fileName, pp_c)

    # Original    
    vcorrection_matrix = -Q_inv.matMult(G)  
    vv_c = vcorrection_matrix.createVecLeft()
    vcorrection_matrix.mult(pp_c,vv_c)             
    vv_c_fileName = os.path.join(out_path, "vv_c_outerloop_{:d}.dat".format(outerLoop))
    petscMatrixInterface.writeVec(vv_c_fileName, vv_c)  

    #mem_process = process.memory_info().rss / 1024**3
    #mem_allprocess=comm.gather(mem_process,root=0)
    #if rank==0: print('Mem Before Clearing Matrices: ',mem_allprocess  )

    #Clear memory
    if sys.argv[1] == "diagInverse":
       pass
    else:
       G_copy.destroy()
       aux_1.destroy()
       aux_2.destroy()
       aux_3.destroy()

    D_exasimple.destroy()
    Q_diag.destroy()
    M.destroy()
    Q_inv_diag.destroy()
    Q_inv.destroy()
    FullMassSystem.destroy()
    b.destroy()
    diagScale.destroy()
    pp_c.destroy()
    vcorrection_matrix.destroy()
    vv_c.destroy()

    PETSc.garbage_cleanup()

    #mem_process = process.memory_info().rss / 1024**3
    #mem_allprocess=comm.gather(mem_process,root=0)
    #if rank==0: print('Mem After Clearing Matrices: ',mem_allprocess  )

    #Delete matrices to save space on disk
    filename_D_exasimple_info = os.path.join(in_path,"Divergence_exaSIMPLE_outerloop_{:d}.dat.info".format(outerLoop))
    filename_Q_diag_info      = os.path.join(in_path,"a_mat_momentumTransport_outerloop_{:d}.dat.info".format(outerLoop))
    filename_M_info           = os.path.join(in_path,"a_mat_massTransport_outerloop_{:d}.dat.info".format(outerLoop))
    filename_b_info           = os.path.join(in_path,"b_vec_massTransport_outerloop_{:d}.dat.info".format(outerLoop))
    filename_diagScale_info   = os.path.join(in_path,"DiagScaling_outerloop_{:d}.dat.info".format(outerLoop))
    filename_x_mom            = os.path.join(in_path,"x_vec_momentumTransport_outerloop_{:d}.dat".format(outerLoop))
    filename_x_mom_info       = os.path.join(in_path,"x_vec_momentumTransport_outerloop_{:d}.dat.info".format(outerLoop))
    filename_b_mom            = os.path.join(in_path,"b_vec_momentumTransport_outerloop_{:d}.dat".format(outerLoop))
    filename_b_mom_info       = os.path.join(in_path,"b_vec_momentumTransport_outerloop_{:d}.dat.info".format(outerLoop))

    ksp_iter = ksp.getIterationNumber()
    ksp_max_iter = comm.allreduce(ksp_iter,MPI.MAX)
    if comm.Get_rank() == 0:
        os.remove(filename_D_exasimple)
        os.remove(filename_Q_diag)
        os.remove(filename_M)
        os.remove(filename_b)
        os.remove(filename_diagScale)
        os.remove(filename_x_mom)
        os.remove(filename_D_exasimple_info)
        os.remove(filename_Q_diag_info)
        os.remove(filename_M_info)
        os.remove(filename_b_info)
        os.remove(filename_diagScale_info) 
        os.remove(filename_x_mom_info)

        #Write number of innerloops
        with open(python_counter, 'a') as filecounter:
            filecounter.write("{:d} {:d} \n".format(outerLoop,ksp_max_iter))

    #Write status to let ReFRESCO know files are ready   
    #only master core
    status_out_fileName = os.path.join(out_path, "status.io")
    status_out_fileName_lock = os.path.join(out_path, "status.io.lock")    
    if comm.Get_rank() == 0:
        with open(status_out_fileName_lock, 'w') as lock:
            lock.write("LOCK_FILE")
      
        with(FileLock(status_out_fileName)):
            with open(status_out_fileName, 'w') as f:
                f.write("TRUE")
   
        #Delete lock file
        os.remove(status_out_fileName_lock)

    if comm.Get_rank() == 0:      
        #Wait for ReFRESCO to execute next outerloop here   
        if(outerLoop + 1 <= max_outer_loops):
            print("Waiting for Status File in outerloop {:d}".format(outerLoop + 1))
            while os.path.exists(status_in_fileName_lock):
                pass
            while not os.path.exists(status_in_fileName):
                #Check if ReFRESCO has stopped
                if os.path.exists(stop_file_python):
                    with open(stop_file_python, 'r') as f:
                        stop = f.readline().strip('\n')
                        if stop == "ReFRESCO converged":
                            print("ReFRESCO converged - stopping Python :-)")
                        elif stop == "ReFRESCO diverged":
                            print("ReFRESCO diverged - stopping Python :-(")
                    #Delete stop file for next run
                    os.remove(stop_file_python)    
                    exit_python = True
                    break
                elif os.path.exists(os.path.join("stopfile")):
                    exit_python = True
                    print("ReFRESCO stopfile exists - stopping Python")
                    break                
                #end if os.path.exists(stop_file):
            #end while not os.path.exists(status_in_fileName):
            while os.path.exists(status_in_fileName_lock):
                pass
            print("Status File Found")
            #delete vectors to save disk space
            os.remove(pp_c_fileName)
            pp_c_info_fileName = os.path.join(out_path, "pp_c_outerloop_{:d}.dat.info".format(outerLoop))
            os.remove(pp_c_info_fileName)
            os.remove(vv_c_fileName)
            vv_c_info_fileName = os.path.join(out_path, "vv_c_outerloop_{:d}.dat.info".format(outerLoop))
            os.remove(vv_c_info_fileName)
        #end if(outerLoop + 1 <= max_outer_loops):
    #end if comm.Get_rank() == 0: 

    #mem_process = process.memory_info().rss / 1024**3
    #mem_allprocess=comm.gather(mem_process,root=0)
    #if rank==0: print('Mem Before Next Loop: ',mem_allprocess  )

    #comm.Barrier()
    exit_python = comm.bcast(exit_python, root = 0)
    sys.stdout.flush()
    if exit_python:
        exit(0)

#END FOR LOOP OVER OUTERLOOPS

#destroy G, D and ksp objects 
G.destroy()
D.destroy()
ksp.destroy()




