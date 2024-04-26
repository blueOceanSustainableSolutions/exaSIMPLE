import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.sparse import coo_matrix 
from scipy.sparse.linalg import spsolve
import niceFigures as nF
import importMatrix

#JMuralha - Import PETSc  and mpi4py
import sys
import petsc4py

petsc4py.init(sys.argv)

from petsc4py import PETSc

OptDB = PETSc.Options()

from mpi4py import MPI
comm = MPI.COMM_WORLD
world_size = comm.Get_size()
rank = comm.Get_rank()

#%% Solver Funtions
def solvePETSc(A, b, x_refresco):
  
  #Convert Matrix from COO format to PETSc format
  A_petsc = importMatrix.convert_coo_to_petsc(A, comm)

  # View matrix from command line using -view_mat option 
  A_petsc.viewFromOptions('-view_mat')
  
  start_time_1 = MPI.Wtime()
  
  # Solve - petsc
  #Create KSP object
  ksp = PETSc.KSP()
  ksp.create(comm=A_petsc.getComm())
  ksp.setType(PETSc.KSP.Type.CG)
  ksp.setOperators(A_petsc, A_petsc)
  ksp.setIterationNumber(10000) #number of iterations
  ksp.setConvergenceHistory(ksp.getIterationNumber()) # convergence history plot
  
  #Create preconditioner
  pc = PETSc.PC().create()
  pc.setOperators(A_petsc)
  pc.setType(PETSc.PC.Type.BJACOBI)
  ksp.setPC(pc)
  
  #Create solution and right hand side vector
  x_petsc, b_petsc = A_petsc.createVecs()
  b_petsc.setValues(range(len(b)),b);
  b_petsc.assemblyBegin();
  b_petsc.assemblyEnd();

  #Solve
  start_time_2 = MPI.Wtime() 
  ksp.solve(b_petsc, x_petsc)
  end_time = MPI.Wtime()
  
  print(rank, ksp.getRhs())
  
  # Compute residuals.
  residuals_petsc=ksp.buildResidual()
  l1_norm_petsc = residuals_petsc.norm(norm_type=PETSc.NormType.NORM_1)
  linf_norm_petsc = residuals_petsc.norm(norm_type=PETSc.NormType.NORM_INFINITY)

  # Extract convergence history  
  ksp_res = ksp.getConvergenceHistory()
  
  #Comparison of ReFRESCO solution and PETSc solution  
  if x_refresco is not None:
    vec_start, vec_end = x_petsc.getOwnershipRange() 
    x_diff_petsc = np.log10(np.abs(x_petsc.getArray()-x_refresco[range(vec_start, vec_end)]))
  else:
    x_diff_petsc = None
  
  x_diff_petsc = comm.gather(x_diff_petsc, root=0)
  x_petsc = comm.gather(x_petsc.getArray(), root=0)
  
  # Print the norms
  if rank == 0:
    if x_refresco is not None: x_diff_petsc = np.concatenate(x_diff_petsc, axis=0 )
    x_petsc = np.concatenate(x_petsc, axis=0 )
    print("PETSc Residuals")
    print("L1 Norm:", l1_norm_petsc)
    print("Linf Norm:", linf_norm_petsc)
    
    print("Time assemble + solve: " + str(end_time-start_time_1))
    print("Time solve: " + str(end_time-start_time_2))

  #Plot matrix owned by each process 
  if rank == 0:   
    rows = A_petsc.getOwnershipRanges()
    columns = A_petsc.getOwnershipRangesColumn()
    size = A_petsc.getSize()
    
    # Plot the matrix.
    fig, ax = nF.niceFig("Column", "Row")
    ax.invert_yaxis()
    ax.set_aspect("equal")   
    for i in range(world_size):
      ax.add_patch(Rectangle((columns[i], rows[i]), columns[i+1]-columns[i], rows[i+1]-rows[i], 
             fill=False,
             lw=2))
      ax.add_patch(Rectangle((0, rows[i]), size[1]-0, rows[i+1]-rows[i], 
             fill=False,
             lw=2))       
      
    cs = ax.scatter(A.col, A.row, c=A.data)
    nF.addColourBar(fig, cs, "Coefficient value")
    if saveFigs:
        plt.savefig("../Figures_0_initialTests/exampleMatrix_{}.png".format(case.split("/")[-1].replace("data_", "")),
            bbox_inches="tight", dpi=200)

  #Plot PETSc solver residuals 
  if rank == 0:
    fig, ax = nF.niceFig("Iteration", "Residual")
    plt.yscale("log")
    ax.plot(range(len(ksp_res)),ksp_res)
    
  ksp.destroy()
  A_petsc.destroy()
  pc.destroy()

  return x_petsc, x_diff_petsc

def solveSCIPY(A, b, x_refresco):
  # Solve - scypi
  x = spsolve(A.tocsr(), b)
  
  # Compute residuals.
  residuals = b - A.dot(x)
  l1_norm = np.linalg.norm(residuals, ord=1)
  linf_norm = np.linalg.norm(residuals, ord=np.inf)
  
  # Print the norms
  if rank == 0:
    print("Scipy Residuals")
    print("L1 Norm:", l1_norm)
    print("Linf Norm:", linf_norm)

  if x_refresco is not None:
    x_diff = np.log10(np.abs(x-x_refresco))
  else:
    x_diff = None
    
  return x, x_diff

#%%
path_file =  "../data_0_simple2Dgrids/" 
files = os.listdir(path_file)
files.sort()
cases = [file for file in files if file.startswith("data_")]

A_ = []
b_ = []
x_ref = []
x_sci = []
x_pet = []

for cs in cases:
    saveFigs = False
    # Read the data.
    # A = importMatrix.readPetscMatrix(os.path.join(case, "a_mat_massTransport_outerloop_1.dat"))
    A = importMatrix.readPetscMatrix(os.path.join("../data_0_simple2Dgrids",cs, "a_mat_ascii_massTransport.dat"))
    # b = importMatrix.readPetscVector(os.path.join(case, "b_vec_massTransport_outerloop_1.dat"))
    b = importMatrix.readPetscVector(os.path.join("../data_0_simple2Dgrids",cs, "b_vec_ascii_massTransport.dat"))
    try:
        x_refresco = importMatrix.readPetscVector(os.path.join("../data_0_simple2Dgrids",cs, "x_vec_ascii_massTransport.dat"))
    except FileNotFoundError:
        x_refresco = None
    
    # Solve with Scipy not sure how to parallelize this part 
    if rank == 0:
      x, x_diff = solveSCIPY(A, b, x_refresco)
    
    comm.barrier()
    x_petsc, x_diff_petsc = solvePETSc(A, b, x_refresco)
    
    # Zero Padding - all matrices with shape (692,692)
    if A.shape[0] % 2 == 0:
        Ap = np.pad(A.toarray(), ((int((692-A.shape[0])/2),int((692-A.shape[0])/2)),(int((692-A.shape[1])/2),int((692-A.shape[1])/2))))
        bp = np.pad(b, (int((692-b.shape[0])/2),int((692-b.shape[0])/2)))
        xp = np.pad(x, (int((692-x.shape[0])/2),int((692-x.shape[0])/2)))
        # x_refp = np.pad(x_refresco, (int((692-x_refresco.shape[0])/2),int((692-x_refresco.shape[0])/2)))
        x_petp = np.pad(x_petsc, (int((692-x_petsc.shape[0])/2),int((692-x_petsc.shape[0])/2)))
    else:
        Ap = np.pad(A.toarray(), ((int((692-A.shape[0])/2),int((692-A.shape[0])/2)+1),(int((692-A.shape[1])/2),int((692-A.shape[1])/2)+1)))
        bp = np.pad(b, (int((692-b.shape[0])/2),int((692-b.shape[0])/2)+1))
        xp = np.pad(x, (int((692-x.shape[0])/2),int((692-x.shape[0])/2)+1))
        # x_refp = np.pad(x_refresco, (int((692-x_refresco.shape[0])/2),int((692-x_refresco.shape[0])/2)+1))
        x_petp = np.pad(x_petsc, (int((692-x_petsc.shape[0])/2),int((692-x_petsc.shape[0])/2)+1))
        
    A_.append(Ap)
    b_.append(bp)
    # x_ref.append(x_refp)
    x_sci.append(xp)
    x_pet.append(x_petp)
    
A = np.asarray(A_)
b = np.asarray(b_)
x = np.asarray(x_sci)
#%% CNN train
import tensorflow as tf
from keras.callbacks import EarlyStopping
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
# import os
# import sys
# cores = sys.argv[1]
# tf.config.threading.set_intra_op_parallelism_threads(1)
# tf.config.threading.set_inter_op_parallelism_threads(1)
# os.environ["OMP_NUM_THREADS"] = cores

def mod1(shape_input): # Simple CNN with variable input shape
    model1 = models.Sequential()
    model1.add(layers.Conv2D(320, (5, 5), padding='same', strides= (1, 1), activation='relu', input_shape=shape_input))
    model1.add(layers.MaxPooling2D((2, 2)))
    model1.add(layers.Conv2D(480, (5, 5), padding='same', strides= (1, 1), activation='relu'))
    model1.add(layers.MaxPooling2D((2, 2)))
    model1.add(layers.Conv2D(1260, (5, 5),strides= (1, 1), activation='relu'))
    model1.add(layers.Flatten())
    model1.add(layers.Dropout(0.3))
    model1.add(layers.Dense(860))
    model1.add(layers.Dropout(0.2))
    model1.add(layers.Dense(692, activation='relu'))

    opt = Adam(learning_rate = 1e-03,#0.001
        beta_1=0.8,#0.9
        beta_2=0.9,#0.999
        epsilon=5e-07,
        amsgrad=False,
        name="Adam")
    
    model1.compile(optimizer = opt,
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                metrics=['accuracy'])
    return model1

early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
X_train, X_test, y_train, y_test = train_test_split(A, x, test_size=0.2, random_state=42)
model = mod1((692,692,1))
histories = model.fit(X_train, y_train, validation_split = 0.1, epochs=20, batch_size=16, verbose=1, callbacks=early_stopping)
 
#%%

x_ml = model.predict(X_test, verbose=0) # yhat_test

#%% Solving

saveFigs = False
# case = "../subcase_0_box/baseCase/data_channel_grid_0_pointwise_structured_np_1"
# case = "../subcase_0_box/baseCase/data_channel_grid_1_pointwise_tri_np_1"
# case = "../subcase_0_box/baseCase/data_channel_grid_2_pointwise_triAndQuad_np_1"
# case = "../subcase_0_box/baseCase/data_channel_grid_3_pointwise_triOrdered_np_1"
case = "../data_0_simple2Dgrids/data_convDiff_grid_3_pointwise_triOrdered_np_1"
#case = "../New_Matrices/"

# Read the data.
# A = importMatrix.readPetscMatrix(os.path.join(case, "a_mat_massTransport_outerloop_1.dat"))
A = importMatrix.readPetscMatrix(os.path.join(case, "a_mat_ascii_massTransport.dat"))
# b = importMatrix.readPetscVector(os.path.join(case, "b_vec_massTransport_outerloop_1.dat"))
b = importMatrix.readPetscVector(os.path.join(case, "b_vec_ascii_massTransport.dat"))
try:
    x_refresco = importMatrix.readPetscVector(os.path.join(case, "x_vec_ascii_massTransport.dat"))
except FileNotFoundError:
    x_refresco = None

# Solve with Scipy not sure how to parallelize this part 
if rank == 0:
  x, x_diff = solveSCIPY(A, b, x_refresco)

comm.barrier()
start_time = MPI.Wtime()
x_petsc, x_diff_petsc = solvePETSc(A, b, x_refresco)
end_time = MPI.Wtime()  

print("Average result time " + str(rank) + ": " + str(end_time-start_time))

#%% Difference plots

if comm.Get_rank() == 0:
  # Plot the matrix.
  fig, ax = nF.niceFig("Column", "Row")
  ax.invert_yaxis()
  ax.set_aspect("equal")
  cs = ax.scatter(A.col, A.row, c=A.data)
  nF.addColourBar(fig, cs, "Coefficient value")
  if saveFigs:
      plt.savefig("../Figures_0_initialTests/exampleMatrix_{}.png".format(case.split("/")[-1].replace("data_", "")),
          bbox_inches="tight", dpi=200)
  
  # Print max difference between ReFRESCO and scipy.
  if x_refresco is not None:
  
      x_diff_petsc_scipy = np.log10(np.abs((x_diff+x_refresco)-(x_diff_petsc+x_refresco)))
      print("Max log10 of difference between ReFRESCO and scipy is", x_diff.max())
      print("Max log10 of difference between ReFRESCO and PETSc is", x_diff_petsc.max())
      print("Max log10 of difference between scipy and PETSc is", x_diff_petsc_scipy.max())  
  
      # Plot values.
      fig, ax = nF.niceFig("scipy", "ReFRESCO")
      ax.set_title("Max log$_{{10}}$(difference) = {:.1f}".format(x_diff.max()))
      ax.set_aspect("equal")
      cs = ax.scatter(x, x_refresco, c=x_diff, cmap=plt.cm.bwr)
      nF.addColourBar(fig, cs, "log$_{{10}}$ of difference")
      if saveFigs:
          plt.savefig("../Figures_0_initialTests/exampleSolution_{}_scipy.png".format(case.split("/")[-1].replace("data_", "")),
              bbox_inches="tight", dpi=200)
  
      fig, ax = nF.niceFig("PETSc", "ReFRESCO")
      ax.set_title("Max log$_{{10}}$(difference) = {:.1f}".format(x_diff_petsc.max()))
      ax.set_aspect("equal")
      cs = ax.scatter(x_petsc, x_refresco, c=x_diff_petsc, cmap=plt.cm.bwr)
      nF.addColourBar(fig, cs, "log$_{{10}}$ of difference")
      if saveFigs:
          plt.savefig("../Figures_0_initialTests/exampleSolution_{}_PETSc.png".format(case.split("/")[-1].replace("data_", "")),
              bbox_inches="tight", dpi=200)

      fig, ax = nF.niceFig("PETSc", "scipy")
      ax.set_title("Max log$_{{10}}$(difference) = {:.1f}".format(x_diff_petsc_scipy.max()))
      ax.set_aspect("equal")
      cs = ax.scatter(x_petsc, x, c=x_diff_petsc_scipy, cmap=plt.cm.bwr)
      nF.addColourBar(fig, cs, "log$_{{10}}$ of difference")
      if saveFigs:
          plt.savefig("../Figures_0_initialTests/exampleSolution_{}_PETSc_scipy.png".format(case.split("/")[-1].replace("data_", "")),
              bbox_inches="tight", dpi=200)
          
  plt.show()
  #plt.close()
  #Write Data
  print("Average result time: " + str(end_time-start_time))
