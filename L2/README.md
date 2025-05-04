# L2 - Pressure-velocity coupling improvements

## Introduction

The SIMPLE algorithm can be written in matrix-vector form as shown in [^1, ^2, ^3]:

![Pressure-correction](images/navier-stokes.png)

This equation indicates that the pressure-velocity coupling in SIMPLE-like algorithms arises from the approximation of the inverse of the momentum matrix (Q^-1). By using a polynomial approximation of the inverse of Q, the L2 development aims at reducing the number of outer loops of the SIMPLE algorithm.

This repository presents the Python scripts created to test the concept and interact with CFD flow solver ReFRESCO.

## Installation

To install and run codeInterface_L2.py, you need to fulfil the following prerequisites:
- Python 3.9 or newer
- A working MPI implementation
- PETSc (https://petsc.org/release/)

The following Python depedencies are also needed:
- mpi4py 
- petsc4py (instruction for installation of petsc4py are available at https://petsc.org/release/petsc4py/install.html)
- numpy 
- scipy 
- pandas 
- filelock

## Usage


## References

[^1] C. M. Klaij and C. Vuik. SIMPLE-type Preconditioners for Cell-Centered, Colocated Finite Volume Discretization of Incompressible Reynolds-Averaged Navier–Stokes Equations. International Journal for Numerical Methods in Fluids, 71(7):830–849, 2013.
[^2] C. M. Klaij. On the stabilization of finite volume methods with co-located variables for incompressible flow. Journal of Computational Physics, 297:84–89, 2015.
[^3] C. M. Klaij, X. He, and C. Vuik. On the design of block preconditioners for maritime engineering. In MARINE VII: proceedings of the VII International Conference on Computational Methods in Marine Engineering, pages 893–904. CIMNE, 2017.


