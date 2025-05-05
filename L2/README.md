# L2 - Pressure-velocity coupling improvements

## Introduction

The SIMPLE algorithm can be written in matrix-vector form as shown in [^1][^2][^3]:

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

Two configuration files are available in the configure directory. One uses a shell script to create a Python enviroment and install and dependencies:
```
#!/bin/bash

echo "Setting up Python virtual environment for exaSIMPLE L2 scripts ..."

# Exit on any error
set -e

# Default values
ENV_NAME="venv"

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --env_name)
            ENV_NAME="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: ./configure.sh [--env_name NAME]"
            echo "Options:"
            echo "  --env_name NAME   Name of the virtual environment directory (default: venv)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run with --help for usage."
            exit 1
            ;;
    esac
done

# Check for Python 3
if ! command -v python3 &> /dev/null
then
    echo "Error: Python 3 is not installed or not in your PATH."
    echo " Please install Python 3.8+ before running this script."
    exit 1
fi

# Create and activate virtual environment
python3 -m venv "$ENV_NAME"
source "$ENV_NAME/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install required Python packages
echo "Installing required packages..."
pip install numpy scipy pandas filelock

echo "Python environment setup complete!"
echo "To activate later, run:"
echo "source $ENV_NAME/bin/activate"

echo ""
echo "PETSc must be installed separately with MPI support."
echo "Set PETSC_DIR and PETSC_ARCH before running the solver:"
echo "export PETSC_DIR=/path/to/petsc"
echo "export PETSC_ARCH=arch-linux-c-opt"
```

The othe configuration file is a YAML file to use with conda as follows:
```
conda env create --file configure.yml --name {Enviroment Name}
```
The YAML file contains:
```
channels:
  - conda-forge
dependencies:
  - python=3.10
  - numpy
  - scipy
  - pandas
  - pip
  - pip:
      - filelock
```

## Usage

exaSIMPLE is design to be used in HPC environments. For use with SLURM job scheduler an example job file is provided below:
```
#!/bin/bash
#SBATCH -n {NUMBER OF TASKS}
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=1
#SBATCH --partition={PARTITION NAME}
#SBATCH --job-name={JOB NAME}
#SBATCH --time={JOB NAME}

#Load ReFRESCO

#Activate python enviroment
conda activate {ENVIROMENT NAME}
#or
source {PATH TO ENV}/bin/activate

srun -n {NUMBER OF TASKS}/2 --exclusive refresco > CFD_code.out &
srun -n {NUMBER OF TASKS}/2 --exclusive python codeInterface_L2.py arg1 > python.out &

wait

```

The script codeInterface.py takes one of three arguments (diagInverse, approxInverse, approxInverse_w), which chooses the type of approximation for the momentum matrix inverse.

## References

[^1] C. M. Klaij and C. Vuik. SIMPLE-type Preconditioners for Cell-Centered, Colocated Finite Volume Discretization of Incompressible Reynolds-Averaged Navier–Stokes Equations. International Journal for Numerical Methods in Fluids, 71(7):830–849, 2013.
[^2] C. M. Klaij. On the stabilization of finite volume methods with co-located variables for incompressible flow. Journal of Computational Physics, 297:84–89, 2015.
[^3] C. M. Klaij, X. He, and C. Vuik. On the design of block preconditioners for maritime engineering. In MARINE VII: proceedings of the VII International Conference on Computational Methods in Marine Engineering, pages 893–904. CIMNE, 2017.


