#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --exclusive
#SBATCH --partition=normal
#SBATCH --job-name=P35662.700_suites
#SBATCH --time="10:00:00"
#SBATCH --nice=0

module load slurm/marclus3/23.02.6 oneapi/tbb/2021.4.0 gcc9/9.2.0 oneapi/mpi/2021.4.0 oneapi/compiler-rt/2021.4.0 gcc/9.2.0 \
    hdf5/intel/1.12.2 boost/oneapi/1.73.0 petsc/oneapi/3.19.4 oneapi/mkl/2021.4.0 \
    cgns/intel/4.1.2 oneapi/compiler/2021.4.0 precice/oneapi/2.5.0_PETSC-3.19.4 xsimulation/Core-2023.10.1

export SUGGARPP_LICENSE_PATH=27031@suggar.license.marin.local
export SUGGARPP_LICENSE_PATH=27031@172.16.202.124
export REFRESCO_CODE_DIR=~/ReFRESCO/trunk/trunk/Code
export SUITES_DIR=/home/alidtke/ReFRESCO/trunk/trunk/Suites
export REFRESCO_INSTALL_DIR=~/ReFRESCO/install
export REFRESCO_EXTLIBS_DIR=/home/alidtke/ReFRESCO/install/extLibs
source $REFRESCO_INSTALL_DIR/bin/refresco-run.sh

mpirun -np $SLURM_NTASKS refresco
