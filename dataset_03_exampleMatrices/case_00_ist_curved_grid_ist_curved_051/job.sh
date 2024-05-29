#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=28
#SBATCH --exclusive
#SBATCH --partition=normal
#SBATCH --job-name=P12345.678_suites
#SBATCH --time="10:00:00"
#SBATCH --nice=0

mpirun -n $SLURM_NTASKS  refresco
