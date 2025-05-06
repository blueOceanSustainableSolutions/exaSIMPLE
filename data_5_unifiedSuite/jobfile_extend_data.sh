#!/bin/bash

#SBATCH --partition=normal-x86
#SBATCH --job-name=exasimple_benchmark
#SBATCH --time=2-00:00:00
#SBATCH --account=f202414686cpcaa3x
#SBATCH --nodes=1
# Note: 128 cores per node
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=16

M=$SLURM_NTASKS
n=16

for ((i=0; i<M; i++)); do
    START_CORE=$(( i * n ))
    END_CORE=$(( START_CORE + n - 1 ))

    echo "Starting instance $i on cores $START_CORE-$END_CORE"

    # Launch separate mpirun instances with CPU binding
    taskset -c $START_CORE-$END_CORE mpirun -np $n python extend_data_reorder.py --instance_id=$i --n_tasks=$M &
done

wait

echo "Done!"
