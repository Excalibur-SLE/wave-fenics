#!/bin/bash
#SBATCH --account=mhdz1996-wavefenics
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --job-name=fenics-gpu
#SBATCH --time=0:30:0
#SBATCH --qos=epsrc

module purge
module load baskerville
module load OpenMPI

let ntasks=$SLURM_NNODES*$SLURM_NTASKS_PER_NODE
echo "Executing on: " $ntasks

nsys profile --capture-range=cudaProfilerApi --trace=cuda,mpi mpirun -n $ntasks ./bp1 --p 2 --s 20

# mpirun -n 4 ./bp1 --s 6

