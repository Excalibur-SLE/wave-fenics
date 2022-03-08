#!/bin/bash
#SBATCH --account=mhdz1996-wavefenics
#SBATCH --reservation=mhdz1996-wavefenics
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --job-name=fenics-gpu
#SBATCH --time=0:30:0
#SBATCH --qos=epsrc

module purge
module load baskerville
module load OpenMPI

# nsys profile --capture-range=cudaProfilerApi --trace=cuda,mpi mpirun -n 4 ./bp1

mpirun -n 4 ./bp1 --s 6

