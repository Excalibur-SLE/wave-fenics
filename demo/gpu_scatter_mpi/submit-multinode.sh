#!/bin/bash
#SBATCH --account=mhdz1996-wavefenics
#SBATCH --nodes=2
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --job-name=fenics-gpu
#SBATCH --time=0:30:0
#SBATCH --qos=epsrc

module purge
module load baskerville
module load OpenMPI

mpirun -n 8 ./planar3d --size=50 --degree=4

