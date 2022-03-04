#!/bin/bash
#SBATCH --account=mhdz1996-wavefenics
#SBATCH --nodes=4
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --job-name=fenics-gpu
#SBATCH --time=0:30:0
#SBATCH --qos=epsrc

module purge
module load baskerville
module load OpenMPI

env
let ntasks=$SLURM_NNODES*$SLURM_NTASKS_PER_NODE
echo "Executing on: " $ntasks

mpirun -n $ntasks ./planar3d --size=100 --degree=4

