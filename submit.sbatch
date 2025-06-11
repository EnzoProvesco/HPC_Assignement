#!/bin/bash
#SBATCH -J assignment-MPI
#SBATCH --time=00:15:00
#SBATCH -n 4

## module load gnu openmpi
mpicc -o main-v2 ./main-v2.c -lm
mpirun ./main-v2
