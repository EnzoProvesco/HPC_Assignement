    #!/bin/bash
#SBATCH -J assignment-MPI
#SBATCH --time=00:05:00
#SBATCH -n 10
## module load gnu openmpi
mpirun --oversubscribe -np 6 /home/denned/hpc_assignment/main-v2