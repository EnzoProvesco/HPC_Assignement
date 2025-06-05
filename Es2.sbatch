#!/bin/bash
#SBATCH -J OpenMP-hola-example
#SBATCH --time=0:05:00
#SBATCH -n 5
#SBATCH --mail-user=s343738@studenti.polito.it
#SBATCH --mail-type=end

module load opencv
g++ main.cpp -o my_app `pkg-config --cflags --libs opencv4`
./Es2