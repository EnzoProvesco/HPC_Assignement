#!/bin/bash
#SBATCH -J cudaExerciseAssignement
#SBATCH --time=0:05:00
#SBATCH -n 5
#SBATCH --mail-user=s343738@studenti.polito.it
#SBATCH --mail-type=end

# load cuda module
module load nvidia/cudasdk/10.1
module load opencv

nvidia-smi

nvcc Es2.cpp -o Es2 \
  -I/share/apps/opencv/4.5.0/include/opencv4 \
  -L/share/apps/opencv/4.5.0/lib64 \
  -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc

./Es2

# nsys profile -o output_report ./CUDA_example/add.o

# ncu -o kernel_profile ./CUDA_example/add.o

