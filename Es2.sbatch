#!/bin/bash
#SBATCH --job-name=GPU_exercise
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s343738@studenti.polito.it
#SBATCH --partition=cuda
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=GPU_exercise.log
#SBATCH --mem-per-cpu=1M
#SBATCH --gres=gpu:1

module load nvidia/cudasdk/10.1
module load opencv




#mixto Compilation
export PATH=/usr/local/cuda-11.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH

nvidia-smi

nvcc filter.cu -o filter \
  -I/usr/include/opencv4 \
  -L/usr/lib/aarch64-linux-gnu \
  -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc \
  -lstdc++ -lcudart


#Hactar Compilation
nvidia-smi

nvcc Es2.cpp -o Es2 \
  -I/share/apps/opencv/4.5.0/include/opencv4 \
  -L/share/apps/opencv/4.5.0/lib64 \
  -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc


./Es2

# nsys profile -o output_report ./CUDA_example/add.o

# ncu -o kernel_profile ./CUDA_example/add.o

