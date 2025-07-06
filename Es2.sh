#mixto Compilation
export PATH=/usr/local/cuda-11.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH
export CUDA_N_THREADS=32
nvidia-smi

nvcc Es2.cu -o Es2 \
  -I/usr/include/opencv4 \
  -L/usr/lib/aarch64-linux-gnu \
  -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc \
  -lstdc++ -lcudart

./Es2 -s ./input/reference/pexels-christian-heitz.jpg ./output/reference/pexels-christian-heitz.jpg
