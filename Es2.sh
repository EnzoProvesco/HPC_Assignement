#!/bin/bash
if [ "$#" -ne 2 ]; then
    echo "Error: Incorrect number of arguments."
    echo "Usage: $0 <input_directory> <output_directory>"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"

#mixto Compilation
export PATH=/usr/local/cuda-11.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH
export CUDA_N_THREADS=32
nvidia-smi
echo "----------  Starting Compiling  ------------------"

nvcc Es2.cu -o Es2 \
  -I/usr/include/opencv4 \
  -L/usr/lib/aarch64-linux-gnu \
  -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc \
  -lstdc++ -lcudart

# Run for each image in noise directory
counter=0
for [$counter -lt 30]; do
    for img in "$INPUT_DIR"/*.jpg; do
        if [ -f "$img" ]; then
            img_directory=$(dirname "$img")
            img_filename=$(basename "$img")
            echo "Running with nsys profiling..."
            nsys profile \
            --trace=cuda \
            --cuda-memory-usage=true\
            -o ./nsysProfile/${img_filename}.nsys-rep \
            --force-overwrite true \
            ./Es2 "$INPUT_DIR"/"$img_filename" "$OUTPUT_DIR"/BLUR"$img_filename"

            nsys stats -f csv -o ./nsysProfile/reportCSV${counter}/${img_filename} -r gpumemsizesum  ./nsysProfile/${img_filename}.nsys-rep
        else
            echo "No images found in $INPUT_DIR."
        fi
    done
done
