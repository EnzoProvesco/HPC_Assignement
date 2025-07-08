#!/bin/bash

SRC="main-v2.c"
BASE_EXE="main-v2"

MATRIX_SIZES=(500 1000 2000)
PROCS_LIST=(1 4 9 16 25 36 49 64)
for ((i=0; i<100; i++)); do
  for size in "${MATRIX_SIZES[@]}"; do
    for procs in "${PROCS_LIST[@]}"; do
      echo "Compiling with MATRIX_DIM=$size MAX_PROCESSES=$procs ..."
      mpicc -DMATRIX_DIM=$size -DMAX_PROCESSES=$procs -o $BASE_EXE $SRC

      echo "Running with $procs processes, rank 0 under massif ..."
      mpirun -np $procs -H node1,master --oversubscribe bash -c '
        if [ "$OMPI_COMM_WORLD_RANK" -eq 0 ]; then
          valgrind --tool=massif --time-unit=ms --massif-out-file=massif_'$size'_'$procs'.out ./'$BASE_EXE'
        else
          ./'$BASE_EXE'
        fi
      '

      echo "Run completed for size=$size procs=$procs"
      echo "-----------------------------"
    done
  done
done
