#!/bin/bash

SRC="main-v2.c"
BASE_EXE="main-v2"

MATRIX_SIZES=(500 1000 2000)
PROCS_LIST=(1 4 9 16)
for ((i=0; i<10; i++)); do
for size in "${MATRIX_SIZES[@]}"; do
  for procs in "${PROCS_LIST[@]}"; do
    echo "Compiling with MATRIX_DIM=$size MAX_PROCESSES=$procs ..."
    mpicc -DMATRIX_DIM=$size -DMAX_PROCESSES=$procs -o $BASE_EXE $SRC

    echo "Running with $procs processes, rank 0 under massif ..."
    mpirun -np $procs --oversubscribe bash -c '
      TMP_LOG="leak_summary_rank_${OMPI_COMM_WORLD_RANK}.tmp"
      echo "===== Rank $OMPI_COMM_WORLD_RANK Leak Summary (Run '$i', Procs '$procs', Size '$size') =====" > $TMP_LOG
      valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes \
        ./'$BASE_EXE' 2>&1 | awk "/==.*LEAK SUMMARY:/,/==.*suppressed:/" >> $TMP_LOG
      echo "" >> $TMP_LOG
      cat $TMP_LOG >> leak_summary.log
      rm $TMP_LOG
    '

    echo "Run completed for size=$size procs=$procs"
    echo "-----------------------------"
  done
done
done