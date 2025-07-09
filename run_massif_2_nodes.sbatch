#!/bin/bash
#SBATCH -J assignment-MPI
#SBATCH --time=06:00:00
#SBATCH -N 2
## module load gnu openmpi
SRC="main-v2.c"
BASE_EXE="main-v2"

MATRIX_SIZES=(500 1000 2000)
PROCS_LIST=(1 4 9 16 25)
echo "MATRIX _SIZE,PROCS,ITERATION,RANK,PEAK_HEAP_BYTES" > massif_summary.csv
for ((i=0; i<30; i++)); do
echo "iteration $i"
for size in "${MATRIX_SIZES[@]}"; do
  for procs in "${PROCS_LIST[@]}"; do
    echo "Compiling with MATRIX_DIM=$size MAX_PROCESSES=$procs ..."
    mpicc -DMATRIX_DIM=$size -DMAX_PROCESSES=$procs -o $BASE_EXE $SRC
    echo "Running with $procs processes, rank 0 under massif ..."
    mpirun -np $procs --oversubscribe bash -c '
 echo "Rank ${OMPI_COMM_WORLD_RANK} running on IP: $(hostname -I | awk '\''{print $1}'\'')"
valgrind --tool=massif --massif-out-file=massif_'$size'_'$procs'_'$i'_rank${OMPI_COMM_WORLD_RANK}.out ./'$BASE_EXE'

  peak=$(awk "
    /snapshot=/ {memheap=\"\"}
    /mem_heap_B=/ {memheap=\$0}
    /heap_tree=peak/ {if(memheap) {match(memheap, /mem_heap_B=([0-9]+)/, a); print a[1]; exit}}
  " massif_'$size'_'$procs'_'$i'_rank${OMPI_COMM_WORLD_RANK}.out)

  echo "'$size','$procs','$i',${OMPI_COMM_WORLD_RANK},$peak" >> massif_summary.csv
  rm -f massif_'$size'_'$procs'_'$i'_rank${OMPI_COMM_WORLD_RANK}.out
'

    echo "Run completed for size=$size procs=$procs"
    echo "-----------------------------"
  done
done
done
