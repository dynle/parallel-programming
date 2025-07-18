#!/bin/bash

# Clean up previous results file
rm -f gaussian_results.txt

echo "--- Compiling the MPI program ---"
mpic++ -o mpi_gaussian_elimination mpi_gaussian_elimination.cpp

echo "--- Running experiments for Figure 4 (Speedup vs. # of Nodes) ---"
echo "Using fixed matrix size N=1024"
# You must modify the C++ code to set N=1024 for these runs
for p in 1 2 4 8 16 32
do
  echo "Running with $p processes..."
  mpirun -n $p ./mpi_gaussian_elimination
done


echo "--- Running experiments for Figure 3 (FLOPS vs. Matrix Size) ---"
echo "Using fixed process count P=8"
# For these runs, you need to edit the C++ code and change the 'const int n' value
# for each run, then re-compile. This is a manual but necessary step.
# Example:
# 1. Change n to 256 in the .cpp file, compile, then run 'mpirun -n 8 ...'
# 2. Change n to 512 in the .cpp file, compile, then run 'mpirun -n 8 ...'
# 3. Change n to 1024 in the .cpp file, compile, then run 'mpirun -n 8 ...'
# 4. Change n to 2048 in the .cpp file, compile, then run 'mpirun -n 8 ...'
echo "NOTE: You must manually change the matrix size 'n' in the C++ code,"
echo "re-compile, and run with 'mpirun -n 8 ./mpi_gaussian_elimination' for each data point."


echo "--- All experiments complete. Data is in gaussian_results.txt ---"