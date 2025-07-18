#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cassert>
#include <mpi.h>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int proc_rank, procs_count;
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procs_count);

    // Set the matrix size
    const int n = 1024;

    if (n % procs_count != 0) {
        if (proc_rank == 0) {
            std::cerr << "Error: Matrix size " << n << " must be divisible by the number of processes " << procs_count << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    // 1-D block distribution
    int rows_per_proc = n / procs_count;

    // Data allocation & initialization
    double* A_full = nullptr;
    double* b_full = nullptr;
    double* x_sol = nullptr;

    if (proc_rank == 0) {
        A_full = new double[n * n];
        b_full = new double[n];
        x_sol = new double[n];
        srand(time(NULL));
        for (int i = 0; i < n; ++i) {
            double row_sum = 0.0;
            for (int j = 0; j < n; ++j) {
                if (i != j) {
                    A_full[i * n + j] = (double)rand() / RAND_MAX;
                    row_sum += A_full[i * n + j];
                }
            }
            A_full[i * n + i] = row_sum + 1.0;
            b_full[i] = (double)rand() / RAND_MAX;
        }
    }

    // Local data allocation
    double* A_local = new double[rows_per_proc * n];
    double* b_local = new double[rows_per_proc];
    double* pivot_row_buf = new double[n];
    double pivot_b_val;

    // Scatter data
    MPI_Scatter(A_full, rows_per_proc * n, MPI_DOUBLE, A_local, rows_per_proc * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(b_full, rows_per_proc, MPI_DOUBLE, b_local, rows_per_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    // Parallel forward elimination
    for (int k = 0; k < n; ++k) {
        int pivot_owner_rank = k / rows_per_proc;
        if (proc_rank == pivot_owner_rank) {
            int local_k = k % rows_per_proc;
            for (int j = 0; j < n; ++j) {
                pivot_row_buf[j] = A_local[local_k * n + j];
            }
            pivot_b_val = b_local[local_k];
        }
        MPI_Bcast(pivot_row_buf, n, MPI_DOUBLE, pivot_owner_rank, MPI_COMM_WORLD);
        MPI_Bcast(&pivot_b_val, 1, MPI_DOUBLE, pivot_owner_rank, MPI_COMM_WORLD);
        for (int i = 0; i < rows_per_proc; ++i) {
            int global_i = proc_rank * rows_per_proc + i;
            if (global_i > k) {
                double factor = A_local[i * n + k] / pivot_row_buf[k];
                for (int j = k; j < n; ++j) {
                    A_local[i * n + j] -= factor * pivot_row_buf[j];
                }
                b_local[i] -= factor * pivot_b_val;
            }
        }
    }

    // Gather results
    MPI_Gather(A_local, rows_per_proc * n, MPI_DOUBLE, A_full, rows_per_proc * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(b_local, rows_per_proc, MPI_DOUBLE, b_full, rows_per_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double end_time = MPI_Wtime();
    double total_time = end_time - start_time;

    // Sequential back substitution & output
    if (proc_rank == 0) {
        for (int i = n - 1; i >= 0; --i) {
            double sum = 0.0;
            for (int j = i + 1; j < n; ++j) {
                sum += A_full[i * n + j] * x_sol[j];
            }
            assert(A_full[i * n + i] != 0.0);
            x_sol[i] = (b_full[i] - sum) / A_full[i * n + i];
        }

        long double gflops = (2.0L * n * n * n) / (3.0L * total_time) / 1e9;

        // Write results to file
        std::ofstream outfile;
        outfile.open("gaussian_results.txt", std::ios::app);
        
        // If the file is empty, write a header
        if (outfile.tellp() == 0) {
            outfile << "MatrixSize ProcessCount ExecutionTime GFLOPS\n";
        }
        
        outfile << n << " " << procs_count << " " << total_time << " " << gflops << "\n";
        outfile.close();

        std::cout << "Results for n=" << n << ", p=" << procs_count << " written to gaussian_results.txt" << std::endl;
    }

    delete[] A_local;
    delete[] b_local;
    delete[] pivot_row_buf;
    if (proc_rank == 0) {
        delete[] A_full;
        delete[] b_full;
        delete[] x_sol;
    }

    MPI_Finalize();
    return 0;
}