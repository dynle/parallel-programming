#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <mpi.h>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int proc_rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Two processes are required
    if (num_procs != 2) {
        if (proc_rank == 0) {
            std::cerr << "Error: Must be run with exactly 2 processes." << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    // --- Performance Measurement Loop ---

    // proc_rank 0 == root process
    if (proc_rank == 0) {
        std::cout << "--- MPI Communication Performance ---" << std::endl;
        std::cout << std::left << std::setw(20) << "Message Size (B)"
                  << std::left << std::setw(25) << "Time per Send (s)"
                  << std::left << std::setw(20) << "Bandwidth (GB/s)" << std::endl;
        std::cout << std::string(65, '-') << std::endl;
    }

    std::ofstream outfile;
    if (proc_rank == 0) {
        outfile.open("mpi_comm_results.txt");
        outfile << "Message_Size\tTime_per_Send\tBandwidth_GBps\n";
    }

    // Test message sizes from 1 byte up to about 1 GB
    const int max_exponent = 30;
    for (int exp = 0; exp <= max_exponent; ++exp) {
        long long message_size = 1LL << exp;

        std::vector<char> buffer(message_size);
        int tag = 0;

        MPI_Barrier(MPI_COMM_WORLD);

        double start_time, end_time, round_trip_time;

        if (proc_rank == 0) {
            start_time = MPI_Wtime();
            // Send the message
            MPI_Send(buffer.data(), message_size, MPI_CHAR, 1, tag, MPI_COMM_WORLD);

            // Receive the message back
            MPI_Recv(buffer.data(), message_size, MPI_CHAR, 1, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            end_time = MPI_Wtime();
        } else {
            // Receive the message
            MPI_Recv(buffer.data(), message_size, MPI_CHAR, 0, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Send it back
            MPI_Send(buffer.data(), message_size, MPI_CHAR, 0, tag, MPI_COMM_WORLD);
        }

        if (proc_rank == 0) {
            round_trip_time = end_time - start_time;
            double one_way_time = round_trip_time / 2.0;
            
            // Handle division by zero for extremely small timings
            double bandwidth_gb_s = (one_way_time > 0) ? (static_cast<double>(message_size) / one_way_time) / (1024 * 1024 * 1024) : 0;

            // Print results
            std::cout << std::left << std::setw(20) << message_size
                      << std::left << std::setw(25) << std::fixed << std::setprecision(10) << one_way_time
                      << std::left << std::setw(20) << std::setprecision(6) << bandwidth_gb_s << std::endl;
            // Write to file
            outfile << message_size << "\t" << std::fixed << std::setprecision(10) << one_way_time << "\t" << std::setprecision(6) << bandwidth_gb_s << "\n";
        }
    }
    if (proc_rank == 0) {
        outfile.close();
    }
    MPI_Finalize();
    return 0;
}