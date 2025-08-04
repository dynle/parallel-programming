#include <iostream>
#include <vector>
#include <string>
#include <fstream>

// CUDA error checking
void checkCudaErrors(cudaError_t err) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Kernel to copy data on the device
__global__ void copyKernel(const float* in, float* out, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = in[i];
    }
}

int main() {
    // For results output
    std::ofstream results_file("bandwidth_results.txt");
    if (!results_file.is_open()) {
        fprintf(stderr, "Error\n");
        return 1;
    }

    std::vector<size_t> test_sizes_mb = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};
    const int ITERS = 20;

    float *h_A; // Host array
    float *d_A, *d_B; // Device arrays

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    // --- Host to Device Transfer Bandwidth ---
    results_file << "# CPU-GPU Transfers\n";
    results_file << "Data_Size_B,Sync_BW_GBs,Async_BW_GBs\n";

    for (size_t mb : test_sizes_mb) {
        size_t bytes = mb * 1024 * 1024;
        size_t n = bytes / sizeof(float);

        h_A = new float[n];
        checkCudaErrors(cudaMalloc(&d_A, bytes));
        
        // Synchronous Test
        cudaEventRecord(start);
        for (int i = 0; i < ITERS; i++) {
            cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms_sync;
        cudaEventElapsedTime(&ms_sync, start, stop);
        double bw_sync = (double)bytes * ITERS / (ms_sync / 1000.0) / 1e9;
        
        // Asynchronous Test
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        cudaEventRecord(start, stream);
        for (int i = 0; i < ITERS; i++) {
            cudaMemcpyAsync(d_A, h_A, bytes, cudaMemcpyHostToDevice, stream);
        }
        cudaEventRecord(stop, stream);
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
        float ms_async;
        cudaEventElapsedTime(&ms_async, start, stop);
        double bw_async = (double)bytes * ITERS / (ms_async / 1000.0) / 1e9;
        
        results_file << bytes << "," << bw_sync << "," << bw_async << "\n";
        
        delete[] h_A;
        cudaFree(d_A);
    }

    // -- Device Memory Bandwidth ---
    results_file << "\n# Device Memory Transfers\n";
    results_file << "Data_Size_B,Device_BW_GBs\n";

    for (size_t mb : test_sizes_mb) {
        size_t bytes = mb * 1024 * 1024;
        size_t n = bytes / sizeof(float);

        checkCudaErrors(cudaMalloc(&d_A, bytes));
        checkCudaErrors(cudaMalloc(&d_B, bytes));

        int threads = 256;
        int blocks = (n + threads - 1) / threads;

        cudaEventRecord(start);
        for (int i = 0; i < ITERS; i++) {
            copyKernel<<<blocks, threads>>>(d_A, d_B, n);
        }
        checkCudaErrors(cudaGetLastError());
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms_kernel;
        cudaEventElapsedTime(&ms_kernel, start, stop);
        double bw_kernel = (2.0 * (double)bytes * ITERS) / (ms_kernel / 1000.0) / 1e9;

        results_file << bytes << "," << bw_kernel << "\n";

        cudaFree(d_A);
        cudaFree(d_B);
    }
    
    results_file.close();
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    printf("Done\n");
    return 0;
}