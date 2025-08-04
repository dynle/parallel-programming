#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <iomanip>

#define H 2048 // Grid height and width
#define BLOCK_SIZE 16 // CUDA block size (16x16)

// Kernel to initialize the grid with boundary conditions
__global__ void init_kernel(float* u, int h) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = iy * h + ix;

    if (ix >= h || iy >= h) return;

    // Circle 1 (100V)
    float c1_x = 0.25f * h;
    float c1_y = 0.75f * h;
    float c1_r = 0.125f * h;

    // Circle 2 (20V)
    float c2_x = 0.875f * h;
    float c2_y = 0.125f * h;
    float c2_r = 0.05f * h;

    float dist1 = sqrtf(powf(ix - c1_x, 2) + powf(iy - c1_y, 2));
    float dist2 = sqrtf(powf(ix - c2_x, 2) + powf(iy - c2_y, 2));

    if (dist1 <= c1_r) {
        u[idx] = 100.0f;
    } else if (dist2 <= c2_r) {
        u[idx] = 20.0f;
    } else if (ix == 0 || ix == h - 1 || iy == 0 || iy == h - 1) {
        u[idx] = 0.0f;
    } else {
        u[idx] = 0.0f;
    }
}

// Kernel to perform one iteration of the Jacobi method for Laplace's equation
__global__ void laplace_kernel(const float* u_old, float* u_new, int h) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = iy * h + ix;

    if (ix >= h || iy >= h) return;

    if (ix == 0 || ix == h - 1 || iy == 0 || iy == h - 1) {
        u_new[idx] = u_old[idx];
        return;
    }
    
    // Check if inside the fixed-voltage circles
    float c1_x = 0.25f * h; float c1_y = 0.75f * h; float c1_r = 0.125f * h;
    float c2_x = 0.875f * h; float c2_y = 0.125f * h; float c2_r = 0.05f * h;
    float dist1 = sqrtf(powf(ix - c1_x, 2) + powf(iy - c1_y, 2));
    float dist2 = sqrtf(powf(ix - c2_x, 2) + powf(iy - c2_y, 2));

    if (dist1 <= c1_r || dist2 <= c2_r) {
        u_new[idx] = u_old[idx];
        return;
    }

    // Apply the finite difference formula for internal points
    float top    = u_old[idx + h];
    float bottom = u_old[idx - h];
    float left   = u_old[idx - 1];
    float right  = u_old[idx + 1];
    
    u_new[idx] = (top + bottom + left + right) / 4.0f;
}

// Kernel to calculate the maximum difference between two grids
__global__ void diff_kernel(const float* u1, const float* u2, float* err, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        err[i] = fabsf(u1[i] - u2[i]);
    }
}

// Kernel to find the maximum value in an array
__global__ void reduce_max_kernel(float* data, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? data[i] : 0.0f;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        data[blockIdx.x] = sdata[0];
    }
}


int main() {
    const int N = H * H;
    const int MAX_ITERS = 10000;
    const float TOLERANCE = 1e-6f;

    float *h_u = new float[N];
    float *d_u_old, *d_u_new, *d_err;

    cudaMalloc(&d_u_old, N * sizeof(float));
    cudaMalloc(&d_u_new, N * sizeof(float));
    cudaMalloc(&d_err, N * sizeof(float));

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(H / threads.x, H / threads.y);

    init_kernel<<<grid, threads>>>(d_u_old, H);
    cudaMemcpy(d_u_new, d_u_old, N * sizeof(float), cudaMemcpyDeviceToDevice);

    printf("Starting Jacobi iterations\n");
    float max_err = 0.0f;
    int iter = 0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (iter = 0; iter < MAX_ITERS; ++iter) {
        laplace_kernel<<<grid, threads>>>(d_u_old, d_u_new, H);

        // Check for convergence every 100 iterations to reduce overhead
        if (iter % 100 == 0) {
            int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
            diff_kernel<<<num_blocks, BLOCK_SIZE>>>(d_u_new, d_u_old, d_err, N);
            
            // First pass of reduction
            reduce_max_kernel<<<num_blocks, BLOCK_SIZE>>>(d_err, N);
            
            // Second pass to reduce the block results
            int num_blocks_pass2 = (num_blocks + BLOCK_SIZE - 1) / BLOCK_SIZE;
            reduce_max_kernel<<<num_blocks_pass2, BLOCK_SIZE>>>(d_err, num_blocks);

            cudaMemcpy(&max_err, d_err, sizeof(float), cudaMemcpyDeviceToHost);
            
            printf("Iter: %4d, Max Error: %e\n", iter, max_err);
            if (max_err < TOLERANCE) {
                printf("Converged after %d iterations.\n", iter);
                break;
            }
        }

        // For next iteration
        float* temp = d_u_old;
        d_u_old = d_u_new;
        d_u_new = temp;
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Total execution time: %.2f ms\n", ms);
    
    cudaMemcpy(h_u, d_u_old, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Save result to a binary file
    std::ofstream outfile("laplace_solution.dat", std::ios::out | std::ios::binary);
    if(outfile.is_open()) {
        outfile.write(reinterpret_cast<char*>(h_u), N * sizeof(float));
        outfile.close();
        printf("Solution saved to laplace_solution.dat\n");
    }

    cudaFree(d_u_old);
    cudaFree(d_u_new);
    cudaFree(d_err);
    delete[] h_u;

    return 0;
}