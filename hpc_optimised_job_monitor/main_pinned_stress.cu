#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

__global__ void matrixMul(float *a, float *b, float *c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += a[row * n + k] * b[k * n + col];
        }
        // Write to C (in a real loop we might accumulate, but overwriting is fine for heating up GPU)
        c[row * n + col] = sum;
    }
}

int main() {
    int n = 1024; 
    size_t bytes = n * n * sizeof(float);
    int ITERATIONS = 5000; // Run enough times to generate heat!

    std::cout << "Initializing STRESS TEST (N=" << n << ", Iterations=" << ITERATIONS << ")...\n";

    float *h_a, *h_b, *h_c;
    cudaHostAlloc(&h_a, bytes, cudaHostAllocDefault);
    cudaHostAlloc(&h_b, bytes, cudaHostAllocDefault);
    cudaHostAlloc(&h_c, bytes, cudaHostAllocDefault);

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    for (int i = 0; i < n * n; ++i) {
        h_a[i] = 1.0f; h_b[i] = 2.0f; h_c[i] = 0.0f;
    }

    // Transfer Data ONCE
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((n + 15) / 16, (n + 15) / 16);

    std::cout << "Starting Stress Loop... (Check monitor logs now!)\n";

    // --- THE STRESS LOOP ---
    for(int i=0; i<ITERATIONS; ++i) {
        matrixMul<<<blocks, threads>>>(d_a, d_b, d_c, n);
    }
    // -----------------------

    cudaDeviceSynchronize();
    std::cout << "Stress Loop Finished.\n";

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    if (std::abs(h_c[0] - (n * 2.0f)) < 1e-5) 
        std::cout << "SUCCESS: Calculation Correct.\n";
    
    cudaFreeHost(h_a); cudaFreeHost(h_b); cudaFreeHost(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    return 0;
}
