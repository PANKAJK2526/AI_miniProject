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
        c[row * n + col] = sum;
    }
}

int main() {
    int n = 1024;
    size_t bytes = n * n * sizeof(float);
    std::cout << "Initializing PINNED Memory Benchmark (N=" << n << ")...\n";

    // 1. Allocate PINNED Host Memory (CPU side, fast transfer)
    // "cudaHostAllocDefault" makes it page-locked
    float *h_a, *h_b, *h_c;
    cudaHostAlloc(&h_a, bytes, cudaHostAllocDefault);
    cudaHostAlloc(&h_b, bytes, cudaHostAllocDefault);
    cudaHostAlloc(&h_c, bytes, cudaHostAllocDefault);

    // 2. Allocate Device Memory (GPU side only)
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Initialize Data on Host
    for (int i = 0; i < n * n; ++i) {
        h_a[i] = 1.0f; h_b[i] = 2.0f; h_c[i] = 0.0f;
    }

    std::cout << "Starting Explicit Transfers and Kernel...\n";

    // 3. Explicit Transfer H -> D
    // This happens in ONE efficient block, not 105 tiny chunks.
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Grid Config
    dim3 threads(16, 16);
    dim3 blocks((n + 15) / 16, (n + 15) / 16);

    // 4. Launch Kernel
    // Note: GPU now has all data localized in VRAM. No page faults possible.
    matrixMul<<<blocks, threads>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();

    // 5. Explicit Transfer D -> H
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // Verify
    if (std::abs(h_c[0] - (n * 2.0f)) < 1e-5) 
        std::cout << "SUCCESS: Calculation Correct.\n";
    
    // Cleanup
    cudaFreeHost(h_a); cudaFreeHost(h_b); cudaFreeHost(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    return 0;
}

