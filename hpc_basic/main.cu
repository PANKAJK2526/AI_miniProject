#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

// Forward declaration of the topology check
void checkTopology();

// GPU Kernel: The actual "Compute"
__global__ void matrixMul(float *a, float *b, float *c, int n) {
    // Calculate global thread ID
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            // Standard O(N^3) matrix multiplication logic
            sum += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = sum;
    }
}

int main() {
    // 1. Run Pre-flight Check
    checkTopology();

    // 2. Set Problem Size (Matrix N x N)
    // Keep it large enough to see a delay, e.g., 1024 or 2048
    int n = 1024; 
    size_t bytes = n * n * sizeof(float);

    std::cout << "Initializing Matrix Multiplication for N=" << n << "...\n";

    // 3. Allocate Unified Memory (Accessible by CPU and GPU)
    float *a, *b, *c;
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);

    // 4. Initialize Data (on CPU)
    // Note: This triggers page faults as CPU touches GPU-allocated memory!
    for (int i = 0; i < n * n; ++i) {
        a[i] = 1.0f;
        b[i] = 2.0f;
        c[i] = 0.0f;
    }

    // 5. Grid/Block Config
    // Standard 16x16 threads per block
    int threads = 16;
    dim3 threadsPerBlock(threads, threads);
    dim3 blocksPerGrid((n + threads - 1) / threads, (n + threads - 1) / threads);

    std::cout << "Launching Kernel...\n";
    
    // 6. Launch Kernel
    matrixMul<<<blocksPerGrid, threadsPerBlock>>>(a, b, c, n);

    // 7. Synchronization
    // CPU must wait for GPU to finish. Unified Memory doesn't auto-sync execution.
    cudaDeviceSynchronize();

    std::cout << "Kernel Finished. Verifying...\n";

    // 8. Verify Result (Check top-left corner)
    // Expected result: 1.0 * 2.0 * N = 2.0 * 1024 = 2048.0
    float expected = n * 2.0f;
    if (std::abs(c[0] - expected) < 1e-5) {
        std::cout << "SUCCESS: Result " << c[0] << " matches expected " << expected << "\n";
    } else {
        std::cout << "FAILURE: Result " << c[0] << " != " << expected << "\n";
    }

    // 9. Cleanup
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    return 0;
}
