#include <iostream>
#include <cuda_runtime.h>
// 1. The Kernel (Runs on GPU)
__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}
int main() {
    int N = 100000; // Number of elements
    size_t size = N * sizeof(float);
// 2. Allocate Host Memory
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
// Initialize inputs
    for (int i = 0; i < N; ++i) { h_A[i] = 1.0f; h_B[i] = 2.0f; }
// 3. Allocate Device Memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);
// 4. Copy Host -> Device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
// 5. Launch Kernel (256 threads per block)
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
// 6. Copy Device -> Host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
// Verify
    std::cout << "First Element: " << h_C[0] << " (Should be 3.0)" << std::endl;
// Free memory
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    
    return 0;
}

