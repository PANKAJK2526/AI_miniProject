
#include <iostream>
#include <cuda_runtime.h>
// Define the size of the tile (16x16)
#define TILE_WIDTH 16
// --- THE KERNEL (Optimized with Shared Memory) ---
__global__ void matrixMulTiled(const float *A, const float *B, float *C, int width) {
    // 1. Define Shared Memory for the tile
    // This memory lives ON the chip (L1 Cache), not in VRAM!
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];
int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    float value = 0.0f;
// Loop over the matrix in "Tiles"
    for (int phase = 0; phase < width / TILE_WIDTH; ++phase) {
        // --- STEP 1: LOAD DATA INTO SHARED MEMORY ---
        // Collaborative loading: Each thread loads one element
        As[threadIdx.y][threadIdx.x] = A[row * width + (phase * TILE_WIDTH + threadIdx.x)];
        Bs[threadIdx.y][threadIdx.x] = B[(phase * TILE_WIDTH + threadIdx.y) * width + col];
// BARRIER: Wait for all threads to finish loading tile
        __syncthreads();
// --- STEP 2: COMPUTE (Using Fast Shared Memory) ---
        for (int k = 0; k < TILE_WIDTH; ++k) {
            value += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
// BARRIER: Wait for computation to finish before overwriting next tile
        __syncthreads();
    }
if (row < width && col < width) {
        C[row * width + col] = value;
    }
}
int main() {
    int width = 1024; // 1024 x 1024 Matrix
    size_t size = width * width * sizeof(float);
// --- HOST ALLOCATION (Topic 3 Optimization: Pinned Memory) ---
    // Using cudaHostAlloc instead of malloc to avoid pageable memory issues.
    float *h_A, *h_B, *h_C;
    cudaHostAlloc(&h_A, size, cudaHostAllocDefault);
    cudaHostAlloc(&h_B, size, cudaHostAllocDefault);
    cudaHostAlloc(&h_C, size, cudaHostAllocDefault);
// Initialize Data
    for (int i = 0; i < width * width; ++i) { h_A[i] = 1.0f; h_B[i] = 1.0f; }
// Device Allocation
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
std::cout << "Initializing TILED + PINNED Memory Benchmark (N=" << width << ")..." << std::endl;
// --- EXPLICIT TRANSFERS (Topic 3 Optimization) ---
    // Transfer 100% of data before compute begins to saturate PCIe bus.
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
// Launch Config
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 blocksPerGrid(width / TILE_WIDTH, width / TILE_WIDTH);
// Launch Kernel
    matrixMulTiled<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, width);
    cudaDeviceSynchronize();
// Copy Result Back
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
// Verification
    std::cout << "Check [0][0]: " << h_C[0] << " (Expect " << width << ")" << std::endl;
// Cleanup (Must use cudaFreeHost for pinned memory)
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFreeHost(h_A); cudaFreeHost(h_B); cudaFreeHost(h_C);
    return 0;
}

