#include <iostream>
#include <cuda_runtime.h>
int main() {
    int deviceCount;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
if (error_id != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount returned " << static_cast<int>(error_id)
                  << "\n-> " << cudaGetErrorString(error_id) << std::endl;
        return 1;
    }
if (deviceCount == 0) {
        std::cout << "There are no available device(s) that support CUDA\n";
    } else {
        std::cout << "Detected " << deviceCount << " CUDA Capable device(s)\n";
    }
for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
std::cout << "\nDevice " << dev << ": \"" << deviceProp.name << "\"\n";
        std::cout << "--------------------------------------------------\n";
// 1. Max Threads per Block
        std::cout << "  Max Threads per Block:      " 
                  << deviceProp.maxThreadsPerBlock << "\n";
// 2. Shared Memory per Block
        std::cout << "  Shared Memory per Block:    " 
                  << deviceProp.sharedMemPerBlock / 1024.0 << " KB\n";
// 3. Total Global Memory
        std::cout << "  Total Global Memory:        " 
                  << deviceProp.totalGlobalMem / (1024.0 * 1024.0 * 1024.0) << " GB\n";
// 4. Warp Size
        std::cout << "  Warp Size:                  " 
                  << deviceProp.warpSize << "\n";
        
        // Extra: Max Block Dimensions (x, y, z)
        std::cout << "  Max Block Dimensions:       [" 
                  << deviceProp.maxThreadsDim[0] << ", "
                  << deviceProp.maxThreadsDim[1] << ", "
                  << deviceProp.maxThreadsDim[2] << "]\n";
    }
return 0;
}

