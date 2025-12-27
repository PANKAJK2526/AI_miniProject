#include <iostream>
#include <cuda_runtime.h>

void checkTopology() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    std::cout << "--- PRE-FLIGHT TOPOLOGY CHECK ---\n";
    std::cout << "Detected " << deviceCount << " CUDA Device(s).\n";

    if (deviceCount < 2) {
        std::cout << "[WARN] Single GPU detected. P2P (NVLink/PCIe) features will be disabled.\n";
        // In a real HPC job, you might exit here if the job requires multi-gpu
        return;
    }

    for (int i = 0; i < deviceCount; ++i) {
        for (int j = 0; j < deviceCount; ++j) {
            if (i != j) {
                int canAccess;
                cudaDeviceCanAccessPeer(&canAccess, i, j);
                std::cout << "Device " << i << " can access Device " << j << ": " 
                          << (canAccess ? "YES (P2P Available)" : "NO") << "\n";
            }
        }
    }
    std::cout << "---------------------------------\n\n";
}
