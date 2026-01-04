#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
// --- THE KERNEL (Runs on GPU) ---
__global__ void rgbToGrayKernel(unsigned char* d_in, unsigned char* d_out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
if (x < width && y < height) {
        int grayOffset = y * width + x;
        int rgbOffset = grayOffset * 3;
unsigned char r = d_in[rgbOffset];
        unsigned char g = d_in[rgbOffset + 1];
        unsigned char b = d_in[rgbOffset + 2];
d_out[grayOffset] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
    }
}
int main() {
    // 1. Load Image
    std::string imagePath = "input.jpg"; 
    cv::Mat h_img = cv::imread(imagePath, cv::IMREAD_COLOR);
if (h_img.empty()) {
        std::cerr << "Could not open input.jpg! Make sure it is in the folder." << std::endl;
        return -1;
    }
int width = h_img.cols;
    int height = h_img.rows;
    size_t rgbSize = width * height * 3 * sizeof(unsigned char);
    size_t graySize = width * height * sizeof(unsigned char);
// 2. Allocate Device Memory
    unsigned char *d_rgb, *d_gray;
    cudaMalloc((void**)&d_rgb, rgbSize);
    cudaMalloc((void**)&d_gray, graySize);
// --- SETUP TIMING EVENTS ---
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
// --- MEASURE 1: Host to Device Transfer ---
    cudaEventRecord(start); // Start timer
    
    // The operation we are timing:
    cudaMemcpy(d_rgb, h_img.data, rgbSize, cudaMemcpyHostToDevice);
    
    cudaEventRecord(stop);  // Stop timer
    cudaEventSynchronize(stop); // Wait for the GPU to confirm it finished
    
    float millisecondsTransfer = 0;
    cudaEventElapsedTime(&millisecondsTransfer, start, stop);
// Define Grid/Block dimensions
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
// --- MEASURE 2: Kernel Execution ---
    cudaEventRecord(start); // Start timer
    
    // The operation we are timing:
    rgbToGrayKernel<<<blocksPerGrid, threadsPerBlock>>>(d_rgb, d_gray, width, height);
    
    cudaEventRecord(stop);  // Stop timer
    cudaEventSynchronize(stop); // Wait for the GPU to confirm it finished
    
    float millisecondsKernel = 0;
    cudaEventElapsedTime(&millisecondsKernel, start, stop);
// 3. Copy Result Back (Device -> Host)
    // We usually don't time this for "algorithm speed", but it is part of the cost.
    cv::Mat h_gray(height, width, CV_8UC1);
    cudaMemcpy(h_gray.data, d_gray, graySize, cudaMemcpyDeviceToHost);
// 4. Report Results
    std::cout << "Image Size: " << width << " x " << height << std::endl;
    std::cout << "-----------------------------------" << std::endl;
    std::cout << "Transfer Time (Host->Device): " << millisecondsTransfer << " ms" << std::endl;
    std::cout << "Kernel Time   (Compute):      " << millisecondsKernel << " ms" << std::endl;
    std::cout << "-----------------------------------" << std::endl;
if (millisecondsTransfer > millisecondsKernel) {
        std::cout << "CONCLUSION: This is MEMORY BOUND (Transfer took longer)." << std::endl;
    } else {
        std::cout << "CONCLUSION: This is COMPUTE BOUND (Calculation took longer)." << std::endl;
    }
// 5. Cleanup
    cv::imwrite("output_gray.jpg", h_gray);
    cudaFree(d_rgb);
    cudaFree(d_gray);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
return 0;
}

