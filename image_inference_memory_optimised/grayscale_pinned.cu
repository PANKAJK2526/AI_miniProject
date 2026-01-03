#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
// --- THE KERNEL (Same as before) ---
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
    // 1. Load Image into temporary standard memory to get dimensions
    std::string imagePath = "input.jpg"; 
    cv::Mat temp_img = cv::imread(imagePath, cv::IMREAD_COLOR);
if (temp_img.empty()) {
        std::cerr << "Could not open input.jpg!" << std::endl;
        return -1;
    }
int width = temp_img.cols;
    int height = temp_img.rows;
    size_t rgbSize = width * height * 3 * sizeof(unsigned char);
    size_t graySize = width * height * sizeof(unsigned char);
// --- NEW: Allocate PINNED Host Memory ---
    // Instead of using standard malloc (or cv::Mat's default allocator), 
    // we use cudaMallocHost. This locks the memory pages in RAM.
    unsigned char *h_pinned_rgb;
    cudaError_t err = cudaMallocHost((void**)&h_pinned_rgb, rgbSize);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate pinned memory!" << std::endl;
        return -1;
    }
// Copy data from the OpenCV object to our new Pinned Buffer
    // (This is a fast CPU-to-CPU copy)
    memcpy(h_pinned_rgb, temp_img.data, rgbSize);
// 2. Allocate Device Memory (VRAM)
    unsigned char *d_rgb, *d_gray;
    cudaMalloc((void**)&d_rgb, rgbSize);
    cudaMalloc((void**)&d_gray, graySize);
// --- SETUP TIMING EVENTS ---
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
// --- MEASURE 1: Host to Device Transfer (Optimized) ---
    cudaEventRecord(start); 
    
    // Transfer from PINNED memory to DEVICE
    cudaMemcpy(d_rgb, h_pinned_rgb, rgbSize, cudaMemcpyHostToDevice);
    
    cudaEventRecord(stop);  
    cudaEventSynchronize(stop);
    
    float millisecondsTransfer = 0;
    cudaEventElapsedTime(&millisecondsTransfer, start, stop);
// Define Grid/Block dimensions
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
// --- MEASURE 2: Kernel Execution ---
    cudaEventRecord(start);
    rgbToGrayKernel<<<blocksPerGrid, threadsPerBlock>>>(d_rgb, d_gray, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float millisecondsKernel = 0;
    cudaEventElapsedTime(&millisecondsKernel, start, stop);
// 3. Copy Result Back
    cv::Mat h_gray(height, width, CV_8UC1);
    cudaMemcpy(h_gray.data, d_gray, graySize, cudaMemcpyDeviceToHost);
// 4. Report Results
    std::cout << "Optimization: PINNED MEMORY" << std::endl;
    std::cout << "Image Size:   " << width << " x " << height << std::endl;
    std::cout << "-----------------------------------" << std::endl;
    std::cout << "Transfer Time (Host->Device): " << millisecondsTransfer << " ms" << std::endl;
    std::cout << "Kernel Time   (Compute):      " << millisecondsKernel << " ms" << std::endl;
    std::cout << "-----------------------------------" << std::endl;
// 5. Cleanup
    cv::imwrite("output_pinned.jpg", h_gray);
    
    // Free Device Memory
    cudaFree(d_rgb);
    cudaFree(d_gray);
    
    // Free Pinned Host Memory (Must use cudaFreeHost, not free!)
    cudaFreeHost(h_pinned_rgb); 
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
return 0;
}

