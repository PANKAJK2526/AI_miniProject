#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
// The CUDA Kernel
__global__ void rgbToGrayKernel(unsigned char* d_in, unsigned char* d_out, int width, int height) {
    // Calculate 2D coordinates from 1D grid
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
if (x < width && y < height) {
        // Calculate the linear index for the gray image (1 channel)
        int grayOffset = y * width + x;
        // Calculate the linear index for the RGB image (3 channels)
        int rgbOffset = grayOffset * 3;
unsigned char r = d_in[rgbOffset];
        unsigned char g = d_in[rgbOffset + 1];
        unsigned char b = d_in[rgbOffset + 2];
// Standard grayscale formula
        d_out[grayOffset] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
    }
}
int main() {
    // 1. Load Image using OpenCV
    std::string imagePath = "input.jpg"; // Make sure you have an image named input.jpg!
    cv::Mat h_img = cv::imread(imagePath, cv::IMREAD_COLOR);
if (h_img.empty()) {
        std::cerr << "Could not open image!" << std::endl;
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
// 3. Copy Host -> Device
    cudaMemcpy(d_rgb, h_img.data, rgbSize, cudaMemcpyHostToDevice);
// 4. Launch Kernel
    // We use a 2D block configuration (32x32 = 1024 threads per block)
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
rgbToGrayKernel<<<blocksPerGrid, threadsPerBlock>>>(d_rgb, d_gray, width, height);
    cudaDeviceSynchronize(); // Wait for GPU to finish
// 5. Copy Device -> Host
    cv::Mat h_gray(height, width, CV_8UC1); // Create empty container for result
    cudaMemcpy(h_gray.data, d_gray, graySize, cudaMemcpyDeviceToHost);
// 6. Save Result
    cv::imwrite("output_gray.jpg", h_gray);
    std::cout << "Success! Saved output_gray.jpg" << std::endl;
// Free memory
    cudaFree(d_rgb);
    cudaFree(d_gray);
return 0;
}

