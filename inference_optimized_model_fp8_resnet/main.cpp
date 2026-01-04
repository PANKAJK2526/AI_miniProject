#include <fstream>
#include <iostream>
#include <vector>
#include <memory>
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <opencv2/opencv.hpp>

// Logger for TensorRT
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) 
            std::cout << "[TRT] " << msg << std::endl;
    }
} gLogger;

// Helper to calculate size
size_t getSizeByDim(const nvinfer1::Dims& dims) {
    size_t size = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        size *= dims.d[i];
    }
    return size * sizeof(float);
}

// Preprocess Image
void preprocessImage(const std::string& imagePath, float* gpuInputBuffer, int inputSize) {
    cv::Mat img = cv::imread(imagePath);
    if (img.empty()) { 
        std::cerr << "Error reading image: " << imagePath << std::endl; 
        exit(1); 
    }

    // Resize and Convert
    cv::resize(img, img, cv::Size(224, 224));
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    // Normalize (ImageNet Mean/Std)
    float mean[] = {0.485, 0.456, 0.406};
    float std[]  = {0.229, 0.224, 0.225};

    int width = 224, height = 224, channels = 3;
    std::vector<float> cpuInput(inputSize / sizeof(float));

    for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                float pixel = img.at<cv::Vec3b>(h, w)[c] / 255.0f;
                cpuInput[c * width * height + h * width + w] = (pixel - mean[c]) / std[c];
            }
        }
    }

    // Copy to GPU
    cudaError_t err = cudaMemcpy(gpuInputBuffer, cpuInput.data(), inputSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Memcpy Error: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

int main() {
    std::string engineFile = "resnet50_int8.plan"; // Correct File
    std::string imageFile = "dog.jpg";

    // 1. Load Engine
    std::ifstream file(engineFile, std::ios::binary);
    if (!file.good()) { 
        std::cerr << "Error loading engine file: " << engineFile << std::endl; 
        return 1; 
    }
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);
    std::vector<char> engineData(size);
    file.read(engineData.data(), size);

    // 2. Initialize Runtime
    std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(gLogger));
    std::unique_ptr<nvinfer1::ICudaEngine> engine(runtime->deserializeCudaEngine(engineData.data(), size));
    std::unique_ptr<nvinfer1::IExecutionContext> context(engine->createExecutionContext());

    // 3. Setup Buffers & Dimensions
    void* buffers[2];

    // --- INPUT: "data" ---
    int inputIndex = engine->getBindingIndex("data");
    if (inputIndex == -1) { std::cerr << "Error: Input 'data' not found." << std::endl; return 1; }
    
    // Force Batch Size 1 (Fixes the -1 issue)
    nvinfer1::Dims inputDims = engine->getBindingDimensions(inputIndex);
    inputDims.d[0] = 1; 
    context->setBindingDimensions(inputIndex, inputDims); // Critical for Dynamic Shapes!

    size_t inputSize = getSizeByDim(inputDims);
    cudaMalloc(&buffers[inputIndex], inputSize);

    // --- OUTPUT: "resnetv17_dense0_fwd" ---
    int outputIndex = engine->getBindingIndex("resnetv17_dense0_fwd");
    if (outputIndex == -1) { std::cerr << "Error: Output 'resnetv17_dense0_fwd' not found." << std::endl; return 1; }

    nvinfer1::Dims outputDims = engine->getBindingDimensions(outputIndex);
    outputDims.d[0] = 1; // Force Batch Size 1

    size_t outputSize = getSizeByDim(outputDims);
    cudaMalloc(&buffers[outputIndex], outputSize);

    std::cout << "Engine Loaded. Running Inference on " << imageFile << "..." << std::endl;

    // 4. Preprocess & Transfer
    preprocessImage(imageFile, (float*)buffers[inputIndex], inputSize);

    // 5. Execute
    if (!context->executeV2(buffers)) {
        std::cerr << "Execution Failed!" << std::endl;
        return 1;
    }

    // 6. Get Output
    std::vector<float> cpuOutput(outputSize / sizeof(float));
    cudaMemcpy(cpuOutput.data(), buffers[outputIndex], outputSize, cudaMemcpyDeviceToHost);

    // 7. Find Max
    int maxIdx = 0;
    float maxVal = -1.0f;
    for (int i = 0; i < cpuOutput.size(); ++i) {
        if (cpuOutput[i] > maxVal) {
            maxVal = cpuOutput[i];
            maxIdx = i;
        }
    }

    // 8. Print Result
    std::cout << "-------------------------" << std::endl;
    std::cout << "Class Index: " << maxIdx << std::endl;
    std::cout << "Confidence:  " << maxVal << std::endl;
    std::cout << "-------------------------" << std::endl;

    cudaFree(buffers[inputIndex]);
    cudaFree(buffers[outputIndex]);
    return 0;
}
