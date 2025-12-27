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
        if (severity <= Severity::kWARNING) // Only log warnings and errors
            std::cout << "[TRT] " << msg << std::endl;
    }
} gLogger;

// Helper to calculate total buffer size
size_t getSizeByDim(const nvinfer1::Dims& dims) {
    size_t size = 1;
    for (int i = 0; i < dims.nbDims; ++i) size *= dims.d[i];
    return size * sizeof(float);
}

// Preprocess Image (Resize & Normalize)
// ResNet expects: (Pixel / 255.0 - Mean) / StdDev
void preprocessImage(const std::string& imagePath, float* gpuInputBuffer, int inputSize) {
    cv::Mat img = cv::imread(imagePath);
    if (img.empty()) { std::cerr << "Error reading image!" << std::endl; exit(1); }

    // Resize to 224x224
    cv::resize(img, img, cv::Size(224, 224));
    
    // Convert BGR to RGB
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    // Convert to Float (CHW format)
    // Mean: {0.485, 0.456, 0.406}, Std: {0.229, 0.224, 0.225}
    float mean[] = {0.485, 0.456, 0.406};
    float std[]  = {0.229, 0.224, 0.225};

    int width = 224, height = 224, channels = 3;
    std::vector<float> cpuInput(inputSize / sizeof(float));

    for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                // Access pixel
                float pixel = img.at<cv::Vec3b>(h, w)[c] / 255.0f;
                // Normalize and store in planar (CHW) order
                cpuInput[c * width * height + h * width + w] = (pixel - mean[c]) / std[c];
            }
        }
    }

    // Copy to GPU
    cudaMemcpy(gpuInputBuffer, cpuInput.data(), inputSize, cudaMemcpyHostToDevice);
}

int main() {
    std::string engineFile = "resnet50_fp16.plan";
    std::string imageFile = "dog.jpg";

    // 1. Load Engine File
    std::ifstream file(engineFile, std::ios::binary);
    if (!file.good()) { std::cerr << "Error loading engine file!" << std::endl; return 1; }
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);
    std::vector<char> engineData(size);
    file.read(engineData.data(), size);

    // 2. Deserialize Engine
    std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(gLogger));
    std::unique_ptr<nvinfer1::ICudaEngine> engine(runtime->deserializeCudaEngine(engineData.data(), size));
    std::unique_ptr<nvinfer1::IExecutionContext> context(engine->createExecutionContext());

    // 3. Prepare Buffers
    // ResNet50 has 1 input ("input") and 1 output ("output")
    void* buffers[2]; 
    
    // Setup Input
    int inputIndex = engine->getBindingIndex("input"); // Might be "data" in older models
    // If "input" returns -1, try "data" (common in some ONNX versions)
    if (inputIndex == -1) inputIndex = engine->getBindingIndex("data");
    
    nvinfer1::Dims inputDims = engine->getBindingDimensions(inputIndex);
    size_t inputSize = getSizeByDim(inputDims);
    cudaMalloc(&buffers[inputIndex], inputSize);

    // Setup Output
    int outputIndex = engine->getBindingIndex("output"); // Might be "resnetv17_dense0_fwd"
    if (outputIndex == -1) outputIndex = 1; // Fallback index
    nvinfer1::Dims outputDims = engine->getBindingDimensions(outputIndex);
    size_t outputSize = getSizeByDim(outputDims);
    cudaMalloc(&buffers[outputIndex], outputSize);

    std::cout << "Engine Loaded. Running Inference on " << imageFile << "..." << std::endl;

    // 4. Preprocess & Transfer
    preprocessImage(imageFile, (float*)buffers[inputIndex], inputSize);

    // 5. Execute Inference
    context->executeV2(buffers);

    // 6. Get Output
    std::vector<float> cpuOutput(outputSize / sizeof(float));
    cudaMemcpy(cpuOutput.data(), buffers[outputIndex], outputSize, cudaMemcpyDeviceToHost);

    // 7. Postprocess (Find Max Probability)
    int maxIdx = 0;
    float maxVal = -1.0f;
    for (int i = 0; i < cpuOutput.size(); ++i) {
        if (cpuOutput[i] > maxVal) {
            maxVal = cpuOutput[i];
            maxIdx = i;
        }
    }

    // 8. Load Labels
    std::ifstream labelFile("imagenet_classes.txt");
    std::vector<std::string> labels;
    std::string line;
    while (std::getline(labelFile, line)) labels.push_back(line);

    std::cout << "-------------------------" << std::endl;
    std::cout << "Prediction: " << labels[maxIdx] << std::endl;
    std::cout << "Confidence: " << maxVal << std::endl;
    std::cout << "-------------------------" << std::endl;

    // Cleanup
    cudaFree(buffers[inputIndex]);
    cudaFree(buffers[outputIndex]);

    return 0;
}

