#include <iostream>
#include <fstream>
#include <vector>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
// Logger for TensorRT (Required to capture errors/warnings)
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << "[TRT] " << msg << std::endl;
    }
} logger;
int main() {
    std::string engineFile = "./yolov8n.engine"; // Path to your engine
// 1. Initialize TensorRT
    // ----------------------------------------------------------------
    std::cout << "1. Creating Runtime..." << std::endl;
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
    
    // 2. Read the Engine File
    // ----------------------------------------------------------------
    std::cout << "2. Reading Engine File..." << std::endl;
    std::ifstream file(engineFile, std::ios::binary | std::ios::ate);
    if (!file.good()) {
        std::cerr << "Error: Could not read engine file! Check path." << std::endl;
        return 1;
    }
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
// 3. Deserialize (Load) the Engine
    // ----------------------------------------------------------------
    std::cout << "3. Deserializing Engine..." << std::endl;
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(buffer.data(), size);
    
    if (!engine) {
        std::cerr << "Error: Failed to load engine." << std::endl;
        return 1;
    }
    std::cout << "Success! Engine loaded. " << std::endl;
// 4. Inspect the Engine (Sanity Check)
    // ----------------------------------------------------------------
    int numBindings = engine->getNbIOTensors();
    std::cout << "Number of IO Tensors: " << numBindings << std::endl;
for (int i = 0; i < numBindings; ++i) {
        const char* name = engine->getIOTensorName(i);
        nvinfer1::TensorIOMode mode = engine->getTensorIOMode(name);
        nvinfer1::Dims dims = engine->getTensorShape(name);
        
        std::cout << " - Tensor " << i << ": " << name 
                  << " [" << (mode == nvinfer1::TensorIOMode::kINPUT ? "Input" : "Output") << "]";
        
        std::cout << " Shape: (";
        for (int d = 0; d < dims.nbDims; ++d) std::cout << dims.d[d] << ",";
        std::cout << ")" << std::endl;
    }
// Clean up (Standard C++ destruction)
    delete engine;
    delete runtime;
    
    return 0;
}


