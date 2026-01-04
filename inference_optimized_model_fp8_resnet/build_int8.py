import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import os

# --- CONFIGURATION ---
ONNX_FILE = "/workspace/AI_miniProject/inference_model_setup/resnet50.onnx"
ENGINE_FILE = "resnet50_int8.plan"
CALIB_CACHE = "calibration.cache"
IMAGE_DIR = "."  # We will use the current folder (dog.jpg)
BATCH_SIZE = 1
CALIB_COUNT = 10 # Ideally 500+, but we use 10 for this demo

# Logger
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

# --- 1. THE CALIBRATOR CLASS ---
# This class feeds data to TensorRT during the build
class EntroyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, cache_file, batch_size=1):
        # Initialize the parent class
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.current_index = 0
        
        # Allocation for the input batch
        self.device_input = cuda.mem_alloc(1 * 3 * 224 * 224 * 4) # 1 image, float32 bytes
        
        # Load one image to reuse (Hack for demo purposes)
        # In production, you would load a list of different file paths here
        img = cv2.imread("dog.jpg")
        img = cv2.resize(img, (224, 224))
        # ... (Previous resize code) ...
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize
        img = img.astype(np.float32) / 255.0
        # Explicitly force constants to float32 to prevent upcasting
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std

        # HWC -> CHW
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)

        # Ensure final array is float32 and contiguous
        self.batch_data = np.ascontiguousarray(img.astype(np.float32))

    def get_batch_size(self):
        return self.batch_size

    # TensorRT calls this repeatedly to get new data
    def get_batch(self, names):
        if self.current_index >= CALIB_COUNT:
            return None # Stop when we reach the limit

        print(f"[Calibrator] Feeding batch {self.current_index}")
        # Copy batch to GPU
        cuda.memcpy_htod(self.device_input, self.batch_data)
        self.current_index += 1
        
        # Return pointer to GPU memory
        return [int(self.device_input)]

    # Read/Write Cache (To save time on subsequent runs)
    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

# ... (Previous imports and Calibrator class remain the same) ...

def build_engine():
    builder = trt.Builder(TRT_LOGGER)
    # Ensure Explicit Batch flag is set
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Load ONNX
    print(f"Loading {ONNX_FILE}...")
    with open(ONNX_FILE, 'rb') as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    # --- NEW: Define Optimization Profile ---
    # This fixes Error Code 4 by defining Min, Opt, and Max shapes
    profile = builder.create_optimization_profile()
    
    # Get the name of the input layer (usually "input" or "data")
    input_tensor = network.get_input(0)
    input_name = input_tensor.name
    print(f"Detected Input Name: {input_name}")

    # Set shape: (Batch, Channels, Height, Width)
    # We enforce Batch Size = 1 for this demo
    profile.set_shape(input_name, (1, 3, 224, 224), (1, 3, 224, 224), (1, 3, 224, 224))
    config.add_optimization_profile(profile)
    # ----------------------------------------

    # Enable INT8
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16) # Fallback
    
    # Attach Calibrator
    config.int8_calibrator = EntroyCalibrator(CALIB_CACHE)

    # Build
    print("Building INT8 Engine... (This involves running inference to calibrate)")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        print("Error: Engine build failed!")
        return

    # Save
    with open(ENGINE_FILE, "wb") as f:
        f.write(serialized_engine)
    print(f"Success! Saved to {ENGINE_FILE}")

if __name__ == "__main__":
    build_engine()
