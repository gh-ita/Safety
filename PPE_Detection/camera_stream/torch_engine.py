import tensorrt as trt
import sys
import argparse

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_file_path, engine_file_path):
    # Create builder and network
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    
    # Create ONNX parser
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Parse ONNX file
    with open(onnx_file_path, 'rb') as model_file:
        if not parser.parse(model_file.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    # Create config
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)  # 8 GiB
    config.set_flag(trt.BuilderFlag.FP16)
    
    # Build serialized network
    serialized_network = builder.build_serialized_network(network, config)
    if serialized_network is None:
        print("ERROR: Failed to build the engine!")
        return None
    
    # Save serialized network to file
    with open(engine_file_path, 'wb') as f:
        f.write(serialized_network)
    
    print(f"TensorRT engine built and saved to {engine_file_path}")
    
    # Create runtime and engine (if needed for immediate use)
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(serialized_network)
    
    return engine

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_file_path', type=str, required=True, help='Path to the ONNX model file')
    parser.add_argument('--engine_file_path', type=str, required=True, help='Path where the TensorRT engine will be saved')
    opt = parser.parse_args()
    
    if not opt.onnx_file_path or not opt.engine_file_path:
        print("ERROR: Please provide both --onnx_file_path and --engine_file_path")
        parser.print_help()
        sys.exit(1)
    
    build_engine(opt.onnx_file_path, opt.engine_file_path)
