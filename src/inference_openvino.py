import numpy as np
import cv2
import time
from openvino.runtime import Core

# Cityscapes color palette for visualization (19 classes)
CITYSCAPES_COLORS = np.array([
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
    [0, 80, 100], [0, 0, 230], [119, 11, 32]
], dtype=np.uint8)

def preprocess_image(image_path, target_height=256, target_width=512):
    """Reads and formats the image exactly as the PyTorch pipeline did."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")
    
    img_resized = cv2.resize(img, (target_width, target_height))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Normalize: ImageNet stats matching the training pipeline
    img_normalized = img_rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_normalized = (img_normalized - mean) / std
    
    # HWC to CHW format, then add batch dimension (1, C, H, W)
    img_transposed = np.transpose(img_normalized, (2, 0, 1))
    return np.expand_dims(img_transposed, axis=0), img_resized

def main():
    model_xml = "deeplabv3_cityscapes.xml"
    test_image_path = "val100.png"  # Replace with a real Cityscapes image path
    
    print("Initializing OpenVINO Runtime Core...")
    core = Core()
    
    # Check available devices (Useful for debugging if NPU is not detected)
    print(f"Available Devices: {core.available_devices}")
    
    print("Reading OpenVINO IR model...")
    model = core.read_model(model=model_xml)

    # Define the device here so it automatically updates your logs
    target_device = "NPU"  # Change to "CPU" or "GPU" to test other hardware
    
    print("Compiling model for Intel Core Ultra NPU...")
    # Change "NPU" to "CPU" or "GPU" if you want to benchmark different hardware
    compiled_model = core.compile_model(model=model, device_name=target_device)
    
    # Get the input and output nodes
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)
    
    print("Preprocessing input image...")
    input_tensor, original_img = preprocess_image(test_image_path)
    
    print("Executing inference...")
    # Warm-up run (hardware initialization often makes the first run slow)
    _ = compiled_model([input_tensor])[output_layer]
    
    # Profiling inference speed
    iterations = 10
    start_time = time.perf_counter()
    
    for _ in range(iterations):
        results = compiled_model([input_tensor])[output_layer]
        
    end_time = time.perf_counter()
    
    avg_latency = ((end_time - start_time) / iterations) * 1000
    fps = 1000 / avg_latency
    
    print(f"\n--- OpenVINO Performance Profiling ---")
    print(f"Target Device: ({target_device})")
    print(f"Average Latency: {avg_latency:.2f} ms")
    print(f"Frames Per Second: {fps:.2f} FPS")
    print(f"-----------------------------\n")
    
    # Post-processing: Convert logits to class IDs
    predictions = np.squeeze(results)  # Shape: (19, 256, 512)
    class_ids = np.argmax(predictions, axis=0)  # Shape: (256, 512)
    
    # Map class IDs to RGB colors
    segmentation_mask = CITYSCAPES_COLORS[class_ids]
    segmentation_mask_bgr = cv2.cvtColor(segmentation_mask, cv2.COLOR_RGB2BGR)
    
    # Overlay mask on the original image
    alpha = 0.5
    overlay = cv2.addWeighted(original_img, 1 - alpha, segmentation_mask_bgr, alpha, 0)
    
    cv2.imwrite("npu_segmentation_output.png", overlay)
    print("Saved visualization to 'npu_segmentation_output.png'.")

if __name__ == "__main__":
    main()