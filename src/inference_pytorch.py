import torch
import numpy as np
import cv2
import time
import yaml
from model import get_deeplabv3_resnet50

# Cityscapes color palette for visualization (19 classes)
CITYSCAPES_COLORS = np.array([
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
    [0, 80, 100], [0, 0, 230], [119, 11, 32]
], dtype=np.uint8)

def preprocess_image(image_path, target_height=256, target_width=512):
    """Reads and formats the image exactly as the PyTorch training pipeline did."""
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
    print("Loading configuration...")
    with open("configs/cityscapes.yaml", "r") as f:
        config = yaml.safe_load(f)

    num_classes = config['dataset']['num_classes']
    img_h = config['transforms']['image_height']
    img_w = config['transforms']['image_width']
    
    # Use GPU if available, otherwise fallback to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Initializing PyTorch architecture and loading best weights...")
    model = get_deeplabv3_resnet50(num_classes=num_classes)
    
    weights_path = "deeplabv3_cityscapes_best_0210_2033.pth"
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()  # CRITICAL: Freeze BatchNorm and Dropout layers

    test_image_path = "val100.png"
    
    print("Preprocessing input image...")
    input_numpy, original_img = preprocess_image(test_image_path, target_height=img_h, target_width=img_w)
    
    # Convert the numpy array from preprocessing into a PyTorch tensor
    input_tensor = torch.from_numpy(input_numpy).float().to(device)
    
    print("Executing PyTorch inference...")
    # Warm-up run to initialize CUDA/CPU kernels
    with torch.no_grad():
        _ = model(input_tensor)['out']
    
    # Profiling inference speed
    iterations = 10
    start_time = time.perf_counter()
    
    with torch.no_grad():
        for _ in range(iterations):
            output = model(input_tensor)['out']
            
    end_time = time.perf_counter()
    
    avg_latency = ((end_time - start_time) / iterations) * 1000
    fps = 1000 / avg_latency
    
    print(f"\n--- PyTorch Performance Profiling ---")
    print(f"Target Device: PyTorch ({device.type.upper()})")
    print(f"Average Latency: {avg_latency:.2f} ms")
    print(f"Frames Per Second: {fps:.2f} FPS")
    print(f"-----------------------------\n")
    
    # Post-processing: Convert logits to class IDs
    predictions = output.squeeze().cpu().numpy()  # Shape: (19, 256, 512)
    class_ids = np.argmax(predictions, axis=0)    # Shape: (256, 512)
    
    # Map class IDs to RGB colors
    segmentation_mask = CITYSCAPES_COLORS[class_ids]
    segmentation_mask_bgr = cv2.cvtColor(segmentation_mask, cv2.COLOR_RGB2BGR)
    
    # Overlay mask on the original image
    alpha = 0.5
    overlay = cv2.addWeighted(original_img, 1 - alpha, segmentation_mask_bgr, alpha, 0)
    
    output_filename = "pytorch_segmentation_output.png"
    cv2.imwrite(output_filename, overlay)
    print(f"Saved visualization to '{output_filename}'.")

if __name__ == "__main__":
    main()