import torch
import yaml
from model import get_deeplabv3_resnet50

# --- Wrapper to simplify output for OpenVINO ---
class DeepLabONNXWrapper(torch.nn.Module):
    """
    Standard PyTorch DeepLabV3 returns a dictionary: {'out': tensor, 'aux': tensor}.
    C++ and edge deployment frameworks (like OpenVINO) prefer raw tensors.
    This wrapper extracts just the main segmentation mask to ensure a clean export.
    """
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        return self.base_model(x)['out']

def main():
    print("Loading configuration...")
    with open("configs/cityscapes.yaml", "r") as f:
        config = yaml.safe_load(f)

    num_classes = config['dataset']['num_classes']
    img_h = config['transforms']['image_height']
    img_w = config['transforms']['image_width']
    
    # Exporting is generally safest and most stable on the CPU
    device = torch.device("cpu")

    print("Initializing architecture and loading best weights...")
    base_model = get_deeplabv3_resnet50(num_classes=num_classes)
    
    # Load the weights from your successful Epoch 5 run
    weights_path = "deeplabv3_cityscapes_best_0210_2033.pth"
    base_model.load_state_dict(torch.load(weights_path, map_location=device))
    
    # Wrap the model and set to evaluation mode (CRITICAL for freezing BatchNorm layers)
    model = DeepLabONNXWrapper(base_model).to(device)
    model.eval()

    # Create a dummy input tensor matching your training pipeline
    # Shape: (Batch Size, Channels, Height, Width) -> (1, 3, 256, 512)
    dummy_input = torch.randn(1, 3, img_h, img_w).to(device)
    
    onnx_filename = "deeplabv3_cityscapes.onnx"
    
    print(f"Exporting model to {onnx_filename}...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_filename,
        export_params=True,
        opset_version=18,          # Updated to match PyTorch's requirement
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
        # Removed dynamic_axes to prevent Dynamo engine conflicts and C++ crashes
    )
    
    print("Export complete. The ONNX graph is ready for Intel OpenVINO optimization.")

if __name__ == "__main__":
    main()