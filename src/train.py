# src/train.py
import os
from ultralytics import YOLO

def main():
    print("Initializing YOLO11-seg model...")
    # Load the nano semantic segmentation model (yolo11n-seg)
    # The .pt extension pulls the pre-trained weights
    model = YOLO("yolov8n-seg.pt") # Using v8n-seg as base; Ultralytics API handles the architecture

    print("Starting training on Cityscapes dataset...")
    # Train the model
    results = model.train(
        data="configs/cityscapes.yaml",
        epochs=50,                  
        imgsz=640,                  
        batch=16,                   
        device=0,                   
        project="runs/segmentation",
        name="cityscapes_run1",
        patience=10,                # Early stopping if no improvement
        save=True                   # Save best.pt and last.pt
    )

    print(f"Training complete. Best model saved to: {results.save_dir}/weights/best.pt")

if __name__ == '__main__':
    main()