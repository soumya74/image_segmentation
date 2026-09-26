import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import CityscapesDataset
from model import get_deeplabv3_resnet50

def main():
    # 1. Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on: {device}")
    
    dataset_root = "/content/datasets/cityscapes"
    batch_size = 8
    epochs = 20
    learning_rate = 1e-4

    # 2. Data Preparation (Resize to fit in T4 GPU memory)
    img_transform = transforms.Compose([
        transforms.Resize((256, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Loading datasets...")
    train_dataset = CityscapesDataset(root_dir=dataset_root, split="train", transform=img_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # 3. Model Initialization
    print("Initializing DeepLabV3...")
    model = get_deeplabv3_resnet50(num_classes=19).to(device)

    # 4. Optimizer and Loss Function
    # ignore_index=255 ensures boundary/void pixels in Cityscapes don't corrupt the loss
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 5. Training Loop
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        # Use tqdm for a clean progress bar
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for images, masks in progress_bar:
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass (DeepLab returns a dict, we want 'out')
            outputs = model(images)['out']
            
            # Loss calculation
            loss = criterion(outputs, masks)

            # Backward pass & weight update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} completed. Average Loss: {epoch_loss:.4f}\n")

    # 6. Save the native weights
    torch.save(model.state_dict(), "deeplabv3_cityscapes_best.pth")
    print("Training complete! Weights saved to deeplabv3_cityscapes_best.pth")

if __name__ == "__main__":
    main()