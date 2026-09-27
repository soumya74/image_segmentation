import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter  # 1. Import TensorBoard Writer
from tqdm import tqdm

from dataset import CityscapesDataset
from model import get_deeplabv3_resnet50

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on: {device}")
    
    dataset_root = "/content/datasets/cityscapes"
    batch_size = 8
    epochs = 10
    learning_rate = 1e-4

    img_transform = transforms.Compose([
        transforms.Resize((256, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Loading datasets...")
    train_dataset = CityscapesDataset(root_dir=dataset_root, split="train", transform=img_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    print("Initializing DeepLabV3...")
    model = get_deeplabv3_resnet50(num_classes=19).to(device)

    # FREEZE THE RESNET BACKBONE
    #for name, param in model.named_parameters():
    #    if 'backbone' in name:
    #        param.requires_grad = False

    # SEPARATE LEARNING RATE FOR DEEPLABV3 AND NEW LAYER
    #print("Initializing DeepLabV3...")
    #model = get_deeplabv3_resnet50(num_classes=19).to(device)

    # Separate the parameters into backbone and head
    #backbone_params = []
    #head_params = []
    #for name, param in model.named_parameters():
    #    if 'backbone' in name:
    #        backbone_params.append(param)
    #    else:
    #        head_params.append(param)

    # Apply a 10x smaller learning rate to the pre-trained backbone
    #optimizer = optim.Adam([
    #    {'params': backbone_params, 'lr': learning_rate * 0.1},
    #    {'params': head_params, 'lr': learning_rate}
    #])
    
    #criterion = nn.CrossEntropyLoss(ignore_index=255)

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 2. Initialize the Writer in the directory Colab is watching
    writer = SummaryWriter('runs/segmentation')
    global_step = 0

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for images, masks in progress_bar:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)['out']
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

            # 3. Log the loss directly to TensorBoard
            writer.add_scalar('Loss/train', loss.item(), global_step)
            global_step += 1

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} completed. Average Loss: {epoch_loss:.4f}\n")

    # 4. Close the writer when training ends
    writer.close()
    
    torch.save(model.state_dict(), "deeplabv3_cityscapes_best.pth")
    print("Training complete! Weights saved to deeplabv3_cityscapes_best.pth")

if __name__ == "__main__":
    main()