import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import CityscapesDataset
from model import get_deeplabv3_resnet50
from metrics import calculate_miou 

def main():
    # Load configuration from the config folder
    with open("configs/cityscapes.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on: {device}")
    
    # Extract variables from config dictionary
    dataset_root = config['dataset']['root_dir']
    num_classes = config['dataset']['num_classes']
    ignore_index = config['dataset']['ignore_index']
    
    batch_size = config['training']['batch_size']
    epochs = config['training']['epochs']
    learning_rate = float(config['training']['learning_rate'])
    
    img_h = config['transforms']['image_height']
    img_w = config['transforms']['image_width']

    img_transform = transforms.Compose([
        transforms.Resize((img_h, img_w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Loading datasets...")
    train_dataset = CityscapesDataset(root_dir=dataset_root, split="train", transform=img_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    val_dataset = CityscapesDataset(root_dir=dataset_root, split="val", transform=img_transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print("Initializing DeepLabV3...")
    model = get_deeplabv3_resnet50(num_classes=num_classes).to(device)

    # Differential Learning Rate
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = optim.Adam([
        {'params': backbone_params, 'lr': learning_rate * 0.1},
        {'params': head_params, 'lr': learning_rate}
    ])
    
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    writer = SummaryWriter('runs/segmentation')
    
    best_miou = 0.0
    global_step = 0

    print("Starting training loop...")
    for epoch in range(epochs):
        
        # --- TRAINING PHASE ---
        model.train()
        running_train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for images, masks in train_bar:
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)['out']
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())
            
            writer.add_scalar('Loss/train_step', loss.item(), global_step)
            global_step += 1

        avg_train_loss = running_train_loss / len(train_loader)
        writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)

        # --- VALIDATION PHASE ---
        model.eval()
        running_val_loss = 0.0
        running_miou = 0.0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
        
        with torch.no_grad():
            for images, masks in val_bar:
                images, masks = images.to(device), masks.to(device)

                outputs = model(images)['out']
                loss = criterion(outputs, masks)
                
                miou = calculate_miou(outputs, masks, num_classes=num_classes, ignore_index=ignore_index)

                running_val_loss += loss.item()
                running_miou += miou
                val_bar.set_postfix(val_loss=loss.item(), mIoU=miou)

        avg_val_loss = running_val_loss / len(val_loader)
        avg_miou = running_miou / len(val_loader)
        
        writer.add_scalar('Loss/val_epoch', avg_val_loss, epoch)
        writer.add_scalar('Metric/mIoU_val', avg_miou, epoch)

        print(f"Epoch {epoch+1} Summary | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val mIoU: {avg_miou:.4f}")

        # --- DYNAMIC CHECKPOINTING ---
        if avg_miou > best_miou:
            print(f"New best mIoU achieved ({best_miou:.4f} -> {avg_miou:.4f}). Saving model...")
            best_miou = avg_miou
            torch.save(model.state_dict(), "deeplabv3_cityscapes_best.pth")
        else:
            print(f"mIoU did not improve from {best_miou:.4f}")
        print("-" * 50)

    writer.close()
    print("Training pipeline completed successfully.")

if __name__ == "__main__":
    main()