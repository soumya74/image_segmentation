import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class CityscapesDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.img_dir = os.path.join(root_dir, split, "img")
        self.label_dir = os.path.join(root_dir, split, "label")
        self.transform = transform
        
        self.images = sorted(os.listdir(self.img_dir))
        self.labels = sorted(os.listdir(self.label_dir))

        # Industry-standard Cityscapes RGB to Class ID mapping
        self.color_map = {
            (128, 64, 128): 0,  (244, 35, 232): 1,  (70, 70, 70): 2, 
            (102, 102, 156): 3, (190, 153, 153): 4, (153, 153, 153): 5, 
            (250, 170, 30): 6,  (220, 220, 0): 7,   (107, 142, 35): 8, 
            (152, 251, 152): 9, (70, 130, 180): 10, (220, 20, 60): 11, 
            (255, 0, 0): 12,    (0, 0, 142): 13,    (0, 0, 70): 14, 
            (0, 60, 100): 15,   (0, 80, 100): 16,   (0, 0, 230): 17, 
            (119, 11, 32): 18
        }

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        label_path = os.path.join(self.label_dir, self.labels[idx])
        
        # Load both as RGB
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(label_path).convert("RGB")
        
        # 1. Resize mask to exactly match the PyTorch transform dimensions (W: 512, H: 256)
        # Image.NEAREST is strictly required so border pixels do not blend into unrecognized colors
        mask = mask.resize((512, 256), Image.NEAREST)
        
        if self.transform:
            image = self.transform(image)
            
        # 2. Convert the 3-channel RGB mask into a 1-channel Integer array
        mask_np = np.array(mask)
        
        # Initialize an empty mask filled with 255 (the ignore_index used in train.py)
        label_mask = np.full((mask_np.shape[0], mask_np.shape[1]), 255, dtype=np.int64)
        
        # Broadcast color mapping across the tensor
        for rgb, class_id in self.color_map.items():
            matches = (mask_np == rgb).all(axis=2)
            label_mask[matches] = class_id
            
        # Return the processed image tensor and the 2D spatial target LongTensor
        return image, torch.from_numpy(label_mask).long()