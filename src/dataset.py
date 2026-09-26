import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class CityscapesDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        """
        Expects root_dir to be the path containing 'train/' and 'val/'.
        Kaggle structure: {root_dir}/{split}/img/ and {root_dir}/{split}/label/
        """
        self.img_dir = os.path.join(root_dir, split, "img")
        self.label_dir = os.path.join(root_dir, split, "label")
        self.transform = transform
        
        # Ensure filenames match between images and labels
        self.images = sorted(os.listdir(self.img_dir))
        self.labels = sorted(os.listdir(self.label_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        label_path = os.path.join(self.label_dir, self.labels[idx])
        
        # Load RGB image and grayscale integer mask
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(label_path)
        
        if self.transform:
            image = self.transform(image)
            
        # Convert mask directly to a LongTensor (required for PyTorch loss function)
        mask = torch.from_numpy(np.array(mask)).long()
        
        return image, mask