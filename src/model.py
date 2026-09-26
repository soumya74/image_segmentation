import torch.nn as nn
from torchvision import models

def get_deeplabv3_resnet50(num_classes=19):
    """
    Loads pretrained DeepLabV3 and replaces the classifier head.
    """
    # Load model with COCO pre-trained backbone
    model = models.segmentation.deeplabv3_resnet50(weights='DEFAULT')
    
    # Replace the final convolutional layer to match Cityscapes classes
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    
    # Also update the auxiliary classifier if it exists
    if model.aux_classifier is not None:
        model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
        
    return model