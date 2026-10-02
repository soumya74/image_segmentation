import torch

def calculate_miou(predictions, labels, num_classes=19, ignore_index=255):
    """
    Calculates the Mean Intersection over Union (mIoU) for a batch.
    Converts logits to class predictions and calculates overlap, ignoring the void class (255).
    """
    preds = torch.argmax(predictions, dim=1)
    
    # Create a mask to ignore the 255 background/void class
    valid_mask = (labels != ignore_index)
    preds = preds[valid_mask]
    labels = labels[valid_mask]

    # Calculate intersection and union per class
    intersection = torch.bincount(labels[preds == labels], minlength=num_classes)
    area_pred = torch.bincount(preds, minlength=num_classes)
    area_label = torch.bincount(labels, minlength=num_classes)
    union = area_pred + area_label - intersection

    # Compute IoU only for classes that are actually present in the image
    valid_classes = union > 0
    iou = intersection[valid_classes].float() / union[valid_classes].float()
    
    return torch.mean(iou).item() if valid_classes.any() else 0.0