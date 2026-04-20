from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import os

def img_check(path_to_check):
    """Detects the primary image format and ensures consistency."""
    detected = set()
    if not os.path.exists(path_to_check):
        raise ValueError(f"Path does not exist: {path_to_check}")
        
    for cls in os.listdir(path_to_check):
        cls_path = os.path.join(path_to_check, cls)
        if not os.path.isdir(cls_path):
            continue
        for img_name in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_name)
            try:
                with Image.open(img_path) as img:
                    detected.add(img.mode)
            except:
                raise ValueError(f"Corrupted or unreadable image: {img_path}")
                
    if not detected:
        raise ValueError(f"No images found in {path_to_check}")
    
    # If we have mixed formats, we prefer RGB as it's the most compatible with pre-trained backbones
    if 'RGB' in detected or 'RGBA' in detected or len(detected) > 1:
        return 'RGB'
    
    return detected.pop()

def get_transform(image_mode):
    """Returns standard ImageNet-style transforms based on image mode."""
    if image_mode == 'L':  # Grayscale
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:  # RGB
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

def get_data(img_path, ret_transform=False):
    """Loads images from a folder as a Dataset/DataLoader."""
    image_format = img_check(img_path)
    transform = get_transform(image_format)
    
    # Custom loader to ensure all images are converted to the target format (RGB or L)
    def fixed_loader(path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert(image_format)

    data = datasets.ImageFolder(root=img_path, transform=transform, loader=fixed_loader)
    
    if ret_transform:
        return data, image_format, transform
    else:
        return data, image_format
