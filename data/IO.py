from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import os

def get_data(support_path, query_path):

    def format_chk(path):
        detected_modes = set()
        for cls in os.listdir(path):
            cls_path = os.path.join(path, cls)
            for img_name in os.listdir(cls_path):
                img_path = os.path.join(cls_path, img_name)
                try:
                    with Image.open(img_path) as img:
                        detected_modes.add(img.mode)
                except:
                    raise ValueError(f"Corrupted or unreadable image: {img_path}")
        
        if len(detected_modes) > 1:
            raise ValueError(f"Mixed image formats detected: {detected_modes}. Please upload consistent image types (e.g., all RGB or all grayscale).")
        
        return detected_modes.pop()
    
    def get_transform(image_mode):
        if image_mode == 'L':  # Grayscale
            return transforms.Compose([
                transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])
        else:  # Assume RGB or compatible
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])

    def load_images(path, image_mode='RGB'):
        data = datasets.ImageFolder(root=path, transform=get_transform(image_mode))
        return data
    

    format = format_chk(support_path)
    if format != format_chk(query_path):
        raise ValueError(f"Mixed image formats detected. Please upload consistent image types for both query and support sets(e.g., all RGB or all grayscale).")
    
    support_data = load_images(support_path, format)
    query_data = load_images(query_path, format)

    return support_data, query_data, format

