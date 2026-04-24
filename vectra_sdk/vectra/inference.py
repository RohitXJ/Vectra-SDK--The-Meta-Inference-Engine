import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os
from .utils.transforms import get_encoder, get_transform

class VectraInference:
    """
    High-level inference class for Vectra Engine models.
    """
    def __init__(self, model_path, use_gpu=True):
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        try:
            checkpoint = torch.load(model_path, map_location="cpu")
        except Exception as e:
            raise RuntimeError(f"Failed to load model checkpoint: {str(e)}")
            
        # Required config fields
        self.labels = checkpoint.get("labels", [])
        self.backbone_name = checkpoint.get("backbone")
        self.image_format = checkpoint.get("image_format", "RGB")
        self.prototypes = checkpoint.get("prototypes").to(self.device)
        self.use_unknown = checkpoint.get("use_unknown", False)
        self.unknown_threshold = checkpoint.get("unknown_threshold")
        
        if not self.backbone_name or self.prototypes is None:
            raise ValueError("Invalid model checkpoint: missing backbone or prototypes.")
            
        # Initialize encoder
        self.encoder = get_encoder(self.backbone_name, self.image_format).to(self.device)
        self.transform = get_transform(self.image_format)
        
    def _preprocess(self, input_data):
        """Converts input (path, PIL, or numpy) to a preprocessed tensor."""
        if isinstance(input_data, str):
            image = Image.open(input_data)
        elif isinstance(input_data, Image.Image):
            image = input_data
        elif isinstance(input_data, np.ndarray):
            # Assume OpenCV BGR format and convert to RGB
            if len(input_data.shape) == 3 and input_data.shape[2] == 3:
                image = Image.fromarray(input_data[:, :, ::-1])
            else:
                image = Image.fromarray(input_data)
        else:
            raise ValueError("Input must be a file path, PIL.Image, or numpy.ndarray.")

        # Ensure correct mode
        if image.mode != self.image_format:
            image = image.convert(self.image_format)
            
        return self.transform(image).unsqueeze(0).to(self.device)

    def predict(self, input_data, return_confidence=True):
        """
        Predicts the class of the input image.
        
        Args:
            input_data: File path, PIL.Image, or numpy.ndarray.
            return_confidence: Whether to include the similarity score.
            
        Returns:
            Dictionary containing 'label', 'index', and optionally 'confidence'.
        """
        tensor = self._preprocess(input_data)
        
        with torch.no_grad():
            embedding = self.encoder(tensor)
            if len(embedding.shape) > 2:
                embedding = embedding.view(embedding.size(0), -1)
            
            # Normalize embedding to unit hypersphere
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
            
            # Compute distances to prototypes
            dists = torch.cdist(embedding, self.prototypes)
            min_dist, pred_idx = torch.min(dists, dim=1)
            
            min_dist = min_dist.item()
            idx = pred_idx.item()
            
            # Handle Unknown category
            is_unknown = False
            if self.use_unknown and self.unknown_threshold is not None:
                if min_dist > self.unknown_threshold:
                    is_unknown = True
            
            label = "Unknown" if is_unknown else self.labels[idx]
            
            # Confidence heuristic (inverse of distance or normalized similarity)
            # For simplicity, we return the raw similarity if asked
            result = {
                "label": label,
                "index": -1 if is_unknown else idx,
                "distance": min_dist
            }
            
            if return_confidence:
                # Naive confidence: 1 / (1 + distance)
                result["confidence"] = 1.0 / (1.0 + min_dist)
                
            return result

    def predict_batch(self, input_list):
        """Helper to predict multiple inputs at once."""
        return [self.predict(item) for item in input_list]
