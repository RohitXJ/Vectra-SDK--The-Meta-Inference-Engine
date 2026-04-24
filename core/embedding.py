import torch
from collections import defaultdict

def get_embeddings(dataloader, encoder):
    """Returns stacked embeddings and labels from a DataLoader"""
    encoder.eval()
    device = next(encoder.parameters()).device
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            embeddings = encoder(images)
            # Ensure embeddings are flattened if not already
            if len(embeddings.shape) > 2:
                embeddings = embeddings.view(embeddings.size(0), -1)
            
            # Normalize embeddings to unit hypersphere for stable distances
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels)
            
    return torch.cat(all_embeddings), torch.cat(all_labels)

def compute_prototypes(embeddings, labels):
    """
    Computes class-wise prototype vectors (mean embeddings).
    
    Args:
        embeddings: Tensor of shape [N, D]
        labels: Tensor of shape [N]

    Returns:
        prototypes: Tensor of shape [n_way, D]
    """
    prototypes = []
    unique_classes = torch.unique(labels)

    for cls in unique_classes:
        class_embeddings = embeddings[labels == cls]   # [k_shot, D]
        class_mean = class_embeddings.mean(dim=0)       # [D]
        prototypes.append(class_mean)

    return torch.stack(prototypes)

def compute_distances_and_predict(query_embeddings, query_labels, prototypes, use_unknown=False, unknown_threshold=None):
    """
    Predicts labels for queries based on distances to prototypes.
    
    Args:
        query_embeddings: Tensor of shape [Q, D]
        query_labels: Tensor of shape [Q]
        prototypes: Tensor of shape [n_way, D]
        use_unknown: Boolean, whether to use a threshold for out-of-distribution rejection.
        unknown_threshold: Float, distances above this value map to index 'n_way'.

    Returns:
        preds: Tensor of shape [Q] — predicted class indices (0 to n_way)
        labels: Tensor of shape [Q] — actual class indices (0 to n_way-1)
    """
    # Compute pairwise distances [Q, n_way]
    dists = torch.cdist(query_embeddings, prototypes)

    # Predicted class is index of nearest prototype
    min_dists, preds = torch.min(dists, dim=1)  # [Q]

    if use_unknown and unknown_threshold is not None:
        # If min distance exceeds threshold, map to the index after the last valid class
        unknown_mask = min_dists > unknown_threshold
        preds[unknown_mask] = prototypes.size(0)

    return preds, query_labels
