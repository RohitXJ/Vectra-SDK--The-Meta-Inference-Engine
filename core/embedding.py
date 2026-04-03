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

def compute_distances_and_predict(query_embeddings, query_labels, prototypes):
    """
    Predicts labels for queries based on distances to prototypes.
    
    Args:
        query_embeddings: Tensor of shape [Q, D]
        query_labels: Tensor of shape [Q]
        prototypes: Tensor of shape [n_way, D]

    Returns:
        preds: Tensor of shape [Q] — predicted class indices (0 to n_way-1)
        labels: Tensor of shape [Q] — actual class indices (0 to n_way-1)
    """
    # Compute pairwise distances [Q, n_way]
    dists = torch.cdist(query_embeddings, prototypes)

    # Predicted class is index of nearest prototype
    preds = torch.argmin(dists, dim=1)  # [Q]
    return preds, query_labels
