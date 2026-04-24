import os
import torch
from torch.utils.data import DataLoader
from data import IO
from core import backbone, embedding, export
from utils import storage

def run_fewshot_pipeline(token, backbone_name, use_unknown=False):
    """
    Runs the few-shot learning pipeline for a specific user session.
    
    Args:
        token: User session token.
        backbone_name: The CNN architecture to use.
        use_unknown: Whether to enable out-of-distribution rejection.
    """
    paths = storage.get_session_paths(token)
    
    # 1. Load data
    try:
        support_data, img_format, transforms_obj = IO.get_data(paths["support"], ret_transform=True)
        query_data, _ = IO.get_data(paths["query"])
    except Exception as e:
        raise ValueError(f"Data loading failed: {str(e)}")

    # 2. Initialize encoder
    encoder = backbone.get_encoder(backbone_name, img_format)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device)

    # 3. Create loaders (one batch for everything in few-shot)
    support_loader = DataLoader(support_data, batch_size=len(support_data))
    query_loader = DataLoader(query_data, batch_size=len(query_data))

    # 4. Get embeddings
    support_embeddings, support_labels = embedding.get_embeddings(support_loader, encoder)
    query_embeddings, query_labels = embedding.get_embeddings(query_loader, encoder)
    
    # 5. Compute prototypes and threshold
    prototypes = embedding.compute_prototypes(support_embeddings, support_labels)
    # Re-normalize prototypes to unit length
    prototypes = torch.nn.functional.normalize(prototypes, p=2, dim=1)
    
    unknown_threshold = None
    if use_unknown:
        # Calculate intra-class distances robustly
        unique_classes = torch.unique(support_labels)
        intra_distances = []
        for i, cls in enumerate(unique_classes):
            class_embeddings = support_embeddings[support_labels == cls]
            class_prototype = prototypes[i]
            # Distances of each sample in this class to its prototype
            dist = torch.norm(class_embeddings - class_prototype, dim=1)
            intra_distances.append(dist)
        
        if intra_distances:
            all_intra_distances = torch.cat(intra_distances)
            # Improved Heuristic: Max distance + margin, with a floor
            # Since embeddings are normalized, max distance is 2.0.
            # A good default for OOD is around 1.0 - 1.2
            max_dist = torch.max(all_intra_distances).item()
            unknown_threshold = max(0.8, 1.5 * max_dist) 
        else:
            unknown_threshold = 1.0 # Default fallback

    preds_labels, true_labels = embedding.compute_distances_and_predict(
        query_embeddings, query_labels, prototypes, 
        use_unknown=use_unknown, unknown_threshold=unknown_threshold
    )

    # 6. Calculate accuracy
    # If using unknown, we need to handle the case where true_labels doesn't contain the unknown index
    # but preds_labels might.
    correct = (preds_labels == true_labels).sum().item()
    total = true_labels.size(0)
    accuracy = correct / total * 100

    # 7. Prepare export config
    class_labels = support_data.classes
    config_out = {
        "labels": class_labels,
        "label_map": {i: label for i, label in enumerate(class_labels)},
        "backbone": backbone_name,
        "image_format": img_format,
        "transforms": str(transforms_obj),  # Store for reference
        "prototypes": prototypes.cpu(),
        "use_unknown": use_unknown,
        "unknown_threshold": unknown_threshold
    }
    
    export_filename = f"fewshot_model_{token}.pt"
    exp_path = export.export_model(config_out, export_dir=paths["export"], filename=export_filename)

    return {
        "export_path": exp_path,
        "accuracy": accuracy,
        "predicted_labels": preds_labels.tolist(),
        "true_labels": true_labels.tolist(),
        "labels": class_labels,
        "num_support": len(support_data),
        "num_query": len(query_data)
    }
