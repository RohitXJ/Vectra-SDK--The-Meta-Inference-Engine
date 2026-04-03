import os
import torch
from torch.utils.data import DataLoader
from data import IO
from core import backbone, embedding, export
from utils import storage

def run_fewshot_pipeline(token, backbone_name):
    """
    Runs the few-shot learning pipeline for a specific user session.
    
    Args:
        token: User session token.
        backbone_name: The CNN architecture to use.
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
    
    # 5. Compute prototypes and predict
    prototypes = embedding.compute_prototypes(support_embeddings, support_labels)
    preds_labels, true_labels = embedding.compute_distances_and_predict(query_embeddings, query_labels, prototypes)

    # 6. Calculate accuracy
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
