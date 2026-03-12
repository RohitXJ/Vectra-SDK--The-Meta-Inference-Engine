import torch
from collections import defaultdict

def get_embeds(support_data, query_data, encoder):
    encoder.eval()
    
    def just_embeds(data):
        with torch.no_grad():
            images, labels = next(iter(data))  # single batch
            embeddings = encoder(images).squeeze()   # [N, D]
        return embeddings, labels

    support_embeds, support_labels = just_embeds(support_data)
    query_embeds, query_labels = just_embeds(query_data)
    return support_embeds, support_labels, query_embeds, query_labels


