import torch
from collections import defaultdict

def get_embeds(support_data, query_data, encoder):
    encoder.eval()

    def just_embeds(data):
        with torch.no_grad():
            images, labels = next(iter(data)) 
            embeddings = encoder(images).squeeze() 
        return embeddings, labels

    support_embeds, support_labels = just_embeds(support_data)
    query_embeds, query_labels = just_embeds(query_data)
    return support_embeds, support_labels, query_embeds, query_labels


def get_prototype(embeds, labels):
    prototype = []
    unique_labels = torch.unique(labels)
    for cls in unique_labels:
        class_embeds = embeds[labels == cls]
        mean_embeds = class_embeds.mean(dim=0)
        prototype.append(mean_embeds)
    return torch.stack(prototype)

    