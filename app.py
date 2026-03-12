from torch.utils.data import DataLoader
import torch
import os
import json
from data.IO import get_data
from core.backbone import get_encoder
from core.embedding import get_embeds

config = {
    "n_way" : 2,
    "k_shot" : 2,
    "support" : "temp_data/support/",
    "query" : "temp_data/query/",
    "backbone" : "resnet18",
    }

def main(config):
    support_data, query_data, format = get_data(config["support"], config["query"])
    print("Data Retrieval Success!")
    encoder = get_encoder(config["backbone"], format)
    print(encoder)
    print("Encoder Retrieval Success!")
    support_embeds, support_labels, query_embeds, query_labels = get_embeds(support_data, query_data, encoder)
    print(support_embeds, support_labels)
    print(query_embeds, query_labels)
    print("Receiving Embeds")


if __name__ == "__main__":
    main(config)
