from torch.utils.data import DataLoader
import torch
import os
import json
from data.IO import get_data

config = {
    "n_way" : 2,
    "k_shot" : 2,
    "support" : "temp_data/support/",
    "query" : "temp_data/query/",
    "backbone" : "resnet18",
    }

def main(config):
    support_data, query_data, format = get_data(config["support"], config["query"])
    
    

if __name__ == "__main__":
    main(config)
