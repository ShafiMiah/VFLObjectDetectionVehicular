import os
import torch
import FeatureExtractionClient 
import GlobalVariables
from torch.utils.data import DataLoader
import numpy as np

def ReadFile(file_path):
    if not os.path.exists(file_path):
        print(f"File {file_path} not found. Creating an empty tensor file.")
        # images, targets = next(iter(GlobalVariables.trainloaders[0]))
        feature_extractor_model = FeatureExtractionClient.FeatureExtractor(GlobalVariables.model)
        feature_extractor_model.train()
        dummy_loader = DataLoader([([0]*640*640, 0)]*100, batch_size = GlobalVariables.BatchSize)
        #images, targets = next(iter(dummy_loader))
        for images, targets in GlobalVariables.trainloaders[1]:
            outputs = feature_extractor_model(images)
            torch.save(outputs, 'feature_outputs.pt')
            torch.save(targets, 'targets.pt')
        # tensor = torch.rand(shape)# torch.zeros(shape)
        # torch.save(tensor, file_path)
    else:
        print(f"File {file_path} already exists.")
    return torch.load(file_path)

def simulate_feature_extraction(data_loader, feature_extractor):
    feature_list = []
    targets_list = []

    with torch.no_grad():
        for images, target in data_loader:
            feature = feature_extractor(images)
            feature_list.append(feature)
            targets_list.append(target)
        features = torch.cat(feature_list)
        # targets = [item for sublist in targets_list for item in sublist]
        targets = torch.cat(targets_list)
    return features, targets

#region Save trainloader and validation loader

#endregion

