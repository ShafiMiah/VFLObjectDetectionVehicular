import os
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import flwr as fl
import GlobalVariables
import argparse
# Load YOLOv8 model
# model = torch.hub.load('ultralytics/yolov8', 'yolov8s')

# Split the model into feature extraction part
class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        self.feature_extractor  = nn.Sequential(*list(model.model.children())[:-1])  # Assuming the last  layers are detection layers

    def forward(self, x):
        return self.feature_extractor(x)

feature_extractor = FeatureExtractor(GlobalVariables.model)

class FeatureExtractionClient(fl.client.NumPyClient):
    def __init__(self, model, data_loader):
        self.model = model
        self.data_loader = data_loader

    def get_parameters(self):
        return [param.data.numpy() for param in self.model.parameters()]

    def set_parameters(self, parameters):
        for param, new_param in zip(self.model.parameters(), parameters):
            param.data = torch.tensor(new_param, dtype=param.data.dtype)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        print("self.model.parameters() printing")
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        for images, targets in self.data_loader:
            optimizer.zero_grad()
            outputs = self.model(images)
            # Save outputs to disk to be used by detection client
            print("feature_outputs.pt saving")
            torch.save(outputs, 'feature_outputs.pt')
            torch.save(targets, 'targets.pt')
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        return self.get_parameters(), len(self.data_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        loss = 0.0
        for images, targets in self.data_loader:
            with torch.no_grad():
                outputs = self.model(images)
                # Save outputs to disk to be used by detection client
                print("feature_outputs.pt saving")
                torch.save(outputs, 'C:\\Users\\shafi\\source\\repos\\VFLFinal\\feature_outputs.pt')
                torch.save(targets, 'C:\\Users\\shafi\\source\\repos\\VFLFinal\\targets.pt')
                loss += criterion(outputs, targets).item()
        return float(loss) / len(self.data_loader), len(self.data_loader.dataset), {}

def client_fn_feature_extraction(cid: str):
  
    # Load model
    print("FeatureExtractorClient calling: print parameters")
    # feature_extractor = FeatureExtractor(GlobalVariables.model)
    print(feature_extractor.parameters())
    print("End printing parameters")
    trainloader = GlobalVariables.trainloaders[int(cid)]
    valloader = GlobalVariables.valloaders[int(cid)]
    # Create a  single Flower client representing a single organization
    client = FeatureExtractionClient(feature_extractor, trainloader)
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)
    
#region comment
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Start server')
    parser.add_argument('--arg1', type=str, required=True, help='First argument')
    args = parser.parse_args()
    client_fn_feature_extraction(args.arg1)
#endregion