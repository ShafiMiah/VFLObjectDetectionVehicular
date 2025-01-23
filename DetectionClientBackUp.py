import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import flwr as fl
import GlobalVariables
import argparse
import Utility
import torch.optim as optim
import LoadKittiDataset2

class ActiveClient(fl.client.NumPyClient):
    def __init__(self, model, data_loader):
        self.model = model.to(GlobalVariables.DEVICE)
        self.data_loader = data_loader
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()  # Example detection loss

    def get_parameters(self):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = dict(zip(self.model.state_dict().keys(), [torch.tensor(val) for val in parameters]))
        self.model.load_state_dict(state_dict)

    def train(self, feature_outputs, labels):
        self.model.train()
        self.optimizer.zero_grad()
        detections = self.model(feature_outputs)
        loss = self.criterion(detections, labels.to(GlobalVariables.DEVICE))
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        for feature_outputs, labels in self.data_loader:
            loss = self.train(feature_outputs.to(GlobalVariables.DEVICE), labels)
        return self.get_parameters(), len(self.data_loader.dataset), {"loss": loss}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss = 0
        for feature_outputs, labels in self.data_loader:
            loss += self.train(feature_outputs.to(GlobalVariables.DEVICE), labels)
        return loss / len(self.data_loader), len(self.data_loader.dataset), {"loss": loss}

def runClient(clientNumber):
    print('Running detection client '+  str(clientNumber))
    detector = GlobalVariables.model.model.model[-1] 
    train_images_dir, train_labels_dir, val_images_dir, val_labels_dir = LoadKittiDataset2.GetDataDirectory(clientNumber)
    train_dataset = LoadKittiDataset2.KITTIYoloDataset(images_dir = train_images_dir, labels_dir = train_labels_dir) 
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=Utility.collate_fn)
    client = ActiveClient(model = GlobalVariables.model, data_loader=train_loader)  # Active client needs one loader for training
    fl.client.start_numpy_client(server_address="localhost:8080", client=client) 
  
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Start server')
    parser.add_argument('--arg1', type=str, required=True, help='First argument')
    args = parser.parse_args()
    runClient(int(args.arg1))