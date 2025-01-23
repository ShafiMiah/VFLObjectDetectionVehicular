
import torch
import torch.nn as nn
from ultralytics import YOLO  # Assuming you are using YOLOv8 from ultralytics

# Define the model structure where YOLOv8 model is split into feature extraction and detection layers
class YOLOv8FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(YOLOv8FeatureExtractor, self).__init__()
        # Extract the feature extraction layers (backbone + neck) of YOLOv8
        self.feature_extractor = nn.Sequential(*list(model.model[:-2]))  # backbone + neck layers

    def forward(self, x):
        return self.feature_extractor(x)


class YOLOv8DetectionHead(nn.Module):
    def __init__(self, model):
        super(YOLOv8DetectionHead, self).__init__()
        # Extract the detection head (head) of YOLOv8
        self.detection_head = model.model[-2:]  # head layers (classify and bbox layers)

    def forward(self, x):
        return self.detection_head(x)

# Assuming the model is pre-trained
model = YOLO('yolov8n.pt')  # Load a pre-trained YOLOv8 model (or any YOLOv8 model)

# Create separate models for feature extraction and detection
feature_extractor = YOLOv8FeatureExtractor(model)
detection_head = YOLOv8DetectionHead(model)

# Simulated federated learning setup with active and passive clients

class PassiveClient(nn.Module):
    def __init__(self, feature_extractor):
        super(PassiveClient, self).__init__()
        self.feature_extractor = feature_extractor

    def train_local(self, data_loader, optimizer, loss_fn):
        self.train()  # Set the model to training mode
        for images, targets in data_loader:
            optimizer.zero_grad()
            features = self.feature_extractor(images)  # Forward pass through the feature extractor
            loss = loss_fn(features, targets)
            loss.backward()
            optimizer.step()
        return self.feature_extractor.state_dict()  # Return updated feature extractor parameters


class ActiveClient(nn.Module):
    def __init__(self, feature_extractor, detection_head):
        super(ActiveClient, self).__init__()
        self.feature_extractor = feature_extractor
        self.detection_head = detection_head

    def train_local(self, data_loader, optimizer, loss_fn):
        self.train()  # Set the model to training mode
        for images, targets in data_loader:
            optimizer.zero_grad()
            features = self.feature_extractor(images)  # Forward pass through the feature extractor
            predictions = self.detection_head(features)  # Forward pass through the detection head
            loss = loss_fn(predictions, targets)
            loss.backward()
            optimizer.step()
        return self.feature_extractor.state_dict(), self.detection_head.state_dict()  # Return updated parameters


# Federated learning server to aggregate model parameters
class FederatedServer:
    def __init__(self):
        self.feature_extractor_aggregated = None
        self.detection_head_aggregated = None

    def aggregate_parameters(self, feature_extractor_params_list, detection_head_params_list):
        # Example: simple averaging of model parameters
        self.feature_extractor_aggregated = self.aggregate_layer_params(feature_extractor_params_list)
        self.detection_head_aggregated = self.aggregate_layer_params(detection_head_params_list)

    def aggregate_layer_params(self, params_list):
        # Example: averaging the weights of feature extractor layers
        avg_params = {}
        for key in params_list[0].keys():
            avg_params[key] = torch.mean(torch.stack([params[key] for params in params_list]), dim=0)
        return avg_params

    def update_clients(self, active_client, passive_clients):
        # Update the feature extractor for all clients
        for client in passive_clients:
            client.feature_extractor.load_state_dict(self.feature_extractor_aggregated)
        active_client.feature_extractor.load_state_dict(self.feature_extractor_aggregated)
        active_client.detection_head.load_state_dict(self.detection_head_aggregated)


# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Simulate the clients
passive_clients = [PassiveClient(feature_extractor).to(device) for _ in range(5)]
active_client = ActiveClient(feature_extractor, detection_head).to(device)

# Assume we have data loaders for training
# For simplicity, here just creating dummy data
dummy_data_loader = [(torch.rand(4, 3, 640, 640).to(device), torch.rand(4, 80, 640, 640).to(device))] * 10  # Dummy data

# Optimizers and loss function
optimizer = torch.optim.Adam(list(feature_extractor.parameters()) + list(detection_head.parameters()), lr=0.001)
loss_fn = nn.MSELoss()  # This is a placeholder, you'd likely use a more suitable loss for object detection

# Create the server
server = FederatedServer()

# Simulate a federated learning round
for round in range(10):
    print(f"Round {round + 1}")

    # Train on passive clients
    passive_feature_extractor_params = []
    for client in passive_clients:
        passive_feature_extractor_params.append(client.train_local(dummy_data_loader, optimizer, loss_fn))

    # Train on the active client
    active_feature_extractor_params, active_detection_head_params = active_client.train_local(dummy_data_loader, optimizer, loss_fn)

    # Aggregation step by the server
    server.aggregate_parameters(passive_feature_extractor_params, [active_detection_head_params])
    server.update_clients(active_client, passive_clients)

    print(f"Round {round + 1} completed.")