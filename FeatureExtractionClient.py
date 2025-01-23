import os
import torch
from torch import nn
# from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import flwr as fl
import GlobalVariables
import argparse
from typing import Dict, List, Tuple
import Utility
import numpy as np
import torch.optim as optim
import LoadKittiDataset2
from sklearn.metrics import average_precision_score
from torchvision.ops import box_iou
from PIL import Image
from ultralytics import YOLO
import ModelParameters
import collections
import tempfile
#new import
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import flwr
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Context
import time
#end new import


class PassiveYOLOClient(NumPyClient):
    def __init__(self, train_sample,val_sample, client_number):
        self.train_sample = train_sample
        self.val_sample = val_sample
        self.model = YOLO('model.yaml')
        self.client_number = client_number 
        self.parameter_path = "WeightPath/"+str(client_number)+"/seed.pkl"
        #Enable GPU computation and initialize model
        self.model.to(GlobalVariables.DEVICE)
        ModelParameters.init_seed(YOLO('model.yaml'), "WeightPath/"+str(client_number)+"/seed.pkl")
        
    #region new get and set parameter
    def get_parameters(self, config):
        """Return the model weights."""
        #get model 
        print('Get parameter called')
        # YOLOModel = ModelParameters.get_model(self.model, self.parameter_path,self.client_number)
        return ModelParameters.get_parameters(self.model,self.parameter_path, self.client_number)
    def set_parameters(self, parameters):
        YOLOModel = ModelParameters.get_model(self.model, self.parameter_path,self.client_number)
        params_dict = zip(YOLOModel.state_dict().keys(), parameters)
        state_dict = collections.OrderedDict({key: torch.tensor(x) for key, x in params_dict})
        YOLOModel.load_state_dict(state_dict, strict=False)
        # Do we need extra loading? Comment out
        with tempfile.TemporaryDirectory() as temp_dir:
            modelName = 'model'+str(self.client_number)+'.pt'
            tmp_file_path = os.path.join(temp_dir, modelName)
            torch.save(YOLOModel, tmp_file_path)  # Save the model
            YOLOModel = YOLO(tmp_file_path)  # Load the model
        ModelParameters.save_parameters(YOLOModel,self.parameter_path)
    #endregion    
    def fit(self, parameters, config):
        """Train the model locally."""
        start_time = time.time()
        self.set_parameters(parameters)
        YOLOModel = ModelParameters.get_model(self.model, self.parameter_path,self.client_number)
        #Try setting the local epoch to 1/2
        print('Config='+'YOLOConfigClient'+str(self.client_number)+'.yaml')
        # with tempfile.TemporaryDirectory() as tmp_dir:
        #     print('Directory train came here')
        #     trainResult = YOLOModel.train(data='YOLOConfigClient'+str(self.client_number)+'.yaml', epochs=1,batch=16,verbose=False,exist_ok=True, project=tmp_dir,device=0 if torch.cuda.is_available() else 'cpu')
        trainResult = YOLOModel.train(data='YOLOConfigClient'+str(self.client_number)+'.yaml', epochs = 5,imgsz=672,rect=True, device = 0 if torch.cuda.is_available() else 'cpu')
        ModelParameters.save_parameters(YOLOModel, self.parameter_path)
        # Return updated weights
        fitParameters = ModelParameters.get_parameters(self.model,self.parameter_path, self.client_number)
        current_time = time.time()
        elapsed_time = current_time - start_time
        print(f"Elapsed Time : {elapsed_time:.2f} seconds")
        return fitParameters, self.train_sample, {}

    def evaluate(self, parameters, config):
        print('Directory train came here')
        """Evaluate the model locally."""
        self.set_parameters(parameters)
        YOLOModel = ModelParameters.get_model(self.model, self.parameter_path,self.client_number)
        # Evaluate the model on the validation dataset
        # with tempfile.TemporaryDirectory() as tmp_dir:
        #     # train_results = model.val(data=data_yaml_path, split='train', verbose=False, exist_ok=True,project=tmp_dir)
        #     result = YOLOModel.val(data='YOLOConfigClient'+str(self.client_number)+'.yaml', split='val', verbose=False, exist_ok=True,project=tmp_dir,device=0 if torch.cuda.is_available() else 'cpu')
        result = YOLOModel.val(data='YOLOConfigClient'+str(self.client_number)+'.yaml', plots=False,imgsz=672,rect=True,device=0 if torch.cuda.is_available() else 'cpu')
        # Use metrics provided by the DetMetrics object
        map50 = result.box.map50  # mAP at IoU 0.5
        map95 = result.box.map    # mAP at IoU 0.5:0.95
        # Return a loss placeholder (if required, add custom loss calculation)
        loss = 1 - map50  # Example placeholder loss (can be replaced with custom logic)
        print('ValidationResult = MAP50: '+str(map50)+ ' MAP95: '+ str(map95))
        return float(loss), self.val_sample, {"map50": map50, "map95": map95}
#end test YOLO client

def RunPassiveClient(clientNumber):
    feature_extractor = nn.Sequential(*GlobalVariables.model.model.model[:])
    train_images_dir, train_labels_dir, val_images_dir, val_labels_dir = LoadKittiDataset2.GetDataDirectory(clientNumber)
    train_sample = len([f for f in os.listdir(train_images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    val_sample = len([f for f in os.listdir(val_images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    client = PassiveYOLOClient(train_sample, val_sample, clientNumber)
    fl.client.start_client(server_address="localhost:8080", client=client.to_client())
    
# def client_fn(context: Context) -> Client:
#     """Create a Flower client representing a single organization."""
#     partition_id = context.node_config["partition-id"]
#     return RunPassiveClient(int(partition_id))


# # Create the ClientApp
# client = ClientApp(client_fn=client_fn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Start server')
    parser.add_argument('--arg1', type=str, required=True, help='First argument')
    args = parser.parse_args()
    print('Running feature extraction client '+  str(args.arg1))
    RunPassiveClient(int(args.arg1))

    # client_fn_feature_extraction(args.arg1)