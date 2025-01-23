import flwr as fl
import argparse
import matplotlib.pyplot as plt
import GlobalVariables
import numpy as np
from ultralytics import YOLO
import torch
import tempfile
import os
#new import
import flwr
import collections
from collections import OrderedDict
from flwr.server import start_server
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Metrics, Context, Parameters, parameters_to_ndarrays
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from CustomStrategy import FedCustom
# from CustomStrategy import CustomFedAvg
#end new import
class CustomVFLFedAvg(FedAvg):
    def __init__(self, fraction_fit : float = 1.0, 
                 fraction_evaluate : float = 1.0,
                 min_fit_clients : int = GlobalVariables.NumberOfClients, 
                 min_evaluate_clients : int = GlobalVariables.NumberOfClients, min_available_clients: int = GlobalVariables.NumberOfClients, **kwargs):
        super().__init__(
            fraction_fit = fraction_fit,
            fraction_evaluate = fraction_evaluate,
            min_fit_clients = min_fit_clients,
            min_evaluate_clients = min_evaluate_clients,
            min_available_clients = min_available_clients,
            **kwargs,
        )
        self.global_model = YOLO("model.yaml")  # Load the YOLO model
        self.average_map50_per_round = []
        self.average_map95_per_round = []
        def evaluate(self, server_round: int, parameters: Parameters):
            """Evaluate model parameters using an evaluation function."""
            print('Server evaluate called')
            params_dict = zip(self.global_model.state_dict().keys(), parameters)
            state_dict = collections.OrderedDict({key: torch.tensor(x) for key, x in params_dict})
            self.global_model.load_state_dict(state_dict, strict=False)
            # Do we need extra loading? Comment out
            with tempfile.TemporaryDirectory() as temp_dir:
                modelName = 'model'+str(self.client_number)+'.pt'
                tmp_file_path = os.path.join(temp_dir, modelName)
                torch.save(self.global_model, tmp_file_path)  # Save the model
                self.global_model = YOLO(tmp_file_path)  # Load the model
            #Evaluate using the central server daatset
            result = self.global_model.val(data='YOLOConfigClient0.yaml', plots=False,imgsz=672,rect=True,device=0 if torch.cuda.is_available() else 'cpu')   
            map50 = result.box.map50  # mAP at IoU 0.5
            map95 = result.box.map    # mAP at IoU 0.5:0.95
            # Return a loss placeholder (if required, add custom loss calculation)
            loss = 1 - map50  # Example placeholder loss (can be replaced with custom logic)
            print('Server ValidationResult round '+str(server_round)+' = MAP50: '+str(map50)+ ' MAP95: '+ str(map95))
            return float(loss), {"map50": map50, "map95": map95}
            # if self.evaluate_fn is None:
            #     # No evaluation function provided
            #     return None
            # parameters_ndarrays = parameters_to_ndarrays(parameters)
            # eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
            # if eval_res is None:
            #     return None
            # loss, metrics = eval_res
            # return loss, metrics
def evaluate(server_round: int,parameters,config):
            YOLOModel = YOLO("model.yaml")       
            print('Evaluate function called')
            params_dict = zip(YOLOModel.state_dict().keys(), parameters)
            state_dict = collections.OrderedDict({key: torch.tensor(x) for key, x in params_dict})
            YOLOModel.load_state_dict(state_dict, strict=False)
            # Do we need extra loading? Comment out
            with tempfile.TemporaryDirectory() as temp_dir:
                modelName = 'modelserver.pt'
                tmp_file_path = os.path.join(temp_dir, modelName)
                torch.save(YOLOModel, tmp_file_path)  # Save the model
                YOLOModel = YOLO(tmp_file_path)  # Load the model
            #Evaluate using the central server daatset
            result = YOLOModel.val(data='YOLOConfigClient0.yaml', plots=False,device=0 if torch.cuda.is_available() else 'cpu')   
            map50 = result.box.map50  # mAP at IoU 0.5
            map95 = result.box.map    # mAP at IoU 0.5:0.95
            # Return a loss placeholder (if required, add custom loss calculation)
            loss = 1 - map50  # Example placeholder loss (can be replaced with custom logic)
            print('Server ValidationResult round '+str(server_round)+' = MAP50: '+str(map50)+ ' MAP95: '+ str(map95))
            log_map_to_file(server_round, map50, map95)
            return float(loss), {"map50": map50, "map95": map95}

def log_map_to_file(server_round: int, map50, map95):
    """Log the mAP scores to a file."""
    with open("global_model_metrics.txt", "a") as f:
        f.write(f"round:{server_round}, mAP50: {map50}, mAP95: {map95}\n")
        
strategy = FedAvg(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=1.0,  # Sample 50% of available clients for evaluation
    min_fit_clients = GlobalVariables.NumberOfClients,  # Never sample less than 10 clients for training
    min_evaluate_clients=GlobalVariables.NumberOfClients,  # Never sample less than 5 clients for evaluation
    min_available_clients=GlobalVariables.NumberOfClients,  # Wait until all 10 clients are available
    evaluate_fn=evaluate,
)

customStrategy = CustomVFLFedAvg(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=1.0,  # Sample 50% of available clients for evaluation
    min_fit_clients=GlobalVariables.NumberOfClients,  # Never sample less than 10 clients for training
    min_evaluate_clients=GlobalVariables.NumberOfClients,  # Never sample less than 5 clients for evaluation
    min_available_clients=GlobalVariables.NumberOfClients,  # Wait until all 10 clients are available
)
def server_fn(context: Context) -> ServerAppComponents:
    # Configure the server for 5 rounds of training
    config = ServerConfig(num_rounds=3)

    return ServerAppComponents(strategy=strategy, config=config)
# Create the ServerApp
server = ServerApp(server_fn=server_fn)

if __name__ == "__main__":
        print('Running Server ')    
        parser = argparse.ArgumentParser(description='Start server')
        parser.add_argument('--arg1', type=str, required=False, help='First argument')    
        # Num round should be 100
        # fl.server.start_server(server_address="localhost:8080", config={"num_rounds": 3}, strategy = strategy)
        start_server(server_address="localhost:8080", config=ServerConfig(num_rounds=50), strategy = strategy)
#         print('MAP comes here: ')
        # customStrategy.plot_average_map()
        # customStrategy.plot_map()
        # plot_map_per_round()
