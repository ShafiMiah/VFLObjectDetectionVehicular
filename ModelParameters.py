# from fedn.utils.helpers.helpers import get_helper
from ultralytics import YOLO
import torch
import collections
import tempfile
import Utility
import os
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict
import numpy as np
helper = Utility.get_helper()

def get_model(uninitialized_model, parameter_path,client_num):
    """Load model parameters from file and populate model.

    param parameter_path: The path to load from.
    :type parameter_path: str
    :return: The loaded model.
    :rtype: torch.nn.Module
    """
    model = uninitialized_model # YOLO('model.yaml').to(device)
    parameters_np = helper.load(parameter_path)
    params_dict = zip(model.state_dict().keys(), parameters_np)
    state_dict = collections.OrderedDict({key: torch.tensor(x) for key, x in params_dict})
    model.load_state_dict(state_dict, strict=True)
    #Do we need extra loading?
    with tempfile.TemporaryDirectory() as temp_dir:
        modelName = 'model'+str(client_num)+'.pt'
        tmp_file_path = os.path.join(temp_dir, modelName)
        torch.save(model, tmp_file_path)  # Save the model
        model = YOLO(tmp_file_path)  # Load the model
    return model

def save_parameters(model, parameter_path):
    """Save model parameters to file.

    :param model: The model to serialize.
    :type model: torch.nn.Module
    :param parameter_path: The path to save to.
    :type parameter_path: str
    """
    parameters_np = [val.cpu().numpy() for _, val in model.state_dict().items()]
    helper.save(parameters_np, parameter_path)

def init_seed(model, parameter_path):
    save_parameters(model, parameter_path)

# if __name__ == "__main__":
#     init_seed('../seed.npz')

#region client parameter computation
def get_parameters(net, parameter_path, client_number) -> List[np.ndarray]:
    YOLOModel = get_model(net, parameter_path,client_number)
    return [val.cpu().numpy() for _, val in YOLOModel.state_dict().items()]

# def set_parameters(net, parameters: List[np.ndarray]):
#     params_dict = zip(net.state_dict().keys(), parameters)
#     state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
#     net.load_state_dict(state_dict, strict=True)
#     return net

# def train(net, configuration_file : str, epochs: int):
#     trainResult = net.train(data = configuration_file, epochs = epochs,device=0 if torch.cuda.is_available() else 'cpu')
#     return net

# def test(net, configuration_file : str, number_sample):
#    result = net.val(data = configuration_file, plots=False,device=0 if torch.cuda.is_available() else 'cpu')
#    map50 = result.box.map50 
#    map95 = result.box.map   
#    loss = 1 - map50  # Example placeholder loss (can be replaced with custom logic)
#    print('ValidationResult = MAP50: '+str(map50)+ ' MAP95: '+ str(map95))
#    return float(loss), number_sample, {"map50": map50, "map95": map95}    
#endregion  
#
def get_model_parameters(model) -> List[np.ndarray]:
    parameters_np = [param.cpu().numpy() for param in model.state_dict().values()]
    return parameters_np 
def save_training_state(model, optimizer, path):
    """Save model and optimizer state."""
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        path,
    )

def load_training_state(path):
    """Load model and optimizer state."""
    return torch.load(path)
 