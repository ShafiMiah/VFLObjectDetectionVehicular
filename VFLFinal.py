import os
import torch
import subprocess

import GlobalVariables
from torch.utils.data import DataLoader
import multiprocessing
import shutil
import LoadKittiDataset2
import FeatureExtractionClient
import Server
#new import
import flwr
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Metrics, Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from collections import OrderedDict
from typing import List, Tuple
import tempfile
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from datasets.utils.logging import disable_progress_bar
from flwr.simulation import run_simulation
from ultralytics import YOLO
import ModelParameters
from collections import OrderedDict
import Utility
import cv2
DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU
print(f"Training on {DEVICE}")
print(f"Flower {flwr.__version__} / PyTorch {torch.__version__}")
disable_progress_bar()
#end new import
average_map50_per_round = []
average_map95_per_round = []
def GetSMVFLEvaluationMetric():
    file_path = 'global_model_metrics.txt'  # Replace with your actual file path

    # Read the file and process each line
    with open(file_path, mode='r') as file:
        for line in file:
            # Split the line into columns by the comma separator
            columns = line.strip().split(',')
        
            # Access each column
            communication_round = columns[0]  # First column
            map50 = float(columns[1])  # Second column
            map95 = float(columns[2])  # Third column
            average_map50_per_round.append(map50)
            average_map95_per_round.append(map95)
def plot_average_map():
    """Plot average mAP50 and mAP95 per communication round."""
    rounds = list(range(1, len(average_map50_per_round) + 1))
    plt.figure(figsize=(12, 6))
    plt.plot(rounds, average_map50_per_round, marker="o", linestyle="-", label="mAP50 evaluation metric", color="b")
    plt.plot(rounds, average_map95_per_round, marker="s", linestyle="--", label="mAP95 evaluation metric", color="r")
    plt.title("mAP per Communication Round")
    plt.xlabel("Communication Round")
    plt.ylabel("mAP")
    plt.grid(True)
    plt.legend()
    plt.show()
def get_parameters(model):
    """
    Extracts the parameters from the YOLO model as a list of NumPy arrays.

    :param model: The YOLO model.
    :type model: ultralytics.YOLO
    :return: List of model parameters as NumPy arrays.
    :rtype: List[np.ndarray]
    """
    # Convert model state_dict to NumPy arrays
    parameters_np = [param.cpu().numpy() for param in model.state_dict().values()]
    return parameters_np
def set_parameters(model, parameters):
    """
    Updates the YOLO model with the given parameters.

    :param model: The YOLO model.
    :type model: ultralytics.YOLO
    :param parameters: List of NumPy arrays representing model weights.
    :type parameters: List[np.ndarray]
    """
    # Map NumPy arrays back to state_dict
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({key: torch.tensor(param) for key, param in params_dict})
    
    # Load the updated state_dict into the model
    model.load_state_dict(state_dict, strict=True)
    
def log_map_to_file(server_round: int, map50, map95):
    """Log the mAP scores to a file."""
    with open("Single_client_metrics.txt", "a") as f:
        f.write(f"round:{server_round}, mAP50: {map50}, mAP95: {map95}\n")
def run_script_in_new_terminal(script_name, args):
     # For Windows
    if os.name == 'nt': 
        # Command to open a new terminal and run the script with arguments
        command = f'start cmd /k "python {script_name} {" ".join(args)}"'
    # For macOS/Linux
    elif os.name == 'posix': 
        # Command to open a new terminal and run the script with arguments
        command = f'gnome-terminal -- bash -c "python3 {script_name} {" ".join(args)}; exec bash"'
    else:
        raise NotImplementedError('Unsupported OS')
    
    subprocess.run(command, shell=True)
class_mapping = {
    "Pedestrian" : 0,
    "Truck" : 1,
    "Car" : 2,
    "Cyclist" : 3,
    "DontCare" : 4,
    "Misc" : 5,
    "Van" : 6,
    "Tram" : 7,
    "Person_sitting" : 8
} 
reverse_class_mapping = {v: k for k, v in class_mapping.items()}
def predictImageObjectYOLO():
        # model = YOLO('CentralServer1000epoch/train/weights/best.pt')
        # #test image 000221.png, 005255.png
        # img = cv2.imread('datasets/KITTIDataset/CentralServer/train/images/000221.png')
        # model = YOLO('SingleCLientTrain/weights/best.pt')
        # img = cv2.imread('datasets/KITTIDataset/CentralServer/train/images/000221.png')
        model = YOLO('Global_Model.pt')
        img = cv2.imread('datasets/KITTIDataset/CentralServer/train/images/000221.png')
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = cv2.resize(img, GlobalVariables.ImageSize)
        results = model.predict(img)
        # Draw image
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf.item()  # Convert tensor to float
                cls = box.cls.item()  # Convert tensor to float
                label = reverse_class_mapping[int(cls)] #str(cls)

                # Draw the bounding box and label
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # cv2.putText(img, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(img, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save or display the image
        output_path = 'output.jpg'
        cv2.imwrite(output_path, img)
        cv2.imshow('YOLO Prediction', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
def SingleClient():
    singleModel = YOLO('model.yaml')
    singleModel.to(GlobalVariables.DEVICE)
    # ModelParameters.init_seed(YOLO('model.yaml'), "WeightPath/"+str(0)+"/seed.pkl")
    singleModel.train(data='YOLOConfigClient0.yaml', epochs = 1000,imgsz=672,rect=True, device = 0 if torch.cuda.is_available() else 'cpu')
    finished = True
    # # singleModel.train(data='YOLOConfigClient0.yaml',imgsz=672,rect=True, epochs=30,device=GlobalVariables.DEVICE)
    # result = singleModel.val(data='YOLOConfigClientCentralServer.yaml',imgsz=672,rect=True,device = 0 if torch.cuda.is_available() else 'cpu')
    # map50 = result.box.map50  # mAP at IoU 0.5
    # map95 = result.box.map    # mAP at IoU 0.5:0.95
    # log_map_to_file(0,map50,map95)
    # print(f"ValidationResultCentral = MAP50: '{str(map50)}' MAP95: '{ str(map95)}'")
    # for i in range(30):
    #     singleModel.train(data='YOLOConfigClient0.yaml',imgsz=672,rect=True, epochs=1,device=GlobalVariables.DEVICE)
    #     # parameters = get_parameters(singleModel)
    #     # set_parameters(singleModel, parameters)
    #     result = singleModel.val(data='YOLOConfigClient0.yaml',imgsz=672,rect=True,device=GlobalVariables.DEVICE)
        
        # params_dict = zip(singleModel.state_dict().keys(), parameters)
        # state_dict = OrderedDict({key: torch.tensor(x) for key, x in params_dict})
        # singleModel.load_state_dict(state_dict, strict=False)
        
        # # with tempfile.TemporaryDirectory() as temp_dir:
        # #     modelName = 'model'+str(0)+'.pt'
        # #     tmp_file_path = os.path.join(temp_dir, modelName)
        # #     torch.save(singleModel, tmp_file_path)  # Save the model
        # #     singleModel = YOLO(tmp_file_path)  # Load the model
            
        # map50 = result.box.map50  # mAP at IoU 0.5
        # map95 = result.box.map    # mAP at IoU 0.5:0.95
        # log_map_to_file(i,map50,map95)
        # print(f"ValidationResult'{str(i)}' = MAP50: '{str(map50)}' MAP95: '{ str(map95)}'")
def StartSimulation():
    
    run_script_in_new_terminal('Server.py', ['--arg1', '0'])
    # passive_client_procs = []
    for i in range(GlobalVariables.NumberOfClients - 1):
        run_script_in_new_terminal('FeatureExtractionClient.py', ['--arg1', str(i)])
    run_script_in_new_terminal('DetectionClient.py', ['--arg1', '4']) 
def calculateElsapedTime():
    averageElapse = []
    elapsed_time1 = 1736170290.354144 - 1736169978.6981387
    averageElapse.append(elapsed_time1)
    elapsed_time1 = 1736170949.3419735 - 1736170290.354144
    averageElapse.append(elapsed_time1)
    elapsed_time1 = 1736171656.3193665 - 1736170949.3419735
    averageElapse.append(elapsed_time1)
    elapsed_time1 = 1736171982.894876 - 1736171656.3193665
    averageElapse.append(elapsed_time1)
    elapsed_time1 = 1736172308.5847924 - 1736171982.894876
    averageElapse.append(elapsed_time1)
    elapsed_time1 = 1736172638.582041 - 1736172308.5847924
    averageElapse.append(elapsed_time1)
    elapsed_time1 = 1736172999.101327 - 1736172638.582041
    averageElapse.append(elapsed_time1)
    elapsed_time1 = 1736173356.8771403 - 1736172999.101327
    averageElapse.append(elapsed_time1)
    p =1
if __name__ == "__main__":
   #region  create configuration
   # run_script_in_new_terminal('CreateConfiguration.py', ['--arg1', "CentralServer"])
   # for i in range(GlobalVariables.NumberOfClients):
   #      run_script_in_new_terminal('CreateConfiguration.py', ['--arg1', str(i)])
   #endregion
   # StartSimulation()
   # SingleClient() 
   # FeatureExtractionClient.RunPassiveClient("0") 
   # print(torch.__version__) 
   # backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 0.0}}

   #  # When running on GPU, assign an entire GPU for each client
   # if DEVICE.type == "cuda":
   #      backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 1.0}}
   # run_simulation(
   #  server_app=Server.server,
   #  client_app=FeatureExtractionClient.client,
   #  num_supernodes=5,
   #  # backend_config=backend_config,
   #  ) 
   # GetSMVFLEvaluationMetric()
   # plot_average_map()
   # predictImageObjectYOLO() 
   # Utility.ReadWriteFile() 
   # calculateElsapedTime()
   print("Hello world")