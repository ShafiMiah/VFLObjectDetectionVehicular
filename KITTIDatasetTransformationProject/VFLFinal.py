import os
import torch
import subprocess
import DetectionClient
import GlobalVariables
from torch.utils.data import DataLoader
import multiprocessing
import flwr as fl
import Utility
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import shutil
import LoadKittiDataset2
import GlobalVariables
import cv2
#region Single YOLO8 train
def GenerateYOLOFormat():
    trainloaders, valloaders = LoadKittiDataset2.loadKITTI_datasets()
    for clientNumber in range((GlobalVariables.NumberOfClients + 1)):
        for images, targets in trainloaders[int(clientNumber)]:
            print(images.shape) 
        for images, targets in valloaders[int(clientNumber)]:
            print(images.shape) 
          
if __name__ == "__main__":
   GenerateYOLOFormat()
