from ultralytics import YOLO
from torchvision import transforms
# from LoadKittiDataset import loadKITTI_datasets
import torch
model = YOLO("yolov8n.pt")
NumberOfClients = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#region KITTI dataset
ClientsTrainFolderPath = 'C:\\Shafi Personal\\Study\\Masters Thesis\\Thesis Project\\ImplementationAndCode\\VFLFinalImaplementation\\VFLFinal\\dataset\\training\\VFLTrain'
#endregion

