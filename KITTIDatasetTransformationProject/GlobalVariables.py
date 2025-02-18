from ultralytics import YOLO
# from LoadKittiDataset import loadKITTI_datasets
import torch
model = YOLO("yolov8n.pt")
NumberOfClients = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ImageSize = (672,224)
BatchSize = 4
#region KITTI dataset
RootKITTIDataSetLocation = 'C:\\Shafi Personal\Study\\Masters Thesis\\Thesis Project\\ImplementationAndCode\\VFLFinalImaplementation\\VFLFinal\\dataset\\training'
ClientsTrainFolderPath = 'C:\\Shafi Personal\Study\\Masters Thesis\\Thesis Project\\ImplementationAndCode\\VFLFinalImaplementation\\VFLFinal\\dataset\\training\\VFLTrain'
ValidationSetSplitRatio = 0.2
# trainloaders, valloaders = loadKITTI_datasets()
#endregion


