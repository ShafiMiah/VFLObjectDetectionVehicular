import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import glob
import random
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO
from torchvision import transforms
from torchvision.transforms import ToTensor
from PIL import Image
from pathlib import Path
# import cv2
import GlobalVariables

# def plot_image_with_boxes(image, boxes):
#     # Ensure image is a NumPy array
#     if isinstance(image, torch.Tensor):
#         # Convert from Tensor (C, H, W) to (H, W, C) and convert to NumPy array
#         image = image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
#     print(image.shape)
#     # Ensure the image is contiguous in memory
#     image = np.ascontiguousarray(image)
    
#     # Ensure the image is in the correct layout (height, width, channels)
#     if len(image.shape) == 3 and image.shape[2] == 3:
#         pass  # Image is correctly formatted
#     else:
#         raise ValueError("Image must be a 3D array with shape (height, width, 3)")
#     #Extract the coOrdinate
    
#     #end extract
#     plt.figure(figsize=(10, 10))
#     for box in boxes:
#         class_id, x1, y1, x2, y2 = convert_from_yolo_format(box,image.shape)
#         # class_id, x1, y1, x2, y2 = box
#         label = str(int(class_id)) #reverse_class_mapping[int(class_id)]
#         # Draw the rectangle
#         cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
#         # Put the text label
#         cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
#     plt.imshow(image)
#     plt.axis('off')
#     plt.show()
# # Displaying a batch of images with bounding boxes
# def DisplayTrainLoaderImage(trainloader):    
#     for imgs, targets in trainloader:
#         imgs = imgs.permute(0, 2, 3, 1).numpy()  # Convert to (batch, height, width, channels)
#         for img, target in zip(imgs, targets):
#             plot_image_with_boxes(img, target)
#         break
# def convert_to_yolo_format(boxes, image_shape):
#     """
#     Convert bounding boxes to YOLO format.

#     Args:
#         boxes (numpy.ndarray): Array of bounding boxes in format [x_min, y_min, x_max, y_max].
#         image_shape (tuple): Shape of the image (height, width).

#     Returns:
#         numpy.ndarray: Array of bounding boxes in YOLO format [center_x, center_y, width, height].
#     """
#     image_height, image_width, channel = image_shape

#     # Calculate center_x, center_y, width, and height in pixels
#     x_min, y_min, x_max, y_max = boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
#     center_x = (x_min + x_max) / 2
#     center_y = (y_min + y_max) / 2
#     width = x_max - x_min
#     height = y_max - y_min

#     # Normalize the bounding box coordinates
#     center_x /= image_width
#     center_y /= image_height
#     width /= image_width
#     height /= image_height

#     # Stack the converted bounding boxes with class ids
#     yolo_boxes = np.column_stack((boxes[:, 0], center_x, center_y, width, height))

#     return yolo_boxes
class KITTIYoloDataset(Dataset):
    def __init__(self, images_dir, labels_dir, class_mapping=None, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_paths = sorted(glob.glob(os.path.join(images_dir, '*.png')))
        self.label_paths = sorted(glob.glob(os.path.join(labels_dir, '*.txt')))
        # self.class_mapping = class_mapping
        self.transform = transform
        # self.device = device
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        # width, height = GlobalVariables.ImageSize
        image = Image.open(img_path).convert("RGB")
        image = torch.tensor(np.array(image) / 255.0, dtype=torch.float32).permute(2, 0, 1)
        
        labels = []
        with open(label_path, 'r') as f:
            for line in f.readlines():
                class_id, x_center, y_center, width, height = map(float, line.strip().split())
                labels.append([class_id, x_center, y_center, width, height])
                
        if len(labels) > 0:
            labels = torch.tensor(labels, dtype=torch.float32)
        else:
            labels = torch.empty((0, 5), dtype=torch.float32)  # Shape [0, 5] for consistency
        # labels = torch.tensor(labels, dtype=torch.float32)        
        # if self.transform:
        #     img, labels = self.transform(img, labels)
        # return img, labels
        return image, labels 
class KITTIDataset(Dataset):
    def __init__(self, image_dir, label_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.png') or f.endswith('.jpg')]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        label_path = os.path.join(self.label_dir, self.image_files[idx].replace('.jpg', '.txt').replace('.png', '.txt'))
        
        img = cv2.imread(img_path)
        img = img[..., ::-1]  # Convert BGR to RGB
        img = img / 255.0  # Normalize to [0, 1]
        img = np.transpose(img, (2, 0, 1))  # Change shape to (C, H, W)
        
        # Load label (YOLO format: class_id center_x center_y width height)
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as file:
                for line in file.readlines():
                    parts = list(map(float, line.strip().split()))
                    labels.append(parts)
        
        labels = np.array(labels)
        return torch.tensor(img, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

# class_mapping = {
#     "Pedestrian" : 0,
#     "Truck" : 1,
#     "Car" : 2,
#     "Cyclist" : 3,
#     "DontCare" : 4,
#     "Misc" : 5,
#     "Van" : 6,
#     "Tram" : 7,
#     "Person_sitting" : 8
# }  


def GetDataDirectory(clientNumber):
        trainloaders = []
        valloaders = []    
        # Paths to the dataset directories
        clientDirectory = str(clientNumber)
        if clientNumber == GlobalVariables.NumberOfClients:
            clientDirectory = 'CentralServer'
        train_images_dir = os.path.join(GlobalVariables.ClientsTrainFolderPath, clientDirectory,'images','train') 
        train_labels_dir = os.path.join(GlobalVariables.ClientsTrainFolderPath, clientDirectory,'labels','train')
        val_images_dir = os.path.join(GlobalVariables.ClientsTrainFolderPath, clientDirectory,'images','val') 
        val_labels_dir = os.path.join(GlobalVariables.ClientsTrainFolderPath, clientDirectory,'labels','val')
        #Transform all the images to same size
        print('return client '+  str(clientNumber) +' directory: '+ train_images_dir)
        return train_images_dir, train_labels_dir, val_images_dir, val_labels_dir