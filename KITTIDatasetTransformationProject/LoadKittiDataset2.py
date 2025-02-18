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
# import cv2
import GlobalVariables
# Custom Dataset for KITTI
# Custom Dataset for KITTI
def convert_to_yolo_format(boxes, image_shape):
    """
    Convert bounding boxes to YOLO format.

    Args:
        boxes (numpy.ndarray): Array of bounding boxes in format [x_min, y_min, x_max, y_max].
        image_shape (tuple): Shape of the image (height, width).

    Returns:
        numpy.ndarray: Array of bounding boxes in YOLO format [center_x, center_y, width, height].
    """
    image_height, image_width, channel = image_shape

    # Calculate center_x, center_y, width, and height in pixels
    x_min, y_min, x_max, y_max = boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min

    # Normalize the bounding box coordinates
    center_x /= image_width
    center_y /= image_height
    width /= image_width
    height /= image_height

    # Stack the converted bounding boxes with class ids
    yolo_boxes = np.column_stack((boxes[:, 0], center_x, center_y, width, height))

    return yolo_boxes
def convert_from_yolo_format(boxes, image_shape):
    """
    Convert YOLO format bounding boxes back to [x_min, y_min, x_max, y_max].

    Args:
        boxes (numpy.ndarray): Array of bounding boxes in YOLO format [center_x, center_y, width, height].
        image_shape (tuple): Shape of the image (height, width).

    Returns:
        numpy.ndarray: Array of bounding boxes in [x_min, y_min, x_max, y_max] format.
    """
    image_height, image_width, channel = image_shape
    boxes = boxes.numpy()
    # Denormalize center_x, center_y, width, and height
    center_x = boxes[1] * image_width
    center_y = boxes[2] * image_height
    width = boxes[ 3] * image_width
    height = boxes[ 4] * image_height

    # Calculate x_min, y_min, x_max, y_max
    x_min = center_x - width / 2
    y_min = center_y - height / 2
    x_max = center_x + width / 2
    y_max = center_y + height / 2

    # Stack the converted bounding boxes with class ids
    # original_boxes = np.column_stack((boxes[0],x_min, y_min, x_max, y_max))

    return boxes[0],x_min, y_min, x_max, y_max
class KITTIDataset2(Dataset):
    def __init__(self, images_dir, labels_dir, class_mapping=None, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_paths = sorted(glob.glob(os.path.join(images_dir, '*.png')))
        self.label_paths = sorted(glob.glob(os.path.join(labels_dir, '*.txt')))
        self.class_mapping = class_mapping
        self.transform = transform
        # self.device = device
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        width, height = GlobalVariables.ImageSize
        
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Image not found at {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_shape = img.shape[:2]
        img = cv2.resize(img, GlobalVariables.ImageSize)
        success = cv2.imwrite(img_path, img)
        # img = Image.open(img_path).convert('RGB')
        # originalWidth, originalheight = img.size
        # img = img.resize((width, height), Image.BILINEAR)
        labels = []
        transformEnable = True
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if(len(parts) > 5):
                
                    class_label = parts[0]
                    if self.class_mapping:
                        class_id = self.class_mapping[class_label]
                    else:
                        class_id = 0  # Use a default value or throw an error
                    bbox = list(map(float, parts[4:8]))
                    # Scale bounding box to the resized image
                    bbox = [
                        bbox[0] * (width / original_shape[1]),  # xmin
                        bbox[1] * (height / original_shape[0]),  # ymin
                        bbox[2] * (width / original_shape[1]),  # xmax
                        bbox[3] * (height / original_shape[0])   # ymax
                    ]
                    labels.append([class_id] + bbox)
                else:
                    transformEnable = False
                    bbox = [
                        parts[0],
                        parts[1],
                        parts[2],
                        parts[3],
                        parts[4],
                    ]
                    labels.append(bbox)
        
        labels = np.array(labels, dtype=np.float32)
        # YOLO format expects normalized bounding box values.
        if len(labels) > 0 and transformEnable:
            labels[:, :5] = convert_to_yolo_format(labels[:, :5], img.shape)  # Apply normalization
        
        # labels = torch.tensor(labels, dtype=torch.float32)        
        if self.transform:
            img, labels = self.transform(img, labels)
        #rewrite label files
        with open(label_path, 'w') as f:
            for array in labels:
                array_str = ' '.join(map(str, array))
                f.write(f"{array_str}\n")
        # end rewrite files
        # return img, labels
        return img, torch.tensor(labels)   
    
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
def collate_fn(batch):
    images, targets = list(zip(*batch))
    images = torch.stack([torch.tensor(img).permute(2, 0, 1) for img in images])
    return images, targets    
 
def loadKITTI_datasets():
        trainloaders = []
        valloaders = []    
        for clientNumber in range((GlobalVariables.NumberOfClients+1)):  
            # Paths to the dataset directories
            clientDirectory = str(clientNumber)
            if clientNumber == GlobalVariables.NumberOfClients:
                clientDirectory = 'CentralServer'
            train_images_dir = os.path.join(GlobalVariables.ClientsTrainFolderPath, clientDirectory,'images','train') 
            train_labels_dir = os.path.join(GlobalVariables.ClientsTrainFolderPath, clientDirectory,'labels','train')
            val_images_dir = os.path.join(GlobalVariables.ClientsTrainFolderPath, clientDirectory,'images','val') 
            val_labels_dir = os.path.join(GlobalVariables.ClientsTrainFolderPath, clientDirectory,'labels','val')
            #Transform all the images to same size

            train_dataset = KITTIDataset2(train_images_dir, train_labels_dir, class_mapping = class_mapping)
            val_dataset = KITTIDataset2(val_images_dir, val_labels_dir, class_mapping = class_mapping)
            train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
            val_loader = DataLoader(val_dataset, batch_size=8, shuffle = False, collate_fn=collate_fn)
            # train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
            # val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
            trainloaders.append(train_loader)
            valloaders.append(val_loader)
            print('Train loader and val loader'+ clientDirectory)
        return trainloaders, valloaders
# Display an image with bounding boxes from the train loader
def plot_image_with_boxes(image, boxes):
    # Ensure image is a NumPy array
    if isinstance(image, torch.Tensor):
        # Convert from Tensor (C, H, W) to (H, W, C) and convert to NumPy array
        image = image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    print(image.shape)
    # Ensure the image is contiguous in memory
    image = np.ascontiguousarray(image)
    
    # Ensure the image is in the correct layout (height, width, channels)
    if len(image.shape) == 3 and image.shape[2] == 3:
        pass  # Image is correctly formatted
    else:
        raise ValueError("Image must be a 3D array with shape (height, width, 3)")
    #Extract the coOrdinate
    
    #end extract
    plt.figure(figsize=(10, 10))
    for box in boxes:
        class_id, x1, y1, x2, y2 = convert_from_yolo_format(box,image.shape)
        # class_id, x1, y1, x2, y2 = box
        label = str(int(class_id)) #reverse_class_mapping[int(class_id)]
        # Draw the rectangle
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        # Put the text label
        cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    plt.imshow(image)
    plt.axis('off')
    plt.show()
# Displaying a batch of images with bounding boxes
def DisplayTrainLoaderImage(trainloader):    
    for imgs, targets in trainloader:
        imgs = imgs.permute(0, 2, 3, 1).numpy()  # Convert to (batch, height, width, channels)
        for img, target in zip(imgs, targets):
            plot_image_with_boxes(img, target)
        break
