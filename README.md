# Vertical Federated Object Detection for Vehicular Networks
Due to the large size of the dataset the images and annotations have not been uploaded to the git repository. Interested readers are requested to download the dataset from corresponding site and pre-process according to the instruction. The SMVFL has been implemented using python. KITTI and ZOD dataset has been used to train the model. 
# Packages and requirement.
1. Python version 3.9 or higher
2. Run following commands to install the corresponding Packages.

   ```sh  
   pip install flower
   pip install ultralytics
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install opencv-python-headless

Note: from the torch site get the corresponding CUDA supported torch version according to your configuration.

# KITTI Dataset preprocessing
Url: https://www.kaggle.com/datasets/klemenko/kitti-dataset

# Selection of Dataset and assign Images to different clients.
# ZOD Dataset preprocessing
# SMVFL Implementation
