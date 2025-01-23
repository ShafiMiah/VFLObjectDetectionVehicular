#!/bin/env bash

#SBATCH -A NAISS2024-5-153    # find your project with the "projinfo" command
#SBATCH -p alvis               # what partition to use (usually not necessary)
#SBATCH -t 0-06:00:00          # how long time it will take to run
#SBATCH --gpus-per-node=A100:1   # choosing no. GPUs and their type
#SBATCH -J modules             # the jobname (not necessary)

# Load PyTorch using the module tree
module purge
module load Python/3.12.3-GCCcore-13.3.0
python -m venv myenv
source myenv/bin/activate
pip install flower==2.0.1
pip install ultralytics
pip install torch
pip install torchvision
pip install opencv-python-headless

flower --port=5555 &
#run VFL training
python VFLFinal.py