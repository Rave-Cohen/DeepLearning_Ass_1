#!/bin/bash

#SBATCH --job-name=dl_ass1
#SBATCH --output=training_log_%j.out
#SBATCH --time=24:00:00              # Exactly as your sinteractive command
#SBATCH --partition=course           # From --part course
#SBATCH --qos=course                 # From --qos course (This fixes the error!)
#SBATCH --gres=gpu:1                 # From --gpu 1
#SBATCH --cpus-per-task=4            # Safe number of CPU cores for data loading
#SBATCH --mem=24G                    # Safe memory limit

# 1. Activate your specific conda environment (using the base path we verified)
source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate neuro_dl

# 2. Run your Python script
python Assignment_1.py