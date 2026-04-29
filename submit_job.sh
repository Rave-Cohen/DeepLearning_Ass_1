#!/bin/bash
#SBATCH --job-name=DL_Optuna_Rave
#SBATCH --partition=course
#SBATCH --qos=course
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=24G
#SBATCH --output=outputs/run_%j.log
#SBATCH --error=outputs/run_%j.err

# Activate environment
source activate neuro_dl

# Ensure output directory exists
mkdir -p outputs

# Run the experiment
python Assignment_1.py
