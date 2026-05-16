#!/bin/bash
#SBATCH --job-name=phase_b_grid
#SBATCH --output=outputs/phase_b_grid_%j.out
#SBATCH --error=outputs/phase_b_grid_%j.err
#SBATCH --partition=course
#SBATCH --qos=course
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=raveco@post.bgu.ac.il

# 1. This line finds where your conda is installed and tells this script how to use it
source $(conda info --base)/etc/profile.d/conda.sh

# 2. Now it will actually recognize this command
conda activate neuro_dl

# 3. Run the script
echo "🚀 Starting Phase B on $(hostname)"
python -u improved_train_standalone.py
echo "🎉 Job finished!"