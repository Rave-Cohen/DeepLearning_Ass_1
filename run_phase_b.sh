#!/bin/bash
#SBATCH --job-name=phase_b_grid
#SBATCH --output=outputs/phase_b_grid_%j.out
#SBATCH --error=outputs/phase_b_grid_%j.err
#SBATCH --time=12:00:00           # 12 hours max (it will likely finish much faster due to early stopping)
#SBATCH --gres=gpu:1              # Request 1 GPU
#SBATCH --cpus-per-task=4         # 4 CPU cores for data loading
#SBATCH --mem=24G                 # 16 GB of RAM

# Load your environment if needed (uncomment and change 'base' to your env name if you use conda)
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate neuro_dl

echo "🚀 Starting Phase B Improved Grid Search on $(hostname)"
python improved_train_standalone.py
echo "🎉 Job finished!"