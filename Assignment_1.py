#!/usr/bin/env python
# coding: utf-8

# # Deep Learning Assignment 1 (Cluster Optimized)
# 
# - Data is loaded from a local `data/` folder.
# - Figures are saved into an `outputs/` folder with informative filenames.
# - Models are saved into an `outputs/models/` folder.
# - Incremental saving ensures no data is lost upon cluster preemption.

import os
import itertools
from pathlib import Path
import random
import copy
import gc
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# CRITICAL FOR CLUSTER: Forces matplotlib to work without a display (headless)
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples

# --- 1. Global Setup, Device, and Paths ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

notebook_dir = Path.cwd()
DATA_ROOT = notebook_dir / "data"
OUTPUTS_DIR = notebook_dir / "outputs"
MODEL_SAVE_DIR = OUTPUTS_DIR / "models"
RESULTS_PATH = OUTPUTS_DIR / "phase_a_grid_results.pkl" # Moved to globals

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
print(f"Weights will be saved to: {MODEL_SAVE_DIR}")

# --- 2. Reproducibility Setup ---
def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

seed_everything(42)
g = torch.Generator()
g.manual_seed(42)

# ==============================================================================
# DATA LOADING & VISUALIZATION
# ==============================================================================

def get_train_val_test_data(validation_size=5000,
                            train_transforms=transforms.ToTensor(),
                            test_transforms=transforms.ToTensor()):
    # Changed download=True as a safety net for cluster nodes
    full_train_dataset = datasets.CIFAR10(
        root=str(DATA_ROOT), train=True, download=True, transform=train_transforms
    )
    full_val_dataset = datasets.CIFAR10(
        root=str(DATA_ROOT), train=True, download=True, transform=test_transforms
    )
    test_dataset = datasets.CIFAR10(
        root=str(DATA_ROOT), train=False, download=True, transform=test_transforms
    )

    total_train_size = len(full_train_dataset)
    train_size = total_train_size - validation_size
    indices = torch.randperm(total_train_size).tolist()

    train_dataset = Subset(full_train_dataset, indices[:train_size])
    val_dataset = Subset(full_val_dataset, indices[train_size:])

    return train_dataset, val_dataset, test_dataset

train_dataset, val_dataset, test_dataset = get_train_val_test_data()
print("Loaded datasets:")
print("  Train images:", len(train_dataset))
print("  Validation images:", len(val_dataset))
print("  Test images:", len(test_dataset))

# ==============================================================================
# MODEL & TRAINING LOGIC
# ==============================================================================

class VanillaMLP(nn.Module):
    def __init__(self, input_size=3072, hidden_dims=[512, 256], num_classes=10, activation_name='relu'):
        super(VanillaMLP, self).__init__()
        acts = {
            'relu': nn.ReLU(), 'leaky_relu': nn.LeakyReLU(0.01), 
            'gelu': nn.GELU(), 'sigmoid': nn.Sigmoid(), 'tanh': nn.Tanh()
        }
        activation = acts.get(activation_name.lower(), nn.ReLU())

        layers = []
        prev_dim = input_size
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(activation)
            prev_dim = dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1) 
        return self.model(x)

def init_weights(model, init_type='kaiming'):
    def init_func(m):
        if isinstance(m, nn.Linear):
            if init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    model.apply(init_func)

def train(model, optimizer, loss_fn, train_loader, val_loader=None, epochs=10, device=device, scheduler=None, exp_name="Experiment"):
    model.to(device)
    history = {
        "train_loss": [], "val_loss": [], 
        "train_acc": [], "val_acc": [],
        "stop_reason": "completed"
    }
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience = 5  
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()

        if scheduler is not None:
            scheduler.step()

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = correct_train / total_train
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        if val_loader:
            model.eval()
            val_running_loss, correct_val, total_val = 0.0, 0, 0
            with torch.no_grad():
                for v_in, v_lab in val_loader:
                    v_in, v_lab = v_in.to(device), v_lab.to(device)
                    v_out = model(v_in)
                    val_running_loss += loss_fn(v_out, v_lab).item() * v_in.size(0)
                    _, v_pred = v_out.max(1)
                    total_val += v_lab.size(0)
                    correct_val += v_pred.eq(v_lab).sum().item()

            val_loss = val_running_loss / len(val_loader.dataset)
            val_acc = correct_val / total_val
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            
            print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

            # --- WEIGHT SAVING LOGIC ---
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                weights_path = MODEL_SAVE_DIR / f"{exp_name}_best.pth"
                torch.save(model.state_dict(), weights_path)

            # --- EARLY STOPPING & PRUNING LOGIC ---
            if epoch == 7 and val_acc < 0.35:
                print(f"⚠️ [{exp_name}] Pruned: Poor performance at epoch 8 (Val Acc: {val_acc:.4f})")
                history["stop_reason"] = "pruned_bad_start"
                break

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0 
            else:
                patience_counter += 1

            massive_overfit = (val_loss - train_loss) > 0.8 

            if patience_counter >= patience or massive_overfit:
                reason = "massive_divergence" if massive_overfit else "patience_exceeded"
                print(f"🛑 [{exp_name}] Early Stopped at Epoch {epoch+1} to prevent overfitting ({reason}).")
                history["stop_reason"] = f"early_stopped_overfit_{reason}"
                break

    return history

# ==============================================================================
# PLOTTING FUNCTIONS
# ==============================================================================

def plot_training_results(results_dict, exp_name=None):
    if exp_name is None:
        exp_name = list(results_dict.keys())[-1]

    data = results_dict[exp_name]
    history = data['history']
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f'Performance Analysis: {exp_name}', fontsize=18, fontweight='bold')

    ax1.plot(epochs, history['train_loss'], 'o-', label='Train Loss', color='#1f77b4', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 's-', label='Val Loss', color='#ff7f0e', linewidth=2)
    ax1.set_title('Cross Entropy Loss', fontsize=14)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.plot(epochs, history['train_acc'], 'o-', label='Train Acc', color='#2ca02c', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 's-', label='Val Acc', color='#d62728', linewidth=2)
    ax2.set_title('Accuracy Score', fontsize=14)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')

    val_acc_val = float(history['val_acc'][-1]) if history['val_acc'] else 0.0
    final_txt = f"Final Val Acc: {val_acc_val:.4f} | Stop Reason: {history['stop_reason']}"
    
    fig.text(0.5, 0.02, final_txt, ha='center', fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='#CCCCCC'))

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    save_path = Path(OUTPUTS_DIR) / f"{exp_name}_performance.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def save_results_as_image(results_dict, outputs_dir="outputs", filename="phase_a_grid_summary_table.png"):
    if not results_dict:
        return

    summary_list = []
    for name, data in results_dict.items():
        h = data['history']
        conf = data['config']
        lr_display = f"{conf['lr']} -> 0.00025" if conf.get('scheduler') == 'step' else f"{conf['lr']:.5f}"

        summary_list.append({
            "Experiment": name,
            "Layers": str(conf['hidden_dims']),
            "Act.": conf['activation'],
            "Optim": conf['optimizer'],
            "LR": lr_display,
            "Best Val Acc": f"{max(h['val_acc']):.4f}" if h['val_acc'] else "0.0000",
            "Final Val Acc": f"{h['val_acc'][-1]:.4f}" if h['val_acc'] else "0.0000",
            "Status": h['stop_reason']
        })

    df = pd.DataFrame(summary_list)
    fig, ax = plt.subplots(figsize=(18, max(4, len(df) * 0.3))) 
    ax.axis('off') 
    
    table = ax.table(
        cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center',
        colColours=["#40466e"] * len(df.columns)
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10) 
    table.scale(1, 2.0) 

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.get_text().set_color('white')
            cell.get_text().set_weight('bold')
        cell.set_edgecolor('#CCCCCC')

    final_save_path = Path(outputs_dir) / filename
    plt.savefig(final_save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"✅ Summary table saved to: {final_save_path}")

# ==============================================================================
# PHASE A: FOCUSED GRID SEARCH
# ==============================================================================

print("🚀 Starting Phase A: Focused Grid Search...")

# Load previous results if resuming a crashed job
if RESULTS_PATH.exists():
    print(f"📂 Found existing results file. Resuming from {RESULTS_PATH}...")
    with open(RESULTS_PATH, "rb") as f:
        results_log = pickle.load(f)
else:
    results_log = {}

# 1. SLIMMED DOWN GRID (180 Experiments)
grid_params = {
    "hidden_dims": [
        [1024],                           # 1 layer
        [512, 256],                       # 2 layers 
        [1024, 512, 128],                 # 3 layers 
        [1024, 512, 256, 128],            # 4 layers 
    ], 
    "activation": ["relu", "leaky_relu", "sigmoid"], 
    "lr": [1e-2, 1e-3, 5e-4, 1e-4, 1e-5],                          
    "scheduler": ["none", "cosine", "step"]                          
}

keys, values = zip(*grid_params.items())
experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

batch_size = 128
epochs = 50

print(f"🔥 TOTAL QUEUED EXPERIMENTS: {len(experiments)}")

loss_fn = nn.CrossEntropyLoss()
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

for i, config in enumerate(experiments):
    act = config['activation']
    lr = config['lr']
    sch = config['scheduler']
    dims_str = f"L{len(config['hidden_dims'])}" 
    width_str = f"W{config['hidden_dims'][0]}"
    
    exp_name = f"Exp_{i:03d}_{dims_str}_{width_str}_{act}_lr{lr}_{sch}"
    
    # RESUME LOGIC: Check if this experiment is already done!
    if exp_name in results_log:
        print(f"⏩ Skipping {exp_name} (already completed in previous run).")
        continue

    print(f"\n" + "="*70)
    print(f"🔹 Running {exp_name} ({i+1}/{len(experiments)})")
    print(f"🔹 Config: {config}")
    print("="*70)

    model = VanillaMLP(hidden_dims=config["hidden_dims"], activation_name=config["activation"]).to(device)
    
    if config["activation"] == "sigmoid":
        init_weights(model, init_type='xavier')
    else:
        init_weights(model, init_type='kaiming')

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    config["optimizer"] = "adam"

    if config["scheduler"] == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    elif config["scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        scheduler = None

    history = train(
        model=model, optimizer=optimizer, loss_fn=loss_fn, 
        train_loader=train_loader, val_loader=val_loader, 
        epochs=epochs, device=device, scheduler=scheduler, exp_name=exp_name
    )

    results_log[exp_name] = {
        "config": config,
        "history": history
    }

    # Plot & save individual experiment
    plot_training_results(results_log, exp_name=exp_name)

    # INCREMENTAL SAVE: Save after EVERY experiment to prevent data loss!
    with open(RESULTS_PATH, "wb") as f:
        pickle.dump(results_log, f)
    save_results_as_image(results_log, filename="phase_a_grid_summary_table.png")

print(f"\n🎉 All Grid Search runs completed successfully!")