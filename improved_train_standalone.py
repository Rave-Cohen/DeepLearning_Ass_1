#!/usr/bin/env python
# coding: utf-8

import os
import random
import pickle
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

OUTPUTS_DIR = Path("outputs")
MODEL_SAVE_DIR = OUTPUTS_DIR / "models"
RESULTS_PATH = OUTPUTS_DIR / "phase_b_results.pkl"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
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

DATA_ROOT = Path("data")
_ = datasets.CIFAR10(root=str(DATA_ROOT), train=True, download=True)

def get_cifar10_stats():
    temp_dataset = datasets.CIFAR10(root=str(DATA_ROOT), train=True, transform=transforms.ToTensor())
    temp_loader = DataLoader(temp_dataset, batch_size=1024, shuffle=False)
    
    mean = 0.0
    std = 0.0
    total_images = len(temp_dataset)
    
    for images, _ in temp_loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        
    return mean / total_images, std / total_images

CIFAR_MEAN, CIFAR_STD = get_cifar10_stats()
print(f"Calculated CIFAR-10 Mean: {CIFAR_MEAN.tolist()}")
print(f"Calculated CIFAR-10 Std:  {CIFAR_STD.tolist()}")

def get_dataloaders(config, validation_size=5000, batch_size=128):
    train_transforms_list = []
    test_transforms_list = [transforms.ToTensor()]
    
    if config['augmentation'] in ['crop', 'both']:
        train_transforms_list.append(transforms.RandomCrop(32, padding=4))
    if config['augmentation'] in ['rotate', 'both']:
        train_transforms_list.append(transforms.RandomRotation(15))
    
    train_transforms_list.append(transforms.ToTensor())
    
    if config['input_norm']:
        norm_layer = transforms.Normalize(CIFAR_MEAN.tolist(), CIFAR_STD.tolist())
        train_transforms_list.append(norm_layer)
        test_transforms_list.append(norm_layer)

    train_transform = transforms.Compose(train_transforms_list)
    test_transform = transforms.Compose(test_transforms_list)

    full_train = datasets.CIFAR10(root=str(DATA_ROOT), train=True, transform=train_transform)
    full_val = datasets.CIFAR10(root=str(DATA_ROOT), train=True, transform=test_transform)
    
    total_train_size = len(full_train)
    train_size = total_train_size - validation_size
    indices = torch.randperm(total_train_size).tolist()

    train_ds = Subset(full_train, indices[:train_size])
    val_ds = Subset(full_val, indices[train_size:])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

class ImprovedMLP(nn.Module):
    def __init__(self, input_size=3072, hidden_dims=[1024, 512, 128], num_classes=10, 
                 activation_name='leaky_relu', use_batchnorm=False, dropout_rate=0.0):
        super(ImprovedMLP, self).__init__()
        
        acts = {'relu': nn.ReLU(), 'leaky_relu': nn.LeakyReLU(0.01), 'gelu': nn.GELU()}
        activation = acts.get(activation_name.lower(), nn.ReLU())

        layers = []
        prev_dim = input_size
        
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(dim))
            layers.append(activation)
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(p=dropout_rate))
            prev_dim = dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1) 
        return self.model(x)

def init_weights(model):
    def init_func(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    model.apply(init_func)

def train(model, optimizer, loss_fn, train_loader, val_loader=None, epochs=50, device=device, exp_name="Experiment", l1_lambda=0.0):
    model.to(device)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "stop_reason": "completed"}
    
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
            
            if l1_lambda > 0:
                l1_penalty = sum(p.abs().sum() for p in model.parameters())
                loss += l1_lambda * l1_penalty

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()

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

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), MODEL_SAVE_DIR / f"{exp_name}_best.pth")

            if epoch == 7 and val_acc < 0.35:
                print(f"⚠️ [{exp_name}] Pruned: Poor performance at epoch 8")
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
                print(f"🛑 [{exp_name}] Early Stopped ({reason}).")
                history["stop_reason"] = f"early_stopped_{reason}"
                break

    return history

def plot_training_results(results_dict, exp_name=None):
    if exp_name is None: exp_name = list(results_dict.keys())[-1]
    data = results_dict[exp_name]
    history = data['history']
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f'Performance: {exp_name}', fontsize=16)

    ax1.plot(epochs, history['train_loss'], 'o-', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 's-', label='Val Loss')
    ax1.set_title('Cross Entropy Loss')
    ax1.legend(); ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.plot(epochs, history['train_acc'], 'o-', label='Train Acc')
    ax2.plot(epochs, history['val_acc'], 's-', label='Val Acc')
    ax2.set_title('Accuracy Score')
    ax2.legend(); ax2.grid(True, linestyle='--', alpha=0.6)

    val_acc_val = float(history['val_acc'][-1]) if history['val_acc'] else 0.0
    fig.text(0.5, 0.02, f"Final Val Acc: {val_acc_val:.4f} | Stop Reason: {history['stop_reason']}", ha='center')

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(OUTPUTS_DIR / f"{exp_name}_performance.png", dpi=300, bbox_inches='tight')
    plt.close()

print("\n🚀 Starting Phase B: Regularization & Augmentation Ablation Study...")

if RESULTS_PATH.exists():
    print(f"📂 Found existing Phase B results. Resuming from {RESULTS_PATH}...")
    with open(RESULTS_PATH, "rb") as f:
        results_log = pickle.load(f)
else:
    results_log = {}

experiments = [
    {"name": "B_00_Baseline", "input_norm": False, "augmentation": "none", "batchnorm": False, "l2_wd": 0.0, "l1_lambda": 0.0, "dropout": 0.0},
    {"name": "B_01_InputNorm", "input_norm": True, "augmentation": "none", "batchnorm": False, "l2_wd": 0.0, "l1_lambda": 0.0, "dropout": 0.0},
    {"name": "B_02_Aug_Crop", "input_norm": True, "augmentation": "crop", "batchnorm": False, "l2_wd": 0.0, "l1_lambda": 0.0, "dropout": 0.0},
    {"name": "B_03_Aug_Rotate", "input_norm": True, "augmentation": "rotate", "batchnorm": False, "l2_wd": 0.0, "l1_lambda": 0.0, "dropout": 0.0},
    {"name": "B_04_BatchNorm", "input_norm": True, "augmentation": "none", "batchnorm": True, "l2_wd": 0.0, "l1_lambda": 0.0, "dropout": 0.0},
    {"name": "B_05_L2_Reg", "input_norm": True, "augmentation": "none", "batchnorm": False, "l2_wd": 1e-4, "l1_lambda": 0.0, "dropout": 0.0},
    {"name": "B_06_L1_Reg", "input_norm": True, "augmentation": "none", "batchnorm": False, "l2_wd": 0.0, "l1_lambda": 1e-5, "dropout": 0.0},

    {"name": "B_07_Dropout_0.1", "input_norm": True, "augmentation": "none", "batchnorm": False, "l2_wd": 0.0, "l1_lambda": 0.0, "dropout": 0.1},
    {"name": "B_08_Dropout_0.2", "input_norm": True, "augmentation": "none", "batchnorm": False, "l2_wd": 0.0, "l1_lambda": 0.0, "dropout": 0.2},
    {"name": "B_09_Dropout_0.3", "input_norm": True, "augmentation": "none", "batchnorm": False, "l2_wd": 0.0, "l1_lambda": 0.0, "dropout": 0.3},
    {"name": "B_10_Dropout_0.4", "input_norm": True, "augmentation": "none", "batchnorm": False, "l2_wd": 0.0, "l1_lambda": 0.0, "dropout": 0.4},
    {"name": "B_11_Dropout_0.5", "input_norm": True, "augmentation": "none", "batchnorm": False, "l2_wd": 0.0, "l1_lambda": 0.0, "dropout": 0.5},

    {"name": "B_12_Combo_Aug_Both", "input_norm": True, "augmentation": "both", "batchnorm": False, "l2_wd": 0.0, "l1_lambda": 0.0, "dropout": 0.0},
    {"name": "B_13_Combo_BN_Drop", "input_norm": True, "augmentation": "none", "batchnorm": True, "l2_wd": 0.0, "l1_lambda": 0.0, "dropout": 0.3},
    {"name": "B_14_Combo_BN_L2", "input_norm": True, "augmentation": "none", "batchnorm": True, "l2_wd": 1e-4, "l1_lambda": 0.0, "dropout": 0.0},
    {"name": "B_15_Combo_Drop_L2", "input_norm": True, "augmentation": "none", "batchnorm": False, "l2_wd": 1e-4, "l1_lambda": 0.0, "dropout": 0.3},
    {"name": "B_16_Combo_AugCrop_Drop", "input_norm": True, "augmentation": "crop", "batchnorm": False, "l2_wd": 0.0, "l1_lambda": 0.0, "dropout": 0.3},
    {"name": "B_17_Combo_BN_AugCrop", "input_norm": True, "augmentation": "crop", "batchnorm": True, "l2_wd": 0.0, "l1_lambda": 0.0, "dropout": 0.0},

    {"name": "B_18_Combo_BN_Drop_L2", "input_norm": True, "augmentation": "none", "batchnorm": True, "l2_wd": 1e-4, "l1_lambda": 0.0, "dropout": 0.3},
    {"name": "B_19_Combo_BN_AugBoth_Drop", "input_norm": True, "augmentation": "both", "batchnorm": True, "l2_wd": 0.0, "l1_lambda": 0.0, "dropout": 0.3},
    {"name": "B_20_Ultimate_Combo", "input_norm": True, "augmentation": "both", "batchnorm": True, "l2_wd": 1e-4, "l1_lambda": 0.0, "dropout": 0.3},
]

loss_fn = nn.CrossEntropyLoss()

for i, config in enumerate(experiments):
    exp_name = config["name"]
    
    if exp_name in results_log:
        print(f"⏩ Skipping {exp_name} (already completed).")
        continue

    print(f"\n" + "="*60)
    print(f"🔹 Running {exp_name} ({i+1}/{len(experiments)})")
    print(f"🔹 Config: {config}")
    print("="*60)

    train_loader, val_loader = get_dataloaders(config)

    model = ImprovedMLP(
        hidden_dims=[1024, 512, 128], 
        activation_name='leaky_relu',
        use_batchnorm=config['batchnorm'],
        dropout_rate=config['dropout']
    ).to(device)
    init_weights(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=config['l2_wd'])

    history = train(
        model=model, 
        optimizer=optimizer, 
        loss_fn=loss_fn, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        epochs=50, 
        device=device, 
        exp_name=exp_name,
        l1_lambda=config['l1_lambda']
    )

    results_log[exp_name] = {"config": config, "history": history}
    plot_training_results(results_log, exp_name=exp_name)
    
    with open(RESULTS_PATH, "wb") as f:
        pickle.dump(results_log, f)

print(f"\n🎉 Phase B Regularization Study completed successfully!")