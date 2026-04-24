
# TODO:Please add your import statements below:
#
# For example:
#   - Data manipulation libraries
#   - Machine learning frameworks
#   - Visualization tools
#
# (Your imports go here)
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch
from torch.utils.data import Subset
import matplotlib.pyplot as plt
import numpy as np

# Using GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def get_train_val_test_data(validation_size=5000,
                            train_transforms=transforms.ToTensor(),
                            test_transforms=transforms.ToTensor()):
    """
    Load the CIFAR10 dataset and split it into training, validation, and test sets.

    Parameters:
        validation_size (int): Number of samples to use for the validation set.
        train_transforms: Transformations to be applied to the training data.
        test_transforms: Transformations to be applied to the validation and test data.
    """

    # -------------------------- FILL HERE --------------------------
    # Overall hints:
    # 1. Load the full CIFAR10 dataset both for train and test.
    # 2. Create separate CIFAR10 dataset instances for training (with train_transforms),
    # 3. Same as #2 for validation, and testing (with test_transforms).
    # 4. Use a method like Subset to create the split datasets.
    # -------------------------------------------------------------

    # Your implementation goes here
        # load the training set twice:
    # one copy with train transforms, one copy with test/validation transforms
    full_train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=False,
        transform=train_transforms
    )

    full_val_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=False,
        transform=test_transforms
    )

    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=False,
        transform=test_transforms
    )

    total_train_size = len(full_train_dataset)
    train_size = total_train_size - validation_size

    indices = torch.randperm(total_train_size).tolist()

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_val_dataset, val_indices)

    return train_dataset, val_dataset, test_dataset


def show_images(dataset, num_images=5, title=""): #shows 5 images from the dataset
    label_names = dataset.dataset.classes if hasattr(dataset, "dataset") else dataset.classes

    plt.figure(figsize=(10, 3))
    plt.suptitle(title)

    for i in range(num_images):
        image, label = dataset[i]
        image = image.permute(1, 2, 0).numpy()

        plt.subplot(1, num_images, i + 1)
        plt.imshow(image)
        plt.title(label_names[label])
        plt.axis("off")

    plt.tight_layout()
    plt.show()

train_dataset, val_dataset, test_dataset = get_train_val_test_data()
show_images(train_dataset, 5, title="Train Samples")
show_images(val_dataset, 5, title="Validation Samples")
show_images(test_dataset, 5, title="Test Samples")

# ===================== DEBUG: IMAGE RGB CHECK =====================
def debug_image_rgb(dataset):
    image, label = dataset[0]  # take first image

    print("Image shape (C, H, W):", image.shape)
    print("Label:", label)

    print("\nChannel-wise stats:")
    for i, color in enumerate(["Red", "Green", "Blue"]):
        print(f"{color} channel -> min: {image[i].min():.4f}, max: {image[i].max():.4f}, mean: {image[i].mean():.4f}")

    # show raw channels
    fig, axs = plt.subplots(1, 3, figsize=(10, 3))
    for i, color in enumerate(["Red", "Green", "Blue"]):
        axs[i].imshow(image[i], cmap="gray")
        axs[i].set_title(f"{color} channel")
        axs[i].axis("off")

    plt.suptitle("RGB Channel Breakdown (Raw)")
    plt.show()


def debug_rgb_overlay(dataset):
    image, _ = dataset[0]

    # convert CHW → HWC for display
    img = image.permute(1, 2, 0).numpy()

    plt.figure(figsize=(4,4))
    plt.imshow(img)
    plt.title("RGB Image (combined)")
    plt.axis("off")
    plt.show()

    print("Final image array shape (H, W, C):", img.shape)
    print("Value range:", img.min(), "to", img.max())


# run debug
train_dataset, val_dataset, test_dataset = get_train_val_test_data()
debug_image_rgb(train_dataset)
debug_rgb_overlay(train_dataset)
# ================================================================

def plot_class_distribution(train_dataset, val_dataset, test_dataset):
    label_names = test_dataset.classes
    num_classes = len(label_names)

    train_counts = np.zeros(num_classes, dtype=int)
    val_counts = np.zeros(num_classes, dtype=int)
    test_counts = np.zeros(num_classes, dtype=int)

    # count train labels
    for idx in train_dataset.indices:
        label = train_dataset.dataset.targets[idx]
        train_counts[label] += 1

    # count validation labels
    for idx in val_dataset.indices:
        label = val_dataset.dataset.targets[idx]
        val_counts[label] += 1

    # count test labels
    for label in test_dataset.targets:
        test_counts[label] += 1

    x = np.arange(num_classes)
    bar_width = 0.25

    plt.figure(figsize=(12, 6))

    bars_train = plt.bar(x - bar_width, train_counts, width=bar_width, label="Train")
    bars_val = plt.bar(x, val_counts, width=bar_width, label="Validation")
    bars_test = plt.bar(x + bar_width, test_counts, width=bar_width, label="Test")

    # add numbers on top of bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2,
                height + 20,
                str(int(height)),
                ha='center',
                va='bottom',
                fontsize=8
            )

    add_labels(bars_train)
    add_labels(bars_val)
    add_labels(bars_test)

    plt.xticks(x, label_names, rotation=45)
    plt.ylabel("Number of images")
    plt.xlabel("Class")
    plt.title("Class distribution across train, validation, and test sets")

    plt.grid(axis='y', linestyle='--', alpha=0.7)  
    plt.legend()
    plt.tight_layout()
    plt.show()

train_dataset, val_dataset, test_dataset = get_train_val_test_data()
plot_class_distribution(train_dataset, val_dataset, test_dataset)

# ===================== RGB CLASS ANALYSIS =====================

def compute_class_rgb_stats(dataset):
    """
    Computes per-class RGB mean and variance.

    Args:
        dataset: torch.utils.data.Subset (train/val)

    Returns:
        class_means: (num_classes, 3)
        class_vars: (num_classes, 3)
    """
    num_classes = len(dataset.dataset.classes)

    pixel_sum = torch.zeros(num_classes, 3)
    pixel_sq_sum = torch.zeros(num_classes, 3)
    pixel_count = torch.zeros(num_classes)

    for idx in dataset.indices:
        img, label = dataset.dataset[idx]  # (C,H,W)

        img = img.view(3, -1)  # flatten → (3, pixels)

        pixel_sum[label] += img.sum(dim=1)
        pixel_sq_sum[label] += (img ** 2).sum(dim=1)
        pixel_count[label] += img.shape[1]

    class_means = pixel_sum / pixel_count.unsqueeze(1)
    class_vars = (pixel_sq_sum / pixel_count.unsqueeze(1)) - class_means**2

    return class_means.numpy(), class_vars.numpy()


def plot_rgb_stats(class_means, class_vars, class_names):
    x = np.arange(len(class_names))
    mean_R = class_means[:, 0].mean()
    mean_G = class_means[:, 1].mean()
    mean_B = class_means[:, 2].mean()

    var_R = class_vars[:, 0].mean()
    var_G = class_vars[:, 1].mean()
    var_B = class_vars[:, 2].mean()

    plt.figure(figsize=(12, 5))

    # ---- Mean ----
    plt.subplot(1, 2, 1)
    plt.scatter(x, class_means[:, 0], color='r', label="Red")
    plt.scatter(x, class_means[:, 1], color='g', label="Green")
    plt.scatter(x, class_means[:, 2], color='b', label="Blue")
    plt.title("Mean RGB Intensity per Class")
    plt.xticks(x, class_names, rotation=45)
    plt.legend()
    plt.grid(True)

    plt.axhline(mean_R, color='r', linestyle='--', alpha=0.4)
    plt.axhline(mean_G, color='g', linestyle='--', alpha=0.4)
    plt.axhline(mean_B, color='b', linestyle='--', alpha=0.4)

    # ---- Variance ----
    plt.subplot(1, 2, 2)
    plt.scatter(x, class_vars[:, 0], color='r', label="Red")
    plt.scatter(x, class_vars[:, 1], color='g', label="Green")
    plt.scatter(x, class_vars[:, 2], color='b', label="Blue")
    plt.title("RGB Variance (Color Diversity) per Class")
    plt.xticks(x, class_names, rotation=45)
    plt.legend()
    plt.grid(True)

    plt.axhline(var_R, color='r', linestyle='--', alpha=0.4)
    plt.axhline(var_G, color='g', linestyle='--', alpha=0.4)
    plt.axhline(var_B, color='b', linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.show()


# ---- RUN ANALYSIS ----
class_means, class_vars = compute_class_rgb_stats(train_dataset)
plot_rgb_stats(class_means, class_vars, train_dataset.dataset.classes)

# Optional: print numeric values (useful for report)
for i, cls in enumerate(train_dataset.dataset.classes):
    print(f"{cls}:")
    print(f"  Mean RGB     = {class_means[i]}")
    print(f"  Variance RGB = {class_vars[i]}")
    print()
    
# =============================================================

    
def train(model, optimizer, loss_fn, train_loader, val_loader=None, epochs=10, device="cuda"):
    """
    Train a neural network model and optionally evaluate its performance on validation data.

    Parameters:
        model: The neural network model to be trained.
        optimizer: The optimizer used to update the model parameters.
        loss_fn: The loss function used to compute the training loss.
        train_loader: DataLoader providing the training data in batches.
        val_loader (optional): DataLoader providing the validation data in batches.
        epochs (int): Number of training epochs.
        device (str): Device for computation (e.g., "cuda" or "cpu").
    """

    # -------------------------- FILL HERE --------------------------
    # Overall hints:
    # - Loop through epochs and batches.
    # - For each batch, perform a forward pass, compute loss, and update the model.
    # - If a validation loader is provided, evaluate the model on the validation set.
    # -------------------------------------------------------------
    
    # Your implementation goes here

def test(model, test_loader, loss_fn):
    """
    Evaluate the trained model on the test dataset.

    Parameters:
        model: The trained model to evaluate.
        test_loader: DataLoader providing the test data in batches.
        loss_fn: The loss function used to compute the test loss.

    """

    # -------------------------- FILL HERE --------------------------
    # Overall hints:
    # - Set the model to evaluation mode.
    # - Disable gradient computation.
    # - Loop over the test_loader to compute predictions and loss.
    # - Aggregate the loss and accuracy metrics.
    # -------------------------------------------------------------
    
    # Your implementation goes here


# class VanillaMLP(nn.Module):
#     def __init__(self, input_size=3*32*32, num_class=10):
#         """
#         Initialize the MLP network.

#         Parameters:
#             input_size (int): The total number of input features (e.g., for a 32x32 color image).
#             num_class (int): The number of output classes for classification.
#         """
#         super(VanillaMLP, self).__init__()
        
#         # -------------------------- FILL HERE --------------------------
#         # Overall hints:
#         # - Define a series of fully connected layers.
#         # - Insert activation functions (e.g., GELU or ReLU) between layers.
#         # - Ensure the final layer outputs 'num_class' logits.
#         # -------------------------------------------------------------
        
#     def forward(self, x):
#         # -------------------------- FILL HERE --------------------------
#         # Overall hints:
#         # - Flatten the input tensor.
#         # - Pass the input sequentially through the defined layers.
#         # - Apply activation functions appropriately.
#         # - Return the final output logits.
#         # -------------------------------------------------------------
        



# class ImprovedMLP(nn.Module):
#     def __init__(self, input_size=3*32*32, num_class=10, dropout_p=0.15, use_batchnorm=True):
#         """
#         Initialize the improved MLP network.

#         Parameters:
#             input_size (int): Total number of input features (e.g., for a 32x32 color image).
#             num_class (int): Number of output classes for classification.
#             dropout_p (float): Dropout probability; apply dropout only if this value is > 0.
#             use_batchnorm (bool): Whether to use Batch Normalization between layers.
#         """
#         super(ImprovedMLP, self).__init__()
        
#         # -------------------------- FILL HERE --------------------------
#         # Overall hints:
#         # - Define a sequence of fully connected layers.
#         # - Add activation functions (e.g., GELU or ReLU) between the layers.
#         # - If use_batchnorm is True, include Batch Normalization layers after the corresponding linear layers.
#         # - Only add or activate dropout layers if dropout_p is greater than 0.
#         # - Ensure the final layer produces 'num_class' outputs.
#         # -------------------------------------------------------------
        
#     def forward(self, x):
#         # -------------------------- FILL HERE --------------------------
#         # Overall hints:
#         # - Flatten the input tensor.
#         # - Pass the data sequentially through the defined layers.
#         # - Apply activation functions, and conditionally apply dropout (only if dropout_p > 0).
#         # - Return the final output logits.
#         # -------------------------------------------------------------