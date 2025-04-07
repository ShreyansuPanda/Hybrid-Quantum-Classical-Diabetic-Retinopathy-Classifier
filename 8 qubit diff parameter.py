import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import time
import os

# ================================
# CONFIGURATION
# ================================
config = {
    "data_dir": "D:/study/quantum/Diabetic Retinopathy 224x224 (2019 Data)/colored_images/train",
    "img_size": (224, 224),
    "batch_size": 64,
    "test_size": 0.2,
    "random_state": 50,
    "n_qubits": 8,
    "n_layers": 4,
    # Optimizer hyperparameters:
    "lr_classical": 0.0005,      # Increased classical learning rate
    "lr_quantum": 0.0002,        # Quantum parameters get a slightly lower learning rate
    "weight_decay": 1e-4,
    "epochs": 50,
    "patience": 50
}

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Determine device: GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ================================
# Step 1: Dataset Preparation
# ================================
def create_datasets(data_dir):
    train_transform = transforms.Compose([
        transforms.Resize(config["img_size"]),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize(config["img_size"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    full_dataset = datasets.ImageFolder(root=data_dir)
    targets = [s[1] for s in full_dataset.samples]
    train_idx, val_idx = train_test_split(
        np.arange(len(targets)),
        test_size=config["test_size"],
        stratify=targets,
        random_state=config["random_state"]
    )
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    return train_dataset, val_dataset

# ================================
# Step 2: Class Balancing
# ================================
def create_weighted_sampler(dataset):
    class_counts = Counter([label for _, label in dataset])
    total_samples = len(dataset)
    class_weights = {cls: total_samples/(count * 1.5) for cls, count in class_counts.items()}
    sample_weights = [class_weights[label] for _, label in dataset]
    return WeightedRandomSampler(sample_weights, len(dataset), replacement=True)

# ================================
# Step 3: Hybrid Quantum-Classical Model
# ================================
n_qubits = config["n_qubits"]
n_layers = config["n_layers"]

dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# For TorchLayer, we need to specify the weight shapes.
weight_shapes = {"weights": (n_layers, n_qubits)}
quantum_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

class HybridDRModel(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights='IMAGENET1K_V1')
        self.classical = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, n_qubits),
            nn.BatchNorm1d(n_qubits)
        )
        self.quantum = quantum_layer
        self.fc = nn.Sequential(
            nn.Linear(n_qubits, 32),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(32, 5)
        )

    def forward(self, x):
        x = self.classical(x)
        x = self.quantum(x)
        x = self.fc(x)
        return x

# ================================
# Step 4: Training Configuration (Customizable Hyperparameters)
# ================================
def configure_training(train_dataset, model, config):
    class_counts = Counter([label for _, label in train_dataset])
    num_classes = len(class_counts)
    total_samples = len(train_dataset)
    class_weights = [total_samples/(class_counts[i]*1.5) for i in range(num_classes)]
    
    class FocalLoss(nn.Module):
        def __init__(self, alpha=0.25, gamma=2):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma
        def forward(self, inputs, targets):
            ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
            pt = torch.exp(-ce_loss)
            return (self.alpha * (1-pt)**self.gamma * ce_loss).mean()
    
    criterion = FocalLoss()
    optimizer = optim.AdamW(model.parameters(), 
                            lr=config["lr_classical"],
                            weight_decay=config["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    return {
        'criterion': criterion,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'batch_size': config["batch_size"]
    }

# ================================
# Step 5: Training Loop
# ================================
def train_model(train_loader, val_loader, model, criterion, optimizer, scheduler):
    best_val_acc = 0.0
    early_stop_counter = 0
    patience = config["patience"]
    epochs = config["epochs"]
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        val_loss, val_acc = evaluate_model(val_loader, model, criterion, print_cm=False)
        scheduler.step()
        train_loss = running_loss / len(train_loader)
        train_acc = correct_train / total_train
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_hybrid_model.pth")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        print("-----------------------------------")
        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig("hybrid_training_history.png")
    plt.show()
    return best_val_acc

# ================================
# Step 6: Evaluation Function
# ================================
def evaluate_model(loader, model, criterion, print_cm=True):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=train_dataset.dataset.classes, zero_division=0))
    if print_cm:
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=train_dataset.dataset.classes, yticklabels=train_dataset.dataset.classes)
        plt.xlabel("Predicted Class")
        plt.ylabel("True Class")
        plt.title("Confusion Matrix")
        plt.show()
    return running_loss/len(loader), correct/total

# ================================
# Step 7: Visual Inspection (Random Selection)
# ================================
def visual_inspection_random(loader, model, num_images=8):
    model.eval()
    # Gather all images and labels from the loader
    all_images = []
    all_labels = []
    for images, labels in loader:
        all_images.append(images)
        all_labels.append(labels)
    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    # Randomly select indices
    indices = np.random.choice(len(all_images), num_images, replace=False)
    plt.figure(figsize=(15, 15))
    for i, idx in enumerate(indices):
        img = all_images[idx].cpu().clone()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img.permute(1, 2, 0).numpy() * std + mean
        img = np.clip(img, 0, 1)
        pred = model(all_images[idx:idx+1].to(device))
        _, predicted = torch.max(pred.data, 1)
        plt.subplot(2, num_images//2, i+1)
        plt.imshow(img)
        plt.title(f"True: {train_dataset.dataset.classes[all_labels[idx].item()]}\nPred: {train_dataset.dataset.classes[predicted.item()]}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

# ================================
# Main Execution
# ================================
if __name__ == "__main__":
    data_dir = config["data_dir"]
    train_dataset, val_dataset = create_datasets(data_dir)
    
    # Get class names from the underlying dataset
    class_names = train_dataset.dataset.classes
    
    train_sampler = create_weighted_sampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])
    
    model = HybridDRModel().to(device)
    
    config_train = configure_training(train_dataset, model, config)
    criterion = config_train['criterion']
    optimizer = config_train['optimizer']
    scheduler = config_train['scheduler']
    
    # Train the model
    train_model(train_loader, val_loader, model, criterion, optimizer, scheduler)
    
    model.load_state_dict(torch.load("best_hybrid_model.pth"))
    final_val_loss, final_val_acc = evaluate_model(val_loader, model, criterion, print_cm=True)
    print(f"\nFinal Validation Accuracy: {final_val_acc:.4f}")
    
    visual_inspection_random(val_loader, model, num_images=12)
