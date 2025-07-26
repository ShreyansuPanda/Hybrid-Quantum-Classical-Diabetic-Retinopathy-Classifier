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

# Import for mixed precision training
from torch.cuda.amp import autocast, GradScaler

# ================================
# CONFIGURATION
# ================================
config = {
    "data_dir": "D:/study/quantum/Diabetic Retinopathy 224x224 (2019 Data)/colored_images/train",
    "img_size": (224, 224),
    "batch_size": 32,
    "test_size": 0.2,
    "random_state": 50,
    "n_qubits": 16,
    "n_layers": 8,
    # Optimizer hyperparameters:
    "lr_classical": 0.0005,
    "lr_quantum": 0.0002,
    "weight_decay": 2e-4,
    "epochs": 150,
    "patience": 20
}

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Determine device: GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Set CUDA device
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

# ================================
# Step 1: Dataset Preparation
# ================================
def create_datasets(data_dir):
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize larger then crop for better variation
        transforms.RandomCrop(config["img_size"]),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=10, translate=(0.15, 0.15), scale=(0.8, 1.2)),
        transforms.RandomApply([transforms.GaussianBlur(5)], p=0.5),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2)  # Random erasing for occlusion robustness
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
    # Improved embedding strategy
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    
    # More sophisticated entangling layers with rotations
    for l in range(weights.shape[0]):
        for i in range(n_qubits):
            qml.RY(weights[l, i], wires=i)
        
        # Entanglement pattern: full entanglement for better expressivity
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])
    
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# For TorchLayer, we need to specify the weight shapes.
weight_shapes = {"weights": (n_layers, n_qubits)}
quantum_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

class HybridDRModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Using ResNet50 instead of ResNet18 for better feature extraction
        resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', weights='IMAGENET1K_V1')
        self.classical = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.Flatten(),
            nn.Linear(2048, 512),  # Adjusted for ResNet50's output size
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.4),
            nn.Linear(128, n_qubits),
            nn.BatchNorm1d(n_qubits)
        )
        self.quantum = quantum_layer
        self.fc = nn.Sequential(
            nn.Linear(n_qubits, 64),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
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
    
    # Calculate class weights for imbalanced dataset
    class_weights = torch.tensor([total_samples/(class_counts[i]*1.5) for i in range(num_classes)]).to(device)
    
    class FocalLoss(nn.Module):
        def __init__(self, weight=None, alpha=0.25, gamma=2):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.weight = weight
            
        def forward(self, inputs, targets):
            ce_loss = nn.CrossEntropyLoss(reduction='none', weight=self.weight)(inputs, targets)
            pt = torch.exp(-ce_loss)
            return (self.alpha * (1-pt)**self.gamma * ce_loss).mean()
    
    criterion = FocalLoss(weight=class_weights)
    
    # Create parameter groups with different learning rates
    classical_params = []
    quantum_params = []
    
    for name, param in model.named_parameters():
        if "quantum" in name:
            quantum_params.append(param)
        else:
            classical_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': classical_params, 'lr': config["lr_classical"]},
        {'params': quantum_params, 'lr': config["lr_quantum"]}
    ], weight_decay=config["weight_decay"])
    
    # One cycle learning rate scheduler for better convergence
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=[config["lr_classical"]*2, config["lr_quantum"]*2],
        steps_per_epoch=len(train_dataset) // config["batch_size"] + 1,
        epochs=config["epochs"],
        pct_start=0.3,  # Warm up for first 30% of training
        anneal_strategy='cos'
    )
    
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
    
    # For mixed precision training
    scaler = GradScaler()
    
    # Gradient accumulation steps
    accumulation_steps = 2  # Accumulate gradients over 2 batches
    
    # Create model checkpoint directory
    checkpoint_dir = "model_output/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # For LR finder and tracking best models
    best_models = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        # Reset optimizer gradients at the start of each epoch
        optimizer.zero_grad()
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Use mixed precision training
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels) / accumulation_steps  # Normalize loss
            
            # Scale gradients and accumulate
            scaler.scale(loss).backward()
            
            # Update weights after accumulation_steps
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()  # Reset gradients after update
                
                # Update LR
                scheduler.step()
            
            running_loss += loss.item() * accumulation_steps
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            # Print progress every 10 batches
            if (i+1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}, "
                      f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Evaluate on validation set
        val_loss, val_acc = evaluate_model(val_loader, model, criterion, print_cm=(epoch % 10 == 0))
        
        train_loss = running_loss / len(train_loader)
        train_acc = correct_train / total_train
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc
            }, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")
        
        # Track best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join("model_output", "best_hybrid_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with validation accuracy: {val_acc:.4f}")
            
            # Keep track of top 3 models
            best_models.append((val_acc, epoch+1, best_model_path))
            best_models.sort(key=lambda x: x[0], reverse=True)
            best_models = best_models[:3]  # Keep only top 3
            
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        print(f"LR Classical: {optimizer.param_groups[0]['lr']:.2e} | LR Quantum: {optimizer.param_groups[1]['lr']:.2e}")
        print("-----------------------------------")
        
        # Early stopping check
        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Print best models
    print("\nBest models:")
    for i, (acc, epoch, path) in enumerate(best_models):
        print(f"{i+1}. Epoch {epoch}: {acc:.4f}")
    
    # Save training history plots
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot learning rate
    lr_history = []
    model_path = os.path.join("model_output", "training_history.png") 
    plt.savefig(model_path)
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
    start_time = time.time()
    
    # Create output directory for model artifacts
    os.makedirs("model_output", exist_ok=True)
    model_path = os.path.join("model_output", "best_hybrid_model.pth")
    history_plot_path = os.path.join("model_output", "training_history.png")
    
    print(f"Starting training with {config['n_qubits']} qubits and {config['n_layers']} layers")
    print(f"Using device: {device}")
    
    data_dir = config["data_dir"]
    train_dataset, val_dataset = create_datasets(data_dir)
    
    # Get class names from the underlying dataset
    class_names = train_dataset.dataset.classes
    print(f"Classes: {class_names}")
    print(f"Training with {len(train_dataset)} samples, validating on {len(val_dataset)} samples")
    
    # Analyze class distribution
    train_labels = [label for _, label in train_dataset]
    val_labels = [label for _, label in val_dataset]
    
    train_class_counts = Counter(train_labels)
    val_class_counts = Counter(val_labels)
    
    print("\nClass distribution in training set:")
    for cls, count in train_class_counts.items():
        print(f"{class_names[cls]}: {count} samples ({count/len(train_labels)*100:.2f}%)")
        
    print("\nClass distribution in validation set:")
    for cls, count in val_class_counts.items():
        print(f"{class_names[cls]}: {count} samples ({count/len(val_labels)*100:.2f}%)")
    
    train_sampler = create_weighted_sampler(train_dataset)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["batch_size"], 
        sampler=train_sampler,
        num_workers=4,  # Parallel data loading 
        pin_memory=True  # Faster data transfer to GPU
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config["batch_size"],
        num_workers=4,
        pin_memory=True
    )
    
    model = HybridDRModel().to(device)
    
    # Initialize the model with better weight initialization
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    # Apply weight initialization to the classical part
    model.fc.apply(init_weights)
    
    config_train = configure_training(train_dataset, model, config)
    criterion = config_train['criterion']
    optimizer = config_train['optimizer']
    scheduler = config_train['scheduler']
    
    # Train the model
    best_val_acc = train_model(train_loader, val_loader, model, criterion, optimizer, scheduler)
    
    # Load the best model for evaluation
    model.load_state_dict(torch.load(model_path if os.path.exists(model_path) else "best_hybrid_model.pth"))
    final_val_loss, final_val_acc = evaluate_model(val_loader, model, criterion, print_cm=True)
    print(f"\nFinal Validation Accuracy: {final_val_acc:.4f}")
    
    # Visualize some predictions
    visual_inspection_random(val_loader, model, num_images=12)
    
    # Report training time
    training_time = time.time() - start_time
    hours = int(training_time // 3600)
    minutes = int((training_time % 3600) // 60)
    seconds = int(training_time % 60)
    print(f"Total training time: {hours}h {minutes}m {seconds}s")
