# Hybrid Quantum-Classical Diabetic Retinopathy Classifier

---

## 📝 Introduction

This repository implements a **hybrid quantum-classical model** for multi-class image classification of diabetic retinopathy severity levels. It combines the feature-extraction power of a pretrained ResNet-50 backbone with the expressive capabilities of a variational quantum circuit, wrapped as a TorchLayer via PennyLane. The pipeline includes:

- **Robust data preprocessing** with advanced augmentation and class balancing
- **Classical projection stack** that gradually reduces high-dimensional features to match the quantum circuit input
- **Variational quantum circuit** operating on *n* qubits with layered rotations and entanglement
- **Custom training strategy** featuring focal loss, OneCycleLR scheduling, mixed precision, gradient accumulation, and early stopping
- **Comprehensive evaluation** using classification reports, confusion matrices, and qualitative visual inspection

This approach targets fine-grained classification across **five severity grades** (No_DR, Mild, Moderate, Severe, Proliferate_DR) and addresses class imbalance and overfitting through sampling strategies and strong regularization.

---

## 🚀 Tech Stack

| Layer         | Framework / Library            |
| ------------- | ------------------------------ |
| Deep Learning | PyTorch                        |
| Quantum       | PennyLane (qml.qnn.TorchLayer) |
| Data Augment  | torchvision.transforms         |
| Optimization  | AdamW, OneCycleLR              |
| Loss          | Custom Focal Loss              |
| Visualization | Matplotlib, Seaborn            |
| Python Tools  | NumPy, scikit-learn            |

---

## 📂 Repository Structure

├── 8_qubit_diff_parameter_v2.py # Main training & evaluation script
├── Quantum_model_v2.docx # Detailed architecture & methodology report
├── model_output/ # Checkpoints & training history plots
├── data/ # Dataset directory with class subfolders
├── requirements.txt # Python dependencies
└── README.md # This documentation file

---

## 💡 Code Explanation

1. **Configuration & Environment**  
   - **`config` dictionary**: Single point for all experiment settings, including data paths, image size, batch split ratio, quantum/classical hyperparameters, learning rates, epochs, and early stopping  
   - **Reproducibility**: Fixed random seeds for PyTorch and NumPy  
   - **Device selection**: Automatically detects CUDA GPU and prints device info  

2. **Data Preparation**  
   - **Transforms**:  
     - *Training*: Resize → RandomCrop → Rotation → Flips → ColorJitter → Affine → Blur → Sharpness → Erasing  
     - *Validation*: Resize → Normalize  
   - **Stratified splitting** using scikit-learn’s `train_test_split` to preserve class distribution  
   - **Class balancing** via a `WeightedRandomSampler` with inverse-frequency weights  

3. **Hybrid Model Architecture**  
   - **Classical Backbone**: ResNet-50 pretrained on ImageNet, minus final FC layer  
   - **Projection Head**: Dense layers that reduce from 2048 → 512 → … → *n_qubits*, with ReLU, BatchNorm1d, and Dropout  
   - **Quantum Layer**: PennyLane QNode with:  
     - `AngleEmbedding` for input encoding  
     - Layered RY rotations + full‑ring CNOT entanglement  
     - Measurement of Pauli‑Z expectation on each qubit  
     - Wrapped as `qml.qnn.TorchLayer` for end‑to‑end integration  
   - **Classifier Head**: Dense layers mapping quantum outputs → 64 → 32 → 5 classes  

4. **Training Configuration**  
   - **Loss**: Custom *FocalLoss* to emphasize hard‑to‑classify examples (α=0.25, γ=2)  
   - **Optimizer**: AdamW with separate parameter groups (`lr_classical`, `lr_quantum`)  
   - **Scheduler**: OneCycleLR with warm‑up (30% of total epochs) and cosine annealing  
   - **Mixed Precision**: `autocast` and `GradScaler` for faster GPU training  
   - **Gradient Accumulation**: Virtual batch size increase by accumulating over multiple mini‑batches  

5. **Training Loop**  
   - **Checkpointing** every 10 epochs and saving best models (top‑3 based on validation accuracy)  
   - **Early Stopping** after `patience` epochs without improvement  
   - **Logging** of per‑batch losses, learning rates, and epoch summaries  

6. **Evaluation & Visualization**  
   - **`evaluate_model`**: Computes classification report and confusion matrix heatmap  
   - **`visual_inspection_random`**: Displays random validation images with true vs. predicted labels for qualitative assessment
     
---

## 📦 Outputs

- Checkpoints in `model_output/checkpoints/`
- Best model saved as `model_output/best_hybrid_model.pth`
- Training history plot at `model_output/training_history.png`
- Console logs with metrics and confusion matrices

---

## 🤝 Contributing

1. **Fork** the repository  
2. **Create a feature branch**  
   ```bash
   git checkout -b feature/name
   ```
3.**Commit your changes**
  ```bash
    git commit -m "Add feature"
  ```
4. **Push to your branch**
   ```bash
   git push origin feature/name
   ```
5. **Open a pull request and describe your changes**
   Please ensure code style consistency (PEP8) and include relevant tests or visualizations for new features.
