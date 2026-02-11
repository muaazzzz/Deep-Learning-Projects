# PA2: Optimizers, Autoencoders & Segmentation

This assignment explores optimization techniques, autoencoder architectures, and their applications in image reconstruction and medical image segmentation.

## 📋 Assignment Structure

### Part 1: Gradient Descent & Line Search (50 Points)
**File**: `26100117_PA2_1.ipynb` / `26100117_PA2_1.py`

Implements and compares various gradient descent optimization strategies:
- Constant step size gradient descent
- Exact line search
- Backtracking line search (Armijo condition)

**Objective**: Minimize bivariate quadratic functions and analyze convergence behavior.

**Key Concepts**:
- Line search strategies for adaptive learning rates
- Convergence speed comparison
- Stability analysis of different optimizers

### Part 2: Linear & Convolutional Autoencoders (75 Points)
**File**: `26100117_PA2_2.ipynb` / `26100117_PA2_2.py`

Implements autoencoder architectures for dimensionality reduction and feature learning:

#### Tasks:
1. **CIFAR-10 Reconstruction** (25 points)
   - Linear autoencoder for image reconstruction
   - Encoder: Fully connected layers compressing 32×32×3 images
   - Decoder: Reconstructs original images from compressed representations

2. **MNIST Denoising** (25 points)
   - Linear autoencoder for noise removal
   - Learns to reconstruct clean images from noisy inputs

3. **CNN Autoencoder** (25 points)
   - Convolutional architecture for improved feature learning
   - Uses convolutional and transpose convolutional layers

**Architecture Example**:
```
Encoder: [Input] → Conv2D → ReLU → MaxPool → ... → [Latent Space]
Decoder: [Latent] → ConvTranspose2D → ReLU → Upsample → ... → [Output]
```

### Part 3: Brain Tumor Segmentation (50 Points)
**File**: `26100117_PA2_3.ipynb` / `26100117_PA2_3.py`

Medical image segmentation using U-Net architecture:
- **Dataset**: Brain tumor segmentation dataset from Kaggle
- **Task**: Pixel-level segmentation of tumor regions
- **Architecture**: U-Net style encoder-decoder with skip connections

## 📊 Results

### Overall Results
The file `26100117_Overall_Results.json` contains comprehensive results for all tasks.

### Part 2 Results:
- **Test Loss**: ~0.0009 (Linear autoencoders)
- Successfully reconstructed CIFAR-10 images
- Effective noise removal on MNIST

### Part 3 Results (Brain Tumor Segmentation):
- **Training Accuracy**: 98.32%
- **Test Accuracy**: 73.15%
- **Test Loss**: 0.2454
- Achieved good segmentation on medical images despite class imbalance

## 🚀 How to Run

### Prerequisites
```bash
pip install torch torchvision
pip install numpy matplotlib pandas
pip install kagglehub  # For Part 3 dataset
```

### Running the Notebooks

1. **Part 1 - Optimizers**:
```bash
jupyter notebook 26100117_PA2_1.ipynb
```
Or run the Python script:
```bash
python 26100117_PA2_1.py
```

2. **Part 2 - Autoencoders**:
```bash
jupyter notebook 26100117_PA2_2.ipynb
```

3. **Part 3 - Brain Tumor Segmentation**:
```bash
jupyter notebook 26100117_PA2_3.ipynb
```
Note: First run will download the brain tumor dataset from Kaggle.

### Expected Outputs

- **Part 1**: Convergence plots showing different optimization trajectories
- **Part 2**: 
  - Reconstructed CIFAR-10 images
  - Denoised MNIST images
  - Training/validation loss curves
- **Part 3**:
  - Brain MRI images with predicted tumor segmentation masks
  - Segmentation accuracy metrics
  - Training curves

## 🧪 Key Learnings

1. **Optimization**: Different line search methods have distinct convergence properties
2. **Autoencoders**: 
   - Effective for unsupervised feature learning
   - CNNs outperform linear models for image data
   - Latent space can capture meaningful representations
3. **Medical Imaging**:
   - U-Net architecture is powerful for segmentation tasks
   - Skip connections help preserve spatial information
   - Class imbalance is a common challenge in medical datasets

## 📁 Files Description

| File | Description |
|------|-------------|
| `26100117_PA2_1.ipynb` | Part 1: Optimizers notebook |
| `26100117_PA2_1.py` | Part 1: Optimizers script |
| `26100117_PA2_2.ipynb` | Part 2: Autoencoders notebook |
| `26100117_PA2_2.py` | Part 2: Autoencoders script |
| `26100117_PA2_3.ipynb` | Part 3: Segmentation notebook |
| `26100117_PA2_3.py` | Part 3: Segmentation script |
| `26100117_Overall_Results.json` | Complete results summary |

## 🔗 Dependencies

- PyTorch (>= 1.9.0)
- NumPy
- Matplotlib
- Pandas
- Kagglehub (for dataset download)
- torchvision

---

**Note**: All experiments were run with standard hyperparameters. Results may vary slightly based on random initialization and hardware.
