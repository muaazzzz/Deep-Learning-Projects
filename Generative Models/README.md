# PA3: Generative Models

This assignment explores three major families of generative models: autoregressive models (PixelCNN), variational autoencoders (VAE), and generative adversarial networks (GANs).

## 📋 Assignment Structure

### Part 1: PixelCNN on MNIST (50 Points)
**File**: `26100117_PA3_1.ipynb` / `26100117_PA3_1.py`

Implementation of PixelCNN for autoregressive image generation.

**Key Concepts**:
- **Masked Convolutions**: Ensure autoregressive property (pixel depends only on previous pixels)
- **Residual Blocks**: Deep architecture with skip connections
- **Sequential Generation**: Generate images pixel-by-pixel from top-left to bottom-right

**Architecture**:
```
Input → Masked Conv (Type A) → Residual Blocks (Masked Conv Type B) → Output
```

**Training Configurations**:
- 10 epochs
- 20 epochs  
- 30 epochs

**Outputs**:
- Generated MNIST digit samples
- Likelihood plots (log-probability)
- Visualization of autoregressive pixel dependencies

### Part 2: Variational Autoencoders on CelebA (Variable Points)
**File**: `26100117_PA3_2.ipynb` / `26100117_PA3_2.py`

β-VAE implementation for disentangled representation learning on celebrity faces.

**Key Concepts**:
- **β-VAE**: Weighted KL divergence for disentanglement
- **Reparameterization Trick**: Enable gradient flow through stochastic latent variables
- **Disentanglement**: β controls trade-off between reconstruction and independence

**Architecture**:
```
Encoder: Image → Conv Layers → μ, log(σ²)
Latent Sampling: z ~ N(μ, σ²)
Decoder: z → Deconv Layers → Reconstructed Image
```

**Dataset**: CelebA (178×218 RGB images of celebrity faces)

**Experiments**:
- β = 1 (standard VAE)
- β = 5
- β = 10
- β = 50 (highly disentangled)

**Model Checkpoint**: `vae_generator.pth` (saved trained model)

**Results**:
- Higher β leads to more disentangled features
- Trade-off between reconstruction quality and feature independence
- Latent space traversal shows learned attributes (smile, hair, pose, etc.)

### Part 3: GANs with Latent Space Interpolation (Variable Points)
**File**: `26100117_PA3_3.ipynb` / `26100117_PA3_3.py`

Exploration of pre-trained DCGAN for image synthesis and latent space analysis.

**Key Concepts**:
- **Adversarial Training**: Generator vs. Discriminator minimax game
- **Latent Space Interpolation**: Smooth transitions between generated images
- **Mode Collapse**: Analysis of GAN failure modes
- **DCGAN Architecture**: Deep Convolutional GAN with specific design guidelines

**Tasks**:
1. Use pre-trained DCGAN on CelebA
2. Generate high-quality face images
3. Perform latent space interpolation (linear interpolation in z-space)
4. Analyze:
   - Mode collapse issues
   - Training instability
   - Comparison with diffusion models

**Outputs**:
- Generated celebrity faces
- Interpolation sequences showing smooth attribute transitions
- Analysis of GAN strengths and weaknesses

## 📊 Results

### PixelCNN (Part 1):
- Successfully generates coherent MNIST digits
- Longer training (30 epochs) produces sharper, more realistic digits
- Captures pixel-level dependencies effectively

### β-VAE (Part 2):
- **β = 1**: Best reconstruction, less disentanglement
- **β = 50**: Highly disentangled features, blurrier reconstructions
- Latent space allows controlled attribute manipulation
- Model saved as `vae_generator.pth` for inference

### DCGAN (Part 3):
- Generates realistic celebrity faces
- Smooth interpolation shows meaningful latent space structure
- Observations:
  - Mode collapse can occur with poor hyperparameters
  - Less stable than diffusion models
  - Faster sampling than autoregressive or diffusion models

## 🚀 How to Run

### Prerequisites
```bash
# Core dependencies
pip install torch torchvision
pip install tensorflow keras  # For PixelCNN
pip install numpy matplotlib
pip install kagglehub pillow  # For CelebA dataset
```

### Running the Assignments

1. **Part 1 - PixelCNN**:
```bash
jupyter notebook 26100117_PA3_1.ipynb
```
Or:
```bash
python 26100117_PA3_1.py
```

2. **Part 2 - β-VAE on CelebA**:
```bash
jupyter notebook 26100117_PA3_2.ipynb
```
First run will download CelebA dataset (~1.5GB).

3. **Part 3 - DCGAN**:
```bash
jupyter notebook 26100117_PA3_3.ipynb
```

### Expected Outputs

- **Part 1**: Generated MNIST digits at different training stages
- **Part 2**: 
  - Reconstructed CelebA faces at different β values
  - Latent space traversal visualizations
  - KL divergence and reconstruction loss plots
- **Part 3**:
  - Generated face samples
  - Interpolation grids showing smooth transitions

## 🧪 Key Learnings

### PixelCNN:
- ✅ Explicit likelihood modeling (can compute exact probabilities)
- ✅ No mode collapse
- ❌ Slow sequential generation
- ❌ Limited to small images without modifications

### VAE:
- ✅ Stable training
- ✅ Interpretable latent space
- ✅ Good for representation learning
- ❌ Blurry reconstructions (due to Gaussian likelihood)
- ❌ Can't compute exact likelihood (only lower bound)

### GAN:
- ✅ Sharp, high-quality samples
- ✅ Fast sampling
- ❌ Training instability
- ❌ Mode collapse
- ❌ No explicit likelihood

## 📁 Files Description

| File | Description |
|------|-------------|
| `26100117_PA3_1.ipynb` | PixelCNN implementation notebook |
| `26100117_PA3_1.py` | PixelCNN script |
| `26100117_PA3_2.ipynb` | β-VAE implementation notebook |
| `26100117_PA3_2.py` | β-VAE script |
| `26100117_PA3_3.ipynb` | DCGAN experiments notebook |
| `26100117_PA3_3.py` | DCGAN script |
| `vae_generator.pth` | Saved β-VAE generator model |

## 🎯 Applications

- **PixelCNN**: Density estimation, image compression
- **VAE**: Anomaly detection, data augmentation, semi-supervised learning
- **GAN**: Image synthesis, style transfer, data augmentation

## 🔗 Dependencies

- PyTorch (>= 1.9.0)
- TensorFlow/Keras (>= 2.6.0)
- NumPy, Matplotlib
- Kagglehub (CelebA dataset)
- PIL (Python Imaging Library)

---

**Note**: The pre-trained DCGAN model is assumed to be provided or loaded from a checkpoint. Training GANs from scratch requires careful hyperparameter tuning.
