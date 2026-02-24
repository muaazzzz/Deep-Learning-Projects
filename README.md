# 🧠 Deep Learning Projects Portfolio

This repository showcases deep learning projects covering fundamental and advanced topics in modern AI. Each project includes complete implementations, detailed explanations, and reproducible results.

## 📚 Projects Overview

### Repository Structure
```
Deep-Learning-Projects/
├── Optimizers, Autoencoders & Segmentation/   # Gradient descent, autoencoders, U-Net segmentation
├── Generative Models/                          # PixelCNN, VAE, and GAN implementations
├── Multimodal Learning & Transformers/         # CLIP, Vision Transformers, text generation
├── Diffusion Models/                           # DDPM, classifier-free guidance, Stable Diffusion
├── Graph Neural Networks/                      # GraphSAGE, GCN, spatio-temporal prediction
├── README.md                                   # This file
└── LICENSE
```

### [Optimizers, Autoencoders & Segmentation](./Optimizers,%20Autoencoders%20%26%20Segmentation)
Explores optimization algorithms, autoencoder architectures, and medical image segmentation.
- **Topics**: Gradient descent variants (constant step, exact & backtracking line search), linear and convolutional autoencoders, U-Net for pixel-level segmentation
- **Highlights**: U-Net achieving 98% training accuracy on brain tumor segmentation, CIFAR-10 and MNIST reconstruction with convolutional autoencoders
- **Tech Stack**: PyTorch, NumPy, Matplotlib

### [Generative Models](./Generative%20Models)
Implements and compares three major families of generative models.
- **Topics**: Autoregressive generation with PixelCNN, disentangled representation learning with β-VAE, adversarial training with DCGAN
- **Highlights**: β-VAE trained on CelebA with controllable latent attributes, DCGAN latent space interpolation showing smooth attribute transitions
- **Tech Stack**: PyTorch, TensorFlow, Keras

### [Multimodal Learning & Transformers](./Multimodal%20Learning%20%26%20Transformers)
Builds vision-language models and transformer architectures from scratch.
- **Topics**: Contrastive Language-Image Pre-training (CLIP), Vision Transformers (ViT), encoder-decoder transformers for text generation
- **Highlights**: Custom CLIP implementation achieving 90%+ zero-shot accuracy on Flickr8k, transformer trained on movie scripts for dialogue generation
- **Tech Stack**: PyTorch, HuggingFace Transformers, Timm, DistilBERT, ResNet50

### [Diffusion Models](./Diffusion%20Models)
Implements denoising diffusion probabilistic models and advanced guided generation techniques.
- **Topics**: DDPM with U-Net denoising network, class-conditional and CLIP-guided generation, classifier-free guidance (CFG), image inpainting
- **Highlights**: End-to-end DDPM trained on MNIST, Stable Diffusion text-to-image experiments with varying guidance scales, IP-Adapter style transfer
- **Tech Stack**: PyTorch, HuggingFace Diffusers, Stable Diffusion

### [Graph Neural Networks](./Graph%20Neural%20Networks)
Applies graph-based learning to navigation, molecular classification, and traffic forecasting.
- **Topics**: GraphSAGE for inductive learning, spectral graph convolutions, GCN with Laplacian regularization, spatio-temporal GCN
- **Highlights**: GraphSAGE policy learning for maze navigation, GCN achieving 84% test accuracy on MUTAG molecular graphs, ST-GCN for traffic prediction
- **Tech Stack**: PyTorch Geometric, NetworkX

## 🚀 Getting Started

### Quick Start

1. **Clone the repository**:
```bash
git clone https://github.com/muaazzzz/Deep-Learning-Projects.git
cd Deep-Learning-Projects
```

2. **Choose a project** (e.g., Optimizers, Autoencoders & Segmentation):
```bash
cd "Optimizers, Autoencoders & Segmentation"
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Run the notebook**:
```bash
jupyter notebook 26100117_PA2_1.ipynb
```

Or run the Python script:
```bash
python 26100117_PA2_1.py
```

### Prerequisites
```bash
# Core dependencies (for most projects)
pip install torch torchvision
pip install tensorflow
pip install numpy matplotlib pandas seaborn

# For specific projects
pip install transformers timm diffusers
pip install torch-geometric networkx
pip install kagglehub pillow
```

### What Each Project Contains

Each project folder includes:
- **`.ipynb` files**: Interactive Jupyter notebooks with detailed explanations and code
- **`.py` files**: Python scripts exported from notebooks for direct execution
- **`README.md`**: Project-specific documentation, results, and instructions
- **`requirements.txt`**: Specific dependencies for that project
- **Model files** (where applicable): Pre-trained models for inference

Navigate to any project folder and follow its README for detailed instructions.

## 📊 Key Results Summary

| Project | Best Achievement | Metric |
|---------|-----------------|--------|
| Optimizers, Autoencoders & Segmentation | 98.32% train / 73.15% test | Accuracy |
| Generative Models | High-quality CelebA faces | Visual Quality |
| Multimodal Learning & Transformers | 90%+ | Zero-shot Accuracy |
| Diffusion Models | Stable image generation | Visual Quality |
| Graph Neural Networks | 84.21% test | Accuracy |

## 🛠️ Technologies Used

<div align="center">

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)

</div>

## 📖 Topics Covered

- **Optimization**: Gradient descent, line search, Adam, SGD
- **Computer Vision**: CNNs, U-Net, ResNet, Vision Transformers
- **Generative Models**: VAE, GAN, PixelCNN, Diffusion Models
- **Multimodal Learning**: CLIP, vision-language alignment
- **NLP**: Transformers, BERT, text generation
- **Graph Neural Networks**: GCN, GraphSAGE, spectral methods
- **Applications**: Image segmentation, text-to-image, graph classification

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 About

This repository showcases deep learning projects completed as part of advanced courses in Deep Learning. Each project demonstrates practical implementation of state-of-the-art techniques and algorithms.

**Author**: Muaaz

## 📬 Contact

For questions, suggestions, or collaboration opportunities:
- Open an issue in this repository
- Reach out via GitHub: [@muaazzzz](https://github.com/muaazzzz)

## 🙏 Acknowledgments

- Course instructors and TAs for providing excellent assignments
- Open-source community for amazing frameworks (PyTorch, TensorFlow, HuggingFace)
- Kaggle for providing datasets

---

<div align="center">

⭐ **If you found this repository helpful, please consider giving it a star!** ⭐

Made with ❤️ for the Deep Learning community

</div>
