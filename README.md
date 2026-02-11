# 🧠 Deep Learning Projects Portfolio

This repository contains a comprehensive collection of deep learning assignments covering fundamental and advanced topics in modern AI. Each assignment includes complete implementations, detailed explanations, and reproducible results.

## 📚 Assignments Overview

### Repository Structure
```
Deep-Learning-Projects/
├── 26100117_PA2/          # Optimizers, Autoencoders & Segmentation
├── 26100117_PA3/          # Generative Models (VAE, GAN, PixelCNN)
├── 26100117_PA4/          # CLIP & Transformers
├── 26100117_PA5/          # Diffusion Models
├── 26100117_PA6/          # Graph Neural Networks
├── README.md              # This file
└── LICENSE
```

### [PA2: Optimizers, Autoencoders & Segmentation](./26100117_PA2)
Advanced optimization techniques and autoencoder architectures for various tasks.
- **Topics**: Gradient descent variants, line search methods, dimensionality reduction, medical image segmentation
- **Highlights**: U-Net for brain tumor segmentation (98% train accuracy), CIFAR-10 reconstruction
- **Tech Stack**: PyTorch, NumPy, Matplotlib

### [PA3: Generative Models](./26100117_PA3)
Deep dive into various generative modeling approaches.
- **Topics**: PixelCNN, Variational Autoencoders (VAE), Generative Adversarial Networks (GANs)
- **Highlights**: β-VAE on CelebA, DCGAN latent space interpolation, autoregressive image generation
- **Tech Stack**: PyTorch, TensorFlow, Keras

### [PA4: Multimodal Learning & Transformers](./26100117_PA4)
Vision-language models and transformer architectures.
- **Topics**: CLIP (Contrastive Language-Image Pre-training), Vision Transformers, text generation
- **Highlights**: Custom CLIP implementation (90%+ accuracy), transformer-based text generation
- **Tech Stack**: PyTorch, Transformers, Timm, DistilBERT, ResNet50

### [PA5: Diffusion Models](./26100117_PA5)
Denoising diffusion probabilistic models for image generation.
- **Topics**: DDPM, classifier-free guidance, text-to-image generation, image inpainting
- **Highlights**: Conditional generation, Stable Diffusion, IP-Adapter style transfer
- **Tech Stack**: PyTorch, Diffusers, Stable Diffusion

### [PA6: Graph Neural Networks](./26100117_PA6)
Graph-based learning for structured data.
- **Topics**: GraphSAGE, GCN, spectral filtering, spatio-temporal prediction
- **Highlights**: GNN for maze navigation, MUTAG graph classification (84% test accuracy), traffic prediction
- **Tech Stack**: PyTorch Geometric, NetworkX

## 🚀 Getting Started

### Quick Start

1. **Clone the repository**:
```bash
git clone https://github.com/muaazzzz/Deep-Learning-Projects.git
cd Deep-Learning-Projects
```

2. **Choose an assignment** (e.g., PA2):
```bash
cd 26100117_PA2
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
# Core dependencies (for most assignments)
pip install torch torchvision
pip install tensorflow
pip install numpy matplotlib pandas seaborn

# For specific assignments
pip install transformers timm diffusers
pip install torch-geometric networkx
pip install kagglehub pillow
```

### What Each Assignment Contains

Each assignment folder includes:
- **`.ipynb` files**: Interactive Jupyter notebooks with detailed explanations and code
- **`.py` files**: Python scripts exported from notebooks for direct execution
- **`README.md`**: Assignment-specific documentation, results, and instructions
- **`requirements.txt`**: Specific dependencies for that assignment
- **Model files** (where applicable): Pre-trained models for inference

Navigate to any assignment folder and follow its README for detailed instructions.

## 📊 Key Results Summary

| Assignment | Best Achievement | Metric |
|------------|------------------|--------|
| PA2 - Segmentation | 98.32% train / 73.15% test | Accuracy |
| PA3 - Generative Models | High-quality CelebA faces | Visual Quality |
| PA4 - CLIP | 90%+ | Zero-shot Accuracy |
| PA5 - Diffusion | Stable image generation | Visual Quality |
| PA6 - GCN | 84.21% test | Accuracy |

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

This repository showcases deep learning assignments completed as part of advanced courses in Deep Learning. Each assignment demonstrates practical implementation of state-of-the-art techniques and algorithms.

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
