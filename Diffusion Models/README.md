# PA5: Diffusion Models

This assignment explores Denoising Diffusion Probabilistic Models (DDPM), a powerful class of generative models that learn to generate images by reversing a gradual noising process.

## 📋 Assignment Structure

### Part 1: MNIST Diffusion (Variable Points)
**File**: `Part1_Diffusion.ipynb` / `Part1_Diffusion.py`

Complete implementation of diffusion models on MNIST dataset with various conditioning strategies.

#### Task 1: Basic Diffusion Model
**Concepts**:
- **Forward Process**: Gradually add Gaussian noise to images (q(x_t | x_{t-1}))
- **Reverse Process**: Learn to denoise (p_θ(x_{t-1} | x_t))
- **U-Net Architecture**: Denoising network with skip connections
- **Noise Schedule**: Controls rate of noise addition (β_t)

**Architecture**:
```
U-Net Denoising Network:
  Input: Noisy Image x_t + Time Embedding t
  Encoder: Conv → ResBlock → Down → ... → Bottleneck
  Decoder: Up → ResBlock → Conv → ... → Output (predicted noise)
```

**Training**:
- Sample timestep t uniformly
- Add noise to image: x_t = √(ᾱ_t) x_0 + √(1-ᾱ_t) ε
- Predict noise: ε_θ(x_t, t)
- Minimize: ||ε - ε_θ(x_t, t)||²

**Sampling**:
- Start from random noise x_T ~ N(0, I)
- Iteratively denoise: x_{t-1} = μ_θ(x_t, t) + σ_t z
- Generate clean image x_0

#### Task 2: Class-Conditional Diffusion
**Concepts**:
- **Conditional Generation**: Generate specific digit classes
- **Classifier Guidance**: Use classifier gradients to guide generation
- **Class Embeddings**: Inject class information into U-Net

**Architecture Enhancement**:
```
U-Net + Class Conditioning:
  Input: Noisy Image x_t + Time Embedding t + Class Embedding c
  Process with class-conditional layers
```

**Benefits**:
- Control over generated content
- Higher quality samples
- Targeted generation

#### Task 3: Text-Guided Generation (CLIP + Diffusion)
**Concepts**:
- **CLIP Guidance**: Use CLIP embeddings to guide diffusion
- **Text Prompts**: Generate images from text descriptions
- **Gradient-Based Guidance**: Steer denoising toward text embedding

**Process**:
1. Encode text prompt with CLIP text encoder
2. During denoising, compute CLIP image embedding
3. Use gradient of CLIP similarity to guide generation
4. Generate image matching text description

### Part 2: Guided Generation & Advanced Techniques (Variable Points)
**File**: `Part2_Diffusion.ipynb` / `Part2_Diffusion.py`

Advanced diffusion techniques using Stable Diffusion and guidance mechanisms.

#### Classifier-Free Guidance (CFG)
**Concepts**:
- **Unconditional vs. Conditional**: Train single model for both
- **Guidance Scale**: Control strength of conditioning
- **Formula**: ε̃ = ε_u + s(ε_c - ε_u), where s is guidance scale

**Experiments**:
- Guidance scale = 1.0 (unconditional)
- Guidance scale = 5.0 (moderate guidance)
- Guidance scale = 7.5 (strong guidance, common default)
- Guidance scale = 12.0 (very strong guidance)

**Observations**:
- Low scale (1.0): Diverse but less adherent to prompt
- Medium scale (7.5): Good balance
- High scale (12.0): Strong adherence but less diversity

#### Text-to-Image with Stable Diffusion
**Concepts**:
- **Latent Diffusion**: Operate in compressed latent space
- **VAE Encoder/Decoder**: Convert between pixel and latent space
- **Cross-Attention**: Inject text conditioning via attention

**Architecture**:
```
Text → CLIP Text Encoder → Text Embeddings
Image → VAE Encoder → Latent z
Latent z → U-Net (with cross-attention to text) → Denoised z
Denoised z → VAE Decoder → Generated Image
```

#### Image Inpainting
**Concepts**:
- **Masked Diffusion**: Generate only in masked regions
- **Conditioning on Context**: Use unmasked regions as context
- **Seamless Integration**: Blend generated with original

**Applications**:
- Object removal
- Image completion
- Face editing

#### IP-Adapter for Style Control
**Concepts**:
- **Image Prompt Adapter**: Condition on reference images
- **Style Transfer**: Generate in style of reference
- **Content + Style**: Combine text and image conditioning

## 📊 Results

### Part 1: MNIST Diffusion

**Task 1 - Basic Diffusion**:
- Successfully generates coherent MNIST digits
- 1000 diffusion timesteps
- U-Net architecture with ~10M parameters
- Training time: ~2-3 hours on GPU

**Task 2 - Class-Conditional**:
- Can generate specific digit classes on demand
- Higher quality than unconditional
- Class control is reliable

**Task 3 - CLIP-Guided**:
- Generates MNIST-style images from text prompts
- Examples: "digit five", "number three"
- Demonstrates multimodal guidance

### Part 2: Advanced Techniques

**Classifier-Free Guidance Results**:
| Guidance Scale | Quality | Diversity | Prompt Adherence |
|----------------|---------|-----------|------------------|
| 1.0 | Moderate | High | Low |
| 5.0 | Good | Moderate | Good |
| 7.5 | High | Moderate | High |
| 12.0 | Very High | Low | Very High |

**Best Practice**: Scale 7.5 for general use, adjust based on needs

**Inpainting Results**:
- Seamless object removal
- Context-aware generation
- Natural blending with surroundings

**IP-Adapter Results**:
- Effective style transfer
- Maintains content from text prompt
- Adopts style from reference image

## 🚀 How to Run

### Prerequisites
```bash
# Core dependencies
pip install torch torchvision
pip install diffusers  # HuggingFace Diffusers library
pip install transformers  # For CLIP
pip install numpy matplotlib
pip install accelerate  # For efficient training
```

### Installation
```bash
# Install all dependencies
pip install -r requirements.txt
```

### Running the Assignments

1. **Part 1 - MNIST Diffusion**:
```bash
jupyter notebook Part1_Diffusion.ipynb
```
Or run the script:
```bash
python Part1_Diffusion.py
```

2. **Part 2 - Advanced Techniques**:
```bash
jupyter notebook Part2_Diffusion.ipynb
```

### Expected Outputs

**Part 1**:
- Generated MNIST digits (unconditional)
- Class-conditional generated digits (0-9)
- CLIP-guided generations from text prompts
- Diffusion process visualization (noising and denoising)
- Training loss curves

**Part 2**:
- Text-to-image generations at different guidance scales
- Inpainted images with masked regions filled
- Style-transferred images using IP-Adapter
- Comparison visualizations

### Sample Usage (Part 2)

```python
from diffusers import StableDiffusionPipeline

# Load Stable Diffusion
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16
)

# Generate with CFG
image = pipe(
    "A beautiful sunset over mountains",
    guidance_scale=7.5,
    num_inference_steps=50
).images[0]
```

## 🧪 Key Learnings

### Diffusion Models:
- ✅ State-of-the-art generation quality
- ✅ Stable training (no mode collapse)
- ✅ Flexible conditioning mechanisms
- ❌ Slow sampling (50-1000 steps)
- ❌ Memory intensive

### vs. Other Generative Models:

| Model | Quality | Speed | Training Stability | Likelihood |
|-------|---------|-------|-------------------|------------|
| GAN | High | Fast | Poor | No |
| VAE | Moderate | Fast | Good | Lower Bound |
| Diffusion | Very High | Slow | Excellent | Yes |
| Autoregressive | High | Very Slow | Good | Yes |

### Classifier-Free Guidance:
- 💡 Single model for conditional + unconditional
- 💡 Guidance scale allows quality-diversity trade-off
- 💡 No separate classifier needed

## 📁 Files Description

| File | Description |
|------|-------------|
| `Part1_Diffusion.ipynb` | Basic diffusion + conditioning |
| `Part1_Diffusion.py` | Part 1 script |
| `Part2_Diffusion.ipynb` | Advanced techniques |
| `Part2_Diffusion.py` | Part 2 script |

## 🎯 Applications

- **Text-to-Image**: DALL-E, Midjourney, Stable Diffusion
- **Image Editing**: Inpainting, outpainting, style transfer
- **Super-Resolution**: Enhance image quality
- **Video Generation**: Extend to temporal dimension
- **3D Generation**: Novel view synthesis
- **Audio Generation**: Diffusion for speech/music

## 🔗 Dependencies

- **PyTorch**: Deep learning framework
- **Diffusers**: HuggingFace diffusion models library
- **Transformers**: CLIP and text encoders
- **Accelerate**: Efficient training utilities
- **NumPy, Matplotlib**: Data and visualization

## 📚 References

- [DDPM Paper](https://arxiv.org/abs/2006.11239) - Denoising Diffusion Probabilistic Models
- [Classifier-Free Guidance](https://arxiv.org/abs/2207.12598)
- [Stable Diffusion](https://arxiv.org/abs/2112.10752) - High-Resolution Image Synthesis with Latent Diffusion Models

---

**Training Time**: Part 1 takes ~2-3 hours, Part 2 uses pre-trained models (instant inference).

**GPU Requirements**: NVIDIA GPU with 8GB+ VRAM recommended for Part 1, 12GB+ for Part 2 Stable Diffusion.
