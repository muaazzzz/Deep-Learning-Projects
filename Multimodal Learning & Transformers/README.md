# PA4: Multimodal Learning & Transformers

This assignment explores the intersection of vision and language through CLIP (Contrastive Language-Image Pre-training) and the application of transformers for various tasks.

## 📋 Assignment Structure

### Part 1: CLIP - Vision-Language Model (100 Points)
**File**: `26100117_PA4_1.ipynb` / `26100117_PA4_1.py`

Custom implementation of CLIP from scratch for vision-language understanding.

**Key Concepts**:
- **Contrastive Learning**: Learn aligned image-text embeddings
- **Zero-Shot Classification**: Classify images using text descriptions
- **Dual Encoders**: Separate encoders for vision and language
- **Projection Heads**: Map to shared embedding space

**Architecture**:
```
Image Branch:
  Input Image → ResNet50 (frozen) → Projection Layer → Embedding

Text Branch:
  Input Text → DistilBERT (frozen) → Projection Layer → Embedding

Loss:
  Contrastive Loss = Maximize similarity of matching pairs
                     Minimize similarity of non-matching pairs
```

**Dataset**: Flickr8k
- 8,000 images
- 5 text captions per image
- Diverse scenes and objects

**Training Strategy**:
1. Freeze pre-trained ResNet50 and DistilBERT weights
2. Train only projection layers (efficient training)
3. Use InfoNCE (contrastive) loss
4. Batch-based positive/negative sampling

**Target Performance**: 90%+ accuracy on test set

### Part 2: Transformers (125 Points)

#### Part 2a: Text Generation with Transformers (75 Points)
**File**: `26100117_PA4_2.ipynb` / `26100117_PA4_2.py`

Implementation of transformer model for text generation.

**Key Concepts**:
- **Encoder-Decoder Architecture**: Full transformer with attention mechanisms
- **Self-Attention**: Multi-head attention for sequence modeling
- **Positional Encoding**: Inject position information
- **Autoregressive Generation**: Generate text token-by-token

**Dataset**: Transformers movie script
- Train on movie dialogue
- Learn character interactions and story patterns
- Generate new dialogue sequences

**Architecture**:
```
Encoder:
  Input → Embedding → Positional Encoding → 
  Multi-Head Attention → Feed-Forward → Layer Norm

Decoder:
  Output Embedding → Positional Encoding →
  Masked Multi-Head Attention → Cross Attention → 
  Feed-Forward → Layer Norm → Output
```

**Tasks**:
1. Implement transformer from scratch (no pre-trained models)
2. Train on movie script
3. Generate coherent text sequences
4. Analyze attention patterns

#### Part 2b: Vision Transformers (50 Points)
**File**: Included in `26100117_PA4_2.ipynb` / `26100117_PA4_2.py`

Application of Vision Transformers (ViT) for image classification.

**Key Concepts**:
- **Patch Embeddings**: Split images into patches
- **Transformer Encoder**: Apply self-attention to patch sequences
- **Classification Token**: [CLS] token for image representation

**Architecture**:
```
Image → Patch Embedding → Position Embedding →
Transformer Encoder (12 layers) → [CLS] Token → MLP Head → Classes
```

**Comparison**:
- ViT vs. CNNs (ResNet, VGG)
- Scalability analysis
- Attention visualization

## 📊 Results

### Part 1: CLIP
- **Test Accuracy**: 90%+ (achieved target)
- Successfully aligns image and text embeddings
- Demonstrates zero-shot capabilities
- Cross-modal retrieval:
  - Image → Text: Find captions for images
  - Text → Image: Find images matching descriptions

**Key Observations**:
- Frozen pre-trained encoders work well
- Projection layers are sufficient for alignment
- Batch size matters for contrastive learning
- Larger batches provide more negative samples

### Part 2a: Text Generation
- Generates coherent movie dialogue
- Captures character-specific language patterns
- Attention weights show meaningful focus
- Longer sequences require careful training

### Part 2b: Vision Transformers
- Competitive with CNN baselines
- Requires more data than CNNs
- Attention maps show object-focused patterns
- Scalable to larger datasets

## 🚀 How to Run

### Prerequisites
```bash
# Core dependencies
pip install torch torchvision
pip install transformers  # HuggingFace transformers
pip install timm  # PyTorch Image Models (for ResNet50)
pip install kagglehub  # For Flickr8k dataset
pip install numpy matplotlib pillow
```

### Installation
```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Assignments

1. **Part 1 - CLIP**:
```bash
jupyter notebook 26100117_PA4_1.ipynb
```
Or run the script:
```bash
python 26100117_PA4_1.py
```
First run will download Flickr8k dataset (~1GB).

2. **Part 2 - Transformers**:
```bash
jupyter notebook 26100117_PA4_2.ipynb
```

### Expected Outputs

**Part 1 (CLIP)**:
- Training/validation loss curves
- Accuracy metrics (should exceed 90%)
- Image-text similarity matrices
- Zero-shot classification examples
- Cross-modal retrieval results

**Part 2a (Text Generation)**:
- Generated movie dialogue samples
- Attention visualization
- Perplexity scores
- Training curves

**Part 2b (ViT)**:
- Classification accuracy
- Attention maps showing patch importance
- Comparison with CNN baselines

## 🧪 Key Learnings

### CLIP:
- ✅ Powerful for zero-shot learning
- ✅ Transfer learning from vision + language
- ✅ Robust to distribution shift
- 💡 Key insight: Alignment in shared space is sufficient

### Transformers:
- ✅ Self-attention captures long-range dependencies
- ✅ Parallel computation (vs. RNN sequential)
- ✅ Transfer learning via pre-training
- ❌ Quadratic complexity in sequence length
- 💡 Position encoding is crucial for sequence tasks

### Vision Transformers:
- ✅ Unifies vision and NLP architectures
- ✅ Scalable to large datasets
- ❌ Data-hungry (needs more data than CNNs)
- 💡 Patches = "words" in images

## 📁 Files Description

| File | Description |
|------|-------------|
| `26100117_PA4_1.ipynb` | CLIP implementation notebook |
| `26100117_PA4_1.py` | CLIP script |
| `26100117_PA4_2.ipynb` | Transformers implementation |
| `26100117_PA4_2.py` | Transformers script |

## 🎯 Applications

**CLIP**:
- Zero-shot image classification
- Image-text retrieval
- Content moderation
- Visual question answering

**Transformers**:
- Machine translation
- Text summarization
- Chatbots
- Code generation

**Vision Transformers**:
- Image classification
- Object detection
- Semantic segmentation
- Video understanding

## 🔗 Dependencies

- **PyTorch**: Core deep learning framework
- **Transformers**: HuggingFace library (DistilBERT, BERT)
- **Timm**: PyTorch Image Models (ResNet50)
- **Kagglehub**: Dataset downloading
- **NumPy, Matplotlib**: Data processing and visualization
- **PIL**: Image processing

## 📚 References

- [CLIP Paper](https://arxiv.org/abs/2103.00020) - Learning Transferable Visual Models From Natural Language Supervision
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [ViT Paper](https://arxiv.org/abs/2010.11929) - An Image is Worth 16x16 Words

---

**Training Time**: CLIP training takes ~2-4 hours on GPU, transformers vary by dataset size.
