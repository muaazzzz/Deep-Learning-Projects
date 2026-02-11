# %%
import warnings
warnings.filterwarnings('ignore')
from IPython.core.display import HTML
HTML('<link rel="stylesheet" type="text/css" href="extra/custom_style.css">')

# %% [markdown]
# <hr>
# 
# **<h1><b> Multimodal Learning Experiments</b></h1>**
# 
# <hr>
# 
# %%
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from tqdm import tqdm
!pip install datasets
from datasets import load_dataset
from typing import Tuple
import re
import numpy as np
import pandas as pd
!pip install tiktoken
import tiktoken
import nltk
from dataclasses import dataclass
from nltk.corpus import stopwords
from torchvision.transforms import ToTensor
nltk.download('stopwords')
torch.set_default_device('cuda')
# torch.set_default_device('mps')

# %%

# %% [markdown]
# # Part 2a: Transformer Text Generation

# %% [markdown]
# <h3 style="color: #FFC72C;">A Tale of Transformation: From Data Wars to the Last Knight</h3>
# 
# <p>
# In an era where data was scattered like remnants of forgotten battles, a new hero emerged—much like the legendary warriors from the blockbuster <i>Transformers: The Last Knight</i>. This was no ordinary hero; it was a groundbreaking model that redefined how machines understand language and imagery.
# </p>
# 
# <p>
# Much like the courageous knights fighting to restore honor in a chaotic world, our Transformer model was forged in the fires of innovation. Born from the visionary ideas of Vaswani et al. in <b>"Attention Is All You Need"</b>, it wielded the power of self-attention—a magic that allowed it to see connections across vast expanses of data, transforming confusion into clarity.
# </p>
# 
# <p>
# In a realm where traditional methods faltered, the model stood as a bulwark against disorder. Its layers of encoders and decoders resembled the valiant legions of ancient knights, each unit collaborating to unlock the secrets hidden within streams of raw information. Every token, like a soldier on a battlefield, found its rightful place, ensuring that no detail was lost in the chaos.
# </p>
# 
# <p>
# The saga was one of evolution and resilience. As the model learned to capture context and meaning with every iteration, it mirrored the epic struggles seen in <i>Transformers: The Last Knight</i>—where ancient legacies clashed with modern threats, and heroes rose to defend a future worth fighting for.
# </p>
# 
# <p>
# In this digital epic, the Transformer was not merely a tool but a legend reborn—a guardian of knowledge, destined to lead us into a new era where data and destiny converge. And so, the journey continues, as every byte of information is transformed into a beacon of insight, lighting the path to a smarter, more connected tomorrow.
# </p>

# %% [markdown]
# <figure></figure>
#   <img src="./extra/Prime.jpg" style="width:100%">

# %% [markdown]
# <h3>What is it?</h3>
# 
# <p>
# The Transformer model, introduced in the paper <b>"Attention Is All You Need"</b> by Vaswani et al. (2017),
# is an <b>encoder-decoder</b> neural network primarily used for sequence-to-sequence tasks like machine translation.
# </p>
# 
# <ul>
#     <li>The model consists of an <b>Encoder</b> and <b>Decoder</b>.</li>
#     <li>Instead of recurrence (RNNs), it uses <b>self-attention</b> for global context.</li>
#     <li>Parallelization enables faster training compared to RNNs.</li>
# </ul>
# 
# <h3>Main Components</h3>
# 
# <h3 style="color: #FFC72C;">1. Input Embeddings & Positional Encoding</h3>
# 
# <ul>
#     <li><b>Word Embeddings</b>: Converts words into vectors.</li>
#     <li><b>Pixel Embeddings</b>: Converts pixels into vectors.</li>
#     <li><b>Positional Encoding</b>: Adds position information to word embeddings.</li>
#     <li><b>Tokens</b>: Represent input/output words in the model.</li>
# </ul>
# 
# <h3 style="color: #FFC72C;">2. Encoder (N Layers)</h3>
# 
# <ul>
#     <li><b>Multi-Head Self-Attention</b>: Captures relationships between all tokens.</li>
#     <li><b>Feedforward Network (FFN)</b>: Adds non-linearity for better learning.</li>
#     <li><b>Layer Normalization & Residual Connections</b>: Helps stable training.</li>
# </ul>
# 
# <h3 style="color: #FFC72C;">3. Decoder (N Layers)</h3>
# 
# <ul>
#     <li><b>Masked Multi-Head Attention</b>: Prevents peeking at future tokens.</li>
#     <li><b>Cross-Attention</b>: Attends to encoder outputs.</li>
# </ul>
# 
# <h3 style="color: #FFC72C;">4. Final Output</h3>
# 
# <p>- Decoder outputs logits for a <b>softmax</b> layer, generating predictions.</p>
# 
# <h3>Summary Table</h3>
# 
# <table>
#     <tr>
#         <th>Component</th>
#         <th>Function</th>
#     </tr>
#     <tr>
#         <td>Word Embeddings</td>
#         <td>Converts tokens into dense vectors.</td>
#     </tr>
#     <tr>
#         <td>Pixel Embeddings</td>
#         <td>Converts pixels into dense vectors.</td>
#     </tr>
#     <tr>
#         <td>Positional Encoding</td>
#         <td>Adds sequence order information.</td>
#     </tr>
#     <tr>
#         <td>Encoder</td>
#         <td>Processes input sequences.</td>
#     </tr>
#     <tr>
#         <td>Decoder</td>
#         <td>Generates output sequences.</td>
#     </tr>
#     <tr>
#         <td>Multi-Head Self-Attention</td>
#         <td>Captures relationships between tokens.</td>
#     </tr>
#     <tr>
#         <td>Feedforward Network (FFN)</td>
#         <td>Applies transformations to token representations.</td>
#     </tr>
#     <tr>
#         <td>Layer Normalization & Residuals</td>
#         <td>Stabilizes training and improves gradient flow.</td>
#     </tr>
#     <tr>
#         <td>Masked Attention (Decoder)</td>
#         <td>Prevents looking at future words.</td>
#     </tr>
#     <tr>
#         <td>Final Linear Layer & Softmax</td>
#         <td>Converts hidden states into word probabilities.</td>
#     </tr>
# </table>
# 
# <figure>
#     <img src="extra/figure1.png" alt="Transformer Architecture" style="width: 1200px;"/>
# </figure>

# %% [markdown]
# <hr>
# 
# # <h1> <span style="color:#FFC72C;">**Task 1**</span> | <span style="color: #C0C0C0; font-family: Transformers"> basic Transformers </span>| <span style="color:#DA291C; font-family: Montserrat"> ****</span></h1>  
# 
# 
# <hr>

# %% [markdown]
# <h2 style="color: #FFC72C;">Task Overview</h2>
# <p>
#   In this assignment, you will build a Transformer-based language model using the script of the Transformers Movie (Your choice on the movie hehe).
#   The goal is to implement and integrate key components of the Transformer architecture to perform text generation.
# </p>
# 
# <h3 style="color: #FFC72C;">Key Components</h3>
# <ul>
#   <li><strong>TokenizedDataset</strong>: Reads, cleans, and tokenizes the movie script, providing batches for training.</li>
#   <li><strong>Configuration</strong>: A dataclass that encapsulates hyperparameters such as block size, embedding dimension, head size, number of heads, and more.</li>
#   <li><strong>MHA</strong>: Implements multi-head self-attention with a single QKV projection, causal masking, and dropout.</li>
#   <li><strong>Feedforward</strong>: A three-layer feedforward network with GELU activations and dropout for improved regularization.</li>
#   <li><strong>Block</strong>: Combines the MHA and Feedforward modules with pre-layer normalization and residual connections.</li>
#   <li><strong>Decepticons</strong>: The full Transformer model that integrates word embeddings, positional encodings, stacked Blocks, and a final projection to generate vocabulary logits.</li>
#   <li><strong>Trainer</strong>: Manages the training, evaluation, and autoregressive text generation processes, including gradient clipping and perplexity calculation.</li>
# </ul>
# 
# <h3 style="color: #FFC72C;">Objective</h3>
# <p>
#   The objective is to train a Transformer model on the Transformers Movie script and use it to generate new text.
#   You will implement each module, integrate them into a cohesive model, and then train and evaluate the model using a dedicated Trainer class.
# </p>

# %% [markdown]
# <hr>
# 
# # <h2> <span style="color:#FFC72C;">**Class**</span> | <span style="color: #C0C0C0"> **Tokenized Dataset** </span></h2>  
# 
# <hr>

# %% [markdown]
# <p>
#   Preprocesses the Transformers Movie script by cleaning, normalizing, and tokenizing the text.
# </p>
# 
# <h3 style="color: #FFC72C;">Args</h3>
# <ul>
#   <li><strong>path</strong>: Path to the text file.</li>
#   <li><strong>batch_size</strong>: Number of samples per batch.</li>
#   <li><strong>block_size</strong>: Length of each sample in tokens.</li>
#   <li><strong>encoder_name</strong>: Encoder name (e.g., "gpt2").</li>
#   <li><strong>additional_clean</strong>: (Optional) Apply extra cleaning.</li>
# </ul>
# 
# <h3 style="color: #FFC72C;">Methods</h3>
# <ul>
#   <li>
#     <strong>normalize(txt: str, additional_clean: bool) -&gt; str</strong>:
#     Cleans text by removing newlines, punctuation, digits, URLs, and optionally more.
#   </li>
#   <li>
#     <strong>batch() -&gt; Tuple[torch.Tensor, torch.Tensor]</strong>:
#     Returns a random batch of input tokens <code>x</code> and target tokens <code>y</code> with shapes
#     <code>(batch_size, block_size)</code>.
#   </li>
# </ul>

# %%
class TokenizedDataset:
    def __init__(self, path, batch_size, block_size, encoder_name: str = "gpt2", additional_clean: bool = True):
        """
        Initializes the TokenizedDataset by reading the Autobots Script, normalizing the text,
        and tokenizing it using the specified tiktoken encoder.

        Args:
            path (str): Path to the text file.
            batch_size (int): Number of samples per batch.
            block_size (int): Length of each sample in tokens.
            encoder_name (str, optional): Name of the encoder (default "gpt2").
            additional_clean (bool, optional): Whether to apply additional cleaning.
        """
        # ========================= Write your code below ==========================
        self.batch_size = batch_size
        self.block_size = block_size

        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()

        cleaned_text = self.normalize(text, additional_clean)

        self.encoder = tiktoken.get_encoding(encoder_name)

        self.tokens = self.encoder.encode(cleaned_text)
        print(f"Tokenized {path}: {len(self.tokens)} tokens")

        self.data = torch.tensor(self.tokens, dtype=torch.long)
        # ==========================================================================

    def normalize(self, txt: str, additional_clean: bool = True) -> str:
        """
        Cleans the input text by removing newlines, punctuation, numbers, URLs,
        and more if you want to do it.

        Args:
            txt (str): The input text.
            additional_clean (bool): If True, clean the text further using additional methods.

        Returns:
            str: The cleaned text.
        """
        # ========================= Write your code below ==========================
        txt = re.sub(r'\n+', ' ', txt)
        txt = re.sub(r'\s+', ' ', txt)

        txt = re.sub(r'http\S+|www\.\S+', '', txt)
        txt = re.sub(r'\S+@\S+', '', txt)

        txt = txt.lower()
        txt = re.sub(r'\([^)]*\)', '', txt)
        txt = re.sub(r'^\s*[A-Z][A-Z\s]+:', '', txt, flags=re.MULTILINE)
        txt = re.sub(r'[^\w\s.,!?\'"-]', '', txt)
        txt = re.sub(r'[^\w\s]', '', txt)

        return txt.strip()
        # ==========================================================================

    def batch(self):
        """
        Returns a random batch (x, y) from the tokenized script.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                x: Tensor of shape (batch_size, block_size) with input token IDs.
                y: Tensor of shape (batch_size, block_size) with target token IDs.
        """
        # ========================= Write your code below ==========================
        ix = torch.randint(len(self.data) - self.block_size, (self.batch_size,))

        x = torch.stack([self.data[i:i+self.block_size] for i in ix])
        y = torch.stack([self.data[i+1:i+self.block_size+1] for i in ix])

        return x, y
        # ==========================================================================

# %% [markdown]
# <hr>
# 
# # <h2> <span style="color:#FFC72C;">**Class**</span> | <span style="color: #C0C0C0"> **Configuration** </span></h2>  
# 
# <hr>

# %% [markdown]
# <p>
#   Encapsulates hyperparameters for the Transformer model.
# </p>
# 
# <h3 style="color: #FFC72C;">Attributes</h3>
# <ul>
#   <li><strong>block_size</strong>: Sequence length (T).</li>
#   <li><strong>emb_dim</strong>: Embedding dimension (C).</li>
#   <li><strong>head_size</strong>: Dimension of each attention head.</li>
#   <li><strong>num_heads</strong>: Number of attention heads.</li>
#   <li><strong>num_layers</strong>: Number of transformer layers.</li>
#   <li><strong>vocab_size</strong>: Size of the vocabulary.</li>
#   <li><strong>attn_dropout</strong>: Dropout rate for attention weights.</li>
#   <li><strong>proj_dropout</strong>: Dropout rate after output projection.</li>
#   <li><strong>emb_dropout</strong>: Dropout rate for embeddings.</li>
# </ul>

# %%
@dataclass
class Configuration:
    # ========================= Write your code below ==========================
    block_size: int = 64

    emb_dim: int = 384
    head_size: int = 64
    num_heads: int = 6
    num_layers: int = 6
    vocab_size: int = 50257

    attn_dropout: float = 0.1
    proj_dropout: float = 0.1
    emb_dropout: float = 0.1

    ff_dim_mul: int = 4

    @property 
    def ff_dim(self):
        return self.emb_dim * self.ff_dim_mul
    # ==========================================================================

# %% [markdown]
# <hr>
# 
# # <h2> <span style="color:#FFC72C;">**Class**</span> | <span style="color: #C0C0C0"> **MHA** </span></h2>  
# 
# <hr>

# %% [markdown]
# <h2 style="color: #FFC72C;">MHSA (Multi-Head Self-Attention)</h2>
# <p>
#   Computes scaled dot-product attention with causal masking using a single QKV projection.
# </p>
# 
# <h3 style="color: #FFC72C;">Args</h3>
# <ul>
#   <li>
#     <strong>cfg</strong>: Configuration object containing: <code>emb_dim</code>, <code>head_size</code>, <code>block_size</code>, <code>num_heads</code>, <code>attn_dropout</code>, and <code>proj_dropout</code>.
#   </li>
# </ul>
# 
# <h3 style="color: #FFC72C;">Formulas</h3>
# 
# $$
# \text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
# $$
# 
# <h3 style="color: #FFC72C;">Attributes</h3>
# <ul>
#   <li>
#     <strong>qkv</strong>: Linear layer projecting input to concatenated Q, K, V.
#   </li>
#   <li>
#     <strong>output</strong>: Linear layer projecting concatenated head outputs back to emb_dim.
#   </li>
#   <li>
#     <strong>mask</strong>: Causal mask (lower-triangular) of shape <code>(block_size, block_size)</code>.
#   </li>
#   <li>
#     <strong>dropout</strong>: Dropout applied to attention probabilities and output projection.
#   </li>
# </ul>
# 
# <p>Dimension details:</p>
# 
# $$
# \text{output(qkv)} = \text{ 3} \times{\text{emb\_dim}}
# $$
# 
# $$
# \text{output} = \text{emb\_dim}
# $$
# 
# <h3 style="color: #FFC72C;">Methods</h3>
# <ul>
#   <li>
#     <strong>forward(x: torch.Tensor) -&gt; torch.Tensor</strong>:
#     <br>
#     <em>Input:</em> x of shape as specified below.
#     <br>
#     <em>Output:</em> Tensor of shape as specified below.
#   </li>
# </ul>
# 
# $$
# \text{x} = (B, T, \text{emb\_dim})
# $$
# 
# $$
# \text{output} = (B, T, \text{emb\_dim})
# $$

# %%
class MHA(nn.Module):
    """
    Multi-Head Self-Attention (MHA) module with dropout.

    Projects inputs to queries, keys, and values using one linear layer,
    computes scaled dot-product attention with causal masking and dropout,
    then applies an output projection with dropout.
    """
    def __init__(self, cfg):
        super().__init__()

        # Code to register the mask as a buffer | A buffer is a tensor that is not updated during backpropagation
        # self.register_buffer("mask", torch.tril(torch.ones(self.block_size, self.block_size)))

        # ========================= Write your code below ==========================
        self.emb_dim = cfg.emb_dim
        self.head_size = cfg.head_size
        self.num_heads = cfg.num_heads
        self.block_size = cfg.block_size

        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        self.register_buffer("mask", mask)

        par2 = 3 * cfg.num_heads * cfg.head_size
        self.qkv = nn.Linear(cfg.emb_dim, par2)

        self.output = nn.Linear(cfg.num_heads * cfg.head_size, cfg.emb_dim)

        self.attn_dropout = nn.Dropout(cfg.attn_dropout)
        self.proj_dropout = nn.Dropout(cfg.proj_dropout)
        # ==========================================================================

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Shape (B, T, emb_dim)

        Returns:
            torch.Tensor: Shape (B, T, emb_dim)
        """
        B, T, C = x.shape  # B: batch, T: sequence length, C: emb_dim
        H = self.num_heads
        # Project to Q, K, V and reshape: (B, T, 3 * emb_dim) -> (B, T, H, 3 * head_size)
        # Scaled dot-product attention
        # scores dim: (B, H, T, T)
        # attention dim: (B, H, T, T)
        # Apply dropout to attention probabilities
        # Compute weighted sum of values: (B, H, T, head_size)
        # Concatenate heads: (B, T, H * head_size)
        # Apply dropout after projection

        # ========================= Write your code below ==========================
        qkv = self.qkv(x)

        qkv = qkv.reshape(B, T, 3, H, self.head_size)
        q, k, v = qkv.unbind(dim=2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_size ** 0.5)

        scores = scores.masked_fill(self.mask[:T, :T] == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)

        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)

        out = out.transpose(1, 2)

        out = out.reshape(B, T, H * self.head_size)

        out = self.output(out)
        out = self.proj_dropout(out)

        return out
        # ==========================================================================

# %% [markdown]
# <hr>
# 
# # <h2> <span style="color:#FFC72C;">**Class**</span> | <span style="color: #C0C0C0"> **Feedforward** </span></h2>  
# 
# <hr>

# %% [markdown]
# <p>
#   Processes input embeddings using a multi-layer feedforward neural network block.
#   This block expands the embedding dimension, applies nonlinear activation and dropout, and then projects
#   the representation back to the original embedding dimension.
# </p>
# 
# <h3 style="color: #FFC72C;">Args</h3>
# <ul>
#   <li><strong>config</strong>: A configuration object containing model hyperparameters, such as <code>emb_dim</code>.</li>
# </ul>
# 
# <h3 style="color: #FFC72C;">Attributes</h3>
# <ul>
#   <li>
#     <strong>linear</strong>: Linear layers change dimensions.
#   </li>
#   <li>
#     <strong>dropout</strong>: Dropout layer.
#   </li>
#   <li>
#     <strong>gelu</strong>: GELU activation function.
#   </li>
# </ul>
# 
# <h3 style="color: #FFC72C;">Methods</h3>
# <ul>
#   <li>
#     <strong>forward(x: torch.Tensor) -&gt; torch.Tensor</strong>:  
#     Sequentially applies the linear layers with GELU activations and dropout. The input tensor <code>x</code>
#     with shape <code>(B, T, emb_dim)</code> is transformed , processed,
#     and then reduced back to <code>(B, T, emb_dim)</code> before being returned.
#   </li>
# </ul>

# %%
class Feedforward(nn.Module):
    def __init__(self, config):
        super().__init__()
        # ========================= Write your code below ==========================
        self.linear1 = nn.Linear(config.emb_dim, config.ff_dim)
        self.linear2 = nn.Linear(config.ff_dim, config.ff_dim)
        self.linear3 = nn.Linear(config.ff_dim, config.emb_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.proj_dropout)
        # ==========================================================================

    def forward(self, x):
        # ========================= Write your code below ==========================
        x = self.gelu(self.linear1(x))

        x = self.gelu(self.linear2(x))

        x = self.linear3(x)
        x = self.dropout(x)
        return x
        # ==========================================================================

# %% [markdown]
# <hr>
# 
# # <h2> <span style="color:#FFC72C;">**Class**</span> | <span style="color: #C0C0C0"> **Encoder** </span></h2>  
# 
# <hr>

# %% [markdown]
# <p>
#   This encoder block is designed for processing sequential data in a transformer-based architecture.
#   It combines normalization, self-attention, and a feedforward network to capture and refine contextual information.
# </p>
# 
# <h3 style="color: #FFC72C;">Components</h3>
# <ul>
#   <li><strong>Normalization(nn.LayerNorm):</strong> Standardizes inputs before each major processing step to improve training stability.</li>
#   <li><strong>Multi-Head Self-Attention:</strong> Enables the model to focus on different parts of the sequence simultaneously, capturing long-range dependencies and interactions.</li>
#   <li><strong>Feedforward Network:</strong> Further processes and transforms the representations produced by the attention mechanism to enhance feature learning.</li>
#   <li><strong>Residual Connections with Dropout:</strong> Help maintain gradient flow and prevent overfitting by adding shortcut connections and regularizing the outputs.</li>
# </ul>
# 
# <h3 style="color: #FFC72C;">Purpose</h3>
# <p>
#   The purpose of this block is to enrich the token representations by incorporating contextual information from the entire sequence.
#   The combination of self-attention and feedforward processing, along with normalization and residual paths, allows the model to generate more robust and informative embeddings for downstream tasks.
# </p>

# %%
class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # ========================= Write your code below ==========================
        self.ln1 = nn.LayerNorm(config.emb_dim)
        self.ln2 = nn.LayerNorm(config.emb_dim)

        self.mha = MHA(config)

        self.ff = Feedforward(config)

        self.dropout = nn.Dropout(config.proj_dropout)
        # ==========================================================================

    def forward(self, x):
        # ========================= Write your code below ==========================
        norm_x = self.ln1(x)

        attn_output = self.mha(norm_x)
        x = x + attn_output

        norm_x = self.ln2(x)

        ff_output = self.ff(norm_x)
        x = x + ff_output

        return x
        # ==========================================================================

# %% [markdown]
# <hr>
# 
# # <h2> <span style="color:#FFC72C;">**Class**</span> | <span style="color: #C0C0C0; font-family: Transformers"> Decepticons </span></h2>  
# 
# <hr>

# %% [markdown]
# <h2 style="color: #FFC72C;">Decepticons</h2>
# <p>
#   Implements the full Transformer model for text generation. This model integrates word embeddings, positional embeddings, and a stack of Transformer blocks to produce logits over the vocabulary. The <strong>generate</strong> method uses autoregressive sampling to produce new tokens based on an input prompt.
# </p>
# 
# <h3 style="color: #FFC72C;">Args</h3>
# <ul>
#   <li>
#     <strong>config</strong>: A configuration object with the following attributes:
#     <ul>
#       <li><code>emb_dim</code>: Embedding dimension.</li>
#       <li><code>block_size</code>: Maximum sequence length.</li>
#       <li><code>num_layers</code>: Number of transformer blocks.</li>
#       <li><code>vocab_size</code>: Size of the vocabulary.</li>
#       <li><code>emb_dropout</code>: Dropout rate applied to embeddings.</li>
#     </ul>
#   </li>
# </ul>
# 
# <h3 style="color: #FFC72C;">Attributes</h3>
# <ul>
#   <li><strong>word_embed</strong>: Embedding layer mapping token indices to embeddings (shape: <code>(vocab_size, emb_dim)</code>).</li>
#   <li><strong>position</strong>: Embedding layer providing positional encodings (shape: <code>(block_size, emb_dim)</code>).</li>
#   <li><strong>blocks</strong>: A stack of Transformer blocks (each block includes multi-head attention, feedforward network, and layer normalization).</li>
#   <li><strong>final_layer_norm</strong>: Layer normalization applied after processing through all blocks.</li>
#   <li><strong>logits</strong>: Final linear projection mapping the transformer output to vocabulary logits.</li>
#   <li><strong>emb_dropout</strong>: Dropout applied to the sum of word and positional embeddings.</li>
# </ul>
# 
# <h3 style="color: #FFC72C;">Methods</h3>
# <ul>
#   <li>
#     <strong>forward(idxs: torch.Tensor) -&gt; torch.Tensor</strong>
#     <ul>
#       <li>
#         <strong>Input:</strong> Token indices <code>idxs</code> with shape <code>(B, T)</code>, where <code>B</code> is the batch size and <code>T</code> is the sequence length. It is asserted that <code>T ≤ block_size</code>.
#       </li>
#       <li>
#         <strong>Implementation:</strong>
#         <ul>
#           <li>
#             <em>Word Embeddings:</em> Look up embeddings for each token.
#           </li>
#           <li>
#             <em>Positional Embeddings:</em> Generate position indices with <code>torch.arange(T)</code> and look up positional encodings.
#           </li>
#           <li>
#             <em>Combine and Dropout:</em> Sum word and positional embeddings and apply dropout.
#           </li>
#           <li>
#             <em>Transformer Blocks:</em> Process the combined embeddings sequentially through each block.
#           </li>
#           <li>
#             <em>Final Normalization and Projection:</em> Apply layer normalization, then a linear projection to obtain vocabulary logits.
#           </li>
#         </ul>
#       </li>
#       <li>
#         <strong>Output:</strong> A tensor of vocabulary logits with shape <code>(B, T, vocab_size)</code>.
#       </li>
#     </ul>
#   </li>
#   <li>
#     <strong>generate(idxs: torch.Tensor, max_new_tokens: int = 20) -&gt; torch.Tensor</strong>
#     <ul>
#       <li>
#         <strong>Input:</strong> A tensor of token indices <code>idxs</code> with shape <code>(B, T)</code> and a maximum number of tokens to generate.
#       </li>
#       <li>
#         <strong>Implementation:</strong>
#         <ul>
#           <li>
#             <em>Trimming:</em> For each iteration, trim the input to the last <code>block_size</code> tokens.
#           </li>
#           <li>
#             <em>Forward Pass:</em> Compute logits for the current sequence and select the logits corresponding to the last token.
#           </li>
#           <li>
#             <em>Sampling:</em> Apply softmax to the logits, sample the next token, and concatenate it to the input.
#           </li>
#           <li>
#             Repeat for <code>max_new_tokens</code> iterations.
#           </li>
#         </ul>
#       </li>
#       <li>
#         <strong>Output:</strong> A tensor containing the original token indices appended with the generated tokens.
#       </li>
#     </ul>
#   </li>
# </ul>
# 
# $$
# \text{logits} = \text{Linear}\Bigl(\text{LayerNorm}\bigl(\text{Block}N\bigl(\dots \text{Block}1\bigl(X{\text{word}} + X{\text{pos}}\bigr)\bigr)\bigr)\Bigr)
# $$

# %%
class Decepticons(nn.Module):
    def __init__(self, config):
        super().__init__()
        # ========================= Write your code below ==========================
        self.block_size = config.block_size
        self.emb_dim = config.emb_dim
        self.vocab_size = config.vocab_size
        self.word_embed = nn.Embedding(config.vocab_size, config.emb_dim)
        self.position = nn.Embedding(config.block_size, config.emb_dim)
        self.emb_dropout = nn.Dropout(config.emb_dropout)
        self.blocks = nn.ModuleList()

        for _ in range(config.num_layers):
            self.blocks.append(Encoder(config))

        self.final_layer_norm = nn.LayerNorm(config.emb_dim)
        self.logits = nn.Linear(config.emb_dim, config.vocab_size)
        self.apply(self._init_weights)
        # ==========================================================================

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idxs):
        # idxs: (B, T)
        B, T = idxs.shape
        assert T <= self.block_size, f"Sequence length {T} exceeds block size {self.block_size}"

        # Word embeddings: (B, T, emb_dim)
        # Position indices and embeddings: (1, T, emb_dim)
        # Combine embeddings and apply dropout
        # Pass through each transformer block
        # Apply final layer norm and project to vocabulary logits: (B, T, vocab_size)

        # ========================= Write your code below ==========================

        word_emb = self.word_embed(idxs)

        pos_idxs = torch.arange(0, T, dtype=torch.long, device=idxs.device).unsqueeze(0)
        pos_emb = self.position(pos_idxs)

        x = word_emb + pos_emb
        x = self.emb_dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_layer_norm(x)

        logits = self.logits(x)

        return logits
        # ==========================================================================

    @torch.no_grad()
    def generate(self, idxs, max_new_tokens=20):
        """
        Autoregressively generate new tokens given an initial sequence of token indices.

        Args:
            idxs (torch.Tensor): Tensor of shape (B, T) containing token indices.
            max_new_tokens (int): Maximum number of new tokens to generate.

        Returns:
            torch.Tensor: Tensor with the original indices appended with generated tokens.
        """
        # ========================= Write your code below ==========================
        B, T = idxs.shape

        for _ in range(max_new_tokens):
            idxs_cond = idxs if idxs.size(1) <= self.block_size else idxs[:, -self.block_size:]

            logits = self(idxs_cond)

            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)

            next_idx = torch.multinomial(probs, num_samples=1)

            idxs = torch.cat([idxs, next_idx], dim=1)

        return idxs
        # ==========================================================================

# %% [markdown]
# <hr>
# 
# # <h2> <span style="color:#FFC72C;">**Class**</span> | <span style="color: #C0C0C0"> **Trainer** </span></h2>  
# 
# <hr>

# %% [markdown]
# <h2 style="color: #FFC72C;">Trainer</h2>
# <p>
#   Manages training, evaluation, and text generation for the Transformer model.
# </p>
# 
# <h3 style="color: #FFC72C;">Args</h3>
# <ul>
#   <li>
#     <strong>config</strong>: Configuration object with hyperparameters.
#   </li>
#   <li>
#     <strong>dataset</strong>: Instance of TokenizedDataset.
#   </li>
#   <li>
#     <strong>model</strong>: The Transformer model (e.g., Decepticons).
#   </li>
#   <li>
#     <strong>lr</strong>: Learning rate for the optimizer.
#   </li>
#   <li>
#     <strong>num_train_batches</strong>: Number of batches per training epoch.
#   </li>
#   <li>
#     <strong>num_eval_batches</strong>: Number of batches used during evaluation.
#   </li>
# </ul>
# 
# <h3 style="color: #FFC72C;">Attributes</h3>
# <ul>
#   <li><strong>optimizer</strong>: Adam optimizer.</li>
#   <li><strong>criterion</strong>: Cross-entropy loss function.</li>
#   <li>
#     <strong>num_train_batches</strong> and <strong>num_eval_batches</strong>: Control the number of batches used during training and evaluation.
#   </li>
# </ul>
# 
# <h3 style="color: #FFC72C;">Methods</h3>
# <ul>
#   <li>
#     <strong>train_epoch(epoch: int) -&gt; float</strong>:
#     Trains the model for one epoch and returns the average training loss.
#   </li>
#   <li>
#     <strong>evaluate() -&gt; Tuple[float, float]</strong>:
#     Evaluates the model on a subset of data and returns the average loss and perplexity.
#   </li>
#   <li>
#     <strong>train(epochs: int)</strong>:
#     Runs the complete training loop over a specified number of epochs.
#     You can use gradient clipping as well (Recommended).
#   </li>
#   <li>
#     <strong>generate_text(start_text: str, max_tokens: int = 50) -&gt; str</strong>:
#     Autoregressively generates text starting from a prompt.
#   </li>
# </ul>
# 
# <p>Key formulas:</p>
# 
# $$
# \text{Loss} = \text{CrossEntropy}(\text{logits}, \text{targets})
# $$
# 
# $$
# \text{Perplexity} = \exp(\text{Loss})
# $$

# %%
class Trainer:
    def __init__(self, config, dataset, model, lr, num_train_batches, num_eval_batches, device="cuda"):
        """
        Initializes the Trainer.

        Args:
            config: Configuration object with hyperparameters.
            dataset: An instance of TokenizedDataset.
            model: The Decepticons Transformer model.
            lr (float, optional): Learning rate for the optimizer.
            device (torch.device, optional): Device to use for training. Defaults to CUDA if available.
            num_train_batches (int, optional): Number of batches to use for each training epoch.
            num_eval_batches (int, optional): Number of batches to use during evaluation.
        """
        # ========================= Write your code below ==========================
        self.config = config
        self.dataset = dataset
        self.model = model
        self.lr = lr
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_train_batches = num_train_batches
        self.num_eval_batches = num_eval_batches

        self.model = self.model.to(self.device)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        self.criterion = nn.CrossEntropyLoss()

        self.encoder = dataset.encoder
        # ==========================================================================

    def train_epoch(self, epoch):
        """
        Runs one epoch of training.

        Args:
            epoch (int): Current epoch index.

        Returns:
            float: Average training loss for the epoch.
        """
        # ========================= Write your code below ==========================
        self.model.train()
        total_loss = 0.0

        progress_bar = tqdm(range(self.num_train_batches), desc=f"Epoch {epoch+1}")

        for _ in progress_bar:
            x, y = self.dataset.batch()
            x, y = x.to(self.device), y.to(self.device)

            logits = self.model(x)

            loss = self.criterion(logits.view(-1, self.config.vocab_size), y.view(-1))

            self.optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()

            total_loss += loss.item()

            progress_bar.set_postfix(loss=loss.item())

        return total_loss / self.num_train_batches
        # ==========================================================================

    def evaluate(self):
        """
        Evaluates the model on a subset of the data.

        Returns:
            Tuple[float, float]: Average evaluation loss and perplexity.
        """
        # ========================= Write your code below ==========================
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for _ in range(self.num_eval_batches):
                x, y = self.dataset.batch()
                x, y = x.to(self.device), y.to(self.device)

                logits = self.model(x)

                loss = self.criterion(logits.view(-1, self.config.vocab_size), y.view(-1))

                total_loss += loss.item()

        avg_loss = total_loss / self.num_eval_batches
        perplexity = math.exp(avg_loss)

        return avg_loss, perplexity
        # ==========================================================================

    def train(self, epochs):
        """
        Runs the full training loop over the specified number of epochs.

        Args:
            epochs (int): Total number of training epochs.
        """
        # ========================= Write your code below ==========================
        best_loss = float('inf')

        for epoch in range(epochs):
            train_loss = self.train_epoch(epoch)

            eval_loss, perplexity = self.evaluate()

            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Eval Loss: {eval_loss:.4f} | Perplexity: {perplexity:.4f}")
        # ==========================================================================

    def generate_text(self, start_text, max_tokens=50):
        """
        Autoregressively generates text starting from a given prompt.

        Args:
            start_text (str): The input prompt text.
            max_tokens (int, optional): Maximum number of new tokens to generate.

        Returns:
            str: The generated text (decoded from token IDs).
        """
        # ========================= Write your code below ==========================
        self.model.eval()

        input_ids = torch.tensor(self.encoder.encode(start_text), dtype=torch.long).unsqueeze(0).to(self.device)

        generated_ids = self.model.generate(input_ids, max_new_tokens=max_tokens)

        generated_text = self.encoder.decode(generated_ids[0].tolist())

        return generated_text
        # ==========================================================================

# %% [markdown]
# <hr>
# 
# # <h2> <span style="color:#FFC72C;">**Training**</span> | <span style="color: #C0C0C0"> **Initialize Dataset and Configuration** </span></h2>  
# 
# <hr>

# %%
# ========================= Write your code below ==========================
torch.manual_seed(42)

script_path = "/content/transformers_last_night.txt"
batch_size = 32
block_size = 64

dataset = TokenizedDataset(path=script_path,batch_size=batch_size,block_size=block_size,encoder_name="gpt2",additional_clean=True)

config = Configuration()
# ==========================================================================

# %% [markdown]
# <hr>
# 
# # <h2> <span style="color:#FFC72C;">**Training**</span> | <span style="color: #C0C0C0"> **Initialize Model, Trainer and Train** </span></h2>  
# 
# <hr>

# %%
# ========================= Write your code below ==========================
model = Decepticons(config)

trainer = Trainer(
    config=config,
    dataset=dataset,
    model=model,
    lr=3e-4,
    num_train_batches=100,
    num_eval_batches=10
)

num_epochs = 6
trainer.train(num_epochs)
# ==========================================================================

# %% [markdown]
# <hr>
# 
# # <h2> <span style="color:#FFC72C;">**Generation**</span> | <span style="color: #C0C0C0"> **Generate a sentence** </span></h2>  
# 
# <hr>

# %%
# ========================= Write your code below ==========================
prompt = "Optimus Prime"

generated_text = trainer.generate_text(prompt, max_tokens=100)
print(f"\n\nGenerated: {generated_text}")
# ==========================================================================

# %% [markdown]
# <hr>
# 
# # Part 2b: Vision Transformers for Classification
# 
# <hr>

# %% [markdown]
# <hr>
# 
# # <h2> <span style="color:#FFC72C;">**Class**</span> | <span style="color: #C0C0C0"> **Patchifier** </span></h2>  
# 
# <hr>

# %% [markdown]
# <p>
#   This class is designed to split a batch of square images into smaller patches. Each image is divided into a grid,
#   where the total number of patches per image is <code>n_patches²</code>.
# </p>
# 
# <h3 style="color: #FFC72C;">Attributes</h3>
# <ul>
#   <li>
#     <strong>n_patches</strong>: Number of patches per image dimension. The total number of patches per image will be <code>n_patches²</code>.
#   </li>
# </ul>
# 
# <h3 style="color: #FFC72C;">Methods</h3>
# 
# <h4 style="color: #FFC72C;"><code>__init__(n_patches: int)</code></h4>
# <ul>
#   <li>
#     Initializes the <strong>Patchifier</strong> with the desired number of patches per image dimension.
#   </li>
#   <li>
#     <strong>Input Parameter</strong>:
#     <ul>
#       <li>
#         <code>n_patches</code>: An integer representing the number of patches per image dimension.
#       </li>
#     </ul>
#   </li>
# </ul>
# 
# <h4 style="color: #FFC72C;"><code>patchify(images: torch.Tensor) -&gt; torch.Tensor</code></h4>
# <ul>
#   <li>
#     Splits a batch of images into patches.
#   </li>
#   <li>
#     <strong>Input Parameters</strong>:
#     <ul>
#       <li>
#         <code>images</code>: A tensor containing a batch of images with shape <code>(n, c, h, w)</code>, where:
#         <ul>
#           <li><em>n</em> is the batch size (number of images),</li>
#           <li><em>c</em> is the number of channels,</li>
#           <li><em>h</em> and <em>w</em> are the height and width of the images respectively.</li>
#         </ul>
#       </li>
#     </ul>
#   </li>
#   <li>
#     <strong>Return</strong>:
#     <ul>
#       <li>
#         A tensor of shape <code>(n, n_patches², patch_size * patch_size * c)</code> containing the flattened patches,
#         where <code>patch_size</code> is computed as <code>h / n_patches</code>.
#       </li>
#     </ul>
#   </li>
# </ul>

# %%
class Patchifier:
    """
    A class for splitting images into patches.

    Attributes:
        n_patches (int): Number of patches per dimension. The total number of patches
                         per image will be n_patches**2.
    """

    def __init__(self, n_patches: int):
        """
        Initialize the Patchifier with the desired number of patches per dimension.

        Args:
            n_patches (int): Number of patches per image dimension.
        """
        # ========================= Write your code below ==========================
        self.n_patches = n_patches
        # ==========================================================================

    def patchify(self, images: torch.Tensor) -> torch.Tensor:
        """
        Splits a batch of images into patches.

        Args:
            images (torch.Tensor): A tensor of shape (n, c, h, w), where n is the number
                                   of images, c is the number of channels, and h, w are
                                   the height and width of the images respectively.

        Returns:
            torch.Tensor: A tensor of shape (n, n_patches**2, patch_size * patch_size * c)
                          containing the flattened patches.
        """
        n, c, h, w = images.shape
        # ========================= Write your code below ==========================
        patch_size = h // self.n_patches

        patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)

        patches = patches.contiguous().view(
            n, c, self.n_patches, self.n_patches, patch_size, patch_size
        )
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()

        flattened_patches = patches.view(
            n, self.n_patches**2, c * patch_size * patch_size
        )

        return flattened_patches
        # ==========================================================================

# %% [markdown]
# <hr>
# 
# # <h2> <span style="color:#FFC72C;">**Class**</span> | <span style="color: #C0C0C0"> **PositionalEmbeddings** </span></h2>  
# 
# <hr>

# %% [markdown]
# <p>
#   This class is responsible for generating sinusoidal positional embeddings for a given sequence length and embedding dimension.
#   These embeddings help encode the positional information of tokens in a sequence.
# </p>
# 
# <h3 style="color: #FFC72C;">Attributes</h3>
# <ul>
#   <li>
#     <strong>sequence_length</strong>: Length of the input sequence.
#   </li>
#   <li>
#     <strong>d</strong>: Dimension of the embeddings.
#   </li>
#   <li>
#     <strong>embeddings</strong>: The computed positional embeddings as a <code>torch.Tensor</code>.
#   </li>
# </ul>
# 
# <h3 style="color: #FFC72C;">Methods</h3>
# 
# <h4 style="color: #FFC72C;"><code>__init__(sequence_length: int, d: int)</code></h4>
# <ul>
#   <li>
#     Initializes the <strong>PositionalEmbeddings</strong> class with the specified sequence length and embedding dimension.
#   </li>
#   <li>
#     <strong>Input Parameters</strong>:
#     <ul>
#       <li>
#         <code>sequence_length</code>: An integer representing the length of the sequence.
#       </li>
#       <li>
#         <code>d</code>: An integer representing the embedding dimension.
#       </li>
#     </ul>
#   </li>
# </ul>
# 
# <h4 style="color: #FFC72C;"><code>get_embeddings() -&gt; torch.Tensor</code></h4>
# <ul>
#   <li>
#     Computes and retrieves the positional embeddings.
#   </li>
#   <li>
#     <strong>Return</strong>:
#     <ul>
#       <li>
#         A <code>torch.Tensor</code> of shape <code>(sequence_length, d)</code> containing the sinusoidal positional embeddings.
#       </li>
#     </ul>
#   </li>
# </ul>

# %%
class PositionalEmbeddings:
    """
    A class for generating positional embeddings for a given sequence length and embedding dimension.

    Attributes:
        sequence_length (int): Length of the input sequence.
        d (int): Dimension of the embeddings.
        embeddings (torch.Tensor): The computed positional embeddings.
    """

    def __init__(self, sequence_length: int, d: int):
        """
        Initialize the PositionalEmbeddings class.

        Args:
            sequence_length (int): The length of the sequence.
            d (int): The embedding dimension.
        """
        # ========================= Write your code below ==========================
        self.sequence_length = sequence_length
        self.d = d
        self.embeddings = self._compute_embeddings()
        # ==========================================================================

    def _compute_embeddings(self) -> torch.Tensor:
        positions = torch.arange(self.sequence_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d, 2) * -(math.log(10000.0) / self.d))

        embeddings = torch.zeros(self.sequence_length, self.d)

        embeddings[:, 0::2] = torch.sin(positions * div_term)
        embeddings[:, 1::2] = torch.cos(positions * div_term)

        return embeddings

    def get_embeddings(self) -> torch.Tensor:
        """
        Retrieve the computed positional embeddings.

        Returns:
            torch.Tensor: The positional embeddings tensor.
        """
        # ========================= Write your code below ==========================
        return self.embeddings
        # ==========================================================================


# %% [markdown]
# <hr>
# 
# # <h2> <span style="color:#FFC72C;">**Class**</span> | <span style="color: #C0C0C0"> **MHA** </span></h2>  
# 
# <hr>

# %% [markdown]
# <p>
#   This module splits the input into multiple attention heads, computes scaled dot-product attention for each head,
#   and concatenates the results.
# </p>
# 
# <h3 style="color: #FFC72C;">Args</h3>
# <ul>
#   <li>
#     <strong>d</strong>: Input embedding dimension.
#   </li>
#   <li>
#     <strong>n_heads</strong>: (Optional) Number of attention heads. Default is 2.
#   </li>
# </ul>
# 
# <h3 style="color: #FFC72C;">Attributes</h3>
# <ul>
#   <li>
#     <strong>d_head</strong>: Dimension of each attention head.
#   </li>
#   <li>
#     <strong>q_mappings</strong>: List of linear layers for computing queries.
#   </li>
#   <li>
#     <strong>k_mappings</strong>: List of linear layers for computing keys.
#   </li>
#   <li>
#     <strong>v_mappings</strong>: List of linear layers for computing values.
#   </li>
#   <li>
#     <strong>softmax</strong>: Softmax function applied over the attention scores.
#   </li>
#   <li>
#     <strong>scale</strong>: Precomputed scaling factor used in scaled dot-product attention.
#   </li>
# </ul>
# 
# <h3 style="color: #FFC72C;">Methods</h3>
# 
# <h4 style="color: #FFC72C;"><code>__init__(d: int, n_heads: int = 2)</code></h4>
# <ul>
#   <li>
#     Initializes the <strong>MHA</strong> module.
#   </li>
#   <li>
#     <strong>Input Parameters</strong>:
#     <ul>
#       <li>
#         <code>d</code>: An integer representing the input embedding dimension.
#       </li>
#       <li>
#         <code>n_heads</code>: (Optional) An integer representing the number of attention heads. Defaults to 2.
#       </li>
#     </ul>
#   </li>
#   <li>
#     Asserts that the input dimension <code>d</code> is divisible by <code>n_heads</code> and computes <code>d_head</code>.
#   </li>
# </ul>
# 
# <h4 style="color: #FFC72C;"><code>forward(sequences: torch.Tensor) -&gt; torch.Tensor</code></h4>
# <ul>
#   <li>
#     Performs the forward pass of the multi-head attention module.
#   </li>
#   <li>
#     <strong>Input Parameters</strong>:
#     <ul>
#       <li>
#         <code>sequences</code>: A tensor of shape <code>(batch_size, seq_length, d)</code> representing the input sequences.
#       </li>
#     </ul>
#   </li>
#   <li>
#     <strong>Return</strong>:
#     <ul>
#       <li>
#         A tensor of shape <code>(batch_size, seq_length, d)</code> containing the output after applying multi-head attention.
#       </li>
#     </ul>
#   </li>
# </ul>

# %%
class MHA(nn.Module):
    """
    Multi-Head Attention module.

    This module splits the input into multiple attention heads, computes scaled dot-product
    attention for each head, and concatenates the results.

    Args:
        d (int): Input embedding dimension.
        n_heads (int, optional): Number of attention heads. Default is 2.

    Attributes:
        d_head (int): Dimension of each attention head.
        q_mappings (nn.ModuleList): List of linear layers for computing queries.
        k_mappings (nn.ModuleList): List of linear layers for computing keys.
        v_mappings (nn.ModuleList): List of linear layers for computing values.
        softmax (nn.Softmax): Softmax function applied over the attention scores.
    """

    def __init__(self, d: int, n_heads: int = 2):
        super(MHA, self).__init__()
        self.d = d
        self.n_heads = n_heads
        assert d % n_heads == 0, "Number of heads must evenly divide input dimension"

        # ========================= Write your code below ==========================
        self.d_head = d // n_heads

        self.q_mappings = nn.ModuleList()
        self.k_mappings = nn.ModuleList()
        self.v_mappings = nn.ModuleList()

        for _ in range(n_heads):
            self.q_mappings.append(nn.Linear(d, self.d_head))
            self.k_mappings.append(nn.Linear(d, self.d_head))
            self.v_mappings.append(nn.Linear(d, self.d_head))
                
        self.output_linear = nn.Linear(d, d)

        self.softmax = nn.Softmax(dim=-1)

        self.scale = 1.0 / math.sqrt(self.d_head)
        # ==========================================================================

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the multi-head attention module.

        Args:
            sequences (torch.Tensor): Input tensor of shape (batch_size, seq_length, d).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length, d)
                          after applying multi-head attention.
        """

        # Process each sequence in the batch individually
        # Compute query, key, and value for the current head
        # Compute scaled dot-product attention
        # Concatenate outputs from all heads along the last dimension
        # Stack outputs for all sequences in the batch along a new dimension (batch dimension)

        # ========================= Write your code below ==========================
        batch_size, seq_length, _ = sequences.shape

        head_outputs = []

        for head in range(self.n_heads):
            q = self.q_mappings[head](sequences)
            k = self.k_mappings[head](sequences)
            v = self.v_mappings[head](sequences)

            attention_scores = torch.bmm(q, k.transpose(1, 2)) * self.scale

            attention_weights = self.softmax(attention_scores)

            head_output = torch.bmm(attention_weights, v)

            head_outputs.append(head_output)

        multi_head_output = torch.cat(head_outputs, dim=-1)

        output = self.output_linear(multi_head_output)

        return output
        # ==========================================================================

# %% [markdown]
# <hr>
# 
# # <h2> <span style="color:#FFC72C;">**Class**</span> | <span style="color: #C0C0C0"> **Encoder** </span></h2>  
# 
# <hr>

# %% [markdown]
# <p>
#   Transformer Encoder Block with Multi-Head Self-Attention and MLP. This encoder applies layer normalization
#   followed by multi-head self-attention and a feed-forward network (MLP), with residual connections at each stage.
# </p>
# 
# <h3 style="color: #FFC72C;">Args</h3>
# <ul>
#   <li>
#     <strong>hidden_d</strong>: The hidden embedding dimension.
#   </li>
#   <li>
#     <strong>n_heads</strong>: Number of attention heads.
#   </li>
#   <li>
#     <strong>mlp_ratio</strong>: (Optional) Expansion factor for the MLP. Default is 4.
#   </li>
# </ul>
# 
# <h3 style="color: #FFC72C;">Attributes</h3>
# <ul>
#   <li>
#     <strong>norm1</strong>: <code>nn.LayerNorm</code> applied before self-attention.
#   </li>
#   <li>
#     <strong>mhsa</strong>: Multi-head self-attention module.
#   </li>
#   <li>
#     <strong>norm2</strong>: <code>nn.LayerNorm</code> applied before the MLP.
#   </li>
#   <li>
#     <strong>mlp</strong>: Feed-forward network implemented as an <code>nn.Sequential</code> module.
#   </li>
# </ul>
# 
# <h3 style="color: #FFC72C;">Methods</h3>
# 
# <h4 style="color: #FFC72C;"><code>__init__(hidden_d: int, n_heads: int, mlp_ratio: int = 4)</code></h4>
# <ul>
#   <li>
#     Initializes the <strong>Encoder</strong> block with the specified hidden dimension, number of attention heads,
#     and MLP expansion ratio.
#   </li>
#   <li>
#     <strong>Input Parameters</strong>:
#     <ul>
#       <li>
#         <code>hidden_d</code>: An integer representing the hidden embedding dimension.
#       </li>
#       <li>
#         <code>n_heads</code>: An integer representing the number of attention heads.
#       </li>
#       <li>
#         <code>mlp_ratio</code>: (Optional) An integer representing the expansion factor for the MLP.
#       </li>
#     </ul>
#   </li>
# </ul>
# 
# <h4 style="color: #FFC72C;"><code>forward(x: torch.Tensor) -&gt; torch.Tensor</code></h4>
# <ul>
#   <li>
#     Performs the forward pass of the encoder block.
#   </li>
#   <li>
#     <strong>Input Parameters</strong>:
#     <ul>
#       <li>
#         <code>x</code>: A tensor of shape <code>(batch_size, seq_length, hidden_d)</code> representing the input.
#       </li>
#     </ul>
#   </li>
#   <li>
#     <strong>Return</strong>:
#     <ul>
#       <li>
#         A tensor of shape <code>(batch_size, seq_length, hidden_d)</code> after applying multi-head self-attention,
#         MLP, and residual connections.
#       </li>
#     </ul>
#   </li>
# </ul>

# %%
class Encoder(nn.Module):
    """
    Transformer Encoder Block with Multi-Head Self-Attention and MLP.

    This encoder applies layer normalization followed by multi-head self-attention and a
    feed-forward network (MLP), with residual connections at each stage.

    Args:
        hidden_d (int): The hidden embedding dimension.
        n_heads (int): Number of attention heads.
        mlp_ratio (int, optional): Expansion factor for the MLP. Default is 4.

    Attributes:
        norm1 (nn.LayerNorm): Layer normalization applied before self-attention.
        mhsa (MHA): Multi-head self-attention module.
        norm2 (nn.LayerNorm): Layer normalization applied before the MLP.
        mlp (nn.Sequential): Feed-forward network.
    """

    def __init__(self, hidden_d: int, n_heads: int, mlp_ratio: int = 4):
        super(Encoder, self).__init__()
        # ========================= Write your code below ==========================
        self.norm1 = nn.LayerNorm(hidden_d)

        self.mhsa = MHA(hidden_d, n_heads)

        self.norm2 = nn.LayerNorm(hidden_d)

        mlp_hidden_dim = hidden_d * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_d)
        )
        # ==========================================================================

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the encoder block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, hidden_d).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length, hidden_d).
        """
        # Apply pre-normalization and multi-head self-attention, then add a residual connection.
        # Apply layer normalization, the feed-forward network, and add the residual connection.

        # ========================= Write your code below ==========================
        norm1_output = self.norm1(x)

        attn_output = self.mhsa(norm1_output)

        x = x + attn_output

        norm2_output = self.norm2(x)

        mlp_output = self.mlp(norm2_output)
        x = x + mlp_output

        return x
        # ==========================================================================


# %% [markdown]
# <hr>
# 
# # <h2> <span style="color:#FFC72C;">**Class**</span> | **Vision** <span style="color: #C0C0C0; font-family: Transformers"> Transformer </span></h2>  
# 
# <hr>

# %% [markdown]
# <p>
#   VisionTransformer model for image classification. This model splits the input images into patches, projects them to a hidden dimension,
#   appends a learnable class token and positional embeddings, processes them through a series of encoder blocks, and outputs classification logits.
# </p>
# 
# <h3 style="color: #FFC72C;">Args</h3>
# <ul>
#   <li>
#     <strong>chw</strong>: Tuple representing (channels, height, width) of the input images.
#   </li>
#   <li>
#     <strong>n_patches</strong>: (Optional) Number of patches per image dimension. Default is 7.
#   </li>
#   <li>
#     <strong>n_blocks</strong>: (Optional) Number of encoder blocks. Default is 2.
#   </li>
#   <li>
#     <strong>hidden_d</strong>: (Optional) Hidden dimension size. Default is 8.
#   </li>
#   <li>
#     <strong>n_heads</strong>: (Optional) Number of attention heads for the encoder. Default is 2.
#   </li>
#   <li>
#     <strong>out_d</strong>: (Optional) Output dimension (number of classes). Default is 10.
#   </li>
# </ul>
# 
# <h3 style="color: #FFC72C;">Attributes</h3>
# <ul>
#   <li>
#     <strong>chw</strong>: The input image dimensions as a tuple (channels, height, width).
#   </li>
#   <li>
#     <strong>n_patches</strong>: Number of patches per image dimension.
#   </li>
#   <li>
#     <strong>patch_size</strong>: The number of pixels per patch (flattened).
#   </li>
#   <li>
#     <strong>linear_mapper</strong>: A linear layer that projects flattened patches to the hidden dimension.
#   </li>
#   <li>
#     <strong>class_token</strong>: A learnable parameter that is prepended to the sequence of patch tokens.
#   </li>
#   <li>
#     <strong>positional_embeddings</strong>: Positional embeddings added to the patch tokens to encode spatial information.
#   </li>
#   <li>
#     <strong>blocks</strong>: A list of encoder blocks (each implementing multi-head self-attention and MLP).
#   </li>
#   <li>
#     <strong>mlp</strong>: A final linear layer used for classification.
#   </li>
# </ul>
# 
# <h3 style="color: #FFC72C;">Methods</h3>
# 
# <h4 style="color: #FFC72C;"><code>__init__(chw: Tuple[int, int, int], n_patches: int = 7, n_blocks: int = 2, hidden_d: int = 8, n_heads: int = 2, out_d: int = 10)</code></h4>
# <ul>
#   <li>
#     Initializes the <strong>VisionTransformer</strong> model with the given image dimensions, number of patches, encoder blocks, hidden dimension,
#     number of attention heads, and output dimension.
#   </li>
#   <li>
#     <strong>Input Parameters</strong>:
#     <ul>
#       <li>
#         <code>chw</code>: Tuple of (channels, height, width) of the input images.
#       </li>
#       <li>
#         <code>n_patches</code>: An integer specifying the number of patches per image dimension.
#       </li>
#       <li>
#         <code>n_blocks</code>: An integer specifying the number of encoder blocks.
#       </li>
#       <li>
#         <code>hidden_d</code>: An integer representing the hidden dimension size.
#       </li>
#       <li>
#         <code>n_heads</code>: An integer representing the number of attention heads.
#       </li>
#       <li>
#         <code>out_d</code>: An integer representing the output dimension (number of classes).
#       </li>
#     </ul>
#   </li>
# </ul>
# 
# <h4 style="color: #FFC72C;"><code>forward(images: torch.Tensor) -&gt; torch.Tensor</code></h4>
# <ul>
#   <li>
#     Executes the forward pass of the VisionTransformer model.
#   </li>
#   <li>
#     <strong>Input Parameters</strong>:
#     <ul>
#       <li>
#         <code>images</code>: A tensor of shape <code>(batch_size, C, H, W)</code> representing the input images.
#       </li>
#     </ul>
#   </li>
#   <li>
#     <strong>Return</strong>:
#     <ul>
#       <li>
#         A tensor of shape <code>(batch_size, out_d)</code> containing the output logits for classification.
#       </li>
#     </ul>
#   </li>
# </ul>

# %%
class VisionTransformer(nn.Module):
    """
    VisionTransformer model for image classification.

    This model splits the input images into patches, projects them to a hidden dimension,
    appends a learnable class token and positional embeddings, processes them through a series
    of encoder blocks, and outputs classification logits.

    Args:
        chw (Tuple[int, int, int]): Tuple representing (channels, height, width) of the input images.
        n_patches (int, optional): Number of patches per image dimension. Default is 7.
        n_blocks (int, optional): Number of encoder blocks. Default is 2.
        hidden_d (int, optional): Hidden dimension size. Default is 8.
        n_heads (int, optional): Number of attention heads for the encoder. Default is 2.
        out_d (int, optional): Output dimension (number of classes). Default is 10.
    """

    def __init__(self, chw: Tuple[int, int, int], n_patches: int = 7, n_blocks: int = 2,
                 hidden_d: int = 8, n_heads: int = 2, out_d: int = 10):
        super(VisionTransformer, self).__init__()
        self.chw = chw  # (C, H, W)
        # ========================= Write your code below ==========================
        self.n_patches = n_patches

        c, h, w = chw
        patch_size = h // n_patches
        self.patch_size = patch_size

        self.patchifier = Patchifier(n_patches)

        patch_dim = c * patch_size * patch_size
        self.linear_mapper = nn.Linear(patch_dim, hidden_d)

        self.class_token = nn.Parameter(torch.randn(1, 1, hidden_d))

        self.register_buffer(
            "pos_embeddings",
            PositionalEmbeddings(n_patches * n_patches + 1, hidden_d).get_embeddings()
        )

        self.blocks = nn.ModuleList([
            Encoder(hidden_d, n_heads) for _ in range(n_blocks)
        ])

        self.mlp = nn.Linear(hidden_d, out_d)

        self.norm = nn.LayerNorm(hidden_d)
        # ==========================================================================

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the VisionTransformer model.

        Args:
            images (torch.Tensor): Input tensor of shape (batch_size, C, H, W).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, out_d).
        """
        # Unfold images into patches. Each patch is extracted by sliding a window
        # over the height and width dimensions.
        # Map patches to the hidden dimension
        # Prepend the class token to each batch element
        # Add positional embeddings
        # Process tokens through each encoder block
        # Extract the representation corresponding to the class token
        # Classification output


        # ========================= Write your code below ==========================
        batch_size = images.shape[0]

        patches = self.patchifier.patchify(images)

        patch_embeddings = self.linear_mapper(patches)

        class_tokens = self.class_token.expand(batch_size, -1, -1)
        x = torch.cat([class_tokens, patch_embeddings], dim=1)

        x = x + self.pos_embeddings

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        class_token_final = x[:, 0]

        logits = self.mlp(class_token_final)

        return logits
        # ==========================================================================

# %% [markdown]
# <hr>
# 
# # <h2> <span style="color:#FFC72C;">**Class**</span> | <span style="color: #C0C0C0"> **Visualizer (Helper)** </span></h2>  
# 
# <hr>

# %%
class Visualizer:
    """
    A class responsible for creating directories, saving figures, and visualizing image batches.
    """
    def __init__(self, home_dir: str):
        """
        Args:
            home_dir (str): Base directory to store results.
        """
        self.home_dir = home_dir
        self.device = "cuda"

    def get_dir(self) -> str:
        """Creates and returns a directory based on the current date and time."""
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        current_date = now.strftime("%d_%m_%Y")
        curr_dir = os.path.join(self.home_dir, current_date)
        if not os.path.isdir(curr_dir):
            os.mkdir(curr_dir)
        curr_dir = os.path.join(curr_dir, current_time)
        if not os.path.isdir(curr_dir):
            os.mkdir(curr_dir)
        return curr_dir

    def save_fig(self, epoch_id: int, batch_id: int):
        """
        Save the current figure to a PNG file.

        Args:
            epoch_id (int): Current epoch number.
            batch_id (int): Current batch number.
        """
        figures_dir = self.get_dir()
        fig_name = f"epoch{epoch_id}_batch{batch_id}.png"
        path = os.path.join(figures_dir, fig_name)
        plt.tight_layout()
        plt.savefig(path, format='png', dpi=300)

    def visualize_batch(
        self,
        batch: torch.Tensor,
        pred: torch.Tensor,
        ground: torch.Tensor,
        epoch_id: int,
        batch_id: int
    ):
        """
        Visualize a batch of images with predicted and ground truth labels.

        Args:
            batch (torch.Tensor): Batch of images with shape (B, channels, H, W).
            pred (torch.Tensor): Model predictions.
            ground (torch.Tensor): Ground truth labels.
            epoch_id (int): Epoch number.
            batch_id (int): Batch number.
        """
        classes = ['zero', 'one', 'two', 'three', 'four',
                   'five', 'six', 'seven', 'eight', 'nine']
        BATCH_SIZE = batch.shape[0]
        plt.figure(figsize=(8, 5))
        for i in range(BATCH_SIZE):
            ax = plt.subplot(8, 16, i + 1)
            # Convert image from (channels, H, W) to (H, W) and scale pixel values.
            image = batch[i][0].cpu().numpy()
            image = (image * 255.0).astype("uint8")
            idx = np.argmax(pred[i].cpu().numpy())
            gt_idx = ground[i].cpu().numpy() if isinstance(ground[i], torch.Tensor) else ground[i]
            label = classes[idx]
            plt.imshow(image, cmap="gray")
            # Display label in red if prediction is wrong.
            if idx == gt_idx:
                plt.title(label, color='black', fontsize=6)
            else:
                plt.title(label, color='red', fontsize=6)
            plt.axis("off")
        self.save_fig(epoch_id, batch_id)
        plt.show()


# %% [markdown]
# <hr>
# 
# # <h2> <span style="color:#FFC72C;">**Class**</span> | <span style="color: #C0C0C0"> **TransformerTrainer** </span></h2>  
# 
# <hr>

# %% [markdown]
# <p>
#   This class encapsulates the entire training and evaluation process for a Transformer model.
#   It handles data loading, device setup, model initialization, training, evaluation, and model saving.
# </p>
# 
# <h3 style="color: #FFC72C;">Args</h3>
# <ul>
#   <li>
#     <strong>config</strong>: A configuration dictionary containing:
#     <ul>
#       <li><code>home_dir</code>: Directory for saving results.</li>
#       <li><code>model_path</code>: Path for saving the model.</li>
#       <li><code>n_epochs</code>: Number of training epochs.</li>
#       <li><code>lr</code>: Learning rate.</li>
#     </ul>
#   </li>
# </ul>
# 
# <h3 style="color: #FFC72C;">Attributes</h3>
# <ul>
#   <li>
#     <strong>home_dir</strong>: Directory for saving results.
#   </li>
#   <li>
#     <strong>model_path</strong>: Path for saving the model.
#   </li>
#   <li>
#     <strong>n_epochs</strong>: Number of training epochs.
#   </li>
#   <li>
#     <strong>lr</strong>: Learning rate.
#   </li>
#   <li>
#     <strong>train_set</strong> and <strong>test_set</strong>: Datasets for training and testing (MNIST).
#   </li>
#   <li>
#     <strong>train_loader</strong> and <strong>test_loader</strong>: Data loaders for training and testing.
#   </li>
#   <li>
#     <strong>device</strong>: Device used for training (CPU or CUDA).
#   </li>
#   <li>
#     <strong>model</strong>: The Transformer model instance.
#   </li>
#   <li>
#     <strong>optimizer</strong>: Optimizer for training the model.
#   </li>
#   <li>
#     <strong>criterion</strong>: Loss function used for training.
#   </li>
#   <li>
#     <strong>best_loss</strong>: Best test loss achieved (used for model saving).
#   </li>
#   <li>
#     <strong>visualizer</strong>: Instance of a Visualizer class used for visualizing results.
#   </li>
# </ul>
# 
# <h3 style="color: #FFC72C;">Methods</h3>
# 
# <h4 style="color: #FFC72C;"><code>__init__(config: dict)</code></h4>
# <ul>
#   <li>
#     Initializes the <strong>TransformerTrainer</strong> with the configuration parameters.
#   </li>
#   <li>
#     Sets up data loading for MNIST, device configuration, model initialization, optimizer, loss function, and the visualizer.
#   </li>
# </ul>
# 
# <h4 style="color: #FFC72C;"><code>train_epoch(epoch: int)</code></h4>
# <ul>
#   <li>
#     Runs one training epoch.
#   </li>
#   <li>
#     <strong>Input Parameter</strong>:
#     <ul>
#       <li>
#         <code>epoch</code>: An integer representing the current epoch number.
#       </li>
#     </ul>
#   </li>
#   <li>
#     Processes batches of training data, computes loss, backpropagates, and updates model weights.
#   </li>
# </ul>
# 
# <h4 style="color: #FFC72C;"><code>evaluate(epoch: int)</code></h4>
# <ul>
#   <li>
#     Evaluates the model on the test dataset.
#   </li>
#   <li>
#     <strong>Input Parameter</strong>:
#     <ul>
#       <li>
#         <code>epoch</code>: An integer representing the current epoch number.
#       </li>
#     </ul>
#   </li>
#   <li>
#     Computes test loss and accuracy, visualizes a batch, and saves the model if performance improves.
#   </li>
# </ul>
# 
# <h4 style="color: #FFC72C;"><code>run()</code></h4>
# <ul>
#   <li>
#     Runs the full training and evaluation loop over all epochs.
#   </li>
# </ul>

# %%
class TransformerTrainer:
    """
    A class that encapsulates the entire training and evaluation process for a Transformer model.
    """
    def __init__(self, config: dict):
        """
        Args:
            config (dict): Configuration dictionary containing:
                - home_dir: Directory for saving results.
                - model_path: Path for saving the model.
                - n_epochs: Number of training epochs.
                - lr: Learning rate.
        """
        self.home_dir = config["home_dir"]
        self.model_path = config["model_path"]
        self.n_epochs = config["n_epochs"]
        self.lr = config["lr"]

        # Data loading
        transform = ToTensor()
        self.train_set = MNIST(root="./datasets", train=True, download=True, transform=transform)
        self.test_set = MNIST(root="./datasets", train=False, download=True, transform=transform)
        self.train_loader = DataLoader(self.train_set, shuffle=False, batch_size=128)
        self.test_loader = DataLoader(self.test_set, shuffle=False, batch_size=128)

        # ========================= Write your code below ==========================
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = VisionTransformer(
            chw=(1, 28, 28),
            n_patches=7,
            n_blocks=2,
            hidden_d=64,
            n_heads=8,
            out_d=10
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()

        self.best_loss = float('inf')

        self.visualizer = Visualizer(self.home_dir)
        # ==========================================================================

    def train_epoch(self, epoch: int):
        """Runs one training epoch."""
        # ========================= Write your code below ==========================
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)

            loss.backward()
            self.optimizer.step()

            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            total_loss += loss.item()

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total
        print(f'Epoch {epoch+1}: Train Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
        # ==========================================================================

    def evaluate(self, epoch: int):
        """Evaluates the model on the test dataset and saves the model if improved."""
        # Please visualize the first batch of the test dataset after each epoch
        # ========================= Write your code below ==========================
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)

                loss = self.criterion(output, target)
                test_loss += loss.item()

                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

                if batch_idx == 0:
                    self.visualizer.visualize_batch(data, output, target, epoch, batch_idx)

        avg_loss = test_loss / len(self.test_loader)
        accuracy = 100.0 * correct / total
        print(f'Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
        # ==========================================================================

    def run(self):
        """Runs the full training and evaluation loop."""
        # ========================= Write your code below ==========================
        for epoch in range(self.n_epochs):
            print(f"Epoch {epoch+1}/{self.n_epochs}")
            self.train_epoch(epoch)
            self.evaluate(epoch)
        # ==========================================================================

# %% [markdown]
# <hr>
# 
# # <h2> <span style="color:#FFC72C;">**Results**</span> | **Vision** <span style="color: #C0C0C0; font-family: Transformers"> Transformer </span></h2>  
# 
# <hr>

# %%
from torchvision.datasets import MNIST
from datetime import datetime
import os

config = {
    "home_dir": "./",
    "model_path": "./model.pth",
    "n_epochs": 3,
    "lr": 0.005
}
trainer = TransformerTrainer(config)
trainer.run()


