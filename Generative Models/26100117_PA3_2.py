# %% [markdown]
# # **<span style="color: #87CEEB;">β-VAE: Disentangled Representation Learning</span>**
# 

# %% [markdown]
# ---

# %% [markdown]
# <div style="width: 100%; text-align: center;">
# 
# <h1 style="color:rgb(244, 244, 245); font-size: 40px; font-weight: bold;">
#     <span style="text-decoration-color: white;">THE LAST OF US: 
#     <span style="color:rgb(130, 89, 214);">DEEP LEARNING EDITION</span></span> 🎮
# </h1>
# 
# <hr style="height: 10px; width: 100%; background-color: white; border: none; margin-top: 10px; margin-bottom: 10px;">
# </div>
# 
# 
# 
# <img src="TLOU2.jpg" alt="The Last of Us" style="width: 100%; max-height: 500px; object-fit: cover;">
# 
# 
# ---
# 
# <p style="font-size: 18px;"> In the world of <b>The Last of Us</b>, society has collapsed due to the <b>Cordyceps Brain Infection (CBI)</b>, a fungal outbreak that transforms humans into aggressive, zombie-like creatures known as the <b>Infected</b>. Amidst this chaos, survivors navigate a perilous existence, facing threats from both the Infected and other human factions. </p> <p style="font-size: 18px;"> Our protagonist, <b>Ellie</b>, has endured unimaginable loss and hardship. After tragedy strikes her closest companions, she embarks on a relentless quest for justice. Her journey takes her across the desolate landscapes of a post-apocalyptic America, where she must confront both the horrors of the Infected and the ruthless factions vying for power. </p> <p style="font-size: 18px;"> Ellie's pursuit leads her to the ruins of <b>Hollywood</b>, a city once synonymous with glamour and fame, now a haunting wasteland overrun by the Infected. Intelligence suggests that members of the <b>Washington Liberation Front (WLF)</b> have infiltrated this area, disguising themselves among the hordes to evade detection. To achieve her goal, Ellie must distinguish friend from foe amidst the chaos. </p> <p style="font-size: 18px;"> In this dire scenario, Ellie turns to advanced technology for assistance. She discovers that, much like the Infected emit a distinct scent, humans possess unique facial features that can be analyzed to identify individuals. To leverage this, Ellie employs a <b>Variational Autoencoder (VAE)</b>, a deep learning model capable of learning and disentangling the underlying factors of facial images. By training the VAE on a dataset of celebrity faces, Ellie aims to develop a tool that can differentiate between the Infected, innocent survivors, and hidden WLF operatives. </p> <p style="font-size: 18px;"> This mission is not just about vengeance; it’s about survival, resilience, and the pursuit of justice in a world where morality is no longer black and white. With the VAE as her ally, Ellie embarks on a path fraught with danger, hope, and the unyielding human spirit's fight for redemption. </p>
# 
# 
# <div style="width: 100%; text-align: center;">
# <h1 style="color:rgb(244, 244, 245); font-size: 40px; font-weight: bold;">
#     <span style="text-decoration-color: white;">Let's Start the: 
#     <span style="color:rgb(130, 89, 214);">Hunt</span></span> 🐾
# </h1>
# <hr style="height: 10px; width: 100%; background-color: white; border: none; margin-top: 10px; margin-bottom: 10px;">
# </div>
# 

# %% [markdown]
# # Here is Some Music to help you settle in for the Journey Ahead !

# %%
from IPython.display import Audio

# Replace 'your_audio_file.mp3' with your actual file name
audio_file = "03. The Last of Us.mp3"

# Display an audio player in the notebook
Audio(audio_file)


# %% [markdown]
# ---

# %% [markdown]
# # <u>**Understanding Variational Autoencoders (VAEs)**</u>
# 
# ## 📌 What is a Variational Autoencoder?
# A **Variational Autoencoder (VAE)** is a **generative model** that learns to encode input data into a **latent space representation** and then reconstruct it back. Unlike traditional **Autoencoders**, VAEs introduce **probabilistic sampling** in the latent space, making them effective for generating new data and disentangling representations.
# 
# ### 🔹 **Key Idea**
# Instead of mapping an input $x$ to a fixed latent vector $z$, VAEs model the latent space as a **distribution**:
# 
# - **Encoder**: Learns a **mean** $\mu$ and **variance** $\sigma^2$ for each latent dimension.
# - **Latent Space**: Instead of a fixed vector, we **sample** $z$ from a Gaussian distribution:
#   
#   $$
#   z = \mu + \sigma \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0,1)
#   $$
# 
# - **Decoder**: Takes the sampled $z$ and reconstructs the input image $\hat{x}$.
# 
# This allows for **smooth interpolations** and **better disentangled features** in the latent space.
# 
# ---
# 
# ## 📌 What is the **β-Hyperparameter**? 🛠️
# The **β-VAE** (Beta Variational Autoencoder) introduces a **hyperparameter** $\beta$ in the loss function to **control the trade-off** between:
# 
# - **Reconstruction Quality** (How well the image is reconstructed).
# - **Disentanglement** (How independent the latent factors are).
# 
# The modified **VAE loss function** becomes:
# 
# $$
# \mathcal{L} = \mathbb{E}_{q(z|x)} [\log p(x|z)] - \beta D_{KL}(q(z|x) || p(z))
# $$
# 
# Where:
# - $\mathbb{E}_{q(z|x)} [\log p(x|z)]$ = **Reconstruction Loss** (Measures how well the output matches the input).
# - $D_{KL}(q(z|x) || p(z))$ = **KL Divergence** (Forces latent space to follow a standard Gaussian distribution).
# - $\beta$ = A scaling factor that controls the balance between reconstruction and disentanglement.
# 
# ---
# 
# ## 📌 **Why is KL Divergence Important?**
# **Kullback-Leibler (KL) Divergence** measures how much one probability distribution **differs** from another. In VAEs, we use it to **regularize the latent space**, ensuring it follows a normal distribution.
# 
# ### **🔹 General Formula for KL Divergence**
# For any two probability distributions \( P(x) \) and \( Q(x) \), the KL Divergence is defined as:
# 
# $$
# D_{KL}(P \parallel Q) = \sum_{x \in \mathcal{X}} P(x) \log \frac{P(x)}{Q(x)}
# $$
# 
# This equation measures how much **information is lost** when \( Q(x) \) is used to approximate \( P(x) \). In simpler terms, it **penalizes deviations** from the target distribution.
# 
# ### **🔹 KL Divergence in VAEs**
# In **VAEs**, our encoder learns a latent representation \( q(z|x) \), which we want to keep **close to** a standard Gaussian prior \( p(z) \). Since both distributions are assumed to be **Gaussian**, we can derive a closed-form solution for KL Divergence:
# 
# $$
# D_{KL}(q(z|x) \parallel p(z)) = \frac{1}{2} \sum_{i=1}^{d} \left( 1 + \log(\sigma_i^2) - \mu_i^2 - \sigma_i^2 \right)
# $$
# 
# ### **What do these terms represent?**
# - **$\mu_i$**: The mean of the latent variable $z_i$.
# - **$\sigma_i^2$**: The variance of the latent variable $z_i$.
# - **$d$**: The dimensionality of the latent space.
# - **$D_{KL}(q(z|x) || p(z))$**: Measures how much the learned latent space deviates from the assumed standard normal distribution.
# 
# ### **Why do we need KL regularization?**
# ✅ It prevents **overfitting**.  
# ✅ It ensures **smooth latent representations**.  
# ✅ It allows for **better interpolation and sampling**.  
# 
# ---
# 

# %% [markdown]
# <!-- INSPIRATION PAPER HEADER -->
# <div style="text-align: center;">
#     <h2 style="color:rgb(248, 248, 244); font-size: 40px; font-weight: bold;">
#         📜 <b>Inspiration for This Assignment</b>  
#     </h2>
# </div>
# 
# <!-- INSPIRATION PAPER BLOCK -->
# <div style="padding: 25px; text-align: center; font-size: 22px; font-weight: bold; color:rgb(242, 244, 245);
#             background: rgba(0, 0, 0, 0.6); border-radius: 15px; width: 70%; margin: auto;
#             box-shadow: 0px 0px 30px rgba(213, 223, 16, 0.99);">
#     
# For this assignment, we will be taking **inspiration** from the following paper:  
# 📖 **[[Disentangled Representation Learning](https://arxiv.org/abs/2211.11695)]**  
# 
# Feel free to **read this for insights** and use it as a guide.  
# 
# </div>
# 
# <hr style="height: 10px; width: 80%; background: linear-gradient(to right, white, #FFD700, white); border: none; margin-top: 15px; margin-bottom: 15px;">
# 
# <!-- CHAPTER 1 HEADER -->
# <div style="text-align: center;">
#     <h1 style="color: #FF8C00; font-size: 55px; font-weight: bold;">
#         🍁 <span style="color: white;">Chapter 1</span> -  
#         <span style="color: #FFD700;">Fall</span> 🍂
#         <span style="color: white;"></span>
#     </h1>
#     <hr style="height: 10px; width: 80%; background: linear-gradient(to right, white, #FF8C00, white); border: none; margin-top: 10px; margin-bottom: 10px;">
# </div>
# 
# <!-- INTRODUCTION -->
# <div style="padding: 30px; text-align: justify; font-size: 22px; color: white;
#             background: rgba(0, 0, 0, 0.6); border-radius: 15px;
#             width: 80%; margin: auto;
#             box-shadow: 0px 0px 30px rgba(255, 140, 0, 0.6);">
#     
# <b>The journey begins, not with a model, but with data.</b>  
# This time, **you won’t be given a pre-processed dataset**—  
# Instead, **you must take control.**  
# 
# In this part of the assignment, your task is to **load the raw CelebA dataset and preprocess it yourself.**  
# This is an **essential skill** for anyone in this field:  
# 
# ✅ **Identify, clean, and prepare data**  
# ✅ **Decide on preprocessing techniques**  
# ✅ **Gain full creative freedom** based on task requirements  
# 
# Your **dataset, your rules.**  
# Your **pipeline, your approach.**  
# **Let's begin.**  
# 
# </div>
# 
# <!-- WHAT IS CELEBA? -->
# <div style="padding: 10px; text-align: center;">
#     <h2 style="color: #FFD700; font-size: 35px; font-weight: bold;">
#         🏷️ <b>What is the CelebA Dataset?</b>
#     </h2>
# </div>
# 
# <div style="padding: 30px; text-align: justify; font-size: 22px; color: white;
#             background: rgba(0, 0, 0, 0.6); border-radius: 15px;
#             width: 80%; margin: auto;
#             box-shadow: 0px 0px 30px rgba(255, 140, 0, 0.6);">
# 
# CelebA (**Large-scale CelebFaces Attributes Dataset**) is one of the **most widely used datasets** in **computer vision & deep learning research**.  
# It contains **202,599 images of celebrities** with **40 attribute labels** per image.  
# 
# 💡 **Why is CelebA Important?**  
# - It enables **facial attribute prediction** (e.g., age, gender, hair color, eyeglasses, etc.).  
# - It allows experimentation with **variational autoencoders (VAEs)** for **feature disentanglement**.  
# - It is **diverse**, containing **faces of different expressions, poses, and occlusions**.  
# 
# 💾 **Dataset Structure:**  
# 📂 **img_align_celeba/** → The actual images (cropped & aligned)  
# 📄 **list_attr_celeba.txt** → Binary labels (1 or -1) for each image  
# 📄 **list_eval_partition.txt** → Specifies train/test/val splits  
# 
# **For this task, you will decide which files are needed based on your preprocessing choices.**  
# 
# </div>
# 
# <!-- TASK OVERVIEW HEADER -->
# <div style="text-align: center;">
#     <h2 style="color: #FFD700; font-size: 35px; font-weight: bold;">
#         🎯 <b>Task Overview: Loading & Preprocessing CelebA</b>
#     </h2>
# </div>
# 
# <!-- TASK DETAILS -->
# <div style="padding: 30px; text-align: justify; font-size: 22px; color: white;
#             background: rgba(0, 0, 0, 0.6); border-radius: 15px;
#             width: 80%; margin: auto;
#             box-shadow: 0px 0px 30px rgba(255, 140, 0, 0.6);">
# 
# 📌 **Download the dataset** from this Link: ([CelebA Official Repository](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)).  
# 💡 **Recommendation:** Use the **Aligned & Cropped Zip Folder**, but the choice is **yours**.  
# 📖 **Read the ReadMe File available in the dataset directory** to understand **what’s available**.  
# 
# ### **🛠️ What You Need to Do:**
# 🔍 **Step 1:** Identify **which files** are necessary for your task. (Hint: You will need the Attributes, The Partitions and The Images themselves)
# 📥 **Step 2:** Download the required files from the link.  
# 📂 **Step 3:** Load the dataset into your notebook **(You are free to use ChatGPT, a custom loader, or any other method—Full Creative Freedom).**  
# 🖼️ **Step 4:** Display **5 random samples** from the **train** split & **5 random samples** from the **test** split.  
# 📑 **Step 5:** **Examples of what your visualizations should look like are in the FAQs document.**  
# 
# ⚠️ **IMPORTANT:** The CelebA dataset is **too large** to be used in its entirety.  
# 🚀 **We strongly recommend reducing it to a subset of 50,000 images** (80-20 split for Train/Eval).  
# - **This will significantly reduce training time.**  
# - **With 50,000 images, training will take ~40 minutes (CPU) on all β-values.**  
# - **Ignoring this will slow down your progress!**  
# 
# </div>
# 
# <!-- FINAL GRADING WARNING -->
# <div style="padding: 30px; text-align: center; font-size: 22px; font-weight: bold; color: red;
#             background: rgba(0, 0, 0, 0.6); border-radius: 15px; width: 80%; margin: auto;
#             box-shadow: 0px 0px 30px rgba(255, 0, 0, 0.6);">
# 
# ⚠️ **Your marks for this task depend entirely on your success in loading and visualizing the dataset.**  
# If done incorrectly, you **might not be able to complete the rest of the tasks.**  
# 
# </div>
# 

# %%
# Code Here
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torch.optim as optim

# %%
# Code Here
transform = transforms.Compose([
    transforms.CenterCrop(178),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

data_dir = './data'

celeba_dataset = datasets.CelebA(root=data_dir, split="train", download=True, transform=transform)

batch_size = 128
data_loader = DataLoader(celeba_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

for images, labels in data_loader:
    print("Batch image tensor shape:", images.shape)
    break

# %%
# Code Here
class CelebADataset(Dataset):
    def __init__(self, img_dir, partition_file, transform=None, subset=None, split='train'):
        df = pd.read_csv(partition_file, delim_whitespace=True, header=None, names=['image', 'partition'])
        
        if split == 'train':
            df = df[df['partition'] == 0]
        elif split == 'test':
            df = df[df['partition'] == 2]
        
        if subset is not None and len(df) > subset:
            df = df.sample(n=subset, random_state=42)
        
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['image']
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_name
    
img_dir = './data/celeba/img_align_celeba'
partition_file = './data/celeba/list_eval_partition.txt'

train_dataset = CelebADataset(img_dir, partition_file, transform=transform, subset=40000, split='train')
test_dataset = CelebADataset(img_dir, partition_file, transform=transform, subset=10000, split='test')

# %%
# Code Here
def show_images(dataset, num_samples=5, title='Sample Images'):
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    images = [dataset[i][0] for i in indices]  

    images = [img * 0.5 + 0.5 for img in images]

    fig, axs = plt.subplots(1, num_samples, figsize=(15, 5))
    for i, img in enumerate(images):
        axs[i].imshow(np.transpose(img.numpy(), (1, 2, 0)))
        axs[i].axis('off')
    plt.suptitle(title)
    plt.show()

show_images(train_dataset, num_samples=5, title='Train Samples')
show_images(test_dataset, num_samples=5, title='Test Samples')

# %% [markdown]
# <div style="text-align: center;">
#     <h1 style="color: rgb(255, 183, 76); font-size: 50px; font-weight: bold;">
#         ☀️ <span style="color: white;">Chapter 2</span> -  
#         <span style="color: rgb(255, 183, 76);">Summer</span> ☀️
#         <span style="color: white;"></span>
#     </h1>
#     <hr style="height: 8px; width: 80%; background: linear-gradient(to right, white, rgb(255, 183, 76), white); border: none; margin-top: 10px; margin-bottom: 10px;">
# </div>
# 

# %% [markdown]
# <!-- MISSION HEADER -->
# <div style="text-align: center;">
#     <h1 style="color: #FFA500; font-size: 55px; font-weight: bold;">
#         💻 <span style="color: white;">TIME TO START CODING!</span> 💻
#     </h1>
# </div>
# 
# <!-- CINEMATIC INTRO -->
# <div style="padding: 25px; text-align: center; font-size: 24px; font-style: italic; color: white;
#             background: rgba(0, 0, 0, 0.6); border-radius: 15px; width: 70%; margin: auto;
#             box-shadow: 0px 0px 30px rgba(255, 165, 0, 0.6);">
#     
# "This is where the real challenge begins..."  
# Ellie’s journey led her here. **Your journey begins now.**
# 
# </div>
# 
# <hr style="height: 8px; width: 80%; background: linear-gradient(to right, white, #FFA500, white); border: none; margin-top: 15px; margin-bottom: 15px;">
# 
# <!-- MISSION OBJECTIVE TITLE -->
# <div style="text-align: center;">
#     <h2 style="color: rgb(255, 183, 76); font-size: 35px; font-weight: bold;">
#         🎯 <b>Mission Objective:</b> Train a <b>β-VAE</b> to Disentangle Features and Uncover the WLF Operatives
#     </h2>
# </div>
# 
# <!-- STORY-THEMED CODING CHALLENGE -->
# <div style="padding: 30px; text-align: justify; font-size: 22px; color: white;
#             background: rgba(0, 0, 0, 0.6); border-radius: 15px;
#             width: 80%; margin: auto;
#             box-shadow: 0px 0px 30px rgba(255, 165, 0, 0.6);">
# 
# The **data is ready**, the **tools are set**, and now it’s time to **dive into the unknown**.  
# Just like **Ellie**, you must **navigate through uncertainty**,  
# experiment with different **β-values**, and **uncover the hidden representations** within faces.
# 
# </div>
# 
# <!-- FINAL CALL TO ACTION -->
# <div style="padding: 20px; text-align: center; font-size: 22px; font-weight: bold; color: rgb(255, 183, 76);">
#     You are on your own from this point onwards.  
#     <br> <b>Good luck, survivor.</b> 🏹🔥
# </div>
# 

# %% [markdown]
# ## **🔹 Your Task: Complete the β-VAE Implementation**
# 
# ### **📌 Key Components to Consider**
# 1. **Encoder (`encode` function)**
#    - Takes an input image .
#    - Passes it thorugh **Convolution Layers**
#    - Outputs two vectors:
#      - **Mean (μ):** The predicted mean of the latent distribution.
#      - **Log variance (logσ²):** The predicted variance (log-scale) of the latent distribution.
# 
# 1. **Reparameterization Trick (`reparameterize` function)**
#    - Uses the **mean (μ) and log variance (logσ²)** to create a latent vector **z**.
#    - Adds **random noise** to make the model **stochastic**.
# 
# 2. **Decoder (`decode` function)**
#    - Takes a latent vector **z**.
#    - Passes it through **transposed convolution layers** to reconstruct the image.
# 
# 3. **Forward Pass (`forward` function)**
#    - Takes an input image.
#    - **Encodes → Reparameterizes → Decodes** it.
#    - Returns **the reconstructed image, mean, and log variance**.
# 
# `Note:`: You do not neccseccarily need to follow the structure above , you are free to make whatever functions you require , change constructors values, Adjust Hyperparameters, GO CRAZY (Clicker Symptoms Starting to show xD).Please also look into how different 'latent_dim' values effect your results and why? Currently a default value of 10 is set
# 
# 

# %% [markdown]
# # B-VAE Architecture

# %%
class BetaVAE(nn.Module):
    def __init__(self, latent_dim=10, beta=1.0):
        """
        Beta-VAE Model
        
        Args:
            latent_dim (int): Dimension of the latent space (default = 10)
            beta (float): Weighting factor for KL divergence in loss function
        """
        super(BetaVAE, self).__init__()
        self.beta = beta  
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)
        
        self.fc_decode = nn.Linear(latent_dim, 256 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  
            nn.Tanh()  
        )

    def encode(self, x):
        """
        Encode input image to latent space representations (μ, logσ²).
        """
        batch_size = x.size(0)
        x_enc = self.encoder(x)               
        x_enc = x_enc.view(batch_size, -1)      
        mu = self.fc_mu(x_enc)                  
        logvar = self.fc_logvar(x_enc)          
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample latent vector z from μ and logσ².
        """
        std = torch.exp(0.5 * logvar)           
        eps = torch.randn_like(std)             
        z = mu + std * eps                      
        return z

    def decode(self, z):
        """
        Decode latent vector z back to image space.
        """
        batch_size = z.size(0)
        x_dec = self.fc_decode(z)             
        x_dec = x_dec.view(batch_size, 256, 4, 4) 
        x_recon = self.decoder(x_dec)         
        return x_recon

    def forward(self, x):
        """
        Full forward pass: Encode → Reparameterize → Decode
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


# %% [markdown]
# # B-VAE Loss

# %%
def beta_vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """
    Beta-VAE Loss Function
    
    Args:
        recon_x: Reconstructed image
        x: Original image
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        beta: KL divergence weighting factor

    Returns:
        Total loss
    """
    # 🚀 Compute Reconstruction Loss (Mean Squared Error)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)

    # 🚀 Compute KL Divergence Loss
    kld_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

    # 🚀 Compute Total Loss
    total_loss = recon_loss + beta * kld_loss

    return total_loss



# %% [markdown]
# # 📌 Training Instructions
# Now that you have implemented the Beta-VAE, it's time to train it on the CelebA dataset.
# ### 🎯 Your Tasks
# - ✅ Train the Beta-VAE with at least 4 different β values (1, 5, 10, 50).
# - ✅ Train for a minimum of 3 epochs (you can train for more).
# - ✅ Log and analyze how different β values affect:
# - Pleas use `tqdm` library to track progress as it can take up to `45 mins of training` on a CPU

# %%
# Training 

# 🚀 1. Define Hyperparameters
# 🚀 2. Initialize Model & Optimizer
# 🚀 3. Define Training Function
# 🚀 4. Define Validation Function
# 🚀 5. Train Model for Each β Value
# 🚀 6. Save the Trained Model

batch_size = 128
num_epochs = 3
learning_rate = 1e-3
beta_values = [1, 5, 10, 50]

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

def train_epoch(model, dataloader, optimizer, beta):
    model.train()
    running_loss = 0.0
    for images, _ in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device)
        optimizer.zero_grad()
        
        recon_images, mu, logvar = model(images)
        loss = beta_vae_loss(recon_images, images, mu, logvar, beta=beta)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def validate_epoch(model, dataloader, beta):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="Validation", leave=False):
            images = images.to(device)
            recon_images, mu, logvar = model(images)
            loss = beta_vae_loss(recon_images, images, mu, logvar, beta=beta)
            running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

save_dir = "./saved_models"
os.makedirs(save_dir, exist_ok=True)

for beta in beta_values:
    print(f"\nTraining Beta-VAE with β = {beta}")
    
    model = BetaVAE(latent_dim=10, beta=beta).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(1, num_epochs+1):
        print(f"Epoch {epoch}/{num_epochs}")
        
        train_loss = train_epoch(model, train_loader, optimizer, beta)
        val_loss = validate_epoch(model, val_loader, beta)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    model_save_path = os.path.join(save_dir, f"beta_vae_beta{beta}.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Saved model for β = {beta} at {model_save_path}")

# %%
def compute_losses(recon_x, x, mu, logvar, beta=1.0):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)
    kld_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    total_loss = recon_loss + beta * kld_loss
    return total_loss, recon_loss, kld_loss

def evaluate_model(model, dataloader, beta, device):
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_kld = 0.0
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            recon_images, mu, logvar = model(images)
            t_loss, r_loss, k_loss = compute_losses(recon_images, images, mu, logvar, beta=beta)
            total_loss += t_loss.item() * images.size(0)
            total_recon += r_loss.item() * images.size(0)
            total_kld += k_loss.item() * images.size(0)
    n = len(dataloader.dataset)
    return total_loss/n, total_recon/n, total_kld/n

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

beta_values = [1, 5, 10, 50]

for beta in beta_values:
    model = BetaVAE(latent_dim=10, beta=beta).to(device)
    model_path = f"./saved_models/beta_vae_beta{beta}.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    total_loss, recon_loss, kld_loss = evaluate_model(model, val_loader, beta, device)
    print(f"\nBeta = {beta}:")
    print(f"Total Loss: {total_loss:.4f}")
    print(f"Reconstruction Loss: {recon_loss:.4f}")
    print(f"KL Divergence Loss: {kld_loss:.4f}")


# %% [markdown]
# ## 📊 **Logging Your Results**
# 
# Once you have trained your **β-VAE** model with different β values, log your results in a table format like the one below.
# 
# |   **Beta Value** |   **Total Loss** |   **KL Divergence** |   **Reconstruction Loss** |
# |:--------------:|:-------------:|:----------------:|:----------------------:|
# | 1  | 1189.31 | 32.7504 | 1156.56 |
# | 5  | 1261.73 | 22.6921 | 1148.27 |
# | 10 | 1330.06 | 18.7226 | 1142.84 |
# | 50 | 1837.18 | 10.5795 | 1308.21 |
# 
# ### 📌 **How to Interpret Your Table**
# - **Total Loss**: The overall loss for the VAE combining both **reconstruction loss** and **KL divergence**.
# - **KL Divergence**: How much the learned latent space deviates from a normal Gaussian distribution.
# - **Reconstruction Loss**: Measures how well the model reconstructs the input images.
# 
# 📢 **Your task is to analyze how the KL Divergence and Reconstruction Loss change as β increases!**
# 

# %% [markdown]
# ## LOGGED RESULTS: 
# 
# |   **Beta Value** |   **Total Loss** |   **KL Divergence** |   **Reconstruction Loss** |
# |:--------------:|:-------------:|:----------------:|:----------------------:|
# | 1  | 1049.0309 | 32.5073 | 1016.5235 |
# | 5  | 1136.2131 | 21.2548| 1029.9391 |
# | 10 | 1254.8954| 18.6131 | 1068.7649 |
# | 50 | 1750.7055 | 9.3235 | 1284.5304 |
# 
# 

# %%
## log Your Own results in the same way as shown above

# %% [markdown]
# ### 🧩 **Visualizing Disentangled Latent Factors**  
# 
# Now that you have trained your **β-VAE**, it's time to **explore the latent space** and visualize how different latent factors influence image generation!  
# 
# ### 🔍 **What You Need to Do:**  
# 1️⃣ **Select a Base Latent Vector**:  
#    - Start with a vector **initialized to zeros** (or another meaningful initialization).  
#    - This represents the "average" or neutral latent space representation.  
# 
# 2️⃣ **Vary One Latent Dimension at a Time**:  
#    - Change the value of **one latent dimension** while keeping all others fixed.  
#    - Observe how this affects the generated images.  
# 
# 3️⃣ **Generate Multiple Samples**:  
#    - Choose a **suitable range** for variation (e.g., **between -3 and 3**).  
#    - Sample **at least 10 variations** across this range.  
# 
# 4️⃣ **Plot the Results in a Grid**:  
#    - Each **row** should represent a different latent dimension.  
#    - Each **column** should represent a different sampled value within the chosen range.  
#    - Label the axes appropriately for interpretation.  
# 
# ### 🎯 **Goal:**  
# Your goal is to **interpret how different latent dimensions affect image generation**.  
# - Do some dimensions **control specific attributes** (e.g., smiling, hair color, face shape)?  
# - Are some dimensions **more interpretable** than others?  
# - Does increasing β make disentanglement clearer?  
# 
# 📝 **Experiment with different variation ranges and observe how your β-VAE captures meaningful factors of variation!**  
# 
# 🔥 **Time to unlock the secrets of the latent space!** 🚀
# 

# %%
def visualize_disentangled_latent_factors(model, latent_dim, num_samples=10, variation_range=(-3, 3)):
    """
    Visualize disentangled latent factors by varying one latent dimension.

    Parameters:
    - model: Trained β-VAE model.
    - latent_dim: Number of latent dimensions in the model.
    - num_samples: Number of samples to generate per latent dimension.
    - variation_range: Range of values to vary for each latent dimension.
    """
    model.eval()

    #--------------------------------------------------------------------------------------------------------
    # CODE HERE

    device = next(model.parameters()).device  
    base_latent = torch.zeros(1, latent_dim, device=device)
    
    variation_values = np.linspace(variation_range[0], variation_range[1], num_samples)
    
    fig, axes = plt.subplots(latent_dim, num_samples, figsize=(num_samples * 2, latent_dim * 2))
    
    with torch.no_grad():
        for dim in range(latent_dim):
            for j, val in enumerate(variation_values):
                z = base_latent.clone()
                z[0, dim] = val  
                
                generated = model.decode(z)
                img = generated.squeeze(0).cpu().numpy()
                img = (img * 0.5 + 0.5).clip(0, 1)  
                img = np.transpose(img, (1, 2, 0))   
                
                ax = axes[dim, j] if latent_dim > 1 else axes[j]
                ax.imshow(img)
                ax.axis('off')
                if dim == 0:
                    ax.set_title(f'{val:.2f}', fontsize=10)
            axes[dim, 0].set_ylabel(f"Dim {dim}", fontsize=12)
    plt.tight_layout()
    plt.show()

    #--------------------------------------------------------------------------------------------------------

latent_dim = 10  
num_samples = 10  
variation_range = (-3, 3)  

model = BetaVAE(latent_dim=10, beta=1.0).to(device)

model.load_state_dict(torch.load("saved_models/beta_vae_beta1.pth", map_location=device))
visualize_disentangled_latent_factors(model, latent_dim=latent_dim, num_samples=num_samples, variation_range=variation_range)


# %% [markdown]
# <div style="text-align: center;">
#     <h1 style="color: rgb(173, 216, 230); font-size: 50px; font-weight: bold;">
#         ❄️ <span style="color: white;">Chapter 3</span> - 
#         <span style="color: rgb(173, 216, 230);">Winter</span> ❄️
#         <span style="color: white;"></span>
#     </h1>
#     <hr style="height: 8px; width: 80%; background: linear-gradient(to right, white, rgb(173, 216, 230), white); border: none; margin-top: 10px; margin-bottom: 10px;">
# </div>
# 

# %% [markdown]
# <!-- SPORE WARNING HEADER -->
# <div style="width: 100%; text-align: center;">
#     <h1 style="color: rgb(248, 247, 249); font-size: 50px; font-weight: bold;">
#         ☣️ <b>WATCH OUT!</b>  
#         <span style="color: rgb(130, 89, 214);">
#             🦠 SPORES AHEAD!  
#         </span>  
#         <span style="color: rgb(255, 255, 255);">
#             If You Breathe…....... <b>You’re Done!</b> 💀  
#         </span>
#     </h1>
#     <hr style="height: 8px; width: 100%; background: linear-gradient(to right, white, rgb(130, 89, 214), white); border: none; margin-top: 10px; margin-bottom: 10px;">
# </div>
# 
# <!-- SPORE-INFESTED IMAGES -->
# <div style="text-align: center;">
#     <img src="TLOU8.jpg" style="width: 48%; height: 1100px; filter: brightness(80%) contrast(110%) saturate(120%); margin-right: 10px; border: 5px solid rgb(130, 89, 214);">
#     <img src="TLOU7.jpg" style="width: 48%; height: 1100px; filter: brightness(80%) contrast(110%) saturate(120%); border: 5px solid rgb(130, 89, 214);">
# </div>
# 
# <!-- SPORE LORE WITH DRAMATIC BACKGROUND -->
# <div style="padding: 30px; text-align: justify; font-size: 22px; color: white; 
#             background: rgba(0, 0, 0, 0.6); border-radius: 15px; 
#             width: 80%; margin: auto; 
#             box-shadow: 0px 0px 30px rgba(130, 89, 214, 0.6);">
#     
# <b>Ellie moves cautiously through the ruins of Hollywood.</b>  
# A dead city, swallowed by time. Silent. Abandoned.  
# 
# Until she sees it—  
# 
# A <b>thick, swirling mist</b> floating in the dim light.  
# Particles drift in the air like specks of **death itself**.  
# 
# 🦠 **SPORES.** Thick. Dense. **Lethal.**  
# 
# She tightens her mask.  
# Her breath slows. **She has to move forward.**  
# 
# Every inch of the air is <b>infected</b>,  
# the last whispers of a world long gone.  
# 
# </div>
# 
# <!-- SUBHEADING: WHAT ARE SPORES? -->
# <div style="padding: 10px; text-align: center;">
#     <h2 style="color: white; font-size: 30px; font-weight: bold;">
#         🦠 <b>What Are Spores?</b>  
#     </h2>
# </div>
# 
# <!-- SPORE LORE WITH DRAMATIC BACKGROUND -->
# <div style="padding: 30px; text-align: justify; font-size: 22px; color: white; 
#             background: rgba(0, 0, 0, 0.6); border-radius: 15px; 
#             width: 80%; margin: auto; 
#             box-shadow: 0px 0px 30px rgba(130, 89, 214, 0.6);">
#     
# <b>According to <span style="color: rgb(130, 89, 214);">The Last of Us lore</span></b>, Spores are  
# <b>airborne fungal particles</b> carrying the Cordyceps infection.  
# They <b>linger in enclosed spaces</b>, unseen, yet deadly.  
# 
# <br>  
# 
# 💀 <b>Without protection, exposure to Spores is a death sentence.</b> 💀  
# 
# <br>
# 
# Ellie, immune yet suffocating in the thick air, moves cautiously.  
# 
# <br>
# 
# <b>The air tastes wrong. The silence is suffocating.</b>  
# 
# <br>
# 
# Then—  
# 
# <br>
# 
# 🚨 <b>This is where everything changes.</b> 🚨  
# 
# </div>
# 
# 
# <!-- SUBHEADING: NO MORE SKELETON CODE -->
# <div style="padding: 10px; text-align: center;">
#     <h2 style="color: white; font-size: 30px; font-weight: bold;">
#         💀 <b>NO MORE SKELETON CODE!</b> 💀
#     </h2>
# </div>
# 
# <!-- CINEMATIC BLOCK -->
# <div style="padding: 30px; font-size: 22px; color: white; 
#             background: rgba(0, 0, 0, 0.6); border-radius: 15px; 
#             width: 80%; margin: auto; 
#             box-shadow: 0px 0px 30px rgba(130, 89, 214, 0.6);">
#     <b>With limited visibility, no outside help, and nowhere to run…</b>  
#     Ellie must rely on her instincts.  
#     <b>And so must YOU.</b>  
# </div>
# 
# <!-- FINAL MESSAGE -->
# <div style="padding: 30px; font-size: 22px; color: white; 
#             background: rgba(0, 0, 0, 0.6); border-radius: 15px; 
#             width: 80%; margin: auto; 
#             box-shadow: 0px 0px 30px rgba(130, 89, 214, 0.6);">
#     She <b>must</b> find the WLF. She <b>must</b> finish what she started.  
#     <b>You have unfinished business too.</b>  
# </div>
# 

# %% [markdown]
# <div style="width: 100%; text-align: center;">
# <h1 style="color:rgb(244, 244, 245); font-size: 40px; font-weight: bold;">
#     <span style="text-decoration-color: white;">Let's Continue the: 
#     <span style="color:rgb(130, 89, 214);">Hunt</span></span> 🐾
# </h1>
# <hr style="height: 10px; width: 100%; background-color: white; border: none; margin-top: 10px; margin-bottom: 10px;">
# </div>

# %% [markdown]
# ## 🛠️ **Your Task: Analyzing Disentangled Representations Across β Values**  
# 
# Ellie has learned to distinguish the **Infected from the WLF** using deep learning, but now she must **fine-tune her approach**. Different **β-values** result in different **levels of disentanglement**—your job is to analyze and visualize them.
# 
# ---
# 
# ## **🎯 What You Need to Do:**
# 1️⃣ **Train a β-VAE model** for at least **4 different β values** (1, 5, 10, 50). --> No need to do this again if you already did it above then just use those models.
# 2️⃣ **Run the training process** for **at least 3 epochs** (you can train for longer).  --> Same procedure as above , i am just repeating the steps incase you do not want to scroll above.
# 3️⃣ **Save each trained model** separately. --> Ideally you should have done this before but not my fault if you did not , please train again then. No Way around this. 
# 4️⃣ **Visualize the effect of β** on the learned latent space:  
#    - Pick a **suitable variation range** (e.g., -3 to 3).  
#    - Generate and plot **disentangled latent factors** for **each β value**.  
# 5️⃣ **Compare the results**—observe **how β affects disentanglement** and reconstruction quality.
# 
# ---
# 
# ## **📊 Visualizing the Disentangled Latent Factors**  
# 
# For each trained model, you must **vary each latent dimension** and generate corresponding images. The goal is to see **how different features change** when we tweak the latent variables.
# 
# 🚀 **Your final visualization should include:**
# - **Separate plots for each β value**.
# - **Variations across latent dimensions**.
# - **A comparison of how different β values affect feature disentanglement**.
# 
# ---
# 
# 💀 **Ellie's revenge isn't over yet**—and neither is your task!  
# 🔥 **Push forward, complete the visualizations, and see the true power of β-VAE!**  
# 
# ---

# %%
# Code

def visualize_disentangled_latent_factors(model, latent_dim, num_samples=10, variation_range=(-3, 3)):
    model.eval()
    device = next(model.parameters()).device  
    base_latent = torch.zeros(1, latent_dim, device=device)
    
    variation_values = np.linspace(variation_range[0], variation_range[1], num_samples)
    
    fig, axes = plt.subplots(latent_dim, num_samples, figsize=(num_samples * 2, latent_dim * 2))
    
    with torch.no_grad():
        for dim in range(latent_dim):
            for j, val in enumerate(variation_values):
                z = base_latent.clone()
                z[0, dim] = val  
                
                generated = model.decode(z)
                img = generated.squeeze(0).cpu().numpy()
                img = (img * 0.5 + 0.5).clip(0, 1)  
                img = np.transpose(img, (1, 2, 0))   
                
                ax = axes[dim, j] if latent_dim > 1 else axes[j]
                ax.imshow(img)
                ax.axis('off')
                if dim == 0:
                    ax.set_title(f'{val:.2f}', fontsize=10)
            axes[dim, 0].set_ylabel(f"Dim {dim}", fontsize=12)
    
    plt.suptitle(f"Disentangled Latent Factors (Model: β = {model.beta})", fontsize=16)
    plt.tight_layout()
    plt.show()

# %%
# Code 
beta_values = [1, 5, 10, 50]
latent_dim = 10         
num_samples = 10        
variation_range = (-3, 3)  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for beta in beta_values:
    model = BetaVAE(latent_dim=latent_dim, beta=beta).to(device)
    model_path = f"./saved_models/beta_vae_beta{beta}.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    print(f"\nVisualizing latent factors for β = {beta}")
    visualize_disentangled_latent_factors(model, latent_dim=latent_dim, num_samples=num_samples,variation_range=variation_range)

# %%
# Experimentation with latent dim = 30, beta = 5

device = torch.device("cuda" if torch.cuda.is_available() else "mps")

model = BetaVAE(latent_dim=10, beta=10).to(device)

batch_size = 128
num_epochs = 3
learning_rate = 1e-3
beta_values = [1, 5, 10, 50]

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

def train_epoch(model, dataloader, optimizer, beta):
    model.train()
    running_loss = 0.0
    for images, _ in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device)
        optimizer.zero_grad()
        
        recon_images, mu, logvar = model(images)
        loss = beta_vae_loss(recon_images, images, mu, logvar, beta=beta)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def validate_epoch(model, dataloader, beta):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="Validation", leave=False):
            images = images.to(device)
            recon_images, mu, logvar = model(images)
            loss = beta_vae_loss(recon_images, images, mu, logvar, beta=beta)
            running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

beta = 10
print(f"\nTraining Beta-VAE with β = {beta}")

model = BetaVAE(latent_dim=30, beta=beta).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(1, num_epochs+1):
    print(f"Epoch {epoch}/{num_epochs}")
    
    train_loss = train_epoch(model, train_loader, optimizer, beta)
    val_loss = validate_epoch(model, val_loader, beta)
    
    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

print(f"\nVisualizing latent factors for β = {10}, Latent Dim = 30")

# %%
# Ignore the error above, i just hadn't ran the cell for the function visualize_disentangled_latent_factors.
visualize_disentangled_latent_factors(model, latent_dim=30, num_samples=10,variation_range=(-3,3))

# %% [markdown]
# <!-- EXPLANATION REQUEST HEADER -->
# <div style="width: 100%; text-align: center;">
#     <h2 style="color: white; font-size: 30px; font-weight: bold;">
#         📝 <b>Please Explain Your Results in Full Detail in the Markdown Below</b> 📝
#     </h2>
# </div>
# 
# <!-- INSTRUCTION BLOCK -->
# <div style="padding: 30px; text-align: justify; font-size: 22px; color: white; 
#             background: rgba(0, 0, 0, 0.6); border-radius: 15px; 
#             width: 80%; margin: auto; 
#             box-shadow: 0px 0px 30px rgba(130, 89, 214, 0.6);">
#     
# <b>Provide a thorough breakdown of your results.</b>  
# Explain every aspect, including:  
# 
# - ✅ **Architecture Choice** – Why did you choose this specific architecture? 
# - ✅ **Observations** – What do the numbers and trends indicate?  
# - ✅ **Patterns** – Are there any significant trends in your results?  
# - ✅ **Comparison** – How do your findings compare to expected outcomes?  
# - ✅ **Errors/Anomalies** – Any unexpected results? Why might they have occurred?  
# - ✅ **Insights** – What conclusions can you draw from these results?  
# 
# Take your time to articulate **every detail clearly and concisely**.  
# 
# </div>
# 

# %% [markdown]
# ### Breakdown of Beta-VAE Results
# 
# - **Architecture Choice:**  
#   I used a convolutional encoder and a transposed convolutional decoder with a latent space of 10 dimensions. This design is standard for image reconstruction tasks, as convolutional layers efficiently capture spatial hierarchies, and the symmetric decoder reconstructs the image from compressed representations.
# 
# - **Observations:**  
#     The logged results show that as Beta increases, the total loss increases from ~1049 at Beta=1 to ~1751 at Beta=50. The KL divergence decreases significantly with higher Beta, while the reconstruction loss also increases, but more gradually.
# 
# - **Patterns:**  
#     With higher Beta values, the model places greater emphasis on minimizing the KL divergence (pushing the latent distribution closer to a standard normal), resulting in more disentangled latent features. However, this comes at the expense of reconstruction quality, as seen by the increase in reconstruction loss.
# 
# - **Comparison:**  
#     These findings align with theoretical expectations: lower Beta values produce sharper reconstructions (lower reconstruction loss) but less disentanglement, whereas higher Beta values yield more regularized latent spaces (lower KL divergence) but slightly blurrier images.
# 
# - **Errors/Anomalies:**  
#     No major anomalies were observed, though the gradual increase in reconstruction loss with higher Beta suggests that the trade-off is working as expected.
# 
# - **Insights:**  
#     The results show the key trade-off in Beta-VAE: a higher Beta promotes latent disentanglement at the cost of reconstruction fidelity. A moderate Beta (around 1–5) might offer a good balance for practical applications, depending on whether clarity or interpretability is prioritized.

# %% [markdown]
# <!-- CHAPTER 3 HEADER -->
# <div style="text-align: center;">
#     <h1 style="color: rgb(255, 165, 0); font-size: 50px; font-weight: bold;">
#         🍂 <span style="color: white;">Chapter 4</span> -  
#         <span style="color: rgb(255, 165, 0);">Autumn</span> 🍂
#         <span style="color: white;"></span>
#     </h1>
#     <hr style="height: 8px; width: 80%; background: linear-gradient(to right, white, rgb(255, 165, 0), white); border: none; margin-top: 10px; margin-bottom: 10px;">
# </div>
# 
# <!-- REFLECTION QUESTIONS BLOCK -->
# <div style="padding: 30px; text-align: justify; font-size: 22px; color: white; 
#             background: rgba(0, 0, 0, 0.6); border-radius: 15px; 
#             width: 80%; margin: auto; 
#             box-shadow: 0px 0px 30px rgba(255, 165, 0, 0.6);">
#     
# <b>Now that you have reached Autumn, it's time to reflect.</b>  
# Answer the following questions thoughtfully:  

# %% [markdown]
# ## 🔍 Reflection Question 1
# ### How does increasing β affect disentanglement and reconstruction quality? Please provide a detailed response (Max 100 Words)
# 
# Ans: Increasing beta increases the weight of the KL divergence, pushing the latent space to be closer to a standard normal distribution. This encourages the model to learn more independent and disentangled latent factors, making each dimension capture distinct features. However, because a higher beta also down-weights the reconstruction loss, the model may sacrifice image detail and sharpness, resulting in lower reconstruction quality. Essentially, there's a trade-off: higher beta improves latent interpretability at the expense of precise image reconstruction.

# %% [markdown]
# ## 🔍 Reflection Question 2
# ### Identify and discuss the limitations of β-VAE, such as loss of fine-grained details in reconstructions or failure to disentangle specific factors. Please provide a detailed response (Max 100 Words)
# 
# Ans: One issue is that the reconstructed images can lose fine details, making them blurry or less sharp. Another problem is that it sometimes fails to separate different factors clearly. This means one latent variable might control more than one feature, making it hard to understand what each part of the latent space represents. Also, tuning the beta value is tricky; higher beta improves disentanglement but often at the cost of reconstruction quality, while lower beta may give better images but less clear separation of factors.
# 

# %% [markdown]
# <!-- FINAL MISSION HEADER -->
# <div style="text-align: center;">
#     <h1 style="color: rgb(255, 215, 0); font-size: 55px; font-weight: bold;">
#         🎯 <span style="color: white;">Mission Accomplished!</span>  
#     </h1>
#     <h2 style="color: rgb(255, 140, 0); font-size: 40px; font-weight: bold;">
#         <b>The Last of Us: DL Edition Ends Here</b> 🎮🔥
#     </h2>
#     <hr style="height: 10px; width: 80%; background: linear-gradient(to right, white, rgb(255, 140, 0), white); border: none; margin-top: 10px; margin-bottom: 10px;">
# </div>
# 
# <!-- STORY CLOSURE --> 
# <div style="padding: 30px; text-align: justify; font-size: 22px; color: white; background: rgba(0, 0, 0, 0.7); border-radius: 15px; width: 80%; margin: auto; box-shadow: 0px 0px 40px rgba(255, 140, 0, 0.7);">
# <b>A Tale of Vengeance, Survival, and AI</b>
# 
# The journey was <b>brutal</b>, the road was <b>long</b>, but <b>Ellie never faltered</b>.
# Through <b>blood, sweat, and deep learning</b>, she <b>saw her mission through to the end</b>.
# 
# With the power of Variational Autoencoders (VAEs), she uncovered hidden enemies, infiltrating their ranks and exposing those who lurked in the shadows.
# The very faction that had <b>taken everything from her</b>—that shattered the last remnants of her world—had now been <b>brought to justice</b>.
# 
# 💔 <b>For those she lost. For those still fighting.</b> 💔
# 
# </div> <!-- SUBHEADING: HOW DID ELLIE USE DL? --> <div style="padding: 10px; text-align: center;"> <h2 style="color: white; font-size: 30px; font-weight: bold;"> 🔬 <b>How Did Ellie Use Deep Learning to Complete Her Mission?</b> </h2> </div> <!-- CINEMATIC STORY BLOCK --> <div style="padding: 30px; text-align: justify; font-size: 22px; color: white; background: rgba(0, 0, 0, 0.7); border-radius: 15px; width: 80%; margin: auto; box-shadow: 0px 0px 40px rgba(255, 140, 0, 0.7);">
# 🔥 <b>In a world where survival depends on identifying the enemy</b>, Ellie turned to deep learning for answers.
# 
# - ✅ Using a β-VAE, she <b>disentangled hidden facial attributes</b> and <b>revealed those who wished to stay hidden</b>.
# - ✅ She experimented with different <b>β values</b>, learning how the model traded off between <b>accuracy and interpretability</b>.
# - ✅ With enough training, she <b>decoded the WLF's disguises</b>, proving that <b>no mask, no shadow, no camouflage could stop the power of AI</b>.
# 
# Her mission was not just about revenge, but about ensuring no one else suffered the same fate.
# With the memory of the fallen in her heart and science in her hands, she rewrote fate itself.
# 
# ⚔️ <b>Justice has been served.</b>
# 🎮 The Last of Us: AI Edition has come to a close.
# 
# 🥀 For those we've lost. For those still standing. 🥀
# 
# </div>
# 
# 
# <!-- MARKING CRITERIA -->
# <div style="padding: 10px; text-align: center;">
#     <h2 style="color: white; font-size: 35px; font-weight: bold;">
#         📜 <b>Marking Criteria & Submission Guidelines</b>  
#     </h2>
# </div>
# 
# <!-- CRITERIA BLOCK -->
# <div style="padding: 30px; text-align: justify; font-size: 22px; color: white;
#             background: rgba(0, 0, 0, 0.7); border-radius: 15px;
#             width: 80%; margin: auto;
#             box-shadow: 0px 0px 40px rgba(255, 140, 0, 0.7);">
# 
# ## 📊 **How Will You Be Evaluated?**  
# 
# Unlike previous parts of the assignment, there are **no strict benchmarks** (e.g., specific loss values).  
# This part is **open-ended**, meaning **multiple correct implementations exist**.  
# 
# ✅ **We will assess your work based on the following:**  
# - **Architecture choices**: How well have you structured your VAE model?  
# - **Results**: How effectively does your model learn meaningful latent representations?  
# - **Quality of Visualizations**: The true measure of your model's success is in its ability to **clearly separate different features**. Your plots should **demonstrate disentanglement effectively**.  
# 
# </div>
# 
# <!-- SUBMISSION REQUIREMENTS -->
# <div style="padding: 10px; text-align: center;">
#     <h2 style="color: white; font-size: 30px; font-weight: bold;">
#         📂 <b>Submission Requirements</b>  
#     </h2>
# </div>
# 
# <!-- SUBMISSION INSTRUCTIONS BLOCK -->
# <div style="padding: 30px; text-align: justify; font-size: 22px; color: white;
#             background: rgba(0, 0, 0, 0.7); border-radius: 15px;
#             width: 80%; margin: auto;
#             box-shadow: 0px 0px 40px rgba(255, 140, 0, 0.7);">
# 
# ### 1️⃣ **Model Checkpoints**  
# 📌 **Save all 4 trained models** (one for each β value) and upload them to **Dropbox** under your name.  
# - **File format**: `.zip`  
# - **Naming format**: `<RollNumber>_PA3_2_Models.zip`  
# 
# ### 2️⃣ **Notebook Submission**  
# 📌 Submit **this notebook and its Python file** (`.ipynb` and `.py`) on **LMS** with the rest of the assignment.  
# 
# </div>
# 
# <!-- FINAL CONGRATULATIONS BLOCK -->
# <div style="padding: 30px; font-size: 22px; color: white;
#             background: rgba(0, 0, 0, 0.7); border-radius: 15px;
#             width: 80%; margin: auto;
#             box-shadow: 0px 0px 40px rgba(255, 140, 0, 0.7); text-align: center; font-weight: bold;">
#     
# 🔥 **If you’ve made it this far, congratulations!**  
# You’ve successfully implemented a **β-VAE**, explored its impact, and helped Ellie on her quest.  
# 
# 👑 <b>You are now a true deep learning survivor.</b>  
# 
# 🚀 **See you in the next adventure!**  
# 
# </div>
# 

# %% [markdown]
# ~ A Saad Haroon Production


