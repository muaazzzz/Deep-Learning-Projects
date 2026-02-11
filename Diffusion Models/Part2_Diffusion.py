# %% [markdown]
# # 🎨 Advanced Diffusion: Guidance and Control Techniques
# 
# ## 🚀 Objectives:
# - Use pretrained **Stable Diffusion** for text-to-image generation
# - Experiment with **guidance scales** and **prompt engineering**
# - Explore **inpainting**
# - Explore **Style Transfer**
# 
# ---
# 
# ## 🧰 Setup
# 
# > Run the cell below to install the necessary libraries: `diffusers`, `transformers`, `accelerate`, `safetensors`, `xformers`, and `controlnet_aux`.
# 
# These libraries will enable us to use pre-trained diffusion models and speed up inference using GPU acceleration.
# 
# 

# %%
# !pip install --upgrade diffusers transformers accelerate safetensors xformers controlnet_aux --quiet



# %% [markdown]
# # 🧪 Experiment 1: Classifier-Free Guidance in Stable Diffusion
# 
# In diffusion-based generative models, **Classifier-Free Guidance (CFG)** is a technique used to steer the generation process toward better image-text alignment without requiring an external classifier.
# 
# Here's how it works:
# - During training, the model occasionally replaces the text condition with an empty string (i.e., unconditional).
# - At inference time, it combines the conditional and unconditional predictions to guide the sample.
#   
# The guidance formula is:\
# prediction = uncond + scale * (cond - uncond)
# 
# 
# Where:
# - `cond` is the model's prediction with the prompt.
# - `uncond` is the model's prediction without the prompt.
# - `scale` (a float) controls the strength of the guidance.
# 
# A higher `guidance_scale` encourages the model to follow the prompt more closely, possibly at the cost of image diversity.
# 
# 
# 

# %%
from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline, StableDiffusionImg2ImgPipeline
import torch
from PIL import Image
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from PIL import ImageDraw
import cv2
import numpy as np
import requests
from io import BytesIO
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

# %% [markdown]
# ### ⚙️ Step 1: Load a Pretrained Stable Diffusion Model
# 
# Use `diffusers` from HuggingFace to load a pretrained Stable Diffusion pipeline.
# 
# 📌 **Implementation:**
# - Load the `"runwayml/stable-diffusion-v1-5"` model.
# - Set the pipeline to use `torch_dtype=torch.float16`.
# - Move the model to `"cuda"` and enable attention slicing for memory efficiency.
# 
# ### 🎛️ Step 2: Implement CFG Sampling
# 
# Define a function that:
# - Takes a prompt and a guidance scale.
# - Uses the pipeline to generate an image with the given CFG value.
# - Returns the generated image.
# 
# 📌 This provides the **CFG sampler**.
# 

# %%
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    use_safetensors=True
)
pipe = pipe.to(device)
pipe.enable_attention_slicing()

def generate_with_cfg(prompt, guidance_scale=7.5, num_inference_steps=50, seed=None):

    if seed is not None:
        torch.manual_seed(seed)
        generator = torch.Generator(device=device).manual_seed(seed)
    else:
        generator = None

    image = pipe(
        prompt=prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator
    ).images[0]

    return image

# %% [markdown]
# ### 🧪 Step 3: Analyze the Effect of CFG
# 
# Use the `generate_with_cfg` function to generate images for the same prompt with **different guidance scales**.
# 
# - Prompt: `"a futuristic cityscape at night"`
# - Try values of `guidance_scale`: `[1.0, 5.0, 7.5, 12.0]`
# - Try a custom prompt as well
# 
# 🎨 Display the results in a horizontal row of subplots.
# - Add titles showing the CFG scale.
# - Hide the axes.
# 

# %%
seed = 42
prompt = "a futuristic cityscape at night"
cfg_values = [1.0, 5.0, 7.5, 12.0]

plt.figure(figsize=(20, 5))

for i, cfg in enumerate(cfg_values):
    image = generate_with_cfg(prompt, guidance_scale=cfg, seed=seed)

    plt.subplot(1, len(cfg_values), i+1)
    plt.imshow(image)
    plt.title(f"CFG Scale: {cfg}")
    plt.axis('off')

plt.tight_layout()
plt.show()

custom_prompt = "Kurosaki ichigo fighting Aizen"
plt.figure(figsize=(20, 5))

for i, cfg in enumerate(cfg_values):
    image = generate_with_cfg(custom_prompt, guidance_scale=cfg, seed=seed)

    plt.subplot(1, len(cfg_values), i+1)
    plt.imshow(image)
    plt.title(f"CFG Scale: {cfg}")
    plt.axis('off')

plt.tight_layout()
plt.show()

# %% [markdown]
# # 🧩 Experiment 2: Image Inpainting with Diffusion Models
# 
# ## What is Inpainting?
# 
# **Image Inpainting** is the task of filling in missing or masked-out regions in an image in a way that is coherent and visually plausible. Diffusion models like Stable Diffusion can do this by generating new content in a masked region based on a text prompt.
# 
# You provide:
# - A **base image** (with a region you want to edit)
# - A **binary mask** (white = area to fill, black = area to preserve)
# - A **prompt** describing what should appear in the masked region
# 
# ---
# 
# ## Classifier-Free Guidance (CFG) in Inpainting
# 
# Just like in text-to-image generation, **Classifier-Free Guidance (CFG)** is used to control how strictly the model follows the text prompt during inpainting. A higher `guidance_scale` forces the model to match the prompt more strongly but may sacrifice image quality or realism.
# 
# 

# %% [markdown]
# ### 🧰 Step 1: Load Stable Diffusion Inpainting Pipeline
# 
# Use `StableDiffusionInpaintPipeline` from HuggingFace's `diffusers` library.
# 
# 📌 **Implementation:**
# - Load the pretrained model: `"stabilityai/stable-diffusion-2-inpainting"`.
# - Use the `fp16` revision.
# - Set `torch_dtype` to `float16` and move it to `"cuda"`.
# - Disable the `safety_checker` for faster setup.
# 
# Refer to the HuggingFace documentation or examples for help.

# %%
inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    revision="fp16",
    torch_dtype=torch.float16,
    safety_checker=None
)
inpaint_pipe = inpaint_pipe.to(device)

# %% [markdown]
# ### 🔎 Step 2: Find an Image and a Mask
# 
# Find an **image and corresponding binary mask** for inpainting.
# 
# Requirements:
# - The image must be **RGB** and resized to **512x512**.
# - The mask should be a **black-and-white image** (white = inpaint area).
# - Use any source: upload an image, or use URLs from a dataset or search.
# 
# 📌 Load both using `PIL.Image`, convert to RGB (for image) and resize both to `(512, 512)`.
# 

# %%
import numpy as np

image_url = "https://thumbs.dreamstime.com/b/high-resolution-image-full-moon-isolated-white-background-343595432.jpg"
response = requests.get(image_url)
image = Image.open(BytesIO(response.content))
image = image.convert("RGB").resize((512, 512))

mask = Image.new("RGB", (512, 512), "black")
from PIL import ImageDraw
draw = ImageDraw.Draw(mask)
draw.ellipse((106, 106, 406, 406), fill="white")

mask = mask.convert("L")

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(mask, cmap='gray')
plt.title("Mask (White = Area to Inpaint)")
plt.axis('off')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 🎨 Step 3: Inpaint the Masked Region
# 
# Use the inpainting pipeline to fill the masked region using a text prompt.
# 
# 💬 Prompt idea: `"a futuristic object"` or `"a fantasy landscape"` (just an example)
# 
# Optional:
# - Try different prompts to observe changes. (up to 3)
# - Comment on the guidance classifier value used and changes observed.
# 
# Display the original and inpainted images together.
# 

# %%
prompts = [
    "a futuristic object",
    "a glowing magical orb",
    "a cute cartoon character"
]

plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis('off')

for i, prompt in enumerate(prompts):
    guidance_scale = 7.5

    inpainted_image = inpaint_pipe(
        prompt=prompt,
        image=image,
        mask_image=mask,
        guidance_scale=guidance_scale,
        num_inference_steps=50
    ).images[0]

    plt.subplot(2, 2, i+2)
    plt.imshow(inpainted_image)
    plt.title(f"Prompt: '{prompt}'\nGuidance Scale: {guidance_scale}")
    plt.axis('off')

plt.tight_layout()
plt.show()

prompt = "a death star"
guidance_scales = [1.0, 5.0, 7.5, 12.0]

plt.figure(figsize=(15, 10))
plt.subplot(2, 3, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(mask, cmap='gray')
plt.title("Mask")
plt.axis('off')

for i, guidance_scale in enumerate(guidance_scales):
    inpainted_image = inpaint_pipe(
        prompt=prompt,
        image=image,
        mask_image=mask,
        guidance_scale=guidance_scale,
        num_inference_steps=50
    ).images[0]

    plt.subplot(2, 3, i+3)
    plt.imshow(inpainted_image)
    plt.title(f"Guidance Scale: {guidance_scale}")
    plt.axis('off')

plt.suptitle(f"Prompt: '{prompt}'", fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()

# lower guidance classifier values result in less adherence to the prompt and the images look more natural, overall more creativity
# higher classifier values result in more adherence to the prompt.

# %% [markdown]
# # 🎨 Experiment 3: Style Transfer using ControlNet + IP-Adapter
# 
# ## What is Style Transfer?
# 
# **Style Transfer** refers to the process of applying the artistic style of one image (e.g., a painting) to the content of another image (e.g., a photograph), generating a visually coherent blend of both.
# 
# In this task, we combine:
# - **ControlNet** to preserve the structure or edges of the original image.
# - **IP-Adapter** to influence the visual style using a reference (style) image.
# 
# This gives us fine-grained control over **what** the image contains (via prompts and edge maps) and **how** it looks (via the style image).
# 
# ---
# 
# ### 🔗 Required Models
# 
# You must load the following pre-trained models from Hugging Face:
# 
# - 🔧 **ControlNet Canny Detector**:  
#   [lllyasviel/sd-controlnet-canny](https://huggingface.co/lllyasviel/sd-controlnet-canny)
# 
# - 🖼️ **Base Stable Diffusion Model (Absolute Reality)**:  
#   [Yntec/AbsoluteReality](https://huggingface.co/Yntec/AbsoluteReality)
# 
# - 🎭 **IP Adapter Models** (for style transfer):  
#   [h94/IP-Adapter](https://huggingface.co/h94/IP-Adapter)
# 
# You will use these models with `StableDiffusionControlNetPipeline` from the `diffusers` library.
# 

# %% [markdown]
# # 🧪 Implementation
# 
# Perform style transfer using ControlNet and an IP Adapter. Follow these steps:
# 
# ---
# 
# ### 🔹 Step 1: Load Models
# 
# Load the following using the appropriate functions:
# - `ControlNetModel` for edge detection
# - `StableDiffusionControlNetPipeline` as your generation pipeline
# - Use `.load_ip_adapter()` to load the IP Adapter for style guidance
# 
# ---
# 
# ### 🔹 Step 2: Choose Images
# 
# - Select a **style image** (e.g., a painting, drawing, or themed artwork).
# - Select a **base image** (e.g., a portrait or landscape).
# - Resize the base image to `768x768` for consistent results.
# 
# ---
# 
# ### 🔹 Step 3: Generate Edge Map
# 
# Use the **CannyDetector** from `controlnet_aux` to extract edges from the base image.
# This serves as the structural guide for generation.
# 
# ---
# 
# ### 🔹 Step 4: Define Your Prompt
# 
# Write a rich, descriptive prompt that communicates the **content** of your output (e.g., "girl in a red jacket standing in rain").
# 
# You may also use a `negative_prompt` like `"low quality"` to suppress undesired features.
# 
# ---
# 
# ### 🔹 Step 5: Generate Styled Images
# 
# Use the pipeline to generate new images, blending:
# - Structure from the **edge map**
# - Style from the **style image**
# - Content from your **prompt**
# 
# ---
# 
# ### ✅ Requirements
# 
# - Generate at least **2 different sets** of images.
#   - Each set should use a different **style image** and a different **base image**.
# - For each set:
# 
# 

# %%
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
# !pip install controlnet_aux
from controlnet_aux import CannyDetector
from diffusers.utils import load_image

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.float32

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=torch_dtype
)

pipeline = StableDiffusionControlNetPipeline.from_pretrained(
    "Yntec/AbsoluteReality",
    controlnet=controlnet,
    torch_dtype=torch_dtype,
    safety_checker=None
)

pipeline.to(device)

pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")

canny_detector = CannyDetector()

def display_images(images_list, titles_list, figsize=(15, 10)):
    num_images = len(images_list)
    plt.figure(figsize=figsize)

    for i, (image, title) in enumerate(zip(images_list, titles_list)):
        plt.subplot(1, num_images, i+1)
        plt.imshow(image)
        plt.title(title)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def generate_styled_image(base_image, style_image, prompt, negative_prompt="low quality, blurry, distorted",
                         guidance_scale=7.5, ip_adapter_scale=0.8,
                         num_inference_steps=30, seed=None):

    base_image_resized = base_image.resize((768, 768))

    edge_image = canny_detector(base_image_resized, low_threshold=100, high_threshold=200)

    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    else:
        generator = None

    output = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=edge_image,
        ip_adapter_image=style_image,
        guidance_scale=guidance_scale,
        ip_adapter_scale=ip_adapter_scale,
        num_inference_steps=num_inference_steps,
        generator=generator
    )

    return output.images[0], edge_image, base_image_resized

base_url_1 = "https://static.wikia.nocookie.net/p__/images/4/44/Megumi_Fushiguro_%28Anime_3%29.png/revision/latest?cb=20240526141231&path-prefix=protagonist"
response = requests.get(base_url_1)
base_image_1 = Image.open(BytesIO(response.content)).convert("RGB")

style_url_1 = "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1200px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg"
response = requests.get(style_url_1)
style_image_1 = Image.open(BytesIO(response.content)).convert("RGB")

prompt_1 = "A detailed portrait of a woman with flowing hair, artistic, dreamy atmosphere"
negative_prompt_1 = "low quality, distorted, blurry, deformed, disfigured"

print("Generating first styled image...")
styled_image_1, edge_map_1, base_resized_1 = generate_styled_image(
    base_image_1,
    style_image_1,
    prompt_1,
    negative_prompt_1,
    guidance_scale=7.5,
    ip_adapter_scale=0.7,
    seed=42
)

display_images(
    [base_resized_1, style_image_1, edge_map_1, styled_image_1],
    ["Base Image", "Style Image (Van Gogh)", "Edge Map", "Generated Result"],
    figsize=(20, 5)
)

base_url_2 = "https://www.easyartstart.com/wp-content/uploads/2024/04/sword_drawing_tutorial.png"
response = requests.get(base_url_2)
base_image_2 = Image.open(BytesIO(response.content)).convert("RGB")

style_url_2 = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS_AYXxAF_yxEIeZ73zajuuWG0tkwBHFMLq_g&s"
response = requests.get(style_url_2)
style_image_2 = Image.open(BytesIO(response.content)).convert("RGB")

prompt_2 = "A serene landscape with a tree at sunset, dramatic sky, traditional Japanese art style"
negative_prompt_2 = "low quality, distorted, artificial, cartoon"

print("Generating second styled image...")
styled_image_2, edge_map_2, base_resized_2 = generate_styled_image(
    base_image_2,
    style_image_2,
    prompt_2,
    negative_prompt_2,
    guidance_scale=8.0,
    ip_adapter_scale=0.9,
    seed=123
)

display_images(
    [base_resized_2, style_image_2, edge_map_2, styled_image_2],
    ["Base Image", "Style Image (Hokusai)", "Edge Map", "Generated Result"],
    figsize=(20, 5)
)


