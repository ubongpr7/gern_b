import os
import sys
import modal
import torch
from diffusers import StableDiffusionPipeline

modal.config.token_id = os.getenv("MODAL_TOKEN_ID")
modal.config.token_secret = os.getenv("MODAL_TOKEN_SECRET")

image = modal.Image.from_registry("nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04") \
    .pip_install("torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121") \
    .pip_install("diffusers transformers accelerate")

app = modal.App(
    name="diffusion-image-generator",
    image=image
)

# Function to generate images
@app.function(
    gpu=modal.gpu.H100(), 
    cpu=16,  
    memory=32768,  
    timeout=1800  # 30 minutes
)
def generate_image(prompt: str, output_path: str):
    """Generates an image from text using Stable Diffusion."""
    model_id = "runwayml/stable-diffusion-v1-5"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    pipe.to(device)

    image = pipe(prompt).images[0]
    image.save(output_path)
    return f"Image saved at {output_path}"

