FROM python:3.10-slim

# Set environment variables
ENV MODEL_DIR=/models
ENV PYTHONUNBUFFERED=1

# Install dependencies
RUN apt-get update && apt-get install -y git wget unzip && \
    pip install --no-cache-dir torch torchvision diffusers realesrgan gfpgan rembg

# Create models directory
RUN mkdir -p $MODEL_DIR

# Download models
WORKDIR $MODEL_DIR
RUN wget -O GFPGANv1.4.pth https://github.com/TencentARC/GFPGAN/releases/download/v1.4/GFPGANv1.4.pth && \
    wget -O RealESRGAN_x4plus.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth && \
    wget -O sd-v1-4.ckpt https://huggingface.co/CompVis/stable-diffusion-v1-4/resolve/main/sd-v1-4.ckpt

# Set work directory
WORKDIR /app
COPY . /app

