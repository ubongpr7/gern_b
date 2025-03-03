import modal
import os
import requests

modal.config.token_id = os.getenv("MODAL_TOKEN_ID")
modal.config.token_secret = os.getenv("MODAL_TOKEN_SECRET")
model_volume = modal.Volume.persisted("diffusion-models")

image = modal.Image.debian_slim().pip_install(
    "gfpgan", "realesrgan", "torch", "transformers", "diffusers", "opencv-python"
)

app = modal.App(
    name="diffusion-models",
    image=image,
    volumes={"/models": model_volume},  # Mount volume at /models
)

@app.function()
def download_models():
    def download_file(url, save_path):
        if not os.path.exists(save_path):
            print(f"Downloading {url}...")
            response = requests.get(url, stream=True)
            with open(save_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Saved: {save_path}")
        else:
            print(f"Already exists: {save_path}")

    # Define model paths inside the Modal volume
    model_paths = {
        "GFPGAN": ("/models/GFPGANv1.4.pth", "https://github.com/TencentARC/GFPGAN/releases/download/v1.4/GFPGANv1.4.pth"),
        "RealESRGAN": ("/models/RealESRGAN_x4plus.pth", "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"),
        "Stable Diffusion": ("/models/sd-v1-4.ckpt", "https://huggingface.co/CompVis/stable-diffusion-v1-4/resolve/main/sd-v1-4.ckpt"),
        "CodeFormer": ("/models/codeformer.pth", "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"),
        "RealESRGAN Anime": ("/models/RealESRGAN_x4plus_anime_6B.pth", "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus_anime_6B.pth"),
        "Dlib Face Landmarks": ("/models/shape_predictor_68_face_landmarks.dat", "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"),
    }

    # Download all models
    for name, (path, url) in model_paths.items():
        download_file(url, path)
