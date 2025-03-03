import os
import sys
import modal
import django
from PIL import Image
from django.core.management import call_command
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

modal.config.token_id = os.getenv("MODAL_TOKEN_ID") 
modal.config.token_secret = os.getenv("MODAL_TOKEN_SECRET")

if not modal.config.token_id or not modal.config.token_secret:  
    logger.error("Modal token ID or secret is not set. Please set the environment variables.")    
    sys.exit(1)

MINUTES = 60
CACHE_DIR = "/cache"


# Build the custom image from your Dockerfile
custom_image = modal.Image.from_dockerfile("./Dockerfile.web")
modal_token_id = os.getenv("MODAL_TOKEN_ID")
modal_token_secret = os.getenv("MODAL_TOKEN_SECRET")
modal_secret = modal.Secret.from_dict({
    "MODAL_TOKEN_ID": modal_token_id,
    "MODAL_TOKEN_SECRET": modal_token_secret,
})
app = modal.App("text-to-image-1",secrets=[modal_secret])

# Add additional dependencies if needed
image = (
    custom_image
    .pip_install(
        "accelerate==0.33.0",
        "diffusers==0.31.0",
        "fastapi[standard]==0.115.4",
        "huggingface-hub[hf_transfer]==0.25.2",
        "sentencepiece==0.2.0",
        "torch==2.5.1",
        "torchvision==0.20.1",
        "transformers~=4.44.0",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",  # faster downloads
            "HF_HUB_CACHE": CACHE_DIR,
        }
    )
)

MODEL_ID = "adamo1139/stable-diffusion-3.5-large-turbo-ungated"
MODEL_REVISION_ID = "9ad870ac0b0e5e48ced156bb02f85d324b7275d2"

cache_volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)

def _setup_django():
    """Set up Django environment."""
    if not django.apps.apps.ready:
        try:
            sys.path.insert(0, "/app")
            os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings")
            django.setup()
        except Exception as e:
            logger.error(f"Error setting up Django: {e}")
            raise

def _execute_django_command(command: str, task_id: int):
    """Execute a Django management command."""
    try:
        _setup_django()
        call_command(command, task_id)
    except Exception as e:
        logger.error(f"Error executing {command}: {e}")
        raise

@app.function(
    image=image,
    gpu="H100",
    timeout=10 * MINUTES,
    volumes={CACHE_DIR: cache_volume},
)
def text_to_image(task_id: int):
    """Generate image from text using Django command."""
    _execute_django_command("convert_text_to_image", task_id)

# @app.function(
#     image=image,
#     gpu="H100",
#     timeout=10 * MINUTES,
#     volumes={CACHE_DIR: cache_volume},
# )
# def text_to_video( task_id: int):
#     """Generate video from text using Django command."""
#     _execute_django_command("convert_text_to_video", task_id)


# @app.function(
#     image=image,
#     gpu="H100",
#     timeout=10 * MINUTES,
#     volumes={CACHE_DIR: cache_volume},
# )
# def image_inpainting_call( task_id: int):
#     """Perform image inpainting using Django command."""
#     _execute_django_command("image_inpainting", task_id)


# @app.function(
#     image=image,
#     gpu="H100",
#     timeout=10 * MINUTES,
#     volumes={CACHE_DIR: cache_volume},
# )
# def object_detection_call( job_id: int):
#     """Detect objects in an image using Django command."""
#     _execute_django_command("object_detection", job_id)


# @app.function(
#     image=image,
#     gpu="H100",
#     timeout=10 * MINUTES,
#     volumes={CACHE_DIR: cache_volume},
# )
# def background_removal_call( job_id: int):
#     """Remove background from an image using Django command."""
#     _execute_django_command("background_removal", job_id)


# @app.function(
#     image=image,
#     gpu="H100",
#     timeout=10 * MINUTES,
#     volumes={CACHE_DIR: cache_volume},
# )
# def style_transfer_call( job_id: int):
#     """Apply style transfer using Django command."""
#     _execute_django_command("style_transfer", job_id)

# @app.function(
#     image=image,
#     gpu="H100",
#     timeout=10 * MINUTES,
#     volumes={CACHE_DIR: cache_volume},
# )
# def image_to_video_call( task_id: int):
#     """Generate video from image using Django command."""
#     _execute_django_command("image_to_video", task_id)


# @app.function(
#     image=image,
#     gpu="H100",
#     timeout=10 * MINUTES,
#     volumes={CACHE_DIR: cache_volume},
# )
# def controlnet_call( task_id: int):
#     """Generate image with ControlNet using Django command."""
#     _execute_django_command("controlnet_generation", task_id)


# @app.function(
#     image=image,
#     gpu="H100",
#     timeout=10 * MINUTES,
#     volumes={CACHE_DIR: cache_volume},
# )
# def upscale_video_call( job_id: int):
#     """Upscale video using Django command."""
#     _execute_django_command("upscale_video", job_id)
