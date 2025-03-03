import os
from django.core.management.base import BaseCommand
from diffusers import AnimateDiffPipeline
import torch

class Command(BaseCommand):
    help = "Generate an animation from text input"

    def handle(self, *args, **kwargs):
        model = AnimateDiffPipeline.from_pretrained(
            "TencentARC/AnimateDiff", torch_dtype=torch.float16
        ).to("cuda")

        text_prompt = "A futuristic city with flying cars at sunset."
        self.stdout.write(self.style.SUCCESS(f"Generating animation for: {text_prompt}"))

        # Generate animation
        video = model(text_prompt, num_inference_steps=50)
        output_path = "output_animation.mp4"
        video.save(output_path)

        self.stdout.write(self.style.SUCCESS(f"Animation saved to {output_path}"))





import os
from django.core.management.base import BaseCommand
from diffusers import AnimateDiffPipeline
import torch
from PIL import Image

class Command(BaseCommand):
    help = "Generate an animation from an image input"

    def add_arguments(self, parser):
        parser.add_argument("image_path", type=str, help="Path to the input image")

    def handle(self, *args, **options):
        image_path = options["image_path"]

        if not os.path.exists(image_path):
            self.stdout.write(self.style.ERROR(f"Image file not found: {image_path}"))
            return

        model = AnimateDiffPipeline.from_pretrained(
            "TencentARC/AnimateDiff", torch_dtype=torch.float16
        ).to("cuda")

        image = Image.open(image_path).convert("RGB")

        self.stdout.write(self.style.SUCCESS(f"Generating animation for image: {image_path}"))

        # Generate animation
        video = model(image, num_inference_steps=50)
        output_path = "output_image_animation.mp4"
        video.save(output_path)

        self.stdout.write(self.style.SUCCESS(f"Animation saved to {output_path}"))






import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

class StableDiffusionGenerator:
    def __init__(self, model_name="stabilityai/stable-diffusion-xl-base-1.0"):
        """Initialize the Stable Diffusion pipeline with custom settings."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model with a chosen scheduler
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16
        ).to(self.device)

        # Change scheduler to Euler (better for quality)
        self.pipe.scheduler = EulerDiscreteScheduler.from_pretrained(model_name, subfolder="scheduler")

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        height: int = 512,
        width: int = 512,
        steps: int = 50,
        guidance_scale: float = 7.5,
        seed: int = None,
        batch_size: int = 1,
    ):
        """Generate an image using Stable Diffusion with user-defined settings."""

        # Set the seed for reproducibility
        generator = torch.manual_seed(seed) if seed else None

        # Generate image
        images = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            num_images_per_prompt=batch_size,
            generator=generator,
        ).images  # Returns a list of PIL images

        # Save the first image
        images[0].save("output_image.png")
        return "output_image.png"

# Example Usage
generator = StableDiffusionGenerator()
output_path = generator.generate_image(
    prompt="A futuristic cyberpunk city at night",
    negative_prompt="blurry, low quality",
    height=1024,
    width=1024,
    steps=40,
    guidance_scale=8.0,
    seed=1234
)
print(f"Image saved at {output_path}")




from django.core.management.base import BaseCommand
from your_app.stable_diffusion import StableDiffusionGenerator  # Import your class

class Command(BaseCommand):
    help = "Generate an image using Stable Diffusion"

    def add_arguments(self, parser):
        parser.add_argument("prompt", type=str, help="Text prompt for image generation")
        parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt")
        parser.add_argument("--height", type=int, default=512, help="Image height")
        parser.add_argument("--width", type=int, default=512, help="Image width")
        parser.add_argument("--steps", type=int, default=50, help="Number of inference steps")
        parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale")
        parser.add_argument("--seed", type=int, help="Random seed")

    def handle(self, *args, **options):
        generator = StableDiffusionGenerator()
        output_path = generator.generate_image(
            prompt=options["prompt"],
            negative_prompt=options["negative_prompt"],
            height=options["height"],
            width=options["width"],
            steps=options["steps"],
            guidance_scale=options["guidance_scale"],
            seed=options.get("seed")
        )
        self.stdout.write(self.style.SUCCESS(f"Image saved at {output_path}"))



from django.http import JsonResponse
from your_app.stable_diffusion import StableDiffusionGenerator

def generate_image_api(request):
    """API to generate an image based on user input."""
    prompt = request.GET.get("prompt", "A beautiful landscape")
    negative_prompt = request.GET.get("negative_prompt", "")
    height = int(request.GET.get("height", 512))
    width = int(request.GET.get("width", 512))
    steps = int(request.GET.get("steps", 50))
    guidance_scale = float(request.GET.get("guidance_scale", 7.5))
    seed = int(request.GET.get("seed", 42))

    generator = StableDiffusionGenerator()
    image_path = generator.generate_image(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        steps=steps,
        guidance_scale=guidance_scale,
        seed=seed
    )

    return JsonResponse({"image_url": f"/media/{image_path}"})



import torch
import os
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from gfpgan import GFPGAN
from realesrgan import RealESRGAN

class StableDiffusionGenerator:
    def __init__(self, model_name="stabilityai/stable-diffusion-xl-base-1.0"):
        """Initialize Stable Diffusion with optimizations"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load pipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_name, torch_dtype=torch.float16
        ).to(self.device)

        # Use a different scheduler for better quality
        self.pipe.scheduler = EulerDiscreteScheduler.from_pretrained(model_name, subfolder="scheduler")

        # Enable memory optimizations
        self.pipe.enable_xformers_memory_efficient_attention()
        self.pipe.enable_attention_slicing()
        self.pipe.enable_vae_tiling()

        # Load face enhancer and upscaler
        model_dir = "/models"
        self.face_enhancer = GFPGAN(model_path=os.path.join(model_dir, "GFPGANv1.4.pth"), upscale=2)
        self.esrgan = RealESRGAN(model_path=os.path.join(model_dir, "RealESRGAN_x4plus.pth"), scale=4)

    def generate_image(self, prompt, negative_prompt="", height=512, width=512, steps=50, guidance_scale=7.5, seed=None, batch_size=1):
        """Generate an image with all advanced settings."""
        generator = torch.manual_seed(seed) if seed else None

        images = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            num_images_per_prompt=batch_size,
            generator=generator,
        ).images

        # Apply face enhancement and upscaling
        enhanced_image = self.face_enhancer.enhance(images[0])
        upscaled_image = self.esrgan.enhance(enhanced_image)

        output_path = "output_image.png"
        upscaled_image.save(output_path)
        return output_path

# Example Usage
generator = StableDiffusionGenerator()
output_path = generator.generate_image(
    prompt="A cyberpunk city at night",
    negative_prompt="blurry, low quality",
    height=1024, width=1024, steps=50, guidance_scale=8.0, seed=1234
)
print(f"Image saved at {output_path}")





import os
import sys
import modal
import django
import torch
from PIL import Image
from django.core.management import call_command
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

modal.config.token_id = os.getenv("MODAL_TOKEN_ID")
modal.config.token_secret = os.getenv("MODAL_TOKEN_SECRET")

if not modal.config.token_id or not modal.config.token_secret:
    logger.error("Modal token ID or secret is not set. Please set the environment variables.")
    sys.exit(1)

# Define Modal Image
image = modal.Image.debian_slim().pip_install(
    "gfpgan", "realesrgan", "diffusers", "rembg", "torchvision"
)

# Define Modal App
stub = modal.Stub("ai_generation_service")

@stub.cls(
    image=image,
    gpu=modal.gpu.A100(),
    # cpu=32,
    # memory=65536,
    timeout=3600
)
class AIGeneration:
    @modal.build
    @modal.enter
    def setup(self):
        """Preload all necessary models for faster execution"""
        from diffusers import StableDiffusionPipeline, AnimateDiffPipeline
        from realesrgan import RealESRGAN
        from gfpgan import GFPGAN
        import rembg
        
        model_dir = "/models"
        
        if not os.path.exists(model_dir):
            logger.error(f"Model directory {model_dir} does not exist.")
            sys.exit(1)

        try:
            # Load pre-installed models
            self.sdxl_1_0 = StableDiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
            ).to("cuda")

            self.modelscope = AnimateDiffPipeline.from_pretrained(
                "TencentARC/AnimateDiff", torch_dtype=torch.float16
            ).to("cuda")

            self.face_enhancer = GFPGAN(model_path=os.path.join(model_dir, "GFPGANv1.4.pth"), upscale=2)
            self.esrgan = RealESRGAN(model_path=os.path.join(model_dir, "RealESRGAN_x4plus.pth"), scale=4)
            self.bg_remover = rembg.new_session()
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            sys.exit(1)
    
    @modal.method()
    def generate_animation(self, prompt: str, steps: int = 50, output_path: str = "output_animation.mp4"):
        """Generate an animation based on a text prompt."""
        if not hasattr(self, "modelscope"):
            raise RuntimeError("Model is not loaded. Run setup() first.")

        try:
            logger.info(f"Generating animation for prompt: {prompt}")
            video = self.modelscope(prompt, num_inference_steps=steps)
            video.save(output_path)
            logger.info(f"Animation saved at {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error generating animation: {e}")
            raise

    @modal.method()
    def text_to_image(self, task_id: int):
        """Generate an image from text input"""
        try:
            self._setup_django()
            call_command("convert_text_to_image", task_id)
        except Exception as e:
            logger.error(f"Error generating image from text: {e}")
            raise
    
    @modal.method()
    def text_to_video(self, task_id: int):
        """Generate a video from text input"""
        try:
            self._setup_django()
            call_command("convert_text_to_video", task_id)
        except Exception as e:
            logger.error(f"Error generating video from text: {e}")
            raise
    
    @modal.method()
    def upscale_image(self, image_path: str):
        """Upscale an image using Real-ESRGAN"""
        try:
            image = Image.open(image_path).convert("RGB")
            enhanced_image = self.esrgan.enhance(image)
            output_path = image_path.replace(".png", "_upscaled.png")
            enhanced_image.save(output_path)
            return output_path
        except Exception as e:
            logger.error(f"Error upscaling image: {e}")
            raise
    
    @modal.method()
    def restore_face(self, image_path: str):
        """Restore faces in an image using GFPGAN"""
        try:
            image = Image.open(image_path).convert("RGB")
            restored_image, _ = self.face_enhancer.enhance(image)
            output_path = image_path.replace(".png", "_restored.png")
            restored_image.save(output_path)
            return output_path
        except Exception as e:
            logger.error(f"Error restoring face: {e}")
            raise
    
    @modal.method()
    def remove_background(self, image_path: str):
        """Remove background from an image"""
        try:
            image = Image.open(image_path).convert("RGBA")
            output = self.bg_remover.process(image)
            output_path = image_path.replace(".png", "_nobg.png")
            output.save(output_path)
            return output_path
        except Exception as e:
            logger.error(f"Error removing background: {e}")
            raise

    @modal.method()
    def apply_style_transfer(self, content_image: str, style_image: str):
        """Apply artistic style transfer"""
        try:
            import torchvision.transforms as transforms
            from torchvision.models.segmentation import deeplabv3_resnet50

            content = Image.open(content_image).convert("RGB")
            style = Image.open(style_image).convert("RGB")
            model = deeplabv3_resnet50(pretrained=True).eval()

            transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor()
            ])
            content_tensor = transform(content).unsqueeze(0)
            style_tensor = transform(style).unsqueeze(0)
            
            output = model(content_tensor)
            output_path = content_image.replace(".png", "_styled.png")
            transforms.ToPILImage()(output["out"]).save(output_path)
            return output_path
        except Exception as e:
            logger.error(f"Error applying style transfer: {e}")
            raise
    
    @modal.method()
    def image_inpainting(self, image_path: str, mask_path: str):
        """Fill missing parts of an image"""
        try:
            from diffusers import StableDiffusionInpaintPipeline
            inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting"
            ).to("cuda")
            
            image = Image.open(image_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")
            result = inpaint_pipe(image, mask)
            output_path = image_path.replace(".png", "_inpainted.png")
            result.save(output_path)
            return output_path
        except Exception as e:
            logger.error(f"Error inpainting image: {e}")
            raise
    
    @modal.method()
    def object_detection(self, image_path: str):
        """Detect objects in an image"""
        try:
            from torchvision.models.detection import fasterrcnn_resnet50_fpn
            from torchvision.transforms import functional as F
            
            model = fasterrcnn_resnet50_fpn(pretrained=True).eval()
            image = Image.open(image_path).convert("RGB")
            img_tensor = F.to_tensor(image).unsqueeze(0)
            
            output = model(img_tensor)
            detected_objects = output[0]['labels'].tolist()
            return detected_objects
        except Exception as e:
            logger.error(f"Error detecting objects: {e}")
            raise
    
    def _setup_django(self):
        try:
            sys.path.insert(0, "/app")
            os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings")
            django.setup()
        except Exception as e:
            logger.error(f"Error setting up Django: {e}")
            raise