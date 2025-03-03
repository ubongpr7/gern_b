import os
import logging
import torch
from django.core.management.base import BaseCommand
from django.core.files.base import ContentFile
from django.conf import settings
from gern_apps.media_gen.models import MediaGenerationTask
from diffusers import DiffusionPipeline
from PIL import Image
import tempfile

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = "Generate an image from text input using Stable Diffusion"

    def add_arguments(self, parser):
        parser.add_argument("task_id", type=str, help="ID of the MediaGenerationTask")

    def handle(self, *args, **kwargs):
        task_id = kwargs["task_id"]

        try:
            task = MediaGenerationTask.objects.get(id=task_id)
        except MediaGenerationTask.DoesNotExist:
            logger.error(f"MediaGenerationTask with ID {task_id} not found.")
            self.stdout.write(self.style.ERROR(f"MediaGenerationTask with ID {task_id} not found."))
            return

        task.status = "processing"
        task.save()
        self.stdout.write(self.style.SUCCESS(f"Start processing task {task_id}"))

        try:
            self.process_text_to_image(task)
            task.status = "completed"
            task.progress = 100
            task.save()
            self.stdout.write(self.style.SUCCESS(f"Task {task_id} completed successfully."))

        except Exception as e:
            logger.exception(f"Error processing task {task_id}: {e}")
            task.status = "failed"
            task.error_message = str(e)
            task.save()
            self.stdout.write(self.style.ERROR(f"Task {task_id} failed: {e}"))

    def save_image_to_task_output(self, task: MediaGenerationTask, image_content: bytes, file_name: str):
        """
        Saves image content to the task's output_file field.

        Args:
            task (MediaGenerationTask): The task instance.
            image_content (bytes): The content of the image to save.
            file_name (str): The name of the file to save.
        """
        try:
            # Save the file locally
            media_dir = os.path.join(settings.MEDIA_ROOT)
            os.makedirs(media_dir, exist_ok=True)  # Create the directory if it doesn't exist

            local_file_path = os.path.join(media_dir, file_name)
            with open(local_file_path, "wb") as f:
                f.write(image_content)

            # Save the file path to the task's output_file field
            task.output_file.name = file_name
            task.save()
            logger.info(f"Image saved to task output_file: {file_name}")
        except Exception as e:
            logger.error(f"Failed to save image to task output_file: {e}")
            raise

    def process_text_to_image(self, task: MediaGenerationTask):
        """
        Generates an image based on the MediaGenerationTask parameters using Stable Diffusion.

        Args:
            task (MediaGenerationTask): The task instance containing the prompt and settings.
        """
        MODEL_ID = "adamo1139/stable-diffusion-3.5-large-turbo-ungated"
        MODEL_REVISION_ID = "9ad870ac0b0e5e48ced156bb02f85d324b7275d2"
        import diffusers

        try:
            pipe = diffusers.StableDiffusion3Pipeline.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION_ID,
            torch_dtype=torch.bfloat16,
        ).to("cuda")

            # Load the Stable Diffusion model
            logger.info("Loading Stable Diffusion model...")
            # pipe = DiffusionPipeline.from_pretrained(
            #     # "stabilityai/stable-diffusion-xl-base-1.0",
            #     "runwayml/stable-diffusion-v1-5",
            #     torch_dtype=torch.float16,
            # ).to("cuda")
            logger.info("Stable Diffusion model loaded successfully.")

            # Generate the image
            logger.info(f"Generating image with prompt: {task.prompt}")
            image = pipe(
                prompt=task.prompt,
                negative_prompt=task.negative_prompt,
                height=task.height,
                width=task.width,
                num_inference_steps=task.num_inference_steps,
                guidance_scale=task.guidance_scale,
                generator=torch.manual_seed(task.seed) if task.seed != -1 else None,
            ).images[0]

            # Save the generated image
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_image:
                image.save(temp_image.name)

                # Save the generated image locally
                with open(temp_image.name, "rb") as image_file:
                    file_content = image_file.read()
                    task.output_file.save(f"text_to_image_{task.id}.png", ContentFile(file_content))

                logger.info(f"Image saved locally")

                # Clean up the temporary image file
                os.remove(temp_image.name)
                logger.info(f"Temp image file {temp_image.name} removed")

        except RuntimeError as re:
            logger.error(f"RuntimeError: {re}")
            raise

        except Exception as e:
            logger.error(f"Error generating or saving image: {e}")
            raise