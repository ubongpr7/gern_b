import os
import logging
import torch
from django.core.management.base import BaseCommand
from django.core.files.base import ContentFile
from django.conf import settings
from gern_apps.media_gen.models import MediaGenerationTask
from core.gern_app import AIGeneration
import tempfile
from typing import Optional
from PIL import Image

logger = logging.getLogger(__name__)

class BaseCommandUtils:
    """
    Utility class for common functionalities in management commands.
    """
    def download_from_s3(self, file_key: str) -> Optional[bytes]:
        """
        Download a file from S3 and return its content as bytes.
        """
        try:
            s3 = boto3.client(
                "s3",
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            )
            response = s3.get_object(Bucket=settings.AWS_STORAGE_BUCKET_NAME, Key=file_key)
            object_content = response["Body"].read()
            logger.info(f"Downloaded {file_key} from S3.")
            return object_content
        except Exception as e:
            logger.error(f"Failed to download {file_key} from S3: {e}")
            return None
    
    def save_image_to_task_output(self, task: MediaGenerationTask, image_content: bytes, file_name: str):
        """
        Saves image content to the task's output_file field.
        """
        try:
            task.output_file.save(file_name, ContentFile(image_content))
            logger.info(f"Image saved to task output_file: {file_name}")
        except Exception as e:
            logger.error(f"Failed to save image to task output_file: {e}")
            raise

class Command(BaseCommand, BaseCommandUtils):
    help = "Generate an image guided by a ControlNet image and text prompt"

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
            self.process_controlnet_generation(task)
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

    def process_controlnet_generation(self, task: MediaGenerationTask):
        """
        Generates an image guided by ControlNet based on the MediaGenerationTask parameters.

        Args:
            task (MediaGenerationTask): The task instance containing the prompt, ControlNet type, and image.
        """
        if not task.control_image or not task.controlnet_type:
            raise ValueError("control_image and controlnet_type are required for ControlNet generation.")
        
        try:
            # Get the AI generation class from Modal
            ai_generation_instance = AIGeneration()
            #TODO: load the controlnet pipeline here
            # if not ai_generation_instance.controlnet_pipeline:
            #     raise RuntimeError("Model is not loaded.")

            # Download input image
            control_image_content = self.download_from_s3(task.control_image.name)
            if not control_image_content:
                raise ValueError("Failed to download control_image from S3.")

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_image:
                temp_image.write(control_image_content)
                temp_image.flush()
                control_image = Image.open(temp_image.name)

                logger.info(f"Generating image with ControlNet type: {task.controlnet_type}, prompt: {task.prompt}, and image: {task.control_image.name}")

                #TODO: use ai_generation_instance.controlnet_pipeline
                # output_image = ai_generation_instance.controlnet_pipeline(
                #         prompt=task.prompt,
                #         negative_prompt=task.negative_prompt,
                #         image=control_image,
                #         width=task.width,
                #         height=task.height,
                #         num_inference_steps=task.num_inference_steps,
                #     ).images[0]
                # save the generated image
                # with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_output:
                #     output_image.save(temp_output.name)
                #     with open(temp_output.name, "rb") as image_file:
                #         file_content = image_file.read()
                #         self.save_image_to_task_output(task, file_content, f"controlnet_{task.id}.png")
                #         
                #     os.remove(temp_output.name)
                #     logger.info(f"Temp image file {temp_output.name} removed")
                os.remove(temp_image.name)
                logger.info(f"Temp file {temp_image.name} removed")
                task.update_progress(100)

        except RuntimeError as re:
            logger.error(f"RuntimeError: {re}")
            raise
        except Exception as e:
            logger.error(f"Error generating image with ControlNet: {e}")
            raise

