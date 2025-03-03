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

    def save_video_to_task_output(self, task: MediaGenerationTask, video_path: str, file_name: str):
        """
        Saves a generated video to the task's output_file field.
        """
        try:
            with open(video_path, "rb") as video_file:
                task.output_file.save(file_name, ContentFile(video_file.read()))
            logger.info(f"Video saved to task output_file: {file_name}")
        except Exception as e:
            logger.error(f"Failed to save video to task output_file: {e}")
            raise

class Command(BaseCommand, BaseCommandUtils):
    help = "Generate a video from an image input and text prompt using AnimateDiff"

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
            self.process_image_to_video(task)
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

    def process_image_to_video(self, task: MediaGenerationTask):
        """
        Generates a video based on the MediaGenerationTask parameters.

        Args:
            task (MediaGenerationTask): The task instance containing the image and prompt.
        """
        if not task.image_input:
            raise ValueError("image_input is required for image to video.")

        try:
            # Get the AI generation class from Modal
            ai_generation_instance = AIGeneration()
            if not ai_generation_instance.modelscope:
                raise RuntimeError("Model is not loaded.")
            
            # Download input image
            image_content = self.download_from_s3(task.image_input.name)
            
            if not image_content:
                raise ValueError("Failed to download image_input from S3.")

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_image, tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
                temp_image.write(image_content)
                temp_image.flush()
                # Save the image to the temporary file.
                image = Image.open(temp_image.name)
                
                logger.info(f"Generating video with image: {task.image_input.name} and prompt: {task.prompt}")
                ai_generation_instance.modelscope.to("cpu") #use cpu, since we are using temp files.
                video = ai_generation_instance.generate_image_animation(
                        prompt=task.prompt,
                        image=image,
                        steps=task.num_inference_steps,
                        video_output_path=temp_video.name
                    )
                ai_generation_instance.modelscope.to("cuda")# return to gpu, after generating the animation.

                # Save the generated video to s3.
                self.save_video_to_task_output(task, temp_video.name, f"image_to_video_{task.id}.mp4")

                os.remove(temp_video.name)
                logger.info(f"Temp video file {temp_video.name} removed")
                os.remove(temp_image.name)
                logger.info(f"Temp file {temp_image.name} removed")

        except RuntimeError as re:
            logger.error(f"RuntimeError: {re}")
            raise
        except Exception as e:
            logger.error(f"Error generating or saving video: {e}")
            raise

