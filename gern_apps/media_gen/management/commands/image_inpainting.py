import os
import logging
import boto3
from django.core.management.base import BaseCommand
from django.core.files.base import ContentFile
from django.conf import settings
from gern_apps.media_gen.models import MediaGenerationTask
from core.gern_app import AIGeneration
import tempfile

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = "Perform image inpainting/outpainting using Stable Diffusion"

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
            self.process_image_inpainting(task)
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

    def download_from_s3(self, file_key: str) -> bytes | None:
        """
        Download a file from S3 and return its content as bytes.

        Args:
            file_key (str): The S3 object key (file path in the bucket).

        Returns:
            bytes | None: The file content as bytes if successful, None otherwise.
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

        Args:
            task (MediaGenerationTask): The task instance.
            image_content (bytes): The content of the image to save.
            file_name (str): The name of the file to save.
        """
        try:
            task.output_file.save(file_name, ContentFile(image_content))
            logger.info(f"Image saved to task output_file: {file_name}")
        except Exception as e:
            logger.error(f"Failed to save image to task output_file: {e}")
            raise

    def process_image_inpainting(self, task: MediaGenerationTask):
        """
        Performs image inpainting based on the MediaGenerationTask parameters.

        Args:
            task (MediaGenerationTask): The task instance containing the image and mask paths.
        """
        if not task.image_input or not task.mask_image:
            raise ValueError("image_input and mask_image are required for inpainting.")
        
        try:
            # Get the AI generation class from Modal
            ai_generation_instance = AIGeneration()
            if not ai_generation_instance.inpaint_pipe:
                raise RuntimeError("Model is not loaded.")

            # Download input image and mask image
            image_content = self.download_from_s3(task.image_input.name)
            mask_content = self.download_from_s3(task.mask_image.name)

            if not image_content or not mask_content:
                raise ValueError("Failed to download image_input or mask_image from S3.")

            # Create temp files.
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_image, tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_mask:
                temp_image.write(image_content)
                temp_image.flush()

                temp_mask.write(mask_content)
                temp_mask.flush()

                logger.info(f"Performing inpainting on image: {task.image_input.name} with mask: {task.mask_image.name}")
                inpainted_image_path = ai_generation_instance.image_inpainting(temp_image.name, temp_mask.name)
            
                with open(inpainted_image_path, "rb") as image_file:
                    file_content = image_file.read()
                    # save the inpainted image to s3.
                    self.save_image_to_task_output(task, file_content, f"inpainted_{task.id}.png")
                    
                os.remove(inpainted_image_path)
                logger.info(f"Temp file {inpainted_image_path} removed")
                os.remove(temp_image.name)
                logger.info(f"Temp file {temp_image.name} removed")
                os.remove(temp_mask.name)
                logger.info(f"Temp file {temp_mask.name} removed")

        except RuntimeError as re:
            logger.error(f"RuntimeError: {re}")
            raise
        
        except Exception as e:
            logger.error(f"Error inpainting image: {e}")
            raise
