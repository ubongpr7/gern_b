import os
import logging
import boto3
from django.core.management.base import BaseCommand
from django.core.files.base import ContentFile
from django.conf import settings
from gern_apps.media_gen.models import GeneralAIJob
from core.gern_app import AIGeneration
import tempfile
from typing import Optional

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = "Remove the background from an image using AI"

    def add_arguments(self, parser):
        parser.add_argument("job_id", type=str, help="ID of the GeneralAIJob")

    def handle(self, *args, **kwargs):
        job_id = kwargs["job_id"]

        try:
            job = GeneralAIJob.objects.get(id=job_id)
        except GeneralAIJob.DoesNotExist:
            logger.error(f"GeneralAIJob with ID {job_id} not found.")
            self.stdout.write(self.style.ERROR(f"GeneralAIJob with ID {job_id} not found."))
            return

        job.status = "processing"
        job.save()
        self.stdout.write(self.style.SUCCESS(f"Start processing job {job_id}"))

        try:
            self.process_background_removal(job)
            job.status = "completed"
            job.progress = 100
            job.save()
            self.stdout.write(self.style.SUCCESS(f"Job {job_id} completed successfully."))

        except Exception as e:
            logger.exception(f"Error processing job {job_id}: {e}")
            job.status = "failed"
            job.error_message = str(e)
            job.save()
            self.stdout.write(self.style.ERROR(f"Job {job_id} failed: {e}"))

    def download_from_s3(self, file_key: str) -> Optional[bytes]:
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

    def save_image_to_job_output(self, job: GeneralAIJob, image_content: bytes, file_name: str):
        """
        Saves image content to the job's output_file field.

        Args:
            job (GeneralAIJob): The job instance.
            image_content (bytes): The content of the image to save.
            file_name (str): The name of the file to save.
        """
        try:
            job.output_file.save(file_name, ContentFile(image_content))
            logger.info(f"Image saved to job output_file: {file_name}")
        except Exception as e:
            logger.error(f"Failed to save image to job output_file: {e}")
            raise

    def process_background_removal(self, job: GeneralAIJob):
        """
        Removes the background of an image based on the GeneralAIJob parameters.

        Args:
            job (GeneralAIJob): The job instance containing the input file.
        """
        if not job.input_file:
            raise ValueError("input_file is required for background removal.")

        try:
            # Get the AI generation class from Modal
            ai_generation_instance = AIGeneration()
            if not ai_generation_instance.bg_remover:
                raise RuntimeError("Model is not loaded.")

            # Download input image
            image_content = self.download_from_s3(job.input_file.name)

            if not image_content:
                raise ValueError("Failed to download input_file from S3.")

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_image:
                temp_image.write(image_content)
                temp_image.flush()

                logger.info(f"Removing background from image: {job.input_file.name}")
                output_path = ai_generation_instance.remove_background(temp_image.name)
                
                with open(output_path, "rb") as image_file:
                    file_content = image_file.read()
                    # save the image to s3.
                    self.save_image_to_job_output(job, file_content, f"nobg_{job.id}.png")

                os.remove(output_path)
                logger.info(f"Temp file {output_path} removed")
                os.remove(temp_image.name)
                logger.info(f"Temp file {temp_image.name} removed")
                job.update_progress(100)

        except RuntimeError as re:
            logger.error(f"RuntimeError: {re}")
            raise
        except Exception as e:
            logger.error(f"Error removing background: {e}")
            raise

