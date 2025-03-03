import os
import logging
from typing import Optional

import boto3
from django.core.management.base import BaseCommand
from django.core.files.base import ContentFile
from django.conf import settings

from gern_apps.media_gen.models import GeneralAIJob
from core.gern_app import AIGeneration
import tempfile
from PIL import Image

logger = logging.getLogger(__name__)

class BaseCommandUtils:
    """
    Utility class for common functionalities in management commands.
    """
    def download_file_from_s3(self, file_key: str) -> Optional[bytes]:
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

    def save_image_to_output(self, obj, image_content: bytes, file_name: str):
        """
        Saves image content to the object's output_file field.
        """
        try:
            obj.output_file.save(file_name, ContentFile(image_content))
            logger.info(f"Image saved to output_file: {file_name}")
        except Exception as e:
            logger.error(f"Failed to save image to output_file: {e}")
            raise

    def save_video_to_output(self, obj, video_path: str, file_name: str):
        """
        Saves a generated video to the object's output_file field.
        """
        try:
            with open(video_path, "rb") as video_file:
                obj.output_file.save(file_name, ContentFile(video_file.read()))
            logger.info(f"Video saved to output_file: {file_name}")
        except Exception as e:
            logger.error(f"Failed to save video to output_file: {e}")
            raise

class Command(BaseCommand, BaseCommandUtils):
    help = "Upscale a video using Real-ESRGAN"

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
            self.process_upscale_video(job)
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

    def process_upscale_video(self, job: GeneralAIJob):
        """
        Upscales a video based on the GeneralAIJob parameters.

        Args:
            job (GeneralAIJob): The job instance containing the video file.
        """
        if not job.input_file:
            raise ValueError("input_file is required for upscaling.")

        try:
            # Get the AI generation class from Modal
            ai_generation_instance = AIGeneration()
            if not ai_generation_instance.esrgan:
                raise RuntimeError("Model is not loaded.")

            # Download input video
            video_content = self.download_file_from_s3(job.input_file.name)
            if not video_content:
                raise ValueError("Failed to download input_file from S3.")

            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
                temp_video.write(video_content)
                temp_video.flush()

                logger.info(f"Upscaling video: {job.input_file.name}")
                #TODO: use ai_generation_instance.upscale_video() method, that will need to be implemented.
                # video_output_path = ai_generation_instance.upscale_video(temp_video.name)
                # with open(video_output_path, "rb") as video_file:
                #     file_content = video_file.read()
                #     self.save_video_to_output(job, video_output_path, f"upscaled_{job.id}.mp4")
                # os.remove(video_output_path)
                # logger.info(f"Temp file {video_output_path} removed")
                os.remove(temp_video.name)
                logger.info(f"Temp video file {temp_video.name} removed")
                job.update_progress(100)

        except RuntimeError as re:
            logger.error(f"RuntimeError: {re}")
            raise
        except Exception as e:
            logger.error(f"Error upscaling video: {e}")
            raise

