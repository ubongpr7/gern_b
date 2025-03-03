import os
import logging
import torch
from django.core.management.base import BaseCommand
from django.core.files.base import ContentFile
from django.conf import settings
from gern_apps.media_gen.models import MediaGenerationTask
from core.gern_app import AIGeneration
import tempfile
import modal
modal.config.token_id = os.getenv("MODAL_TOKEN_ID")
modal.config.token_secret = os.getenv("MODAL_TOKEN_SECRET")

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = "Generate a video from text input using AnimateDiff"

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
            self.process_text_to_video(task)
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

    def save_video_to_task_output(self, task: MediaGenerationTask, video_path: str, file_name: str):
        """
        Saves a generated video to the task's output_file field.

        Args:
            task (MediaGenerationTask): The task instance.
            video_path (str): The path to the video file.
            file_name (str): The name to save the file as.
        """
        try:
            with open(video_path, "rb") as video_file:
                task.output_file.save(file_name, ContentFile(video_file.read()))
            logger.info(f"Video saved to task output_file: {file_name}")
        except Exception as e:
            logger.error(f"Failed to save video to task output_file: {e}")
            raise
        
    def process_text_to_video(self, task: MediaGenerationTask):
        """
        Generates a video based on the MediaGenerationTask parameters.

        Args:
            task (MediaGenerationTask): The task instance containing the prompt and settings.
        """
        try:
            # Get the AI generation class from Modal
            ai_generation_instance = AIGeneration()
            if not ai_generation_instance.modelscope:
                raise RuntimeError("Model is not loaded.")

            # Run video generation
            logger.info(f"Generating video with prompt: {task.prompt}")
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
                ai_generation_instance.modelscope.to("cpu") #use cpu, since we are using temp files.
                video = ai_generation_instance.modelscope(
                    prompt=task.prompt,
                    negative_prompt=task.negative_prompt,
                    num_inference_steps=task.num_inference_steps,
                ).frames[0]
                video.save(temp_video.name)
                ai_generation_instance.modelscope.to("cuda")# return to gpu, after generating the animation.

                # Save the generated video to s3.
                self.save_video_to_task_output(task, temp_video.name, f"text_to_video_{task.id}.mp4")

            task.update_progress(100)
            os.remove(temp_video.name)
            logger.info(f"Temp video file {temp_video.name} removed")

        except RuntimeError as re:
            logger.error(f"RuntimeError: {re}")
            raise
        except Exception as e:
            logger.error(f"Error generating or saving video: {e}")
            raise

