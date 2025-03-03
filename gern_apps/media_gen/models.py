from django.db import models
from django.contrib.auth import get_user_model
import uuid
from django.conf import settings

User = get_user_model()


class AIModel(models.Model):
    """Represents the different AI models available for use."""

    MODEL_TYPE_CHOICES = [
        ("image", "Image"),
        ("video", "Video"),
        ("inpaint", "Inpaint"),
        ("controlnet", "Controlnet"),
    ]

    name = models.CharField(max_length=100, unique=True, help_text="Name of the AI model")
    description = models.TextField(help_text="Detailed description of the model")
    base_model_name = models.CharField(
        max_length=255, help_text="Name of the underlying model (e.g., stabilityai/stable-diffusion-xl-base-1.0)"
    )
    model_type = models.CharField(
        max_length=50, choices=MODEL_TYPE_CHOICES, help_text="Type of AI model (e.g., image, video)"
    )
    is_default = models.BooleanField(default=False, help_text="Is this the default model for its type?")

    class Meta:
        verbose_name = "AI Model"
        verbose_name_plural = "AI Models"

    def __str__(self):
        return self.name


class LoRAModel(models.Model):
    """Represents a LoRA model that can be used for fine-tuning."""

    name = models.CharField(max_length=100, unique=True, help_text="Name of the LoRA model")
    file = models.FileField(upload_to="lora_models/", help_text="LoRA model file")
    description = models.TextField(help_text="Description of what the LoRA model does")
    base_model = models.ForeignKey(AIModel, on_delete=models.CASCADE, help_text="Base model this LoRA is compatible with")

    class Meta:
        verbose_name = "LoRA Model"
        verbose_name_plural = "LoRA Models"

    def __str__(self):
        return self.name


class BaseGenerationTask(models.Model):
    """Abstract base model for all AI generation tasks."""

    STATUS_CHOICES = [
        ("pending", "Pending"),
        ("processing", "Processing"),
        ("completed", "Completed"),
        ("failed", "Failed"),
    ]

    SCHEDULER_CHOICES = [
        ("DDIM", "DDIM"),
        ("DPM++ 2M Karras", "DPM++ 2M Karras"),
        ("Euler a", "Euler a"),
        ("Heun", "Heun"),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, help_text="User who requested the task")
    prompt = models.TextField(help_text="Text prompt to guide the AI")
    negative_prompt = models.TextField(blank=True, null=True, help_text="Text to guide the model on what to avoid")
    model = models.ForeignKey(
        AIModel, on_delete=models.SET_NULL, null=True, help_text="AI model used for this task"
    )
    # Core image settings
    width = models.PositiveIntegerField(default=512, help_text="Width of the generated image")
    height = models.PositiveIntegerField(default=512, help_text="Height of the generated image")
    num_inference_steps = models.PositiveIntegerField(default=50, help_text="Number of denoising steps")
    guidance_scale = models.FloatField(default=7.5, help_text="Higher values make the model follow the prompt more closely")

    # Seed & scheduler
    seed = models.BigIntegerField(
        default=-1, help_text="Random seed (-1 for random, specific value for reproducibility)"
    )
    scheduler = models.CharField(
        max_length=50,
        choices=SCHEDULER_CHOICES,
        default="DPM++ 2M Karras",
        help_text="The diffusion scheduler used for image generation",
    )
    fixed_seed = models.BooleanField(default=False, help_text="Keep the seed fixed even if regenerating.")

    # Status and Output
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="pending", help_text="Current task status")
    output_file = models.FileField(
        upload_to="media_outputs/", blank=True, null=True, help_text="Path to the generated output file"
    )
    progress = models.IntegerField(default=0, help_text="Progress of the task (0-100)")
    error_message = models.TextField(blank=True, null=True, help_text="Error message if the task failed")

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True, help_text="Timestamp of task creation")
    updated_at = models.DateTimeField(auto_now=True, help_text="Timestamp of last update")

    # Advanced settings
    upscale = models.BooleanField(default=False, help_text="Enable upscaling after generation")
    face_enhancement = models.BooleanField(default=False, help_text="Enhance faces using GFPGAN")

    priority = models.IntegerField(default=1, help_text="Job priority, from higher to lower. Higher value runs sooner.")
    class Meta:
        abstract = True
        ordering = ["-priority", "-created_at"]

    def __str__(self):
        return f"{self.__class__.__name__} - {self.user} - {self.status}"

    def update_progress(self, progress_value):
        """Update the progress field and broadcast the update."""
        self.progress = progress_value
        self.save(update_fields=["progress", "updated_at"])


class MediaGenerationTask(BaseGenerationTask):
    """Handles all media generation tasks, including text-to-image, text-to-video, etc."""

    TASK_TYPE_CHOICES = [
        ("text_to_image", "Text to Image"),
        ("text_to_video", "Text to Video"),
        ("image_to_video", "Image to Video"),
        ("image_inpainting", "Image Inpainting"),
        ("image_outpainting", "Image Outpainting"),
        ("controlnet", "Controlnet Image Generation"),
    ]

    RESOLUTION_CHOICES = [
        ("480p", "480p"),
        ("720p", "720p"),
        ("1080p", "1080p"),
    ]

    task_type = models.CharField(
        max_length=50, choices=TASK_TYPE_CHOICES, help_text="Type of media generation task"
    )
    # Task specific settings
    duration = models.FloatField(default=5.0, help_text="Duration of the generated video in seconds")
    resolution = models.CharField(
        max_length=20, choices=RESOLUTION_CHOICES, default="720p", blank=True, null=True, help_text="Resolution of the video"
    )
    frame_rate = models.PositiveIntegerField(default=24, blank=True, null=True, help_text="Frames per second of the video")
    image_input = models.FileField(
        upload_to="image_inputs/", help_text="Input image for transformation", blank=True, null=True
    )
    mask_image = models.FileField(upload_to="mask_images/", blank=True, null=True, help_text="Mask image for inpainting/outpainting")
    control_image = models.FileField(upload_to="control_images/", blank=True, null=True, help_text="ControlNet input image")
    controlnet_type = models.CharField(max_length=50, blank=True, null=True, help_text="Type of ControlNet processing")
    lora_model = models.ForeignKey(LoRAModel, on_delete=models.SET_NULL, null=True, blank=True, help_text="LoRA model to use")
    batch_size = models.PositiveIntegerField(default=1, help_text="Number of images to generate in a batch")

    class Meta:
        verbose_name = "Media Generation Task"
        verbose_name_plural = "Media Generation Tasks"


class GeneralAIJob(models.Model):
    """General model for tracking AI-related jobs, including upscaling, style transfer, etc."""

    JOB_TYPE_CHOICES = [
        ("upscale", "Image/Video Upscaling"),
        ("style_transfer", "Style Transfer"),
        ("background_removal", "Background Removal"),
        ("object_detection", "Object Detection"),
        # Add more job types here
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="ai_jobs", help_text="User who requested the job")
    task_type = models.CharField(max_length=50, choices=JOB_TYPE_CHOICES, help_text="Type of AI job")
    input_file = models.FileField(upload_to="ai_jobs/", help_text="Input file for the AI job")
    output_file = models.FileField(upload_to="ai_jobs_output/", blank=True, null=True, help_text="Output file of the AI job")
    status = models.CharField(
        max_length=20, choices=BaseGenerationTask.STATUS_CHOICES, default="pending", help_text="Current status of the job"
    )
    created_at = models.DateTimeField(auto_now_add=True, help_text="Timestamp of job creation")
    updated_at = models.DateTimeField(auto_now=True, help_text="Timestamp of last update")
    error_message = models.TextField(blank=True, null=True, help_text="Error message if the job failed")
    priority = models.IntegerField(default=1, help_text="Job priority, from higher to lower. Higher value runs sooner.")
    progress = models.IntegerField(default=0, help_text="Progress of the task (0-100)")

    class Meta:
        verbose_name = "General AI Job"
        verbose_name_plural = "General AI Jobs"
        ordering = ["-priority", "-created_at"]

    def __str__(self):
        return f"{self.task_type} - {self.user} - {self.status}"
    
    def update_progress(self, progress_value):
        """Update the progress field and broadcast the update."""
        self.progress = progress_value
        self.save(update_fields=["progress", "updated_at"])
