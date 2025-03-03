from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from gern_apps.media_gen.models import MediaGenerationTask, GeneralAIJob
from gern_apps.media_gen.api.serializers import MediaGenerationTaskSerializer, GeneralAIJobSerializer
import logging
import modal
import os

logger = logging.getLogger(__name__)
modal.config.token_id = os.getenv("MODAL_TOKEN_ID")
modal.config.token_secret = os.getenv("MODAL_TOKEN_SECRET")

class TextToImageView(APIView):
    """
    API view to generate an image from a text prompt.
    """

    def post(self, request):
        """
        Handles POST requests to generate an image.
        """
        prompt = request.data.get("prompt")
        user_id = request.data.get("user")
        if not prompt:
            return Response({"error": "Prompt is required."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # Create a MediaGenerationTask
            task_data = {
                "user": user_id,
                "prompt": prompt,
                "task_type": "text_to_image",
                "status": "pending"
            }
            serializer = MediaGenerationTaskSerializer(data=task_data)
            if serializer.is_valid():
                task = serializer.save()
            else:
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
            logger.info(f"Created text-to-image task: {task.id}")
            ai_generation_app = modal.Function.lookup("text-to-image-1", "text_to_image")
            ai_generation_app.remote(str(task.id))

            logger.info(f"Triggered text-to-image task: {task.id}")

            return Response({"task_id": str(task.id), "message": "Text-to-image task created successfully."}, status=status.HTTP_201_CREATED)
        except Exception as e:
            logger.error(f"Error creating text-to-image task: {e}")
            return Response({"error": "An error occurred while creating the task."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

