from rest_framework import serializers
from gern_apps.media_gen.models import MediaGenerationTask, GeneralAIJob
from django.contrib.auth import get_user_model

User = get_user_model()

class MediaGenerationTaskSerializer(serializers.ModelSerializer):
    class Meta:
        model = MediaGenerationTask
        fields = "__all__"
        read_only_fields = ["id", "status", "progress", "created_at", "updated_at", "error_message", "output_file"]

class GeneralAIJobSerializer(serializers.ModelSerializer):
    class Meta:
        model = GeneralAIJob
        fields = "__all__"
        read_only_fields = ["id", "status", "progress", "created_at", "updated_at", "error_message", "output_file"]
