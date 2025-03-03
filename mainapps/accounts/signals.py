# from django.db.models.signals import post_save
# from django.dispatch import receiver
# from .models import VerificationCode,User

# @receiver(post_save,sender=User)
# def post_save_create_user_code(sender, instance, created,**kwargs):
#     if created:
#         VerificationCode.objects.create(user=instance)