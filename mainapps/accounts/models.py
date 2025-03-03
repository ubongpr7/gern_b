from datetime import datetime
import random
from django.contrib.auth.models import AbstractUser, BaseUserManager,PermissionsMixin
from django.db import models
from django.conf import settings



class CustomUserManager(BaseUserManager):
    def create_user(self, email, password=None, **extra_fields):
        if not email:
            raise ValueError('The Email field must be set')
        email = self.normalize_email(email).lower()

        user = self.model(email=email, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email, password=None, **extra_fields):
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)
        return self.create_user(email, password, **extra_fields)


class User(AbstractUser,PermissionsMixin):
    email = models.EmailField(unique=True, null=False, blank=False)
    picture = models.ImageField(upload_to='profile_pictures/%y/%m/%d/' , blank=True, null=True)
    verification_token=models.CharField(max_length=255,null=True,blank=True)
    USERNAME_FIELD = "email"
    REQUIRED_FIELDS = []
    objects = CustomUserManager()

    def save(self, *args, **kwargs):
        self.username = self.email
        super().save(*args, **kwargs)

