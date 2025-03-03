from rest_framework import serializers,exceptions
from rest_framework.validators import UniqueValidator
from mainapps.accounts.models import User
from django.contrib.auth import authenticate
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer

from rest_framework import generics
from rest_framework.permissions import IsAuthenticated
import stripe
from django.conf import settings
import uuid
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer

from django.contrib.auth import get_user_model

User=get_user_model()

class MyTokenObtainPairSerializer(TokenObtainPairSerializer):
    
    def validate(self, attrs):
        data = super().validate(attrs)

        user = self.user  
        data.update({
            'id': user.id,
            'username': user.username,
            'first_name': user.first_name,
            'access_token': user.access_token,
        })
        
        return data 
class UserRegistrationSerializer(serializers.ModelSerializer):
    email=serializers.EmailField(required=True,validators=[UniqueValidator(queryset=User.objects.all())])
    password=serializers.EmailField(required=True,validators=[UniqueValidator(queryset=User.objects.all())])
    password=serializers.CharField(required=True,write_only=True)
    class Meta:
        model=User
        fields=(
            "email",
            "username",
            "password",
            )
        
    
    def create(self, validated_data):
        user=User.objects.create(username=validated_data['username'],email=validated_data['email'])
        user.set_password(validated_data['password'])
        user.save()
        return user


class LoginSerializer(serializers.Serializer):
    username = serializers.EmailField()
    password = serializers.CharField(write_only=True)

    def validate(self, data):
        username = data.get("username", "")
        password = data.get("password", "")

        if username and password:
            user = authenticate(username=username, password=password)
            if user is not None:
                if user.is_active:
                    data["user"] = user
                else:
                    raise exceptions.ValidationError("You need to verify your account")
            else:
                raise exceptions.ValidationError("Invalid credentials")
        else:
            raise exceptions.ValidationError("All fields are required")

        return data

class ProfilePictureUpdateSerializer(serializers.ModelSerializer):
    class Meta:
        model = get_user_model()
        fields = ['picture']


