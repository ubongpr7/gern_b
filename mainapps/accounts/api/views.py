import json
import logging
import os
import uuid
from django.conf import settings
from django.contrib.auth import (
    login as api_login,
    logout as api_logout,
    authenticate,
    get_user_model,
    tokens,
)
from django.contrib.auth.models import User
from django.contrib.auth.tokens import default_token_generator
from django.core.mail import send_mail
from django.db import transaction
from django.http import HttpResponseRedirect
from django.shortcuts import redirect, render
from django.template.loader import render_to_string
from django.urls import reverse
from django.utils import timezone
from django.utils.encoding import force_bytes
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt


import requests
from rest_framework import (
    viewsets,
    status,
    permissions,
    generics,
)
from rest_framework.authtoken.models import Token
from rest_framework.decorators import action, api_view
from rest_framework.generics import CreateAPIView
from rest_framework.parsers import (
    FormParser,
    MultiPartParser,
    JSONParser,
    FileUploadParser,
)
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.views import TokenObtainPairView

from mainapps.accounts.models import User
from mainapps.accounts.utils import send_confirmation_email, send_html_email2
from .serializers import *
import stripe

from rest_framework import status
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import api_view, permission_classes, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
import os


class RegistrationAPI(APIView):
    
    def post(self,request):
        serializer=UserRegistrationSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        user=User.objects.get(username=request.data['username'])
        return Response({'message':f"User with the email {request.COOKIES.get('email')}  created"},status=201)

class LoginAPIView(APIView):
    
    def post(self,request):
        print(request.data)
        serializer=LoginSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user=serializer.validated_data["user"]
        

        return Response("Verify Your Identity",status=200)



@api_view(['GET'])
def ge_route(request):
    route=['/api/token','api/token/refresh']
    return Response(route,status=201)

        
class TokenGenerator(TokenObtainPairView):
    def post(self, request: Request, *args, **kwargs)  :
        username=request.data.get('username')
        password=request.data.get('password')
        user=authenticate(username=username,password=password)
        if user is not None:
            response=super().post(request,*args,**kwargs)
            response.status_code=200
            return response
        else:
            return Response(status=400)

class LogoutAPI(APIView):
    permission_classes=[permissions.IsAuthenticated,]
    def post(self,request):        
        response=Response()
        response.data={
            'message':'Logged Out Successfully'
        }
        response.status_code=200
        api_logout(request)
        return response
    


