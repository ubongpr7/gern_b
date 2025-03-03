from django.shortcuts import redirect, render
from django.conf import settings
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
    TokenVerifyView
)
from django.contrib.auth import login as api_login, logout as api_logout

from rest_framework.views import APIView
from rest_framework import permissions
from rest_framework.response import Response
from rest_framework import status
from djoser.social.views import ProviderAuthView
from django.contrib.auth import get_user_model


def login_view(request):
    return render(request,'login.html')

class AccountTokenObtainPairView(TokenObtainPairView):
    def post(self, request,*args,**kwargs):
        response =super().post(request,*args,**kwargs)
        if response.status_code==200:
            access_token=response.data.get('access')
            refresh_token=response.data.get('refresh')
            response.set_cookie(
                'access',
                access_token,
                max_age=settings.AUTH_COOKIE_ACCESS_MAX_AGE,
                path=settings.AUTH_COOKIE_PATH, 
                secure=settings.AUTH_COOKIE_SECURE,
                samesite=settings.AUTH_COOKIE_SAMESITE,
                httponly=settings.AUTH_COOKIE_HTTP_ONLY,
            )
            response.set_cookie(
                'refresh',
                refresh_token,
                max_age=settings.AUTH_COOKIE_REFRESH_MAX_AGE,
                path=settings.AUTH_COOKIE_PATH, 
                secure=settings.AUTH_COOKIE_SECURE,
                samesite=settings.AUTH_COOKIE_SAMESITE,
                httponly=settings.AUTH_COOKIE_HTTP_ONLY,
            )
            return response

class AccountTokenRefreshView(TokenRefreshView):
    def post(self,request, *args,**kwargs):
        refresh_token=request.COOKIES.get('refresh')
        if refresh_token:
            request.data['refresh']=refresh_token

        response=super().post(request,*args,**kwargs)
        if response.status_code == 200:
            access_token=response.data.get('access')
            response.set_cookie(
                'access',
                access_token,
                max_age=settings.AUTH_COOKIE_ACCESS_MAX_AGE,
                path=settings.AUTH_COOKIE_PATH, 
                secure=settings.AUTH_COOKIE_SECURE,
                samesite=settings.AUTH_COOKIE_SAMESITE,
                httponly=settings.AUTH_COOKIE_HTTP_ONLY,
            )
        return response

class AccountTokenVerify(TokenVerifyView):

    def post(self, request, *args, **kwargs):
        access_token=request.COOKIES.get('access')
        if access_token:
            request.data['token']= access_token

            return super().post(request,*args,**kwargs)
        
class AccountProviderAuthView(ProviderAuthView):
    def post(self,request,*args,**kwargs):
        response=super().post(request,*args,**kwargs)
        if response.status_code==200:
            access_token=response.data.get('access')
            refresh_token=response.data.get('refresh')
            response.set_cookie(
                'access',
                access_token,
                max_age=settings.AUTH_COOKIE_ACCESS_MAX_AGE,
                path=settings.AUTH_COOKIE_PATH, 
                secure=settings.AUTH_COOKIE_SECURE,
                samesite=settings.AUTH_COOKIE_SAMESITE,
                httponly=settings.AUTH_COOKIE_HTTP_ONLY,
            )
            response.set_cookie(
                'refresh',
                refresh_token,
                max_age=settings.AUTH_COOKIE_REFRESH_MAX_AGE,
                path=settings.AUTH_COOKIE_PATH, 
                secure=settings.AUTH_COOKIE_SECURE,
                samesite=settings.AUTH_COOKIE_SAMESITE,
                httponly=settings.AUTH_COOKIE_HTTP_ONLY,
            )
            return response

class LogoutAPI(APIView):
    permission_classes=[permissions.IsAuthenticated,]
    def post(self,request,*args,**kwargs):        
        response=Response(status=status.HTTP_204_NO_CONTENT)
        response.delete_cookie("access")
        response.delete_cookie("refresh")
        response.data={
            'message':'Logged Out Successfully'
        }
        response.status_code=200
        api_logout(request)
        return response
    
def verify(request, token):
  try:
    user = get_user_model().objects.get(verification_token=token)

    if user is not None:
      user.verification_token = None
      user.save()

      api_login(request, user)
      return redirect('/')
  except:
    return redirect('/')
