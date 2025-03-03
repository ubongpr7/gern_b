from django.urls import path,include
from rest_framework import routers
from .views import LoginAPIView,LogoutAPI,TokenGenerator,ge_route

urlpatterns=[
    path("login/",LoginAPIView.as_view(),name="login"),
    path("logout/",LogoutAPI.as_view(),name="logout"),
    path("token/",TokenGenerator.as_view(),name="token"),
    # path("api_route/",ge_route,name="api_route"),
    
]



