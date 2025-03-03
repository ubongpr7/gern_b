from django.conf import settings
from django.contrib import admin
from django.urls import path,include
from django.urls import re_path
from rest_framework import permissions
from drf_yasg.views import get_schema_view
from drf_yasg import openapi
from django.views.decorators.csrf import csrf_exempt
from django.conf.urls.static import static
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
)

schema_view = get_schema_view(
   openapi.Info(
      title="Gernal API",
      default_version='v1',
      description="Test description",
      terms_of_service="https://www.google.com/policies/terms/",
      contact=openapi.Contact(email="contact@snippets.local"),
      license=openapi.License(name="BSD License"),
   ),
   public=True,
   permission_classes=(permissions.AllowAny,),
)

urlpatterns = [
   path('admin/', admin.site.urls),
   
   path('accounts/',include('mainapps.accounts.urls',namespace='accounts'),),
   path('acccount-api/',include('mainapps.accounts.api.urls'),),
   
   path('auth-api/', include('djoser.urls')),
   path('', include('djoser.urls.jwt')),
   path('social_auth/', include('djoser.social.urls')),
   path('swagger<format>/', schema_view.without_ui(cache_timeout=0), name='schema-json'),
   path('swagger/', schema_view.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),
   path('redoc/', schema_view.with_ui('redoc', cache_timeout=0), name='schema-redoc'),
   # path('ses/event-webhook/', SESEventWebhookView.as_view(), name='handle-event-webhook'),
# gern apps urls
   # path('media-gen/',include('gern_apps.media_gen.urls',namespace='media-gen'),),
   path('media-gen-api/',include('gern_apps.media_gen.api.urls'),),   
]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)


