"""
URL configuration for Aircraft Tracking backend project.
"""
from django.contrib import admin
from django.urls import path, re_path, include
from rest_framework import permissions
from drf_yasg.views import get_schema_view
from drf_yasg import openapi
from django.conf.urls.static import static
from django.conf import settings
from rest_framework.routers import DefaultRouter
from rest_framework.authtoken.views import obtain_auth_token

from aircraft_detection.urls import router as aircraft_detection_router
from cameras.urls import router as cameras_router

schema_view = get_schema_view(
    openapi.Info(
        title="Aircraft Tracking API",
        default_version="v1",
        description="Aircraft Tracking Backend API's for AI-powered surveillance",
        terms_of_service="https://skylarklabs.ai/privacy",
        contact=openapi.Contact(email="support@skylarklabs.ai"),
        license=openapi.License(name="BSD License"),
    ),
    public=True,
    permission_classes=[permissions.AllowAny],
)

router = DefaultRouter()
router.registry.extend(aircraft_detection_router.registry)
router.registry.extend(cameras_router.registry)

urlpatterns = [
    path('admin/', admin.site.urls),
    path("get-token/", obtain_auth_token),

    path('auth/accounts/', include('accounts.urls')),
    path('camera/', include('aircraft_detection.urls')),
    path('', include(router.urls)),

    re_path(
        r"^swagger(?P<format>\.json|\.yaml)$",
        schema_view.without_ui(cache_timeout=0),
        name="schema-json",
    ),
    re_path(
        r"^api/docs/swagger/$",
        schema_view.with_ui("swagger", cache_timeout=0),
        name="schema-swagger-ui",
    ),
    re_path(
        r"^api/docs/redoc/$",
        schema_view.with_ui("redoc", cache_timeout=0),
        name="schema-redoc",
    ),
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)



