"""Root URL configuration for lagos_flood."""
from django.urls import include, path

urlpatterns = [
    path("", include("mapapp.urls")),
]
