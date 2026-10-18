"""URL routes for mapapp."""
from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("api/point/", views.point_query, name="point_query"),
    path("api/health/", views.health_check, name="health_check"),
]
