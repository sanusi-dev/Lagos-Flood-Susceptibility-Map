"""URL routes for mapapp."""
from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("api/point/", views.point_query, name="point_query"),
    path("api/health/", views.health_check, name="health_check"),
    path("api/search/", views.search_places, name="search_places"),
    path("api/validation/", views.validation_points, name="validation_points"),
]
