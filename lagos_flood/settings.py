"""Minimal Django settings for the Lagos Flood Susceptibility Map prototype."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = "django-insecure-lagos-flood-susceptibility-prototype-key"
DEBUG = True
ALLOWED_HOSTS = ["*"]

# Minimal setup: no auth, no database, no session middleware required.
INSTALLED_APPS = [
    "django.contrib.staticfiles",
    "mapapp",
]

MIDDLEWARE = [
    "django.middleware.common.CommonMiddleware",
]

ROOT_URLCONF = "lagos_flood.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
            ],
        },
    },
]

WSGI_APPLICATION = "lagos_flood.wsgi.application"

DATABASES = {}

LANGUAGE_CODE = "en-us"
TIME_ZONE = "Africa/Lagos"
USE_I18N = True
USE_TZ = True

STATIC_URL = "/static/"
STATICFILES_DIRS = [
    BASE_DIR / "data" / "outputs",
]
