import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "lagos_flood.settings")
os.environ.setdefault("DJANGO_DEBUG", "true")

import django

django.setup()
