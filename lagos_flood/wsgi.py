"""WSGI config for lagos_flood."""
import os
from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "lagos_flood.settings")
application = get_wsgi_application()
