"""
WSGI config for ctpro project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/3.2/howto/deployment/wsgi/
"""
print('wsgi called for CATARACT ONLY')

import os
import sys
sys.path.append('/home/eye_scan//ctpro')
from django.core.wsgi import get_wsgi_application

print(os.getcwd())

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ctpro.settings')

application = get_wsgi_application()
