# urls.py
from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static
from django.contrib.auth import views as auth_views


urlpatterns = [
    #path(' ', views.mainhome, name='mainhome'),
    path('home/',views.home, name='home'),
    path('test/', views.test, name='test'),
    path('login/', views.login_view, name='login'),
    path('signup/', views.signup, name='signup'),
    path('form/', views.form, name='form'), # Fixed misplaced closing parenthesis 
    path('analyze_image/', views.analyze_image, name='analyze_image'),
    path('get_cataract_analysis/', views.analyze_image_alternate, name='get_cataract_analysis'),
    path('check_username/', views.check_username, name='check_username'),
    
]

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATICFILES_DIRS[0])
