from django.db import models
import PIL
from django.contrib.auth.models import User



class DrishtiUser(models.Model):
    username = models.CharField(max_length=255)
    name = models.CharField(max_length=255)
    age = models.IntegerField()
    password = models.CharField(max_length=255)
    email = models.CharField(max_length=255)
   

class Catscreen(models.Model):
    user = models.ForeignKey(DrishtiUser, on_delete=models.CASCADE)
    name = models.CharField(max_length=255)
    age = models.IntegerField()
    visual_symptom = models.CharField(max_length=255)
    image = models.ImageField(upload_to='drishti/images')
    result = models.CharField(max_length=255, null=True, blank=True)
    Corrected_label = models.CharField(max_length=255, null=True, blank=True)
    score = models.CharField(max_length=255, null=True, blank=True)
    path_of_file = models.FilePathField(path='/home/bitnami/ctpro_april24/ctpro/staticfiles/drishti/images', max_length=255, null=True, blank=True)
