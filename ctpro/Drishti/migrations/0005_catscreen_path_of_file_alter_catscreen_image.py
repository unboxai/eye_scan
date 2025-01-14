# Generated by Django 5.1 on 2024-08-15 14:08

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('Drishti', '0004_catscreen_score_alter_catscreen_user'),
    ]

    operations = [
        migrations.AddField(
            model_name='catscreen',
            name='path_of_file',
            field=models.FilePathField(blank=True, max_length=255, null=True, path='/home/bitnami/ctpro_april24/ctpro/staticfiles/drishti/images'),
        ),
        migrations.AlterField(
            model_name='catscreen',
            name='image',
            field=models.ImageField(upload_to='drishti/images'),
        ),
    ]
