from django.contrib import admin
from django.utils.html import format_html
from .models import DrishtiUser, Catscreen

@admin.register(DrishtiUser)
class DrishtiUserAdmin(admin.ModelAdmin):
    list_display = ('username', 'name', 'age', 'email')
    search_fields = ('username', 'name', 'email')

from django.utils.safestring import mark_safe

@admin.register(Catscreen)
class CatscreenAdmin(admin.ModelAdmin):
    list_display = ('username', 'patient_name', 'age', 'visual_symptom', 'result', 'Corrected_label', 'score')
    list_filter = ('result', 'visual_symptom')
    search_fields = ('patient_name', 'user__username')
    list_editable = ('Corrected_label',)

    def username(self, obj):
        return obj.user.username
    username.short_description = 'Username'

    def patient_name(self, obj):
        return obj.name
    patient_name.short_description = 'Patient name'
