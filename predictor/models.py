from django.db import models
from django.contrib.auth.models import User

class Symptom(models.Model):
    name = models.CharField(max_length=100, unique=True)
    
    def __str__(self):
        return self.name

class Disease(models.Model):
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(blank=True, null=True)
    precautions = models.TextField(blank=True, null=True)
    recommended_doctor = models.CharField(max_length=100, blank=True, null=True)
    
    def __str__(self):
        return self.name

class PredictionHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    symptoms_selected = models.TextField() # Comma-separated list or JSON string of symptoms
    predicted_disease = models.CharField(max_length=100)
    confidence = models.FloatField()
    date = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.user.username} - {self.predicted_disease} ({self.date.strftime('%Y-%m-%d %H:%M')})"
