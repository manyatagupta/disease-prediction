from django.urls import path
from . import views

app_name = 'predictor'

urlpatterns = [
    path('', views.home_view, name='home'),
    path('checker/', views.checker_view, name='checker'),
    path('history/', views.history_view, name='history'),
]
