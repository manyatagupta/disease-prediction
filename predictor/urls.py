from django.urls import path
from . import views
from . import ai_views

app_name = 'predictor'

urlpatterns = [
    path('', views.home_view, name='home'),
    path('checker/', views.checker_view, name='checker'),
    path('history/', views.history_view, name='history'),
    path('history/delete/<int:item_id>/', views.delete_history_item, name='delete_history_item'),
    path('history/clear/', views.clear_history, name='clear_history'),
    path('ai/chat/', ai_views.chat_with_grok, name='chat_with_grok'),
    path('ai/analyze/', ai_views.analyze_report, name='analyze_report'),
]
