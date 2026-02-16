from django.urls import path
from . import views

urlpatterns = [
    # Main chat UI
    path('', views.index, name='index'),

    # Conversation API
    path('api/conversations/', views.conversation_list, name='conversation_list'),
    path('api/conversations/<uuid:conversation_id>/', views.conversation_detail, name='conversation_detail'),

    # Message API
    path('api/messages/', views.send_message, name='send_message'),
    path('api/messages/<uuid:message_id>/regenerate/', views.regenerate_message, name='regenerate_message'),
]
