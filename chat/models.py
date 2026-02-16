import uuid
from django.db import models
from django.utils import timezone


class Conversation(models.Model):
    """
    Represents a chat conversation/thread.
    Groups messages together and stores user context for personalization.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=255, default="New Chat")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    user_context = models.JSONField(
        default=dict, blank=True,
        help_text="Stores user preferences, brand info, past style choices, etc."
    )

    class Meta:
        ordering = ['-updated_at']

    def __str__(self):
        return f"{self.title} ({self.id})"


class Message(models.Model):
    """
    A single message in a conversation.
    Can be from the user or the assistant.
    """
    ROLE_CHOICES = [
        ('user', 'User'),
        ('assistant', 'Assistant'),
        ('system', 'System'),
    ]
    MESSAGE_TYPE_CHOICES = [
        ('text', 'Text'),
        ('image_generation', 'Image Generation'),
        ('image_transformation', 'Image Transformation'),
        ('poster_design', 'Poster Design'),
        ('vision_board', 'Vision Board'),
        ('brand_artwork', 'Brand Artwork'),
        ('video_loop', 'Video Loop'),
        ('story_sequence', 'Story Sequence'),
        ('mixed', 'Mixed Content'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    conversation = models.ForeignKey(
        Conversation, on_delete=models.CASCADE, related_name='messages'
    )
    role = models.CharField(max_length=10, choices=ROLE_CHOICES)
    content = models.TextField(help_text="The text content of the message")
    message_type = models.CharField(
        max_length=25, choices=MESSAGE_TYPE_CHOICES, default='text'
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created_at']

    def __str__(self):
        return f"[{self.role}] {self.content[:50]}..."


class GeneratedContent(models.Model):
    """
    Represents a piece of generated visual content attached to an assistant message.
    Could be an image, poster, artwork, video frame, etc.
    """
    CONTENT_TYPE_CHOICES = [
        ('image', 'Image'),
        ('poster', 'Poster'),
        ('artwork', 'Artwork'),
        ('photo', 'Photo'),
        ('video_frame', 'Video Frame'),
        ('vision_board', 'Vision Board'),
        ('signage', 'Signage'),
        ('brand_asset', 'Brand Asset'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    message = models.ForeignKey(
        Message, on_delete=models.CASCADE, related_name='generated_contents'
    )
    content_type = models.CharField(max_length=20, choices=CONTENT_TYPE_CHOICES, default='image')
    title = models.CharField(max_length=255, blank=True, default="")
    description = models.TextField(blank=True, default="")
    image_url = models.URLField(
        max_length=500,
        help_text="URL of the generated image (external or local media)"
    )
    prompt_used = models.TextField(
        blank=True, default="",
        help_text="The actual prompt used for generation"
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created_at']

    def __str__(self):
        return f"{self.content_type}: {self.title or self.id}"
