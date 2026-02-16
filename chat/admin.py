from django.contrib import admin
from .models import Conversation, Message, GeneratedContent


class GeneratedContentInline(admin.TabularInline):
    model = GeneratedContent
    extra = 0
    readonly_fields = ('id', 'created_at')


class MessageInline(admin.TabularInline):
    model = Message
    extra = 0
    readonly_fields = ('id', 'created_at')
    show_change_link = True


@admin.register(Conversation)
class ConversationAdmin(admin.ModelAdmin):
    list_display = ('title', 'created_at', 'updated_at')
    search_fields = ('title',)
    inlines = [MessageInline]


@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ('role', 'content_preview', 'message_type', 'conversation', 'created_at')
    list_filter = ('role', 'message_type')
    search_fields = ('content',)
    inlines = [GeneratedContentInline]

    def content_preview(self, obj):
        return obj.content[:80] + '...' if len(obj.content) > 80 else obj.content
    content_preview.short_description = 'Content'


@admin.register(GeneratedContent)
class GeneratedContentAdmin(admin.ModelAdmin):
    list_display = ('title', 'content_type', 'message', 'created_at')
    list_filter = ('content_type',)
