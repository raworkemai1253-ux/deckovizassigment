"""
Vizzy Chat API Views

Provides JSON API endpoints for the chat frontend:
- Conversation CRUD
- Message creation with AI response generation
- Response regeneration
"""

import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.shortcuts import get_object_or_404, render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required

from .models import Conversation, Message, GeneratedContent
from .services import generate_response


# ---------------------------------------------------------------------------
# Auth views
# ---------------------------------------------------------------------------

def login_view(request):
    """Handle user login."""
    if request.user.is_authenticated:
        return redirect('index')
    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('index')
        else:
            return render(request, 'chat/login.html', {'error': 'Invalid username or password.'})
    return render(request, 'chat/login.html')


def register_view(request):
    """Handle user registration."""
    if request.user.is_authenticated:
        return redirect('index')
    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        email = request.POST.get('email', '').strip()
        password1 = request.POST.get('password1', '')
        password2 = request.POST.get('password2', '')
        errors = []
        if not username:
            errors.append('Username is required.')
        if len(password1) < 6:
            errors.append('Password must be at least 6 characters.')
        if password1 != password2:
            errors.append('Passwords do not match.')
        if User.objects.filter(username=username).exists():
            errors.append('Username already taken.')
        if errors:
            return render(request, 'chat/register.html', {
                'errors': errors, 'username': username, 'email': email
            })
        user = User.objects.create_user(username=username, email=email, password=password1)
        login(request, user)
        return redirect('index')
    return render(request, 'chat/register.html')


def logout_view(request):
    """Log user out and redirect to login."""
    logout(request)
    return redirect('login')


@login_required
def index(request):
    """Serve the main chat UI."""
    return render(request, 'chat/index.html')


# ---------------------------------------------------------------------------
# Conversation endpoints
# ---------------------------------------------------------------------------

@csrf_exempt
@require_http_methods(["GET", "POST"])
def conversation_list(request):
    """
    GET  /api/conversations/  → list all conversations
    POST /api/conversations/  → create new conversation
    """
    if request.method == 'GET':
        conversations = Conversation.objects.filter(user=request.user)
        data = [
            {
                'id': str(c.id),
                'title': c.title,
                'created_at': c.created_at.isoformat(),
                'updated_at': c.updated_at.isoformat(),
                'message_count': c.messages.count(),
            }
            for c in conversations
        ]
        return JsonResponse({'conversations': data})

    # POST — create new conversation
    try:
        body = json.loads(request.body) if request.body else {}
    except json.JSONDecodeError:
        body = {}

    title = body.get('title', 'New Chat')
    conversation = Conversation.objects.create(title=title, user=request.user)

    return JsonResponse({
        'id': str(conversation.id),
        'title': conversation.title,
        'created_at': conversation.created_at.isoformat(),
        'updated_at': conversation.updated_at.isoformat(),
    }, status=201)


@csrf_exempt
@require_http_methods(["GET", "PUT", "DELETE"])
def conversation_detail(request, conversation_id):
    """
    GET    /api/conversations/<id>/  → get conversation with messages
    PUT    /api/conversations/<id>/  → rename conversation
    DELETE /api/conversations/<id>/  → delete conversation
    """
    conversation = get_object_or_404(Conversation, id=conversation_id)

    if request.method == 'GET':
        messages = conversation.messages.all()
        messages_data = []
        for msg in messages:
            contents = msg.generated_contents.all()
            messages_data.append({
                'id': str(msg.id),
                'role': msg.role,
                'content': msg.content,
                'message_type': msg.message_type,
                'created_at': msg.created_at.isoformat(),
                'generated_contents': [
                    {
                        'id': str(gc.id),
                        'content_type': gc.content_type,
                        'title': gc.title,
                        'description': gc.description,
                        'image_url': gc.image_url,
                    }
                    for gc in contents
                ],
            })

        return JsonResponse({
            'id': str(conversation.id),
            'title': conversation.title,
            'created_at': conversation.created_at.isoformat(),
            'updated_at': conversation.updated_at.isoformat(),
            'messages': messages_data,
        })

    if request.method == 'PUT':
        try:
            body = json.loads(request.body)
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)

        title = body.get('title')
        if title:
            conversation.title = title
            conversation.save()

        return JsonResponse({
            'id': str(conversation.id),
            'title': conversation.title,
        })

    if request.method == 'DELETE':
        conversation.delete()
        return JsonResponse({'status': 'deleted'})


@csrf_exempt
@require_http_methods(["POST"])
def send_message(request):
    """
    POST /api/messages/
    Body: { "conversation_id": "...", "content": "..." }

    Creates user message, generates AI response, returns both.
    """
    # Handle both JSON and Multipart requests
    conversation_id = None
    content = ''
    image_file = None

    if request.content_type.startswith('multipart/form-data'):
        content = request.POST.get('content', '').strip()
        conversation_id = request.POST.get('conversation_id')
        image_file = request.FILES.get('image')
    else:
        try:
            body = json.loads(request.body)
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)
        content = body.get('content', '').strip()
        conversation_id = body.get('conversation_id')

    if not conversation_id or (not content and not image_file):
        return JsonResponse(
            {'error': 'conversation_id and content (or image) are required'}, status=400
        )

    conversation = get_object_or_404(Conversation, id=conversation_id)

    # Create user message
    user_message = Message.objects.create(
        conversation=conversation,
        role='user',
        content=content if content else '[Image Uploaded]',
        message_type='text' if not image_file else 'image_transformation',
        image=image_file
    )

    # Auto-title conversation from first message
    if conversation.messages.filter(role='user').count() == 1:
        title_text = content if content else 'Image Upload'
        title = title_text[:50] + ('...' if len(title_text) > 50 else '')
        conversation.title = title
        conversation.save()

    # Generate AI response
    ai_result = generate_response(content, image_file=user_message.image if image_file else None)

    assistant_message = Message.objects.create(
        conversation=conversation,
        role='assistant',
        content=ai_result['response_text'],
        message_type=ai_result['message_type'],
    )

    # Create generated content items
    generated_contents = []
    for item in ai_result['content_items']:
        gc = GeneratedContent.objects.create(
            message=assistant_message,
            content_type=item['content_type'],
            title=item['title'],
            description=item['description'],
            image_url=item['image_url'],
            prompt_used=item['prompt_used'],
        )
        generated_contents.append({
            'id': str(gc.id),
            'content_type': gc.content_type,
            'title': gc.title,
            'description': gc.description,
            'image_url': gc.image_url,
        })

    # Touch conversation to update timestamp
    conversation.save()

    return JsonResponse({
        'user_message': {
            'id': str(user_message.id),
            'role': 'user',
            'content': user_message.content,
            'message_type': user_message.message_type,
            'created_at': user_message.created_at.isoformat(),
            'generated_contents': [],
        },
        'assistant_message': {
            'id': str(assistant_message.id),
            'role': 'assistant',
            'content': assistant_message.content,
            'message_type': assistant_message.message_type,
            'created_at': assistant_message.created_at.isoformat(),
            'generated_contents': generated_contents,
        },
        'conversation_title': conversation.title,
    })


@csrf_exempt
@require_http_methods(["POST"])
def regenerate_message(request, message_id):
    """
    POST /api/messages/<id>/regenerate/

    Regenerates the AI response for a given assistant message.
    Finds the preceding user message and generates a new response.
    """
    assistant_message = get_object_or_404(Message, id=message_id, role='assistant')
    conversation = assistant_message.conversation

    # Find the user message that preceded this assistant message
    user_message = (
        conversation.messages
        .filter(role='user', created_at__lt=assistant_message.created_at)
        .order_by('-created_at')
        .first()
    )

    if not user_message:
        return JsonResponse({'error': 'No user message found to regenerate from'}, status=400)

    # Delete old generated contents
    assistant_message.generated_contents.all().delete()

    # Generate new response
    ai_result = generate_response(user_message.content)

    assistant_message.content = ai_result['response_text']
    assistant_message.message_type = ai_result['message_type']
    assistant_message.save()

    generated_contents = []
    for item in ai_result['content_items']:
        gc = GeneratedContent.objects.create(
            message=assistant_message,
            content_type=item['content_type'],
            title=item['title'],
            description=item['description'],
            image_url=item['image_url'],
            prompt_used=item['prompt_used'],
        )
        generated_contents.append({
            'id': str(gc.id),
            'content_type': gc.content_type,
            'title': gc.title,
            'description': gc.description,
            'image_url': gc.image_url,
        })

    return JsonResponse({
        'id': str(assistant_message.id),
        'role': 'assistant',
        'content': assistant_message.content,
        'message_type': assistant_message.message_type,
        'created_at': assistant_message.created_at.isoformat(),
        'generated_contents': generated_contents,
    })
