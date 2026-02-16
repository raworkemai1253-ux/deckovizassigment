"""
Vizzy Chat AI Service

This module provides the AI creative engine that:
1. Classifies user intent from the message text
2. Determines the appropriate creative pathway
3. Generates visual content using Google Imagen 3 (via google-genai SDK)
4. Generates text responses using Google Gemini (via google-genai SDK)

When GEMINI_API_KEY is set in .env:
- Text: Uses gemini-2.0-flash (or fallback to 1.5)
- Images: Uses imagen-3.0-generate-002

When not set, uses the mock keyword-based classifier with placeholder images.
"""

import random
import hashlib
import logging
import os
import uuid
import base64
from datetime import datetime
from pathlib import Path
import requests
import io

from django.conf import settings
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Initialize Google GenAI Client
# ---------------------------------------------------------------------------
_client = None

try:
    from google import genai
    from google.genai import types

    if settings.GEMINI_API_KEY:
        try:
            _client = genai.Client(api_key=settings.GEMINI_API_KEY)
            print("DEBUG: Google GenAI Client initialized successfully.")
            logger.info("Google GenAI Client initialized successfully.")
        except Exception as e:
            print(f"DEBUG: Failed to initialize GenAI Client: {e}")
            logger.error(f"Failed to initialize GenAI Client: {e}")
    else:
        print("DEBUG: No GEMINI_API_KEY set. Using mock AI service.")
        logger.info("No GEMINI_API_KEY set. Using mock AI service.")

except ImportError:
    print("DEBUG: google-genai package not installed.")
    logger.warning("google-genai package not installed. Using mock AI service.")
except Exception as e:
    print(f"DEBUG: General Error init GenAI: {e}")
    logger.warning(f"Failed to initialize GenAI: {e}. Using mock AI service.")


# ---------------------------------------------------------------------------
# Intent classification keywords
# ---------------------------------------------------------------------------
INTENT_MAP = {
    'image_generation': [
        'paint', 'draw', 'create', 'generate', 'make', 'imagine', 'design',
        'sketch', 'illustrate', 'render', 'visualize', 'show me',
        'image', 'photo', 'picture', 'pic',
    ],
    'image_transformation': [
        'transform', 'turn this', 'convert', 'reimagine', 'restyle',
        'renaissance', 'style transfer', 'remake', 'enhance', 'edit',
    ],
    'poster_design': [
        'poster', 'signage', 'sign', 'banner', 'flyer', 'quote poster',
        'sale poster', 'menu', 'advertisement',
    ],
    'vision_board': [
        'vision board', 'moodboard', 'mood board', 'goals', 'collage',
        'inspiration board',
    ],
    'brand_artwork': [
        'brand', 'logo', 'branding', 'brand-themed', 'product video',
        'apple-esque', 'premium', 'campaign', 'marketing',
    ],
    'story_sequence': [
        'story', 'storybook', 'scene by scene', 'sequence', 'narrative',
        'chapter', 'tale',
    ],
    'video_loop': [
        'video loop', 'animation', 'animated', 'motion', 'looping',
        'cinematic',
    ],
}

STYLES = [
    'ethereal', 'vibrant', 'moody', 'minimalist', 'surrealist',
    'impressionist', 'photorealistic', 'watercolor', 'oil painting',
    'digital art', 'abstract', 'cinematic', 'dreamy', 'bold',
    'vintage', 'futuristic', 'noir', 'pastel', 'geometric',
]


def classify_intent(message_text):
    """
    Classify user intent using Gemini (if available) or fallback to keywords.
    Returns: (intent_type, confidence)
    """
    # 1. Try Gemini Classification if available
    if _client:
        try:
            prompt = f"""
            Analyze the following user message and classify the intent into ONE of these categories:
            - image_generation (user wants to see/create/generate an image, painting, drawing)
            - image_transformation (user wants to edit/transform an existing image)
            - poster_design (user wants a poster, sign, flyer)
            - vision_board (user wants a moodboard, collage, vision board)
            - brand_artwork (user wants logos, branding, marketing visuals)
            - story_sequence (user wants a storyboard, scene-by-scene visualization)
            - video_loop (user wants an animation concept, video loop)
            - text_only (user just wants to chat, ask questions, write text, no visuals needed)

            User Message: "{message_text}"

            Respond ONLY with the category name.
            """
            
            # Try 2.5-flash first, fallback to flash-latest
            model_name = 'gemini-2.5-flash'
            try:
                response = _client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
            except Exception:
                model_name = 'gemini-flash-latest'
                response = _client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )

            intent = response.text.strip().lower()
            
            # validated intent
            valid_intents = list(INTENT_MAP.keys()) + ['text_only']
            if intent in valid_intents:
                return intent, 0.95
                
        except Exception as e:
            print(f"DEBUG: Gemini classification failed: {e}")
            logger.warning(f"Gemini classification failed: {e}. Falling back to keywords.")

    # 2. Fallback to Keyword Matching
    text_lower = message_text.lower()
    scores = {}

    for intent, keywords in INTENT_MAP.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[intent] = score

    if not scores:
        return 'text_only', 0.5

    best_intent = max(scores, key=scores.get)
    confidence = min(scores[best_intent] / 3.0, 1.0)
    return best_intent, confidence


# ---------------------------------------------------------------------------
# Gemini Text Generation
# ---------------------------------------------------------------------------
GEMINI_SYSTEM_PROMPT = """You are Vizzy, an AI creative studio assistant. You help users create visual content like artworks, posters, brand visuals, vision boards, and more.

When the user asks you to create something, respond with:
1. A creative, enthusiastic response (2-3 sentences) describing what you've created
2. Mention the styles and moods you've explored
3. Offer to iterate or refine

Keep your tone warm, creative, and professional. Be concise."""

TEXT_ONLY_SYSTEM_PROMPT = """You are Vizzy, a creative AI assistant. 
You can assign tasks like writing poems, stories, scripts, or answering questions.
You are witty, helpful, and imaginative.

If the user asks for a poem, story, or creative text, WRITE IT. Do not refuse.
If the user asks general questions, answer them helpfuly.
Your goal is to be a versatile creative companion."""


def _generate_gemini_response(message_text, intent):
    """Generate a text response using Google Gemini API."""
    if not _client:
        return None

    system_prompt = GEMINI_SYSTEM_PROMPT
    if intent == 'text_only':
        system_prompt = TEXT_ONLY_SYSTEM_PROMPT

    try:
        # Try 2.5 first
        model_name = 'gemini-2.5-flash'
        try:
            response = _client.models.generate_content(
                model=model_name,
                contents=f"{system_prompt}\n\nUser request: {message_text}"
            )
            return response.text
        except Exception:
             # Fallback
             model_name = 'gemini-flash-latest'
             response = _client.models.generate_content(
                model=model_name,
                contents=f"{system_prompt}\n\nUser request: {message_text}"
            )
             return response.text

    except Exception as e:
        print(f"DEBUG: Gemini Text API error: {e}")
        logger.error(f"Gemini Text API error: {e}")
        return None





# ---------------------------------------------------------------------------
# Hugging Face Image Generation (Fallback)
# ---------------------------------------------------------------------------
def _generate_huggingface_image(prompt):
    """Generate image using Hugging Face Inference API (Stable Diffusion XL)."""
    if not settings.HUGGINGFACE_API_KEY:
        return None

    api_url = "https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-xl-base-1.0"
    headers = {"Authorization": f"Bearer {settings.HUGGINGFACE_API_KEY}"}

    try:
        print(f"DEBUG: Requesting Hugging Face Image: {prompt[:50]}...")
        response = requests.post(api_url, headers=headers, json={"inputs": prompt}, timeout=90)
        
        if response.status_code == 200:
            image_bytes = response.content
            # Verify if it's an image (sometimes they return JSON error with 200?)
            # Usually 200 is image bytes.
            filename = f"generated/{uuid.uuid4()}.png"
            image_content = ContentFile(image_bytes)
            saved_path = default_storage.save(filename, image_content)
            return default_storage.url(saved_path)
        else:
            print(f"DEBUG: Hugging Face Error: {response.status_code} - {response.text}")
            logger.error(f"Hugging Face Error: {response.status_code} - {response.text}")
            return None

    except Exception as e:
        print(f"DEBUG: Hugging Face Exception: {e}")
        logger.error(f"Hugging Face Exception: {e}")
        return None


# ---------------------------------------------------------------------------
# Imagen Image Generation
# ---------------------------------------------------------------------------
def _generate_imagen_images_batch(prompt, count=1, aspect_ratio='1:1'):
    """
    Generate multiple images using Google Imagen 3 via google-genai SDK.
    Returns a list of local URLs.
    """
    if not _client:
        return []

    try:
        print(f"DEBUG: Generating {count} images with Imagen 3...")
        # Config for image generation
        config = types.GenerateImagesConfig(
            number_of_images=count,
            aspect_ratio=aspect_ratio,
            safety_filter_level="BLOCK_LOW_AND_ABOVE",
            person_generation="ALLOW_ADULT",
        )

        # Use standard Imagen 3.0 model
        response = _client.models.generate_images(
            model='imagen-3.0-generate-001',
            prompt=prompt,
            config=config
        )

        if not response.generated_images:
            print("DEBUG: Imagen returned no images.")
            logger.error("Imagen returned no images.")
            return []

        saved_urls = []
        for img_obj in response.generated_images:
            # Save the image
            filename = f"generated/{uuid.uuid4()}.png"
            
            # The image data is in img_obj.image.image_bytes usually
            # SDK documentation says: response.generated_images[].image.image_bytes 
            # or just img_obj.image_bytes depending on structure.
            # Let's inspect slightly carefully or handle the PIL Image if it returns it.
            # google-genai returns GeneratedImage which has 'image' attribute (PIL Image) or 'image_bytes'.
            
            # Check what we got. Assuming we get PIL Image property or bytes.
            # According to SDK: response.generated_images[0].image is PIL.Image.Image if PIL installed.
            
            image_content = None
            if hasattr(img_obj, 'image') and img_obj.image:
                 image = img_obj.image
                 from io import BytesIO
                 buffer = BytesIO()
                 image.save(buffer, format="PNG")
                 image_content = ContentFile(buffer.getvalue())
            elif hasattr(img_obj, 'image_bytes'):
                 image_content = ContentFile(img_obj.image_bytes)
            
            if image_content:
                saved_path = default_storage.save(filename, image_content)
                saved_urls.append(default_storage.url(saved_path))
        
        return saved_urls

    except Exception as e:
        error_msg = str(e)
        if "400" in error_msg or "404" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
             print("DEBUG: Real image generation unavailable (Billing/Quota/Model). using Mock Images.")
             logger.warning(f"Imagen unavailable: {e}. Falling back to mock images.")
        else:
             print(f"DEBUG: Imagen API error: {e}")
             logger.error(f"Imagen API error: {e}")
        return []


def _edit_huggingface_image(image_file, prompt):
    """
    Edit an image using Hugging Face Inference API (InstructPix2Pix).
    """
    if not settings.HUGGINGFACE_API_KEY:
        return None

    api_url = "https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-xl-base-1.0"
    headers = {"Authorization": f"Bearer {settings.HUGGINGFACE_API_KEY}"}

    try:
        print(f"DEBUG: Requesting Image Edit: {prompt[:50]}...")
        
        # Encode image to base64
        image_file.seek(0)
        img_bytes = image_file.read()
        b64_img = base64.b64encode(img_bytes).decode('utf-8')
        
        # Construct payload for img2img
        payload = {
            "inputs": b64_img,
            "parameters": {
                "prompt": prompt,
                "negative_prompt": "blurry, low quality, distorted",
                "strength": 0.7  # Balance between original and prompt
            }
        }

        print(f"DEBUG: Sending img2img request to {api_url}...")
        response = requests.post(api_url, headers=headers, json=payload, timeout=90)

        if response.status_code == 200:
            image_bytes = response.content
            filename = f"generated/edited_{uuid.uuid4()}.png"
            image_content = ContentFile(image_bytes)
            saved_path = default_storage.save(filename, image_content)
            return default_storage.url(saved_path)
        else:
            print(f"DEBUG: Hugging Face Edit Error: {response.status_code} - {response.text}")
            logger.error(f"Hugging Face Edit Error: {response.status_code} - {response.text}")
            return None

    except Exception as e:
        print(f"DEBUG: Hugging Face Edit Exception: {e}")
        logger.error(f"Hugging Face Edit Exception: {e}")
        return None


def _generate_real_content_items(intent, message_text, image_file=None):
    """Generate REAL content items using Imagen 3 or Hugging Face."""
    
    # Handle Image Editing (Transformation)
    if intent == 'image_transformation' and image_file:
         print("DEBUG: Processing Image Transformation...")
         edited_url = _edit_huggingface_image(image_file, message_text)
         
         if edited_url:
             return [{
                 'content_type': 'photo',
                 'title': 'AI Edited Image',
                 'description': f"Edited based on: {message_text}",
                 'image_url': edited_url,
                 'prompt_used': message_text,
             }]
         else:
             print("DEBUG: Image editing failed.")
             # Fallback to mock?
             pass

    if intent == 'text_only':
        return []

    # Determine number of images
    count = 1  
    if intent in ('vision_board', 'story_sequence'):
        count = 2 # limited to 4 usually
    
    # Determine aspect ratio
    aspect_ratio = '1:1'
    if intent == 'poster_design':
        aspect_ratio = '3:4'
    elif intent in ('brand_artwork', 'video_loop'):
        aspect_ratio = '16:9'

    content_type_map = {
        'image_generation': 'artwork',
        'image_transformation': 'photo',
        'poster_design': 'poster',
        'vision_board': 'vision_board',
        'brand_artwork': 'brand_asset',
        'story_sequence': 'artwork',
        'video_loop': 'artwork',
    }
    content_type = content_type_map.get(intent, 'image')

    # Batch generation is efficient
    style = random.choice(STYLES)
    enhanced_prompt = f"{message_text}, {style} style, high quality, detailed, {content_type}"
    
    image_urls = _generate_imagen_images_batch(enhanced_prompt, count=count, aspect_ratio=aspect_ratio)

    if not image_urls and settings.HUGGINGFACE_API_KEY:
        print("DEBUG: Google Imagen failed/unavailable. Falling back to Hugging Face...")
        # Generate images one by one for HF
        for i in range(count):
             # Add variation if multiple
             prompt_var = enhanced_prompt
             if i > 0: prompt_var += f", {random.choice(['different angle', 'variation'])}"
             
             url = _generate_huggingface_image(prompt_var)
             if url:
                 image_urls.append(url)
             else:
                 # If HF fails once, maybe stop? or continue?
                 # HF free tier has rate limits.
                 # Let's try to get at least one.
                 pass

    items = []
    for url in image_urls:
         items.append({
                'content_type': content_type,
                'title': f"{content_type.title()} — {style.title()}",
                'description': f"Generated with AI. Prompt: {enhanced_prompt}",
                'image_url': url,
                'prompt_used': enhanced_prompt,
            })
    
    return items


# ---------------------------------------------------------------------------
# Mock Generation (Fallback)
# ---------------------------------------------------------------------------
def _generate_seed(text):
    return int(hashlib.md5(text.encode()).hexdigest()[:8], 16)


def _get_placeholder_url(seed, width=800, height=600):
    img_id = (seed % 1000) + 1
    return f"https://picsum.photos/seed/{img_id}/{width}/{height}"


def _generate_mock_content_items(intent, message_text, count=None):
    """Generate mock content items with placeholder images."""
    if intent == 'text_only':
        return []

    seed = _generate_seed(message_text + str(datetime.now().timestamp()))

    if count is None:
        count = 2

    content_type_map = {
        'image_generation': 'artwork',
        'image_transformation': 'photo',
        'poster_design': 'poster',
        'vision_board': 'vision_board',
        'brand_artwork': 'brand_asset',
        'story_sequence': 'artwork',
        'video_loop': 'artwork',
    }
    content_type = content_type_map.get(intent, 'image')

    items = []
    for i in range(count):
        style = random.choice(STYLES)
        item_seed = seed + i * 137
        w, h = 800, 600
        if intent == 'poster_design': w, h = 600, 900
        elif intent == 'vision_board': w, h = 400, 400

        items.append({
            'content_type': content_type,
            'title': f"{content_type.title()} — {style.title()} (Mock)",
            'description': f"Mock generated image using picsum.photos.",
            'image_url': _get_placeholder_url(item_seed, w, h),
            'prompt_used': f"[{style}] {message_text}",
        })

    return items


def _build_mock_response_text(intent, message_text, content_count):
    if intent == 'text_only':
        return random.choice([
            "I'm listening. How can I help you?",
            "That sounds interesting. Tell me more.",
            "I'm here to chat. What's on your mind?",
            "Understood. Is there anything specific you'd like to discuss?",
        ])

    responses = [
        f"I've created {content_count} visuals based on your request.",
        f"Here are {content_count} creative interpretations.",
        f"I've generated {content_count} artworks for you.",
    ]
    return random.choice(responses)


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------
def generate_response(message_text, conversation_context=None, image_file=None):
    """
    Generate response using Gemini (Text) + Imagen (Images) or Hugging Face.
    Falls back to mock if API key is missing or calls fail.
    """
    if image_file:
        intent = 'image_transformation'
        confidence = 1.0
    else:
        intent, confidence = classify_intent(message_text)
    
    content_items = []
    response_text = None

    # 1. Generate Images/Edits (so we know count/results for text description)
    
    # Generate Images (if client available or HF key)
    # We check _client OR settings.HUGGINGFACE_API_KEY for edits
    if (_client or settings.HUGGINGFACE_API_KEY) and intent != 'text_only':
        content_items = _generate_real_content_items(intent, message_text, image_file=image_file)
    
    # Fallback to Mock Images
    if not content_items and intent != 'text_only':
        # If real generation failed (or client missing), use mock
        # We check _client again inside mock? No.
        # Just use mock if items are empty and intent is visual.
        content_items = _generate_mock_content_items(intent, message_text)
    
    # Generate Text
    if _client:
        text_prompt = message_text
        if image_file:
            text_prompt = f"[User uploaded an image for editing] {message_text}"
        response_text = _generate_gemini_response(text_prompt, intent)
    
    # Fallback for Text
    if not response_text:
        response_text = _build_mock_response_text(intent, message_text, len(content_items))

    return {
        'intent': intent,
        'confidence': confidence,
        'response_text': response_text,
        'content_items': content_items,
        'message_type': intent,
    }
