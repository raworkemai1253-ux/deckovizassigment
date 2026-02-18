"""
Vizzy Chat AI Service

This module provides the AI creative engine that:
1. Classifies user intent from the message text
2. Determines the appropriate creative pathway
3. Generates visual content using Google Imagen 3 (via google-genai SDK)
4. Generates text responses using Google Gemini (via google-genai SDK)

When GEMINI_API_KEY is set in .env:
- Text: Uses gemini-2.0-flash (or fallback to 1.5)
- Images: Uses imagen-4.0-generate-001

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
from PIL import Image, ImageDraw, ImageFont

from django.conf import settings
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Initialize Google GenAI Client
# ---------------------------------------------------------------------------
_client = None

def get_genai_client():
    global _client
    if _client:
        return _client
        
    if not settings.GEMINI_API_KEY:
        print("DEBUG: No GEMINI_API_KEY set. Using mock AI service.")
        logger.info("No GEMINI_API_KEY set. Using mock AI service.")
        return None

    try:
        from google import genai
        _client = genai.Client(api_key=settings.GEMINI_API_KEY)
        print("DEBUG: Google GenAI Client initialized successfully.")
        logger.info("Google GenAI Client initialized successfully.")
        return _client
    except ImportError:
        print("DEBUG: google-genai package not installed.")
        logger.warning("google-genai package not installed. Using mock AI service.")
        return None
    except Exception as e:
        print(f"DEBUG: Failed to initialize GenAI Client: {e}")
        logger.error(f"Failed to initialize GenAI Client: {e}")
        return None


# ---------------------------------------------------------------------------
# Intent classification & Constants
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
        'cinematic', 'video', 'movie', 'clip',
    ],
    'product_mockup': [
        'mockup', 'mock up', 'product', 't-shirt', 'tshirt', 'mug',
        'phone case', 'hoodie', 'merchandise', 'merch', 'put this on',
        'place on', 'print on',
    ],
    'text_only': [
        'text', 'chat', 'ask', 'question', 'write', 'poem', 'essay',
        'story', 'script', 'code', 'explain', 'tell me', 'lyrics',
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
    client = get_genai_client()
    if client:
        try:
            prompt = f"""
            Analyze the following user message and classify the intent into ONE of these categories:
            - image_generation (user wants to see/create/generate an image, painting, drawing)
            - image_transformation (user wants to edit/transform an existing image)
            - poster_design (user wants a poster, sign, flyer)
            - vision_board (user wants a moodboard, collage, vision board)
            - brand_artwork (user wants logos, branding, marketing visuals)
            - story_sequence (user wants a storyboard, scene-by-scene visualization)
            - video_loop (user wants an animation concept, video loop, movie, clip)
            - product_mockup (user wants to place a design on a product like t-shirt, mug, phone case)
            - text_only (user just wants to chat, ask questions, write text, no visuals needed)

            User Message: "{message_text}"

            Respond ONLY with the category name.
            """
            
            # Try 2.5-flash first, fallback to flash-latest
            model_name = 'gemini-2.5-flash'
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
            except Exception:
                model_name = 'gemini-flash-latest'
                response = client.models.generate_content(
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
        # Weighted scoring: Video and Text keywords are more specific, so give them higher weight
        weight = 2 if intent in ['video_loop', 'text_only'] else 1
        score = sum(weight for kw in keywords if kw in text_lower)
        
        if score > 0:
            scores[intent] = score

    print(f"DEBUG: Intent Scores for '{message_text}': {scores}")

    if not scores:
        return 'text_only', 0.5
    
    best_intent = max(scores, key=scores.get)
    confidence = min(scores[best_intent] / 3.0, 1.0)
    return best_intent, confidence


# ---------------------------------------------------------------------------
# NVIDIA NIM Generation (Prioritized)
# ---------------------------------------------------------------------------
def _generate_nvidia_text(prompt, system_prompt, history=None):
    """Generate text using NVIDIA NIM (Llama 3 70B)."""
    if not settings.NVIDIA_API_KEY:
        return None

    try:
        url = "https://integrate.api.nvidia.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {settings.NVIDIA_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Build messages array with conversation history
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": "meta/llama-3.3-70b-instruct",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1024
        }
        
        print(f"DEBUG: Requesting NVIDIA Text: {prompt[:50]}... (history: {len(history) if history else 0} msgs)")
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            return data['choices'][0]['message']['content']
        else:
            print(f"DEBUG: NVIDIA Text Error: {response.status_code} - {response.text}")
            return None

    except Exception as e:
        print(f"DEBUG: NVIDIA Text Exception: {e}")
        return None


def _generate_nvidia_image(prompt):
    """Generate image using NVIDIA NIM (Stable Diffusion XL)."""
    if not settings.NVIDIA_API_KEY:
        return None

    try:
        url = "https://ai.api.nvidia.com/v1/genai/stabilityai/stable-diffusion-xl"
        headers = {
            "Authorization": f"Bearer {settings.NVIDIA_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        payload = {
            "text_prompts": [{"text": prompt}],
            "cfg_scale": 7,
            "seed": random.randint(0, 100000),
            "sampler": "K_DPM_2_ANCESTRAL",
            "steps": 25
        }
        
        print(f"DEBUG: Requesting NVIDIA Image: {prompt[:50]}...")
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            # SDXL NIM might return base64
            artifacts = data.get('artifacts', [])
            if artifacts:
                image_b64 = artifacts[0].get('base64')
                if image_b64:
                    image_bytes = base64.b64decode(image_b64)
                    filename = f"generated/nvidia_{uuid.uuid4()}.png"
                    image_content = ContentFile(image_bytes)
                    saved_path = default_storage.save(filename, image_content)
                    return default_storage.url(saved_path)
            
            print(f"DEBUG: NVIDIA Image Response malformed: {data.keys()}")
            return None
        else:
            print(f"DEBUG: NVIDIA Image Error: {response.status_code} - {response.text}")
            return None

    except Exception as e:
        print(f"DEBUG: NVIDIA Image Exception: {e}")
        return None


def _generate_nvidia_image_to_image(prompt, image_file):
    """
    Generate image from image using NVIDIA NIM (SDXL Refiner/Img2Img).
    """
    if not settings.NVIDIA_API_KEY:
        return None

    try:
        print(f"DEBUG: NVIDIA Img2Img - Processing base image...")
        
        # Process input image
        # image_file is an UploadedFile object or similar from Django
        image_data = image_file.read()
        
        # Resize to max 1024x1024 to fit API limits/costs and convert to base64
        img = Image.open(io.BytesIO(image_data))
        img = img.convert("RGB")
        img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        url = "https://ai.api.nvidia.com/v1/genai/stabilityai/stable-diffusion-xl"
        headers = {
            "Authorization": f"Bearer {settings.NVIDIA_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        payload = {
            "text_prompts": [{"text": prompt}],
            "init_image": img_b64,
            "cfg_scale": 5,
            "seed": random.randint(0, 100000),
            "sampler": "K_DPM_2_ANCESTRAL",
            "steps": 25,
            "strength": 0.4
        }
        
        print(f"DEBUG: Requesting NVIDIA Img2Img: {prompt[:50]}...")
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            artifacts = data.get('artifacts', [])
            if artifacts:
                image_b64 = artifacts[0].get('base64')
                if image_b64:
                    image_bytes = base64.b64decode(image_b64)
                    filename = f"generated/nvidia_img2img_{uuid.uuid4()}.png"
                    image_content = ContentFile(image_bytes)
                    saved_path = default_storage.save(filename, image_content)
                    return default_storage.url(saved_path)
            
            print(f"DEBUG: NVIDIA Img2Img Response malformed: {data.keys()}")
            return None
        else:
            print(f"DEBUG: NVIDIA Img2Img Error: {response.status_code} - {response.text}")
            return None

    except Exception as e:
        print(f"DEBUG: NVIDIA Img2Img Exception: {e}")
        return None


# ---------------------------------------------------------------------------
# AI Horde Img2Img (Free, No Signup, Community-Powered Stable Diffusion)
# ---------------------------------------------------------------------------
def _generate_aihorde_img2img(prompt, image_file):
    """
    Generate image-to-image using AI Horde (free, no API key required).
    Uses community-powered Stable Diffusion workers.
    Flow: POST /generate/async → poll /generate/status/{id} → download result.
    """
    import time as _time
    
    try:
        print(f"DEBUG: AI Horde Img2Img — Processing base image...")
        
        # Read and resize input image
        image_data = image_file.read()
        img = Image.open(io.BytesIO(image_data))
        img = img.convert("RGB")
        
        # AI Horde requires dimensions to be multiples of 64, max 1024x1024
        w, h = img.size
        max_dim = 512  # Keep small for faster processing on community workers
        if w > h:
            new_w = min(w, max_dim)
            new_h = int(h * new_w / w)
        else:
            new_h = min(h, max_dim)
            new_w = int(w * new_h / h)
        # Round to nearest multiple of 64
        new_w = max(64, (new_w // 64) * 64)
        new_h = max(64, (new_h // 64) * 64)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Convert to webp base64 (AI Horde requires webp)
        buffered = io.BytesIO()
        img.save(buffered, format="WEBP", quality=90)
        img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Step 1: Submit async generation request
        api_url = "https://stablehorde.net/api/v2/generate/async"
        headers = {
            "apikey": "0000000000",  # Anonymous access
            "Client-Agent": "VizzyChat:1.0:vizzy",
            "Content-Type": "application/json",
        }
        
        payload = {
            "prompt": prompt,
            "params": {
                "sampler_name": "k_euler_a",
                "cfg_scale": 7.0,
                "denoising_strength": 0.45,  # 0.0=identical, 1.0=completely new
                "height": new_h,
                "width": new_w,
                "steps": 25,
                "n": 1,
            },
            "source_image": img_b64,
            "source_processing": "img2img",
            "nsfw": False,
            "censor_nsfw": False,
            "models": ["stable_diffusion"],
            "r2": True,
            "shared": True,  # Reduces kudos cost for anonymous users
            "slow_workers": True,
            "trusted_workers": False,
        }
        
        print(f"DEBUG: AI Horde — Submitting img2img request: {prompt[:50]}...")
        response = requests.post(api_url, headers=headers, json=payload, timeout=10)
        
        if response.status_code not in (200, 202):
            print(f"DEBUG: AI Horde submit error: {response.status_code} - {response.text}")
            return None
        
        data = response.json()
        request_id = data.get('id')
        if not request_id:
            print(f"DEBUG: AI Horde — No request ID returned: {data}")
            return None
        
        print(f"DEBUG: AI Horde — Request queued: {request_id}")
        
        # Step 2: Poll for completion (max 120 seconds)
        status_url = f"https://stablehorde.net/api/v2/generate/status/{request_id}"
        max_wait = 120
        poll_interval = 3
        elapsed = 0
        
        while elapsed < max_wait:
            _time.sleep(poll_interval)
            elapsed += poll_interval
            
            try:
                status_resp = requests.get(status_url, headers={
                    "Client-Agent": "VizzyChat:1.0:vizzy"
                }, timeout=15)
                
                if status_resp.status_code != 200:
                    print(f"DEBUG: AI Horde poll error: {status_resp.status_code}")
                    continue
                
                status_data = status_resp.json()
                
                if status_data.get('faulted'):
                    print(f"DEBUG: AI Horde — Request faulted")
                    return None
                
                if status_data.get('done'):
                    generations = status_data.get('generations', [])
                    if generations:
                        gen = generations[0]
                        img_url = gen.get('img')
                        
                        if img_url and img_url.startswith('http'):
                            # Download from R2 URL
                            print(f"DEBUG: AI Horde — Downloading result from R2...")
                            img_resp = requests.get(img_url, timeout=30)
                            if img_resp.status_code == 200:
                                filename = f"generated/aihorde_img2img_{uuid.uuid4()}.webp"
                                image_content = ContentFile(img_resp.content)
                                saved_path = default_storage.save(filename, image_content)
                                result_url = default_storage.url(saved_path)
                                print(f"DEBUG: AI Horde — img2img complete: {result_url}")
                                return result_url
                        elif img_url:
                            # Base64 encoded result
                            image_bytes = base64.b64decode(img_url)
                            filename = f"generated/aihorde_img2img_{uuid.uuid4()}.webp"
                            image_content = ContentFile(image_bytes)
                            saved_path = default_storage.save(filename, image_content)
                            result_url = default_storage.url(saved_path)
                            print(f"DEBUG: AI Horde — img2img complete: {result_url}")
                            return result_url
                    
                    print(f"DEBUG: AI Horde — Done but no generations returned")
                    return None
                
                wait_time = status_data.get('wait_time', '?')
                queue_pos = status_data.get('queue_position', '?')
                print(f"DEBUG: AI Horde — Waiting... pos={queue_pos}, eta={wait_time}s, elapsed={elapsed}s")
                
            except requests.RequestException as poll_err:
                print(f"DEBUG: AI Horde poll exception: {poll_err}")
                continue
        
        print(f"DEBUG: AI Horde — Timed out after {max_wait}s")
        # Cancel the request
        try:
            requests.delete(status_url, headers={"Client-Agent": "VizzyChat:1.0:vizzy"}, timeout=5)
        except Exception:
            pass
        return None
        
    except Exception as e:
        print(f"DEBUG: AI Horde Img2Img Exception: {e}")
        import traceback
        traceback.print_exc()
        return None

# ---------------------------------------------------------------------------
# Cloudflare Workers AI Img2Img (Free — no credit card needed)
# ---------------------------------------------------------------------------
def _generate_cloudflare_img2img(prompt, image_file):
    """
    Generate image-to-image using Cloudflare Workers AI.
    Model: @cf/runwayml/stable-diffusion-v1-5-img2img
    Free tier: 10,000 neurons/day, no credit card required.
    Synchronous API — returns image bytes directly (no polling).
    """
    if not settings.CLOUDFLARE_ACCOUNT_ID or not settings.CLOUDFLARE_API_TOKEN:
        print("DEBUG: Cloudflare Workers AI not configured — skipping")
        return None
    
    try:
        print(f"DEBUG: Cloudflare Workers AI Img2Img — Processing base image...")
        
        # Read and resize input image
        image_data = image_file.read()
        img = Image.open(io.BytesIO(image_data))
        img = img.convert("RGB")
        img.thumbnail((512, 512), Image.Resampling.LANCZOS)
        
        # Convert to base64 string
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        api_url = f"https://api.cloudflare.com/client/v4/accounts/{settings.CLOUDFLARE_ACCOUNT_ID}/ai/run/@cf/runwayml/stable-diffusion-v1-5-img2img"
        headers = {
            "Authorization": f"Bearer {settings.CLOUDFLARE_API_TOKEN}",
            "Content-Type": "application/json",
        }
        
        # Prepare payload with tuned parameters for better quality
        # strength: 0.0=identical, 1.0=completely new. 0.7 gives good balance for "refining"
        # guidance: 7.5 is standard for Stable Diffusion
        # num_steps: 20 (Max limit for Cloudflare Free Tier img2img)
        
        # Add quality boosters to prompt if not present
        if "high quality" not in prompt.lower():
            prompt = f"{prompt}, high quality, detailed, sharp focus"
            
        payload = {
            "prompt": prompt,
            "image_b64": img_b64,
            "strength": 0.7, 
            "num_steps": 20, 
            "guidance": 7.5,
        }
        
        print(f"DEBUG: Cloudflare Workers AI — Requesting img2img: {prompt[:50]}...")
        response = requests.post(api_url, headers=headers, json=payload, timeout=60)
        
        print(f"DEBUG: Cloudflare Status: {response.status_code}")
        print(f"DEBUG: Cloudflare Content-Type: {response.headers.get('content-type', 'N/A')}")
        print(f"DEBUG: Cloudflare Content-Length: {len(response.content) if response.content else 0} bytes")
        
        if response.status_code == 200:
            content_type = response.headers.get('content-type', '')
            
            if 'image' in content_type:
                # Direct image response
                if not response.content:
                    print(f"DEBUG: Cloudflare Workers AI — Received empty image content!")
                    return None
                    
                filename = f"generated/cloudflare_img2img_{uuid.uuid4()}.png"
                image_content = ContentFile(response.content)
                saved_path = default_storage.save(filename, image_content)
                result_url = default_storage.url(saved_path)
                print(f"DEBUG: Cloudflare Workers AI — img2img complete: {result_url} ({len(response.content)} bytes)")
                return result_url
            else:
                # JSON response (might contain base64 or URL)
                try:
                    data = response.json()
                    print(f"DEBUG: Cloudflare Workers AI — JSON response keys: {list(data.keys())}")
                    
                    if data.get('result'):
                        img_result = data['result']
                        if isinstance(img_result, str):
                            # It might be base64
                            try:
                                image_bytes = base64.b64decode(img_result)
                                if not image_bytes:
                                    print(f"DEBUG: Cloudflare Workers AI — Decoded base64 is empty!")
                                    return None
                                    
                                filename = f"generated/cloudflare_img2img_{uuid.uuid4()}.png"
                                image_content = ContentFile(image_bytes)
                                saved_path = default_storage.save(filename, image_content)
                                result_url = default_storage.url(saved_path)
                                print(f"DEBUG: Cloudflare Workers AI — img2img complete: {result_url} ({len(image_bytes)} bytes)")
                                return result_url
                            except Exception as b64e:
                                print(f"DEBUG: Cloudflare Workers AI — Base64 decode error: {b64e}")
                                return None
                    
                    if data.get('success') is False:
                        print(f"DEBUG: Cloudflare Workers AI — Success=False. Errors: {data.get('errors')}")
                        return None
                        
                except Exception as json_e:
                    print(f"DEBUG: Cloudflare Workers AI — JSON parsing error: {json_e}")
                    print(f"DEBUG: Response text (first 200 chars): {response.text[:200]}")
                
                print(f"DEBUG: Cloudflare Workers AI — Unexpected response format")
                return None
        else:
            error_text = response.text[:300] if response.text else 'No response body'
            print(f"DEBUG: Cloudflare Workers AI Error: {response.status_code} - {error_text}")
            return None
    
    except Exception as e:
        print(f"DEBUG: Cloudflare Workers AI Img2Img Exception: {e}")
        import traceback
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Replicate Video Generation (Direct HTTP API — no SDK needed)
# ---------------------------------------------------------------------------
def _generate_replicate_video(prompt):
    """
    Generate video using Replicate HTTP API (minimax/video-01).
    Uses direct HTTP calls instead of the replicate SDK to avoid
    Pydantic V1 incompatibility with Python 3.14.
    Returns: URL to the saved video file, or None.
    """
    if not settings.REPLICATE_API_TOKEN:
        print("DEBUG: Replicate API token not configured")
        return None

    try:
        import time as _time

        api_token = settings.REPLICATE_API_TOKEN
        headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
            "Prefer": "wait",  # Tells Replicate to wait up to 60s before returning
        }

        print(f"DEBUG: Replicate Video — generating for: '{prompt[:60]}'...")

        # Step 1: Create prediction via HTTP API
        create_url = "https://api.replicate.com/v1/models/minimax/video-01/predictions"
        payload = {
            "input": {
                "prompt": prompt,
                "prompt_optimizer": True,
            }
        }

        print(f"DEBUG: Replicate — creating prediction...")
        resp = requests.post(create_url, headers=headers, json=payload, timeout=120)

        if resp.status_code not in (200, 201, 202):
            print(f"DEBUG: Replicate create failed: {resp.status_code} - {resp.text[:200]}")
            return None

        prediction = resp.json()
        pred_id = prediction.get("id")
        status = prediction.get("status")
        print(f"DEBUG: Replicate — prediction {pred_id}, status: {status}")

        # Step 2: Poll for completion (if not already done via Prefer: wait)
        poll_url = f"https://api.replicate.com/v1/predictions/{pred_id}"
        poll_headers = {
            "Authorization": f"Bearer {api_token}",
        }

        max_wait = 300  # 5 minutes max
        waited = 0
        while status not in ("succeeded", "failed", "canceled") and waited < max_wait:
            _time.sleep(5)
            waited += 5
            poll_resp = requests.get(poll_url, headers=poll_headers, timeout=30)
            if poll_resp.status_code == 200:
                prediction = poll_resp.json()
                status = prediction.get("status")
                print(f"DEBUG: Replicate — polling... status: {status} ({waited}s)")
            else:
                print(f"DEBUG: Replicate poll error: {poll_resp.status_code}")
                break

        if status != "succeeded":
            error = prediction.get("error", "Unknown error")
            print(f"DEBUG: Replicate prediction failed: {status} — {error}")
            return None

        # Step 3: Get output URL
        output = prediction.get("output")
        video_url = None

        if isinstance(output, str):
            video_url = output
        elif isinstance(output, list) and len(output) > 0:
            video_url = output[0] if isinstance(output[0], str) else str(output[0])
        elif isinstance(output, dict):
            video_url = output.get("url") or output.get("video")

        if not video_url or not video_url.startswith("http"):
            print(f"DEBUG: Replicate — unexpected output format: {output}")
            return None

        print(f"DEBUG: Replicate Video — downloading from: {video_url[:80]}...")

        # Step 4: Download and save locally
        dl_resp = requests.get(video_url, timeout=120)
        if dl_resp.status_code != 200:
            print(f"DEBUG: Replicate download failed: {dl_resp.status_code}")
            return None

        video_data = dl_resp.content

        ext = "mp4"
        if ".webm" in video_url:
            ext = "webm"

        filename = f"generated/replicate_video_{uuid.uuid4().hex[:8]}.{ext}"
        saved_path = default_storage.save(filename, ContentFile(video_data))
        local_url = f"{settings.MEDIA_URL}{saved_path}"
        print(f"DEBUG: Replicate Video saved: {local_url}")
        return local_url

    except Exception as e:
        print(f"DEBUG: Replicate Video Exception: {e}")
        import traceback
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Product Mockup (Composite Approach)
# ---------------------------------------------------------------------------
def _generate_product_mockup(prompt, image_file):
    """
    Generate a product mockup by creating a base product image
    and overlaying the uploaded design using PIL.
    """
    try:
        # 1. Determine product type from prompt
        prompt_lower = prompt.lower()
        if any(w in prompt_lower for w in ['t-shirt', 'tshirt', 'shirt']):
            product_type = 't-shirt'
            base_prompt = 'a blank white t-shirt on a flat surface, studio lighting, product photography'
            overlay_area = (0.25, 0.20, 0.75, 0.65)  # x1, y1, x2, y2 as fractions
        elif any(w in prompt_lower for w in ['mug', 'cup']):
            product_type = 'mug'
            base_prompt = 'a blank white ceramic coffee mug, studio lighting, product photography'
            overlay_area = (0.20, 0.25, 0.80, 0.75)
        elif any(w in prompt_lower for w in ['phone', 'case']):
            product_type = 'phone case'
            base_prompt = 'a blank white phone case, studio lighting, product photography'
            overlay_area = (0.15, 0.10, 0.85, 0.90)
        elif any(w in prompt_lower for w in ['hoodie', 'sweater']):
            product_type = 'hoodie'
            base_prompt = 'a blank white hoodie on a flat surface, studio lighting, product photography'
            overlay_area = (0.25, 0.20, 0.75, 0.60)
        else:
            product_type = 't-shirt'
            base_prompt = 'a blank white t-shirt on a flat surface, studio lighting, product photography'
            overlay_area = (0.25, 0.20, 0.75, 0.65)

        print(f"DEBUG: Product mockup — type={product_type}")

        # 2. Generate base product image
        base_url = None
        if settings.NVIDIA_API_KEY:
            base_url = _generate_nvidia_image(base_prompt)
        if not base_url:
            base_url = _generate_pollinations_image(base_prompt)
        if not base_url:
            print("DEBUG: Could not generate base product image")
            return None

        # 3. Download base image
        import urllib.request
        if base_url.startswith('/media/'):
            base_path = Path(settings.MEDIA_ROOT) / base_url.replace('/media/', '')
            base_img = Image.open(base_path).convert('RGBA')
        else:
            req = urllib.request.urlopen(base_url, timeout=15)
            base_img = Image.open(io.BytesIO(req.read())).convert('RGBA')

        # 4. Load overlay (user's uploaded image)
        image_file.seek(0)
        overlay_img = Image.open(image_file).convert('RGBA')

        # 5. Calculate overlay position
        bw, bh = base_img.size
        ox1, oy1, ox2, oy2 = overlay_area
        target_w = int(bw * (ox2 - ox1))
        target_h = int(bh * (oy2 - oy1))

        # Resize overlay to fit target area, maintaining aspect ratio
        overlay_img.thumbnail((target_w, target_h), Image.LANCZOS)
        ow, oh = overlay_img.size

        # Center within the target area
        paste_x = int(bw * ox1) + (target_w - ow) // 2
        paste_y = int(bh * oy1) + (target_h - oh) // 2

        # 6. Composite
        composite = base_img.copy()
        # Reduce overlay opacity slightly for realism
        overlay_with_alpha = overlay_img.copy()
        alpha = overlay_with_alpha.split()[3]
        alpha = alpha.point(lambda p: int(p * 0.85))
        overlay_with_alpha.putalpha(alpha)
        composite.paste(overlay_with_alpha, (paste_x, paste_y), overlay_with_alpha)

        # 7. Save
        composite_rgb = composite.convert('RGB')
        buffer = io.BytesIO()
        composite_rgb.save(buffer, format='JPEG', quality=90)
        buffer.seek(0)

        filename = f"mockups/mockup_{uuid.uuid4().hex[:8]}.jpg"
        saved_path = default_storage.save(filename, ContentFile(buffer.read()))
        url = f"{settings.MEDIA_URL}{saved_path}"
        print(f"DEBUG: Product mockup saved: {url}")
        return url

    except Exception as e:
        print(f"DEBUG: Product mockup error: {e}")
        return None


# ---------------------------------------------------------------------------
# Context & Memory Management
# ---------------------------------------------------------------------------
def _update_user_context(conversation, message_text, intent):
    """
    Extract preference/style keywords from the message and update conversation context.
    """
    if not conversation:
        return

    # specific styles we want to remember
    interesting_keywords = [
        'minimalist', 'cyberpunk', 'watercolor', 'noir', 'vintage', 
        'abstract', 'photorealistic', '3d render', 'flat design',
        'apple-esque', 'premium', 'dark mode', 'pastel'
    ]
    
    found_styles = [kw for kw in interesting_keywords if kw in message_text.lower()]
    
    if found_styles:
        current_context = conversation.user_context or {}
        # Update 'preferred_styles'
        existing_styles = current_context.get('preferred_styles', [])
        # Add new styles
        updated_styles = list(set(existing_styles + found_styles))
        current_context['preferred_styles'] = updated_styles
        
        conversation.user_context = current_context
        conversation.save()
        print(f"DEBUG: Updated User Context: {current_context}")


def _get_context_prompt(conversation):
    """Generate a prompt snippet based on user context."""
    if not conversation or not conversation.user_context:
        return ""
    
    context = conversation.user_context
    styles = context.get('preferred_styles', [])
    if styles:
        return f"User prefers these styles: {', '.join(styles)}. Incorporate them if relevant."
    return ""


# ---------------------------------------------------------------------------
# GIF Generation
# ---------------------------------------------------------------------------


def _create_gif_from_images(image_urls, fps=2):
    """
    Download images from URLs and stitch them into a GIF.
    Returns the URL of the generated GIF.
    """
    try:
        frames = []
        
        for url in image_urls:
            # Handle local or remote URLs
            if url.startswith('/media/'):
                # Local file
                 # Ensure MEDIA_ROOT is a Path object or string
                 media_root = Path(settings.MEDIA_ROOT)
                 rel_path = url.replace('/media/', '').lstrip('/')
                 path = media_root / rel_path
                 
                 if path.exists():
                     try:
                         img = Image.open(path)
                         frames.append(img)
                     except Exception:
                         pass

            elif url.startswith('http'):
                # Download remote image
                try:
                    resp = requests.get(url, timeout=10)
                    if resp.status_code == 200:
                        img = Image.open(io.BytesIO(resp.content))
                        frames.append(img)
                except Exception:
                    pass

        if not frames:
            return None

        # Resize to match first frame
        base_size = frames[0].size
        resized_frames = [f.resize(base_size) for f in frames]

        # Save GIF
        blob = io.BytesIO()
        resized_frames[0].save(
            blob, 
            format='GIF', 
            save_all=True, 
            append_images=resized_frames[1:], 
            duration=500, # 500ms per frame
            loop=0
        )
        
        filename = f"generated/animation_{uuid.uuid4()}.gif"
        gif_content = ContentFile(blob.getvalue())
        saved_path = default_storage.save(filename, gif_content)
        final_url = default_storage.url(saved_path)
        return final_url

    except Exception:
        # Silently fail or log if needed, but for now we just return None
        return None
        print(f"DEBUG: GIF Generation failed: {e}")
        logger.error(f"GIF Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Gemini Text Generation
# ---------------------------------------------------------------------------
GEMINI_SYSTEM_PROMPT = """You are Vizzy, an AI creative studio assistant. You help users create visual content like artworks, posters, brand visuals, vision boards, and more.

When the user asks you to create something, respond with:
1. A creative, enthusiastic response (2-3 sentences) describing what you've created
2. Mention the styles and moods you've explored
3. Offer to iterate or refine

Keep your tone warm, creative, and professional. Be concise.
"""

TEXT_ONLY_SYSTEM_PROMPT = """You are Vizzy, a creative AI assistant. 
You can assign tasks like writing poems, stories, scripts, or answering questions.
You are witty, helpful, and imaginative.

If the user asks for a poem, story, or creative text, WRITE IT. Do not refuse.
If the user asks general questions, answer them helpfuly.
Your goal is to be a versatile creative companion."""


def _generate_gemini_response(message_text, intent, context_prompt="", history=None):
    """Generate a text response using Google Gemini API."""
    client = get_genai_client()
    if not client:
        return None

    system_prompt = GEMINI_SYSTEM_PROMPT
    if intent == 'text_only':
        system_prompt = TEXT_ONLY_SYSTEM_PROMPT
    
    # Build conversation context from history
    history_text = ""
    if history:
        history_text = "\n\nConversation history:\n"
        for msg in history[-8:]:  # Last 8 messages max
            role_label = "User" if msg['role'] == 'user' else "Assistant"
            history_text += f"{role_label}: {msg['content'][:200]}\n"
        history_text += "---\n"
    
    full_prompt = f"{system_prompt}\n\n{context_prompt}{history_text}\n\nUser request: {message_text}"

    try:
        # Try 2.5 first
        model_name = 'gemini-2.5-flash'
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=full_prompt
            )
            return response.text
        except Exception:
             # Fallback
             model_name = 'gemini-flash-latest'
             response = client.models.generate_content(
                model=model_name,
                contents=full_prompt
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
    client = get_genai_client()
    if not client:
        return []

    try:
        print(f"DEBUG: Generating {count} images with Imagen 3...")
        # Config for image generation
        # Use standard Imagen 3.0 model
        from google.genai import types 
        
        config = types.GenerateImagesConfig(
            number_of_images=count,
            aspect_ratio=aspect_ratio,
            safety_filter_level="BLOCK_LOW_AND_ABOVE",
            person_generation="ALLOW_ADULT",
        )

        response = client.models.generate_images(
            model='imagen-4.0-generate-001',
            prompt=prompt,
            config=config
        )

        if not response.generated_images:
            print("DEBUG: Imagen returned no images.")
            logger.error("Imagen returned no images.")
            return []

        saved_urls = []
        for img_obj in response.generated_images:
            filename = f"generated/{uuid.uuid4()}.png"
            
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
        # Fallback handling
        print(f"DEBUG: Imagen API error: {e}")
        logger.error(f"Imagen API error: {e}")
        return []


def _edit_huggingface_image(image_file, prompt):
    """
    Edit an image using Hugging Face Inference API (InstructPix2Pix).
    """
    if not settings.HUGGINGFACE_API_KEY:
        return None

    api_url = "https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-xl-base-1.0" # SDXL isn't img2img edit strictly, but let's assume it works or swap to instruct-pix2pix
    # Better model for editing: "timbrooks/instruct-pix2pix"
    api_url = "https://router.huggingface.co/hf-inference/models/timbrooks/instruct-pix2pix"

    headers = {"Authorization": f"Bearer {settings.HUGGINGFACE_API_KEY}"}

    try:
        print(f"DEBUG: Requesting Image Edit: {prompt[:50]}...")
        
        image_file.seek(0)
        img_bytes = image_file.read()
        b64_img = base64.b64encode(img_bytes).decode('utf-8')
        
        payload = {
            "inputs": b64_img,
            "parameters": {
                "prompt": prompt,
                "negative_prompt": "blurry, low quality, distorted",
                # instruct-pix2pix specific params
                "image_guidance_scale": 1.5,
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


# ---------------------------------------------------------------------------
# Pollinations AI Image Generation (Unlimited Fallback)
# ---------------------------------------------------------------------------
def _generate_pollinations_image(prompt):
    """
    Generate image using Pollinations.AI (Free, Unlimited).
    """
    try:
        # URL encode the prompt
        import urllib.parse
        encoded_prompt = urllib.parse.quote(prompt)
        
        # Pollinations API: https://image.pollinations.ai/prompt/{prompt}
        # We can add parameters like width, height, model, seed
        # seed = random.randint(0, 10000)
        # url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&seed={seed}&nologo=true"
        
        api_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}"
        params = {
            "width": 1024,
            "height": 1024,
            "nologo": "true",
            "seed": random.randint(0, 50000),
            "model": "flux" # or 'turbo'
        }
        
        print(f"DEBUG: Requesting Pollinations Image: {prompt[:50]}...")
        response = requests.get(api_url, params=params, timeout=60)

        if response.status_code == 200:
            image_bytes = response.content
            filename = f"generated/pollinations_{uuid.uuid4()}.jpg"
            image_content = ContentFile(image_bytes)
            saved_path = default_storage.save(filename, image_content)
            return default_storage.url(saved_path)
        else:
            print(f"DEBUG: Pollinations Error: {response.status_code}")
            return None

    except Exception as e:
        print(f"DEBUG: Pollinations Exception: {e}")
        logger.error(f"Pollinations Exception: {e}")
        return None


def _generate_real_content_items(intent, message_text, conversation=None, image_file=None):
    """Generate REAL content items using Imagen 3 or Hugging Face."""
    
    # Context injection
    context_style = ""
    if conversation and conversation.user_context:
        styles = conversation.user_context.get('preferred_styles', [])
        if styles:
            context_style = f", infusing styles: {', '.join(styles)}"

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
             pass

    if intent == 'text_only':
        return []

    # Determine number of images
    # Default high for variety
    count = 4
    
    # Determine aspect ratio details
    aspect_ratio = '1:1'
    keyword_suffix = ""
    
    if intent == 'poster_design':
        aspect_ratio = '3:4'
        keyword_suffix = ", poster design, typography, graphic design"
    elif intent == 'brand_artwork':
        aspect_ratio = '16:9'
        keyword_suffix = ", logo, vector art, minimal, professional branding"
        if "apple" in message_text.lower():
            keyword_suffix += ", apple-style, minimalist, sleek, white background, premium lighting"
    elif intent == 'video_loop':
        aspect_ratio = '16:9'
        keyword_suffix = ", cinematic shot, movie scene, keyframe, detailed 8k"
    elif intent == 'vision_board':
        aspect_ratio = '1:1' # or 3:4
        keyword_suffix = ", moodboard, collage, grid layout, aesthetic"

    content_type_map = {
        'image_generation': 'artwork',
        'image_transformation': 'photo',
        'poster_design': 'poster',
        'vision_board': 'vision_board',
        'brand_artwork': 'brand_asset',
        'story_sequence': 'artwork',
        'video_loop': 'video', # Special case
    }
    content_type = content_type_map.get(intent, 'image')

    # Batch generation
    style = random.choice(STYLES)
    enhanced_prompt = f"{message_text}, {style} style{context_style}{keyword_suffix}, high quality, detailed"
    
    image_urls = []

    # Priority: NVIDIA NIM -> Google Imagen -> Hugging Face -> Pollinations
    if settings.NVIDIA_API_KEY:
         print(f"DEBUG: Generating {count} images with NVIDIA NIM...")
         for i in range(count):
             prompt_var = enhanced_prompt + f" variation {i+1}"
             url = _generate_nvidia_image(prompt_var)
             if url:
                 image_urls.append(url)
    
    if not image_urls:
        image_urls = _generate_imagen_images_batch(enhanced_prompt, count=count, aspect_ratio=aspect_ratio)

    # GIF Creation for Video Loop
    if intent == 'video_loop' and len(image_urls) >= 2:
        print("DEBUG: Generating GIF for video loop...")
        gif_url = _create_gif_from_images(image_urls)
        if gif_url:
            # Return the GIF as the main item, maybe others as frames?
            # For now, just return the GIF as a single "Video" item
            return [{
                'content_type': 'video',
                'title': f"AI Generated Concept Loop",
                'description': f"Animated GIF from concept frames. Style: {style}",
                'image_url': gif_url,
                'prompt_used': enhanced_prompt,
            }]

    # Fallback to HF if no images from Imagen
    if not image_urls and settings.HUGGINGFACE_API_KEY:
        print("DEBUG: Google Imagen failed/unavailable. Falling back to Hugging Face...")
        for i in range(2): # limit fallback to 2
             prompt_var = enhanced_prompt + f", variation {i+1}"
             url = _generate_huggingface_image(prompt_var)
             if url:
                 image_urls.append(url)

    # Final Fallback to Pollinations AI (Unlimited)
    if not image_urls:
        print("DEBUG: Hugging Face failed/unavailable. Falling back to Pollinations AI...")
        for i in range(count): # Generates requested count (usually 4)
             prompt_var = enhanced_prompt + f" variation {i+1}"
             url = _generate_pollinations_image(prompt_var)
             if url:
                 image_urls.append(url)

    items = []
    for url in image_urls:
         items.append({
                'content_type': content_type,
                'title': f"{content_type.title()} — {style.title()}",
                'description': f"Generated with AI.",
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

def _generate_mock_content_items(intent, message_text, count=4):
    """Generate mock content items with placeholder images."""
    if intent == 'text_only':
        return []

    seed = _generate_seed(message_text + str(datetime.now().timestamp()))

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
    
    # Pre-generate URLs for GIF creation if needed
    mock_urls = []
    for i in range(count):
        item_seed = seed + i * 137
        w, h = 800, 600
        if intent == 'poster_design': w, h = 600, 900
        elif intent == 'vision_board': w, h = 400, 400
        elif intent == 'video_loop': w, h = 1280, 720 # Cinematic aspect

        mock_urls.append(_get_placeholder_url(item_seed, w, h))

    # Handle Video Loop specifically for Mock
    if intent == 'video_loop':
        print("DEBUG: Generating Mock GIFs for video loop...")
        video_items = []
        for i in range(count):
             # Generate a unique sequence for each video variant
             variant_seed = seed + (i * 999)
             variant_urls = []
             for j in range(4): # 4 frames per video
                 frame_seed = variant_seed + j
                 variant_urls.append(_get_placeholder_url(frame_seed, 1280, 720))
            
             gif_url = _create_gif_from_images(variant_urls)
             if gif_url:
                 video_items.append({
                    'content_type': 'video',
                    'title': f"Concept Loop {i+1} (Mock)",
                    'description': f"Variation {i+1}: Animated GIF from mock frames.",
                    'image_url': gif_url,
                    'prompt_used': f"[Mock Video Var {i+1}] {message_text}",
                })
        
        if video_items:
            return video_items

    for i, url in enumerate(mock_urls):
        style = random.choice(STYLES)
        items.append({
            'content_type': content_type,
            'title': f"{content_type.title()} — {style.title()} (Mock)",
            'description': f"Mock generated image.",
            'image_url': url,
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
    
    base_response = random.choice(responses)
    
    # If the user wanted text (but got fallback), give them some options
    if intent == 'story_sequence' or intent == 'brand_artwork':
        base_response += "\n\n**Option 1:** A bold, modern approach focusing on minimalism.\n\n**Option 2:** A vibrant, energetic style with high contrast.\n\n**Option 3:** A sophisticated, premium look with subtle details."

    return base_response


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------
def generate_response(message_text, conversation=None, image_file=None, mode=None, refinement_url=None):
    """
    Generate response using Gemini (Text) + Imagen (Images).
    
    Args:
        message_text (str): User prompt
        conversation (Conversation): Context
        image_file (File): Uploaded image
        mode (str): 'image', 'video' or None (auto)
        refinement_url (str): URL of a previously generated image to refine
    """
    
    # 0. Determine Intent
    # Detect short follow-ups like "yes", "more", "another" — use last message's intent
    SHORT_FOLLOWUPS = {'yes', 'yeah', 'yep', 'sure', 'ok', 'okay', 'more', 'another', 
                       'another one', 'do it', 'go ahead', 'please', 'continue', 'yes please',
                       'one more', 'again', 'try again'}
    is_short_followup = message_text.strip().lower().rstrip('.!') in SHORT_FOLLOWUPS
    
    if refinement_url:
        intent = 'refinement'
        confidence = 1.0
    elif image_file:
        intent = 'image_transformation'
        confidence = 1.0
    elif is_short_followup and conversation:
        # Look at the last assistant message to determine what to continue
        try:
            last_assistant = conversation.messages.filter(role='assistant').order_by('-created_at').first()
            if last_assistant and last_assistant.message_type != 'text':
                intent = last_assistant.message_type
                confidence = 0.9
                print(f"DEBUG: Short follow-up detected — reusing last intent: {intent}")
            else:
                intent = 'text_only'
                confidence = 0.85
        except Exception:
            intent = 'text_only'
            confidence = 0.5
    elif mode == 'video':
        intent = 'video_loop'
        confidence = 1.0
    elif mode == 'image':
        pred_intent, conf = classify_intent(message_text)
        if pred_intent == 'text_only':
            intent = 'image_generation'
            confidence = 1.0
        else:
            intent = pred_intent
            confidence = conf
    else:
        intent, confidence = classify_intent(message_text)
    
    print(f"DEBUG: Intent detected: {intent} (Mode: {mode}, Refinement: {bool(refinement_url)}, ShortFollowup: {is_short_followup})")

    # 1. Update Context
    if conversation:
        _update_user_context(conversation, message_text, intent)
    
    content_items = []
    response_text = None

    # 2. Generate Content
    client = get_genai_client()
    
    # [REFINEMENT] User selected a previously generated image and wants to modify it
    if intent == 'refinement' and refinement_url:
        print(f"DEBUG: Refinement mode — downloading selected image from {refinement_url}")
        try:
            # Download the selected image from local media storage
            from pathlib import Path
            from django.core.files.uploadedfile import SimpleUploadedFile
            
            rel_path = refinement_url.replace(settings.MEDIA_URL, '').lstrip('/')
            media_root = Path(settings.MEDIA_ROOT)
            file_path = media_root / rel_path
            
            if file_path.exists():
                with open(file_path, 'rb') as f:
                    image_data = f.read()
                
                # Create a file-like object for the img2img function
                fake_file = SimpleUploadedFile(
                    name=file_path.name,
                    content=image_data,
                    content_type='image/png'
                )
                
                # Try Cloudflare Workers AI img2img first (primary — free, no credit card)
                if settings.CLOUDFLARE_ACCOUNT_ID and settings.CLOUDFLARE_API_TOKEN:
                    print(f"DEBUG: Refining with Cloudflare Workers AI img2img (primary)...")
                    img_url = _generate_cloudflare_img2img(message_text, fake_file)
                    if img_url:
                        content_items.append({
                            'content_type': 'image_generation',
                            'title': f"Refined Image",
                            'description': f"Refined: {message_text[:80]}",
                            'image_url': img_url,
                            'prompt_used': message_text,
                        })
                
                # Fallback 1: Try AI Horde img2img (free, no key)
                if not content_items:
                    print(f"DEBUG: Refinement fallback — trying AI Horde img2img...")
                    fake_file.seek(0)
                    img_url = _generate_aihorde_img2img(message_text, fake_file)
                    if img_url:
                        content_items.append({
                            'content_type': 'image_generation',
                            'title': f"Refined Image",
                            'description': f"Refined: {message_text[:80]}",
                            'image_url': img_url,
                            'prompt_used': message_text,
                        })
                
                # Fallback 2: Try NVIDIA img2img
                if not content_items and settings.NVIDIA_API_KEY:
                    print(f"DEBUG: Refinement fallback — trying NVIDIA SDXL img2img...")
                    fake_file.seek(0)
                    img_url = _generate_nvidia_image_to_image(message_text, fake_file)
                    if img_url:
                        content_items.append({
                            'content_type': 'image_generation',
                            'title': f"Refined Image",
                            'description': f"Refined: {message_text[:80]}",
                            'image_url': img_url,
                            'prompt_used': message_text,
                        })
                
                # Fallback 2: Try HuggingFace InstructPix2Pix
                if not content_items and settings.HUGGINGFACE_API_KEY:
                    print(f"DEBUG: Refinement fallback — trying HuggingFace InstructPix2Pix...")
                    fake_file.seek(0)
                    edited_url = _edit_huggingface_image(fake_file, message_text)
                    if edited_url:
                        content_items.append({
                            'content_type': 'image_generation',
                            'title': f"Refined Image",
                            'description': f"Refined: {message_text[:80]}",
                            'image_url': edited_url,
                            'prompt_used': message_text,
                        })
                
                # Final fallback: use Pollinations to generate a new image with a descriptive prompt
                if not content_items:
                    print(f"DEBUG: Refinement fallback — using Pollinations text-to-image")
                    merged_prompt = f"{message_text}, maintaining the same composition, colors, and subject as the original image, photorealistic, high quality"
                    poll_url = _generate_pollinations_image(merged_prompt)
                    if poll_url:
                        content_items.append({
                            'content_type': 'image_generation',
                            'title': f"Refined Image",
                            'description': f"Refined (via text prompt): {message_text[:80]}",
                            'image_url': poll_url,
                            'prompt_used': message_text,
                        })
                    else:
                        # Last resort: try regular content generation
                        content_items = _generate_real_content_items('image_generation', merged_prompt, conversation, None)
            else:
                print(f"DEBUG: Refinement image not found at {file_path}")
        except Exception as e:
            print(f"DEBUG: Refinement error: {e}")
    
    # [VIDEO GENERATION via Replicate]
    elif intent == 'video_loop' and settings.REPLICATE_API_TOKEN:
        print(f"DEBUG: Generating Video with Replicate...")
        video_url = _generate_replicate_video(message_text)
        if video_url:
             content_items.append({
                'content_type': 'video',
                'title': "AI Video",
                'description': f"Generated video: {message_text[:60]}",
                'image_url': video_url,
                'prompt_used': message_text,
            })
    
     # [IMAGE TRANSFORMATION via img2img]
    elif intent == 'image_transformation' and image_file:
         print(f"DEBUG: Transforming Image with img2img...")
         img_url = None
         # Try Cloudflare Workers AI first (primary — free, no credit card)
         if settings.CLOUDFLARE_ACCOUNT_ID and settings.CLOUDFLARE_API_TOKEN:
             img_url = _generate_cloudflare_img2img(message_text, image_file)
         # Fallback to AI Horde
         if not img_url:
             image_file.seek(0)
             img_url = _generate_aihorde_img2img(message_text, image_file)
         # Fallback to NVIDIA
         if not img_url and settings.NVIDIA_API_KEY:
             image_file.seek(0)
             img_url = _generate_nvidia_image_to_image(message_text, image_file)
         # Fallback to Pollinations (text-to-image, always works)
         if not img_url:
             img_url = _generate_pollinations_image(f"{message_text}, photorealistic, high quality")
         if img_url:
             content_items.append({
                'content_type': 'image_transformation',
                'title': "Remixed Image",
                'description': "Transformed using img2img AI.",
                'image_url': img_url,
                'prompt_used': message_text,
            })

    # [PRODUCT MOCKUP] Composite: generate base product + overlay uploaded design
    elif intent == 'product_mockup' and image_file:
        print(f"DEBUG: Creating product mockup...")
        mockup_url = _generate_product_mockup(message_text, image_file)
        if mockup_url:
            content_items.append({
                'content_type': 'image_generation',
                'title': f"Product Mockup",
                'description': f"Your design placed on a product.",
                'image_url': mockup_url,
                'prompt_used': message_text,
            })

    if not content_items and (client or settings.HUGGINGFACE_API_KEY or settings.NVIDIA_API_KEY) and intent != 'text_only':
        content_items = _generate_real_content_items(intent, message_text, conversation, image_file)
    
    # Fallback/Mock
    if not content_items and intent != 'text_only':
        content_items = _generate_mock_content_items(intent, message_text)
    
    # 3. Generate Text Response
    # Build conversation history for context
    chat_history = None
    if conversation:
        try:
            recent_msgs = conversation.messages.order_by('-created_at')[:10]
            chat_history = []
            for m in reversed(list(recent_msgs)):
                chat_history.append({
                    'role': 'user' if m.role == 'user' else 'assistant',
                    'content': m.content[:300]
                })
        except Exception as e:
            print(f"DEBUG: Could not load history: {e}")
    
    if client or settings.NVIDIA_API_KEY:
        text_prompt = message_text
        if image_file:
            text_prompt = f"[User uploaded an image for editing] {message_text}"
        
        context_prompt = _get_context_prompt(conversation)
        
        # Priority: NVIDIA -> Gemini -> Mock
        if settings.NVIDIA_API_KEY:
            system_prompt = GEMINI_SYSTEM_PROMPT if intent != 'text_only' else TEXT_ONLY_SYSTEM_PROMPT
            response_text = _generate_nvidia_text(text_prompt, system_prompt, history=chat_history)

        if not response_text and client:
            response_text = _generate_gemini_response(text_prompt, intent, context_prompt, history=chat_history)
    
    if not response_text:
        response_text = _build_mock_response_text(intent, message_text, len(content_items))

    return {
        'intent': intent,
        'confidence': confidence or 1.0,
        'response_text': response_text,
        'content_items': content_items,
        'message_type': intent,
    }
