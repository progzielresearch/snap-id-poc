import os
from io import BytesIO
from typing import List, Optional
from uuid import uuid4

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from google import genai
from google.genai import types
from google.genai.errors import ClientError
from google.genai.types import GenerateContentResponse

from PIL import Image, ImageFilter, ImageDraw, ImageFont
import numpy as np
import cv2

from rembg import remove, new_session

import base64

import math

from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.lib.units import mm
import math

# --- Config ---
app = FastAPI(title="Frontalizer (Gemini 2.5 Flash Image)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://194.59.165.64:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
BG_REMOVAL_SESSION = new_session("u2net_human_seg")
API_KEY = "AIzaSyBQlSVvpHCfYiLiZ-0TFkjt_QzSg84D5ls"
MODEL_ID = "models/gemini-2.5-flash-image-preview"

# ---------- Utils ----------
def _parse_rgb(rgb: str) -> tuple[int, int, int]:
    """
    Parse a single RGB triplet string "R,G,B" into (r,g,b) ints.
    Raises 400 if invalid.
    """
    if not rgb:
        raise HTTPException(400, "background_rgb is required (format: 'R,G,B').")
    try:
        parts = [p.strip() for p in rgb.split(",")]
        if len(parts) != 3:
            raise ValueError
        r, g, b = map(int, parts)
        if not all(0 <= v <= 255 for v in (r, g, b)):
            raise ValueError
        return r, g, b
    except Exception:
        raise HTTPException(400, "background_rgb must be like '0,102,204' with values 0–255.")

# ---------- Prompt ----------
def build_prompt_rgb(r: int, g: int, b: int) -> str:
    return (
        "Task: From the provided 1–3 angled photos, synthesize ONE passport-style head-and-shoulders image.\n"
        "IDENTITY: Treat the FIRST input as the identity anchor. Preserve facial geometry and fine detail (eyes/iris color, hairline, moles/scars) within ±3% of the anchor. "
        "Do NOT borrow features from other photos or prior requests.\n"
        "Pose: STRICTLY FRONTAL - face must be directly facing the camera with NO rotation. "
        "Both ears must be visible and equidistant from image edges. "
        "Nose should be centered vertically in the frame. "
        "Head rotation: yaw=0° (no left/right turn), pitch=0° (no up/down tilt), roll=0° (no head tilt). "
        "Neutral expression; eyes looking directly at camera; mouth closed.\n"
        f"Background: exact solid color RGB({r},{g},{b}). Uniform fill only—no gradient, vignette, texture, or lighting shifts.\n"
        "Attire: if not formal, render plain professional attire appropriate to gender presentation (e.g., suit/blazer or collared blouse). Solid colors; no logos/patterns.\n"
        "No beautification/retouching (no skin smoothing/whitening or reshaping features).\n"
        "Framing/quality: perfectly centered head-and-shoulders with symmetrical composition, even studio lighting, no harsh shadows, no accessories, no text.\n"
        "The subject must face EXACTLY forward like a passport photo or ID card - absolutely no side angles or profile views.\n"
        "Output: one photorealistic PNG image."
    )

# ---------- Build request payload for Gemini ----------
async def _build_contents(prompt: str, files: List[UploadFile]) -> List:
    parts = [prompt]
    for f in files[:3]:
        data = await f.read()
        if data:
            parts.append(types.Part.from_bytes(data=data, mime_type=f.content_type or "image/jpeg"))
    return parts

# ---------- Extract image from Gemini response ----------
def _first_image_bytes(resp: GenerateContentResponse) -> Optional[bytes]:
    for cand in getattr(resp, "candidates", []) or []:
        content = getattr(cand, "content", None)
        if not content:
            continue
        for part in getattr(content, "parts", []) or []:
            inline = getattr(part, "inline_data", None)
            if inline and inline.data:
                return bytes(inline.data)
    return None

# ---------- Post-process: face-centered crop to requested canvas ----------
def _resize_cover(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Resize image to cover target dimensions (crop to fit)"""
    iw, ih = img.size
    scale = max(target_w / iw, target_h / ih)
    nw, nh = int(round(iw * scale)), int(round(ih * scale))
    img = img.resize((nw, nh), Image.LANCZOS)
    left, top = max(0, (nw - target_w) // 2), max(0, (nh - target_h) // 2)
    return img.crop((left, top, left + target_w, top + target_h))

def _face_centered_to_size(
    img: Image.Image,
    target_w: int,
    target_h: int,
    face_ratio: float = 0.52,   # face crown→chin ≈ 52% of canvas height
    vertical_bias: float = -0.06,  # move crop slightly up to keep more shoulders
    pad_factor: float = 1.10       # a bit of breathing room
) -> Image.Image:
    """
    Detect face and crop image to center it properly for passport photos.
    If no face is detected, falls back to center crop.
    """
    arr = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

    if len(faces) == 0:
        # Fallback to center crop if no face detected
        return _resize_cover(img, target_w, target_h)

    # Use largest detected face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    cx, cy = x + w / 2.0, y + h / 2.0

    aspect = target_w / target_h
    crop_h = (h / max(face_ratio, 1e-6)) * pad_factor
    crop_w = crop_h * aspect

    src_h, src_w = arr.shape[:2]
    # Ensure the crop fits within image bounds
    scale = min(src_w / crop_w, src_h / crop_h, 1.0)
    crop_w, crop_h = crop_w * scale, crop_h * scale

    x0 = int(round(cx - crop_w / 2.0))
    y0 = int(round(cy - crop_h / 2.0 + vertical_bias * crop_h))
    x0 = max(0, min(x0, src_w - int(round(crop_w))))
    y0 = max(0, min(y0, src_h - int(round(crop_h))))
    x1 = int(round(x0 + crop_w))
    y1 = int(round(y0 + crop_h))

    return img.crop((x0, y0, x1, y1)).resize((target_w, target_h), Image.LANCZOS)

# ---------- Background Replacement ----------
def replace_background(img: Image.Image, r: int, g: int, b: int) -> Image.Image:
    """
    Replace background with exact RGB color while preserving sharp edges.
    Uses pre-initialized session for better performance.
    """
    # Remove background using portrait-specific model
    img_no_bg = remove(img, session=BG_REMOVAL_SESSION)

    # Extract alpha channel
    alpha = img_no_bg.split()[3]

    # Minimal blur for anti-aliasing only (keeps edges sharp)
    alpha = alpha.filter(ImageFilter.GaussianBlur(radius=0.2))

    # Create solid color background
    bg = Image.new('RGB', img_no_bg.size, (r, g, b))
    img_rgb = img_no_bg.convert('RGB')

    # Composite subject onto background
    bg.paste(img_rgb, (0, 0), alpha)

    return bg


# ---------- Watermark (diagonal tiled outline "Snap ID") ----------
def add_watermark(
    img: Image.Image,
    bg_rgb: tuple[int, int, int] | None = None,
) -> Image.Image:
    base = img.convert("RGBA")
    W, H = base.size

    # Contrast-aware outline color (white on dark bg, dark gray on light bg)
    if bg_rgb is not None:
        r, g, b = bg_rgb
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        color_val = 30 if luminance > 160 else 255
    else:
        color_val = 255

    opacity = 75         # outline opacity
    angle = -30.0
    text = "Snap ID"

    # Font
    min_side = min(W, H)
    font_size = max(20, int(min_side * 0.08))
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

    # Big layer so rotation doesn't clip
    LW, LH = int(W * 1.6), int(H * 1.6)
    layer = Image.new("RGBA", (LW, LH), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    # Measure + spacing (keep your slightly tighter grid)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    step_x = max(1, int(text_w * 2.0))
    step_y = max(1, int(text_h * 2.0))

    # Outline-only: transparent fill + visible stroke
    stroke_px = max(2, int(font_size * 0.06))
    stroke_rgba = (color_val, color_val, color_val, opacity)

    for y in range(-step_y, LH + step_y, step_y):
        x_offset = 0 if ((y // step_y) % 2 == 0) else step_x // 2
        for x in range(-step_x, LW + step_x, step_x):
            draw.text(
                (x + x_offset, y),
                text,
                font=font,
                fill=(0, 0, 0, 0),          # transparent interior
                stroke_width=stroke_px,     # outline thickness
                stroke_fill=stroke_rgba,    # outline color/alpha
            )

    # Rotate and crop back to canvas
    rotated = layer.rotate(angle, resample=Image.BICUBIC, expand=True)
    RW, RH = rotated.size
    left = (RW - W) // 2
    top = (RH - H) // 2
    tiled = rotated.crop((left, top, left + W, top + H))

    out = Image.alpha_composite(base, tiled)
    return out.convert("RGB")


# ---------- PDF (diagonal tiled outline "Snap ID") ----------
L_SIZE = (89 * mm, 127 * mm)
def generate_passport_pdf(img: Image.Image, canvas_width: int, canvas_height: int, num_photos: int = 4) -> BytesIO:
    """
    Generate L-Size PDF with passport photos.
    L-Size: 89mm × 127mm (3.5" × 5") - International standard
    Universal spacing optimized for passport, visa, and driving license photos.

    Args:
        img: PIL Image object
        canvas_width: Width in pixels
        canvas_height: Height in pixels
        num_photos: Number of photos to generate (default: 4)
    """
    # Create PDF with L-Size page
    pdf_buffer = BytesIO()
    c = pdf_canvas.Canvas(pdf_buffer, pagesize=L_SIZE)
    page_width, page_height = L_SIZE

    # Calculate EXACT print dimensions at 300 DPI
    DPI = 300
    img_width_inches = canvas_width / DPI
    img_height_inches = canvas_height / DPI

    # Convert to points
    img_width_pt = img_width_inches * inch
    img_height_pt = img_height_inches * inch

    # Universal padding and spacing (in mm, converted to points)
    padding_mm = 5  # 5mm edge padding (universal standard)
    photo_spacing_mm = 3  # 3mm between photos (standard for cutting)

    padding_pt = padding_mm * mm
    photo_spacing_pt = photo_spacing_mm * mm

    # Calculate how many fit per row and column
    available_width = page_width - (2 * padding_pt)
    available_height = page_height - (2 * padding_pt)

    cols = int((available_width + photo_spacing_pt) / (img_width_pt + photo_spacing_pt))
    rows = int((available_height + photo_spacing_pt) / (img_height_pt + photo_spacing_pt))

    photos_per_page = cols * rows
    total_pages = math.ceil(num_photos / photos_per_page)

    # Use ImageReader
    img_reader = ImageReader(img)

    photo_count = 0

    for page in range(total_pages):
        if page > 0:
            c.showPage()  # Create new page

        photos_on_this_page = min(photos_per_page, num_photos - photo_count)

        # Draw photos from top-left, row by row
        for i in range(photos_on_this_page):
            row = i // cols
            col = i % cols

            x = padding_pt + col * (img_width_pt + photo_spacing_pt)
            y = page_height - padding_pt - (row + 1) * img_height_pt - row * photo_spacing_pt

            c.drawImage(
                img_reader,
                x, y,
                width=img_width_pt,
                height=img_height_pt,
                preserveAspectRatio=False,
                mask='auto'
            )

            photo_count += 1

    c.save()
    pdf_buffer.seek(0)
    return pdf_buffer


# ---------- Endpoint ----------
@app.post("/frontalize")
async def frontalize(
    background_rgb: str = Form(..., description="Exact background color as 'R,G,B' (e.g., '0,102,204')"),
    canvas_width: int  = Form(..., description="Final canvas width in px"),
    canvas_height: int = Form(..., description="Final canvas height in px"),
    files: List[UploadFile] = File(..., description="Upload 1–3 face images"),
):
    """
    Generate a passport-style frontal portrait from 1-3 angled photos.
    Returns a PNG image with exact background color and face-centered composition.
    """
    if not files:
        raise HTTPException(400, "Please upload 1–3 images.")
    if canvas_width <= 0 or canvas_height <= 0:
        raise HTTPException(400, "canvas_width and canvas_height must be positive integers.")
    r, g, b = _parse_rgb(background_rgb)
    prompt = build_prompt_rgb(r, g, b)
    client = genai.Client(api_key=API_KEY)
    try:
        contents = await _build_contents(prompt, files)
        resp = client.models.generate_content(
            model=MODEL_ID,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.0,
                top_p=0.1,
                top_k=32,
                seed=42
            ),
        )
    except ClientError as e:
        return JSONResponse(
            status_code=e.status_code,
            content=e.response_json or {"error": str(e)}
        )
    img_bytes = _first_image_bytes(resp)
    if not img_bytes:
        msg = getattr(resp, "text", "Model returned no image.")
        raise HTTPException(502, f"No image in response. {msg}")

    # Load generated image
    generated_img = Image.open(BytesIO(img_bytes)).convert("RGB")
    # Replace background with exact RGB color
    replaced_background_img = replace_background(generated_img, r, g, b)
    # Face-center and resize to requested canvas dimensions
    final_img = _face_centered_to_size(
        replaced_background_img, canvas_width, canvas_height)
    # WATERMARK
    water_marked_img = add_watermark(final_img, bg_rgb=(r, g, b))

    # Convert both images to base64
    final_out = BytesIO()
    final_img.save(final_out, format="PNG")
    final_out.seek(0)
    final_base64 = base64.b64encode(final_out.read()).decode('utf-8')

    watermark_out = BytesIO()
    water_marked_img.save(watermark_out, format="PNG")
    watermark_out.seek(0)
    watermark_base64 = base64.b64encode(watermark_out.read()).decode('utf-8')

    # Generate PDF with 4 images
    pdf_buffer = generate_passport_pdf(final_img, canvas_width, canvas_height)
    pdf_base64 = base64.b64encode(pdf_buffer.read()).decode('utf-8')

    # Return as JSON with both images and PDF
    return JSONResponse(
        content={
            "final_image": f"data:image/png;base64,{final_base64}",
            "watermarked_image": f"data:image/png;base64,{watermark_base64}",
            "pdf": f"data:application/pdf;base64,{pdf_base64}",
            "session_id": str(uuid4())
        },
        headers={
            "Cache-Control": "no-store"
        },
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": MODEL_ID,
        "bg_removal_model": "u2net_human_seg"
    }