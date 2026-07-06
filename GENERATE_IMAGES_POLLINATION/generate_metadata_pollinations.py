"""
Metadata + Thumbnail Generator
  • Loads story brief from video_package.json for character/setting consistency
  • Gemini primary → Groq fallback for metadata refinement
  • Brief injected into refine prompt and thumbnail render call
  • Pollinations renders the background illustration
  • LLM generates thumbnail text copy (main, sub, highlight word)
  • Pillow composites bold text overlay onto the rendered image
"""

import os
import json
import requests
import urllib.parse
from PIL import Image, ImageDraw, ImageFont

from google import genai
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

BASE_DIR=r"D:\Automated_YouTube_Pipeline"
OUTPUTS_DIR=os.path.join(BASE_DIR, "outputs")

AESTHETIC_ANCHOR=(
    "A professional premium vector graphic layout, high-contrast YouTube thumbnail poster design, "
    "smooth bold black clean outlines, vibrant flat color palette, solid light beige backdrop, "
    "perfect clean cartoon canvas composition, native 16:9 cinematic aspect ratio, NO text."
)

# font paths — Impact primary (classic thumbnail font), Arial Bold fallback
FONT_PRIMARY=r"C:\Windows\Fonts\impact.ttf"
FONT_FALLBACK=r"C:\Windows\Fonts\arialbd.ttf"

# thumbnail text layout constants
MAIN_FONT_MAX=90                   # starting size — auto-shrinks to fit
MAIN_FONT_MIN=40                   # never go below this
SUB_FONT_MAX=44
SUB_FONT_MIN=24
SAFE_MARGIN=80                     # horizontal safe zone each side in px
HIGHLIGHT_COLOR=(255, 220, 0)      # bold yellow for highlight word
MAIN_TEXT_COLOR=(255, 255, 255)    # white for remaining main text
SUB_TEXT_COLOR=(230, 230, 230)     # light grey for subtitle
SHADOW_COLOR=(0, 0, 0)            # black drop shadow
SHADOW_OFFSET=3                   # shadow offset in pixels
BOX_PADDING=14                    # padding around highlight word box
BOX_RADIUS=10                     # rounded corner radius for highlight box


def select_project_folder() -> str:
    if not os.path.exists(OUTPUTS_DIR):
        raise FileNotFoundError(f"Outputs directory not found at: {OUTPUTS_DIR}")

    folders=[d for d in os.listdir(OUTPUTS_DIR) if os.path.isdir(os.path.join(OUTPUTS_DIR, d))]
    if not folders:
        raise FileNotFoundError("No project workspace folders found inside outputs directory.")

    full_paths=[os.path.join(OUTPUTS_DIR, f) for f in folders]
    latest_folder=os.path.basename(max(full_paths, key=os.path.getmtime))

    print("\n--- Available Project Workspaces ---")
    for idx, folder in enumerate(folders, 1):
        suffix=" (Most Recent)" if folder==latest_folder else ""
        print(f"[{idx}] {folder}{suffix}")

    user_input=input(f"\nSelect a folder name or number [Press Enter for '{latest_folder}']: ").strip()

    if not user_input:
        return os.path.join(OUTPUTS_DIR, latest_folder)
    if user_input.isdigit():
        idx=int(user_input) - 1
        if 0 <= idx < len(folders):
            return os.path.join(OUTPUTS_DIR, folders[idx])
    if user_input in folders:
        return os.path.join(OUTPUTS_DIR, user_input)

    return os.path.join(OUTPUTS_DIR, latest_folder)


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in [FONT_PRIMARY, FONT_FALLBACK]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    # last resort — Pillow default bitmap font (no size control)
    return ImageFont.load_default()


def _build_brief_block(brief: dict) -> str:
    return (
        f"Main character : {brief.get('main_character', 'N/A')}\n"
        f"Setting        : {brief.get('setting', 'N/A')}\n"
        f"Tone           : {brief.get('tone', 'N/A')}\n"
        f"Key props      : {brief.get('key_props', 'N/A')}"
    )


def _build_refine_prompt(scenes: list, current_meta: dict, brief: dict) -> str:
    brief_block=_build_brief_block(brief)
    return f"""
You are a YouTube growth expert specializing in high-click-through-rate (CTR) animation channels like @Zenn0009.
Overhaul the primitive metadata for this video based on its actual script context and story brief.

STORY BRIEF (use this for character names, setting, and tone in titles and descriptions)
----------------------------------------------------------------------------------------
{brief_block}

ACTUAL VIDEO SCRIPT / SCENES CONTEXT:
{json.dumps(scenes, indent=2)}

CURRENT PRIMITIVE METADATA:
{json.dumps(current_meta, indent=2)}

YOUR INSTRUCTIONS:
1. SUGGESTED TITLES: Generate 3 high-CTR, curiosity-inducing titles. Use the character name and setting from the brief. Avoid clickbait clichés. Use intriguing psychological hooks or bizarre questions raised by the script.
2. SEO DESCRIPTION: Write a detailed, engaging 3-paragraph video description. Hook the viewer with a question from the script, reference the main character and setting from the brief, summarize the story's vibe without spoiling the ending, and include a natural subscription call-to-action.
3. TAGS: Provide an expanded list of 15-20 highly relevant, comma-separated tags. Include the character name and setting as tags.
4. THUMBNAIL CONCEPT: Invent a brilliant visual concept featuring the main character from the brief. Focus on a high-contrast, strange, or ironic visual hook from the script that creates an open loop in the viewer's mind. Describe the scene layout only — no text elements.

Return your response ONLY as a clean, parsable JSON object matching this schema:
{{
  "suggested_titles": ["Title 1", "Title 2", "Title 3"],
  "seo_description": "Detailed multi-paragraph text...",
  "tags": ["tag1", "tag2", "tag3"],
  "thumbnail_concept": "Detailed visual layout description..."
}}
"""


def _build_text_copy_prompt(thumb_concept: str, brief: dict, titles: list) -> str:
    brief_block=_build_brief_block(brief)
    return f"""
You are a YouTube thumbnail text copywriter for a minimalist stickman animation channel.

STORY BRIEF
-----------
{brief_block}

THUMBNAIL VISUAL CONCEPT
-------------------------
{thumb_concept}

SUGGESTED TITLES (pick the strongest hook — you may adapt wording slightly)
----------------------------------------------------------------------------
{chr(10).join(f"{i+1}. {t}" for i, t in enumerate(titles))}

YOUR TASK
---------
Write punchy thumbnail text that will overlay on the illustration above.
Rules:
- main_text: 2-5 words MAX, all caps, bold hook — the thing that stops the scroll.
- sub_text: 6-10 words, sentence case, adds intrigue or context below the main text.
- highlight_word: exactly ONE word from main_text that gets a bright yellow box behind it for emphasis.
- Match the tone from the brief — dramatic, tragic, ironic, shocking etc.
- Do NOT use generic phrases like "The Truth" or "You Won't Believe".

Return ONLY valid JSON, no markdown.

{{
  "main_text": "2-5 WORD HOOK IN CAPS",
  "sub_text": "Shorter supporting line that adds curiosity",
  "highlight_word": "ONEWORD"
}}
"""


def generate_text_copy(thumb_concept: str, brief: dict, titles: list, gemini_client, groq_client: Groq) -> dict:
    prompt=_build_text_copy_prompt(thumb_concept, brief, titles)
    print("[*] Generating thumbnail text copy …")

    try:
        response=gemini_client.models.generate_content(
            model=os.environ["GEMINI_MODEL"],
            contents=prompt,
            config={"response_mime_type": "application/json"},
        )
        result=json.loads(response.text.strip())
        print(f"[✓] Text copy — main: \"{result['main_text']}\" | sub: \"{result['sub_text']}\"")
        return result

    except Exception as e:
        print(f"[!] Gemini failed for text copy: {e}")
        print("[*] Falling back to Groq for text copy …")

    try:
        response=groq_client.chat.completions.create(
            model=os.environ["GROQ_MODEL"],
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.7,
        )
        result=json.loads(response.choices[0].message.content)
        print(f"[✓] Groq text copy — main: \"{result['main_text']}\" | sub: \"{result['sub_text']}\"")
        return result

    except Exception as e:
        print(f"[!] Both failed for text copy: {e} — using fallback text.")
        return {"main_text": "WATCH THIS", "sub_text": "The story you never knew", "highlight_word": "WATCH"}


def _draw_rounded_box(draw: ImageDraw.Draw, x1: int, y1: int, x2: int, y2: int, color: tuple, radius: int):
    draw.rounded_rectangle([x1, y1, x2, y2], radius=radius, fill=color)


def _draw_shadow_text(draw: ImageDraw.Draw, x: int, y: int, text: str, font, color: tuple):
    # draw black shadow offset then main text on top
    draw.text((x + SHADOW_OFFSET, y + SHADOW_OFFSET), text, font=font, fill=SHADOW_COLOR)
    draw.text((x, y), text, font=font, fill=color)


def _fit_font(draw: ImageDraw.Draw, text: str, max_size: int, min_size: int, max_width: int) -> ImageFont.FreeTypeFont:
    # shrink font size until text fits within max_width
    for size in range(max_size, min_size - 1, -2):
        font=_load_font(size)
        bbox=draw.textbbox((0, 0), text, font=font)
        if (bbox[2] - bbox[0]) <= max_width:
            return font
    return _load_font(min_size)


def composite_text_overlay(image_path: str, text_copy: dict) -> str:
    img=Image.open(image_path).convert("RGBA")
    draw=ImageDraw.Draw(img)
    W, H=img.size

    safe_w=W - (SAFE_MARGIN * 2)   # usable width after left+right margins

    main_text=text_copy.get("main_text", "").upper()
    sub_text=text_copy.get("sub_text", "")
    highlight_word=text_copy.get("highlight_word", "").upper()

    # auto-fit both fonts to safe width
    main_font=_fit_font(draw, main_text, MAIN_FONT_MAX, MAIN_FONT_MIN, safe_w)
    sub_font=_fit_font(draw, sub_text, SUB_FONT_MAX, SUB_FONT_MIN, safe_w)

    main_bbox=draw.textbbox((0, 0), main_text, font=main_font)
    main_w=main_bbox[2] - main_bbox[0]
    main_h=main_bbox[3] - main_bbox[1]

    # position: bottom quarter of image with enough room for subtitle below
    sub_bbox_measure=draw.textbbox((0, 0), sub_text, font=sub_font)
    sub_h=sub_bbox_measure[3] - sub_bbox_measure[1]
    total_block_h=main_h + 20 + sub_h + BOX_PADDING * 2
    text_y=H - total_block_h - 60   # 60px breathing room from bottom edge

    # centre main text horizontally
    text_x=(W - main_w) // 2

    # ── draw dark gradient bar behind entire text block for contrast ──
    bar_overlay=Image.new("RGBA", img.size, (0, 0, 0, 0))
    bar_draw=ImageDraw.Draw(bar_overlay)
    bar_draw.rounded_rectangle(
        [SAFE_MARGIN - 20, text_y - BOX_PADDING,
         W - SAFE_MARGIN + 20, H - 20],
        radius=14,
        fill=(0, 0, 0, 175),
    )
    img=Image.alpha_composite(img, bar_overlay)
    draw=ImageDraw.Draw(img)

    # ── draw main text word by word with highlight box ──
    words=main_text.split()
    cursor_x=text_x

    for word in words:
        word_bbox=draw.textbbox((0, 0), word, font=main_font)
        word_w=word_bbox[2] - word_bbox[0]
        word_h=word_bbox[3] - word_bbox[1]

        if word==highlight_word:
            box_x1=cursor_x - BOX_PADDING
            box_y1=text_y - BOX_PADDING
            box_x2=cursor_x + word_w + BOX_PADDING
            box_y2=text_y + word_h + BOX_PADDING
            _draw_rounded_box(draw, box_x1, box_y1, box_x2, box_y2, HIGHLIGHT_COLOR, BOX_RADIUS)
            # dark text on yellow box for max contrast
            draw.text((cursor_x + SHADOW_OFFSET, text_y + SHADOW_OFFSET), word, font=main_font, fill=SHADOW_COLOR)
            draw.text((cursor_x, text_y), word, font=main_font, fill=(15, 15, 15))
        else:
            _draw_shadow_text(draw, cursor_x, text_y, word, main_font, MAIN_TEXT_COLOR)

        space_bbox=draw.textbbox((0, 0), word + " ", font=main_font)
        cursor_x += space_bbox[2] - space_bbox[0]

    # ── subtitle centred below main text ──
    sub_bbox=draw.textbbox((0, 0), sub_text, font=sub_font)
    sub_w=sub_bbox[2] - sub_bbox[0]
    sub_y=text_y + main_h + 20
    sub_x=(W - sub_w) // 2
    _draw_shadow_text(draw, sub_x, sub_y, sub_text, sub_font, SUB_TEXT_COLOR)

    output_path=image_path.replace(".png", "_final.png")
    img.convert("RGB").save(output_path, "PNG")
    return output_path


def refine_metadata(scenes: list, current_meta: dict, brief: dict, gemini_client, groq_client: Groq) -> dict:
    prompt=_build_refine_prompt(scenes, current_meta, brief)

    try:
        print("[*] Trying Gemini for metadata refinement …")
        response=gemini_client.models.generate_content(
            model=os.environ["GEMINI_MODEL"],
            contents=prompt,
            config={"response_mime_type": "application/json"},
        )
        result=json.loads(response.text.strip())
        print("[✓] Gemini metadata refinement complete.")
        return result

    except Exception as e:
        print(f"[!] Gemini failed: {e}")
        print("[*] Falling back to Groq for metadata refinement …")

    try:
        response=groq_client.chat.completions.create(
            model=os.environ["GROQ_MODEL"],
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.7,
        )
        result=json.loads(response.choices[0].message.content)
        print("[✓] Groq fallback metadata refinement complete.")
        return result

    except Exception as e:
        print(f"[!] Groq also failed: {e}")
        print("[!] Using existing metadata as-is.")
        return current_meta


def _build_thumbnail_prompt(thumb_concept: str, brief: dict) -> str:
    return (
        f"{AESTHETIC_ANCHOR} "
        f"Main character: {brief.get('main_character', 'a stickman')}. "
        f"Setting: {brief.get('setting', 'unknown')}. "
        f"Tone: {brief.get('tone', 'dramatic')}. "
        f"Design Concept: {thumb_concept}"
    )


def main():
    try:
        project_path=select_project_folder()
    except FileNotFoundError as e:
        print(f"[X] Error: {e}")
        return

    json_path=os.path.join(project_path, "video_package.json")
    if not os.path.exists(json_path):
        print(f"[X] Error: 'video_package.json' not found at: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        package_data=json.load(f)

    scenes=package_data.get("scenes", [])
    current_meta=package_data.get("metadata", {})
    brief=package_data.get("brief", {})

    if not brief:
        print("[!] Warning: no brief found in video_package.json — character/setting context will be missing.")

    print(f"\n[*] Project   : {os.path.basename(project_path)}")
    print(f"[*] Character : {brief.get('main_character', 'N/A')}")
    print(f"[*] Setting   : {brief.get('setting', 'N/A')}")
    print(f"[*] Tone      : {brief.get('tone', 'N/A')}")
    print(f"[*] Scenes    : {len(scenes)}")

    gemini_client=genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    groq_client=Groq(api_key=os.environ["GROQ_API_KEY"])

    # 1. refine metadata
    print("\n[...] Refining metadata …")
    refined_meta=refine_metadata(scenes, current_meta, brief, gemini_client, groq_client)

    package_data["metadata"]=refined_meta
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(package_data, f, indent=2, ensure_ascii=False)

    # 2. setup output directory
    metadata_dir=os.path.join(project_path, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)

    titles=refined_meta.get("suggested_titles", ["Untitled Project"])
    description=refined_meta.get("seo_description", "No description generated.")
    tags=refined_meta.get("tags", [])
    thumb_concept=refined_meta.get("thumbnail_concept", "An interesting cartoon scene presentation.")

    # 3. write TXT file
    txt_file_path=os.path.join(metadata_dir, "youtube_metadata.txt")
    with open(txt_file_path, "w", encoding="utf-8") as txt_file:
        txt_file.write("=== SUGGESTED YOUTUBE TITLES ===\n")
        for i, t in enumerate(titles, 1):
            txt_file.write(f"{i}. {t}\n")
        txt_file.write("\n=== TARGET SEO TAGS ===\n")
        txt_file.write(", ".join(tags) + "\n")
        txt_file.write("\n=== VIDEO DESCRIPTION ===\n")
        txt_file.write(description + "\n")

    print("\n" + "="*66)
    print(f"[✓] METADATA COMPILED FOR: {os.path.basename(project_path)}")
    print(f"[*] Text file saved to: metadata/youtube_metadata.txt")
    print("="*66)

    # 4. render background illustration via Pollinations
    thumb_name="youtube_thumbnail.png"
    thumb_output_path=os.path.join(metadata_dir, thumb_name)
    final_payload=_build_thumbnail_prompt(thumb_concept, brief)
    encoded_payload=urllib.parse.quote(final_payload)
    api_url=f"https://image.pollinations.ai/p/{encoded_payload}?width=1920&height=1080&model=flux&nologo=true"

    print(f"[...] Rendering thumbnail background via Pollinations …")
    try:
        response=requests.get(api_url, timeout=50)
        response.raise_for_status()
        with open(thumb_output_path, "wb") as fh:
            fh.write(response.content)
        print(f"[✓] Background saved → metadata/{thumb_name}")
    except Exception as e:
        print(f"[X] Thumbnail request failed: {e}")
        return

    # 5. generate text copy via LLM
    text_copy=generate_text_copy(thumb_concept, brief, titles, gemini_client, groq_client)

    # 6. composite text overlay onto background with Pillow
    print(f"[...] Compositing text overlay …")
    try:
        final_path=composite_text_overlay(thumb_output_path, text_copy)
        print(f"[✓] Final thumbnail saved → metadata/{os.path.basename(final_path)}")
    except Exception as e:
        print(f"[X] Text overlay failed: {e} — background image still saved.")

    print("="*66)


if __name__ == "__main__":
    main()