import os
import json
import base64
import requests
import time
from PIL import Image, ImageDraw, ImageFont
from tavily import TavilyClient

from google import genai
from groq import Groq
from dotenv import load_dotenv

from pipeline_config import CHARACTER_DESIGN, AESTHETIC_ANCHOR as STYLE_ANCHOR, RECOMMENDED_IMAGE_MODEL

load_dotenv()

BASE_DIR=r"D:\Automated_YouTube_Pipeline"
OUTPUTS_DIR=os.path.join(BASE_DIR, "outputs")

IMAGE_MODEL = os.environ.get("IMAGE_MODEL_TARGET", RECOMMENDED_IMAGE_MODEL)
ENHANCE_COUNT = int(os.environ.get("IMAGE_ENHANCE_COUNT", 20))
MAX_PROMPT_CHARS=2048  # hard limit enforced by the model's input schema
IMAGE_WIDTH=1920        # YouTube 16:9 widescreen
IMAGE_HEIGHT=1080       # YouTube 16:9 widescreen

# Character + base art style are locked in pipeline_config.py and shared with
# generate_story.py / generate_images.py — this script used to define its own
# separate "cinematic vector-art / graphic-novel" style here, which is why the
# thumbnail didn't visually match the scene renders. Now it can't drift.
AESTHETIC_ANCHOR = f"{CHARACTER_DESIGN} {STYLE_ANCHOR}"

# font paths — Impact primary (classic thumbnail font), Arial Bold fallback
FONT_PRIMARY=r"C:\Windows\Fonts\impact.ttf"
FONT_FALLBACK=r"C:\Windows\Fonts\arialbd.ttf"

# thumbnail text layout constants
MAIN_FONT_MAX=90                   # starting size — auto-shrinks to fit
MAIN_FONT_MIN=40                   # never go below this
SUB_FONT_MAX=44
SUB_FONT_MIN=24
SAFE_MARGIN=80                     # horizontal safe zone each side in px
MAIN_TEXT_COLOR=(255, 220, 0)      # yellow for all main text
SUB_TEXT_COLOR=(255, 255, 255)     # white for subtitle
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
    return ImageFont.load_default()


def _build_brief_block(brief: dict) -> str:
    # Fixed: this used to append brief['color_mood'] a second time right after
    # itself with no separator ("Color mood : X" immediately followed by "X"),
    # producing a mashed-together duplicate on the same line.
    return (
        f"Character's role : {brief.get('character_role', 'an ancient human')}\n"
        f"Setting        : {brief.get('setting', 'N/A')}\n"
        f"Tone           : {brief.get('tone', 'N/A')}\n"
        f"Key props      : {brief.get('key_props', 'N/A')}\n"
        f"Color mood     : {brief.get('color_mood', 'N/A')}"
    )


def research_topic_with_tavily(brief: dict) -> str:
    """Uses Tavily to pull primary academic sources or historical references based on the video setting/topic."""
    tavily_key = os.environ.get("TAVILY_API_KEY")
    if not tavily_key:
        print("[!] No TAVILY_API_KEY found, skipping live web research.")
        return ""

    topic = f"{brief.get('setting', '')} {brief.get('key_props', '')} history archaeology research papers"
    print(f"[*] Researching topic on the web: '{topic}' via Tavily...")

    try:
        tavily = TavilyClient(api_key=tavily_key)
        response = tavily.search(query=topic, max_results=3, include_raw_content=False)

        research_summary = []
        for res in response.get('results', []):
            research_summary.append(f"Source Title: {res.get('title')}\nURL: {res.get('url')}\nSnippet: {res.get('content')}\n---")
        return "\n".join(research_summary)
    except Exception as e:
        print(f"[!] Tavily research failed: {e}")
        return ""


def _build_refine_prompt(scenes: list, current_meta: dict, brief: dict, research_data: str = "") -> str:
    brief_block = _build_brief_block(brief)

    research_block = ""
    if research_data:
        research_block = f"\nREAL-TIME WEB RESEARCH & CITATIONS FOUND:\n{research_data}\n"

    return f"""
You are an elite YouTube growth expert and scriptwriter specializing in high-click-through-rate (CTR) deep-history animation channels like @Zenn0009.
Provide raw, cinematic, visceral copy pieces for this video. Do not write summary blocks or generic marketing.

STORY BRIEF:
----------------------------------------------------------------------------------------
{brief_block}

ACTUAL VIDEO SCRIPT / SCENES CONTEXT:
{json.dumps(scenes, indent=2)}
{research_block}

CRITICAL RULES:
- Absolutely NO hype or narrator talk ("Imagine a world", "Join us on a journey", "In this video we discover").
- Focus purely on deep time, environmental brutality, and biological evolution.
- Keep every text line concise, stark, and punchy. No multi-sentence fields.

Return your response ONLY as a clean, parsable JSON object matching this schema. Fill every single field with custom story strings matching the script context:
{{
  "suggested_titles": ["Title 1", "Title 2", "Title 3"],
  "line_1_deep_time": "A massive, factual statement about deep time establishing historical weight",
  "line_2_visceral": "A jarring, brutal detail of prehistoric environmental reality",
  "line_3_event": "One stark sentence stating the specific narrative event or character struggle from the script",
  "line_4_conflict": "One stark sentence framing the evolutionary conflict or curse",
  "line_5_mechanics": "The mechanical, physical reality of what their bodies or environment faced",
  "line_6_detail": "Another cold, hard survival or anatomical detail",
  "line_7_connection": "Connect this ancient script element directly to modern human biology, brain evolution, or psychology",
  "why_bullets": [
    "Why specific event happened or mattered",
    "Why specific trap was a death sentence",
    "Why specific discovery changed human history forever"
  ],
  "final_punchline": "A definitive, sharp closing summary disavowing generic tropes",
  "tags": ["tag1", "tag2", "tag3"],
  "thumbnail_concept": "Detailed visual layout description..."
}}
"""


def _build_text_copy_prompt(thumb_concept: str, brief: dict, titles: list) -> str:
    brief_block=_build_brief_block(brief)
    return f"""
You are a YouTube thumbnail text copywriter for a minimalist, blank-round-head cartoon character explainer channel.

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
    draw.text((x + SHADOW_OFFSET, y + SHADOW_OFFSET), text, font=font, fill=SHADOW_COLOR)
    draw.text((x, y), text, font=font, fill=color)


def _fit_font(draw: ImageDraw.Draw, text: str, max_size: int, min_size: int, max_width: int) -> ImageFont.FreeTypeFont:
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

    safe_w=W - (SAFE_MARGIN * 2)

    main_text=text_copy.get("main_text", "").upper()
    sub_text=text_copy.get("sub_text", "")
    highlight_word=text_copy.get("highlight_word", "").upper()

    main_font=_fit_font(draw, main_text, MAIN_FONT_MAX, MAIN_FONT_MIN, safe_w)
    sub_font=_fit_font(draw, sub_text, SUB_FONT_MAX, SUB_FONT_MIN, safe_w)

    main_bbox=draw.textbbox((0, 0), main_text, font=main_font)
    main_w=main_bbox[2] - main_bbox[0]
    main_h=main_bbox[3] - main_bbox[1]

    sub_bbox_measure=draw.textbbox((0, 0), sub_text, font=sub_font)
    sub_h=sub_bbox_measure[3] - sub_bbox_measure[1]
    total_block_h=main_h + 20 + sub_h + BOX_PADDING * 2
    text_y=H - total_block_h - 60

    text_x=(W - main_w) // 2

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

    words=main_text.split()
    cursor_x=text_x

    for word in words:
        _draw_shadow_text(draw, cursor_x, text_y, word, main_font, MAIN_TEXT_COLOR)
        space_bbox=draw.textbbox((0, 0), word + " ", font=main_font)
        cursor_x += space_bbox[2] - space_bbox[0]

    sub_bbox=draw.textbbox((0, 0), sub_text, font=sub_font)
    sub_w=sub_bbox[2] - sub_bbox[0]
    sub_y=text_y + main_h + 20
    sub_x=(W - sub_w) // 2
    _draw_shadow_text(draw, sub_x, sub_y, sub_text, sub_font, SUB_TEXT_COLOR)

    output_path=image_path.replace(".png", "_final.png")
    img.convert("RGB").save(output_path, "PNG")
    return output_path


def refine_metadata(scenes: list, current_meta: dict, brief: dict, gemini_client, groq_client: Groq, research_data: str = "") -> dict:
    prompt=_build_refine_prompt(scenes, current_meta, brief, research_data)

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
    # NOTE: no "Avoid: {NEGATIVE_BAN}" — lucid-origin has no negative_prompt
    # parameter, so that text was being read as positive prompt content
    # instead of being suppressed. Same fix as generate_images.py.
    full_prompt=(
        f"{AESTHETIC_ANCHOR} | "
        f"Character's role: {brief.get('character_role', 'an ancient human')}. "
        f"Setting: {brief.get('setting', 'unknown')}. "
        f"Tone: {brief.get('tone', 'dramatic')}. "
        f"Design Concept: {thumb_concept}"
    )

    if len(full_prompt) > MAX_PROMPT_CHARS:
        print(f"[!] Thumbnail prompt too long ({len(full_prompt)} chars) — truncating to {MAX_PROMPT_CHARS}.")
        full_prompt=full_prompt[:MAX_PROMPT_CHARS]
    return full_prompt


def get_cloudflare_credentials():
    """Scans environment variables for pairs of API keys and Account IDs."""
    credentials = []
    i = 1
    while True:
        api_key = os.environ.get(f"CLOUDFLARE_API_KEY_{i}")
        acc_id = os.environ.get(f"CLOUDFLARE_ACC_ID_{i}")

        if not api_key or not acc_id:
            break

        credentials.append({"api_key": api_key, "acc_id": acc_id})
        i += 1

    if not credentials:
        legacy_key = os.environ.get("CLOUDFLARE_API_KEY")
        legacy_id = os.environ.get("CLOUDFLARE_ACC_ID")
        if legacy_key and legacy_id:
            credentials.append({"api_key": legacy_key, "acc_id": legacy_id})

    return credentials

CREDENTIALS_POOL = get_cloudflare_credentials()


def _extract_image_bytes(response: requests.Response) -> bytes:
    """Same fix as generate_images.py — Cloudflare's REST endpoint returns
    JSON with a base64 'image' field, not raw bytes. See that file's version
    of this function for the full explanation."""
    content_type=response.headers.get("Content-Type", "")

    if "application/json" in content_type or response.content.lstrip()[:1] in (b"{", b"["):
        try:
            data=response.json()
        except ValueError as e:
            raise ValueError(f"response labeled JSON but didn't parse: {e}")

        b64=None
        if isinstance(data, dict):
            b64=(data.get("result") or {}).get("image") if isinstance(data.get("result"), dict) else None
            b64=b64 or data.get("image")
        if not b64:
            raise ValueError(f"no 'image' field found in JSON response: {str(data)[:200]}")

        return base64.b64decode(b64)

    return response.content


def generate_image_cloudflare(prompt: str, output_path: str, max_retries: int = 5) -> bool:
    if not CREDENTIALS_POOL:
        print("[X] No Cloudflare credentials found in .env (Check CLOUDFLARE_API_KEY_1 / CLOUDFLARE_ACC_ID_1 etc.)")
        return False

    payload = {
        "prompt": prompt,
        "width": IMAGE_WIDTH,
        "height": IMAGE_HEIGHT,
        "num_steps": ENHANCE_COUNT
    }

    for idx, creds in enumerate(list(CREDENTIALS_POOL), 1):
        api_key = creds["api_key"]
        acc_id = creds["acc_id"]

        image_url = f"https://api.cloudflare.com/client/v4/accounts/{acc_id}/ai/run/{IMAGE_MODEL}"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        print(f"[*] [Account {idx}] Attempting generation with API Key ending in ...{api_key[-4:]}")
        wait_time = 4.0
        account_failed = False
        account_exhausted = False

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    image_url,
                    headers=headers,
                    json=payload,
                    timeout=60,
                )

                ratelimit_header = response.headers.get("Ratelimit")
                if ratelimit_header and response.ok:
                    print(f"[*] [Account {idx}] Quota status: {ratelimit_header}")

                if response.status_code == 429:
                    print(f"[!] Account {idx} is fully exhausted (Daily Free 10k Limit Reached).")
                    account_failed = True
                    account_exhausted = True
                    break

                if response.status_code in (401, 403):
                    print(f"[X] Account {idx} Auth error ({response.status_code}). Moving to next account.")
                    account_failed = True
                    break

                if response.status_code >= 500:
                    print(f"[!] Account {idx} server error ({response.status_code}). Retrying in {wait_time}s…")
                    time.sleep(wait_time)
                    wait_time *= 2
                    continue

                if not response.ok:
                    print(f"[X] Account {idx} error ({response.status_code}): {response.text[:200]}")
                    account_failed = True
                    break

                try:
                    image_bytes=_extract_image_bytes(response)
                except ValueError as e:
                    print(f"[X] Account {idx} returned a response we couldn't decode as an image: {e}")
                    account_failed = True
                    break

                with open(output_path, "wb") as f:
                    f.write(image_bytes)
                print(f"[✓] Image successfully generated using Account {idx}!")
                return True

            except requests.exceptions.RequestException as e:
                print(f"[!] Network error on Account {idx} (attempt {attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    account_failed = True
                time.sleep(3)

        if account_exhausted:
            try:
                CREDENTIALS_POOL.remove(creds)
                print(f"[*] Removed Account {idx} from CREDENTIALS_POOL for the rest of this run.")
            except ValueError:
                pass

        if account_failed:
            print(f"[!] Account {idx} failed or exhausted daily limits. Dropping account and checking failover...")
            continue

    print("[X] Absolute Failure: All accounts in the pool were exhausted or failed.")
    return False


def run(project_path: str):
    """
    Core metadata + thumbnail stage, reusable from the master pipeline
    (not wired into master_pipeline.py by default — call it explicitly
    if you want the refined SEO description + thumbnail for a project).
    """
    json_path = os.path.join(project_path, "video_package.json")
    if not os.path.exists(json_path):
        print(f"[X] Error: 'video_package.json' not found at: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        package_data = json.load(f)

    scenes = package_data.get("scenes", [])
    current_meta = package_data.get("metadata", {})
    brief = package_data.get("brief", {})

    if not brief:
        print("[!] Warning: no brief found in video_package.json — setting/prop context will be missing "
              "(character design still applies from pipeline_config.py).")

    print(f"\n[*] Project   : {os.path.basename(project_path)}")
    print(f"[*] Setting   : {brief.get('setting', 'N/A')}")
    print(f"[*] Tone      : {brief.get('tone', 'N/A')}")
    print(f"[*] Scenes    : {len(scenes)}")
    print(f"[*] Renderer  : Cloudflare Workers AI ({IMAGE_MODEL}, {IMAGE_WIDTH}x{IMAGE_HEIGHT})")

    gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

    research_data = ""
    if brief:
        research_data = research_topic_with_tavily(brief)

    print("\n[...] Refining metadata …")
    refined_meta = refine_metadata(
        scenes, current_meta, brief, gemini_client, groq_client, research_data
    )

    metadata_dir = os.path.join(project_path, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)

    titles = refined_meta.get("suggested_titles", ["Untitled Project"])
    tags = refined_meta.get("tags", [])
    thumb_concept = refined_meta.get("thumbnail_concept", "An interesting cartoon scene presentation.")

    # Watch-next block is defined once here in Python to avoid LLM formatting bugs
    # (no longer duplicated as a dead module-level constant — it was defined twice
    # before, and only this local copy was ever actually used).
    CHANNELS_WATCH_NEXT_BLOCK = (
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "🛑 WATCH NEXT\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "▸ What Did Ancient Humans Do at Night? \n"
        "👉 https://youtu.be/YourVideoIDHere\n\n"
        "▸ SIMILAR VIDEOS: Ancient Humans Theories (Playlist)\n"
        "👉 https://youtube.com/playlist?list=PLLjWGWKfcp5s&si=FV5nY_kX6yh2A7J1\n"
    )

    l1 = refined_meta.get("line_1_deep_time", "For millennia, humanity clawed survival from an unforgiving planet.")
    l2 = refined_meta.get("line_2_visceral", "A single, relentless freeze could erase generations.")
    l3 = refined_meta.get("line_3_event", "Endless cold trapped them inside the dark.")
    l4 = refined_meta.get("line_4_conflict", "Every step through the wild was a desperate gamble against predators.")
    l5 = refined_meta.get("line_5_mechanics", "Raw meat fueled nothing; their bodies were burning energy to survive.")
    l6 = refined_meta.get("line_6_detail", "Anatomical limitations kept their brains locked behind metabolic walls.")
    l7 = refined_meta.get("line_7_connection", "Our primal code for resilience ignites when all seems lost.")

    desc_parts = [
        l1, l2, "",
        l3, l4, "",
        l5, l6, "",
        l7, ""
    ]

    why_bullets = refined_meta.get("why_bullets", [])
    if len(why_bullets) >= 3:
        b1 = why_bullets[0] if why_bullets[0].lower().startswith("why") else f"Why {why_bullets[0]}"
        b2 = why_bullets[1] if why_bullets[1].lower().startswith("why") else f"Why {why_bullets[1]}"
        b3 = why_bullets[2] if why_bullets[2].lower().startswith("why") else f"Why {why_bullets[2]}"
        desc_parts.extend([b1, b2, b3])
    else:
        desc_parts.extend([
            "Why survival demanded adaptation.",
            "Why freezing meat meant extinction.",
            "Why controlled fire changed everything."
        ])

    desc_parts.append("")
    desc_parts.append(refined_meta.get("final_punchline", "This isn't a fairy tale; this is how our species survived."))
    desc_parts.append("\n" + CHANNELS_WATCH_NEXT_BLOCK)

    desc_parts.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nSOURCES & FURTHER READING\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    if research_data:
        desc_parts.append("Historical/Archaeological context derived from Wonderwerk Cave and Gesher Benot Ya’aqov records.")
    else:
        desc_parts.append("Historical/Archaeological context derived from Wonderwerk Cave and Gesher Benot Ya’aqov records.")

    if tags:
        cleaned_hashtags = [f"#{t.replace(' ', '').lower()}" for t in tags[:8]]
        desc_parts.append("\n" + "  ".join(cleaned_hashtags))
    else:
        desc_parts.append("\n#deephistory  #ancienthistory  #prehistoric  #ancienthumans  #animatedhistory")

    description = "\n".join(desc_parts)

    refined_meta["seo_description"] = description
    package_data["metadata"] = refined_meta
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(package_data, f, indent=2, ensure_ascii=False)

    txt_file_path = os.path.join(metadata_dir, "youtube_metadata.txt")
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

    thumb_name = "youtube_thumbnail.png"
    thumb_output_path = os.path.join(metadata_dir, thumb_name)
    final_payload = _build_thumbnail_prompt(thumb_concept, brief)

    if os.path.exists(thumb_output_path):
        print(f"[*] Existing background found → metadata/{thumb_name}, skipping generation.")
    else:
        print(f"[...] Rendering thumbnail background via Cloudflare Workers AI …")
        if not generate_image_cloudflare(final_payload, thumb_output_path):
            print(f"[X] Thumbnail request failed.")
            return
        print(f"[✓] Background saved → metadata/{thumb_name}")

    text_copy = generate_text_copy(thumb_concept, brief, titles, gemini_client, groq_client)

    print(f"[...] Compositing text overlay …")
    try:
        final_path = composite_text_overlay(thumb_output_path, text_copy)
        print(f"[✓] Final thumbnail saved → metadata/{os.path.basename(final_path)}")
    except Exception as e:
        print(f"[X] Text overlay failed: {e} — background image still saved.")

    print("="*66)


def main():
    try:
        project_path=select_project_folder()
    except FileNotFoundError as e:
        print(f"[X] Error: {e}")
        return
    run(project_path)


if __name__ == "__main__":
    main()