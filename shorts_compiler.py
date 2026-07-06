import os
import json
import re
import time
import base64
import requests
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips

from google import genai
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = r"D:\Automated_YouTube_Pipeline"
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

SHORT_W, SHORT_H = 1080, 1920          # 9:16 vertical, 1080p
SHORT_MAX_DURATION = 60.0              # seconds
IMAGE_MODEL = os.environ.get("IMAGE_MODEL_TARGET", "@cf/leonardo/phoenix-1.0")
ENHANCE_COUNT = int(os.environ.get("IMAGE_ENHANCE_COUNT", 20))
MAX_PROMPT_CHARS = 2048  # hard limit enforced by the model's input schema

AESTHETIC_ANCHOR = (
    "Minimalist 2D doodle stickman illustration. "
    "Every human is drawn as a tiny round head attached to extremely thin stick limbs like a stick. "
    "No full human anatomy, no rounded body, no realistic proportions. "
    "Body consists only of a simple white circular head and thin black stick arms and legs. "
    "Simple oval or rectangular torso with flat white fill and bold black outline. "
    "Very expressive body poses with simple gestures. "
    "Tiny dot eyes only. Small straight mouth. "
    "No nose. No eyelashes. No blush. No cheeks. "
    "Hair is drawn as simple cartoon doodle shapes. "
    "Objects are simple flat doodles with bold black outlines. "
    "Backgrounds are simple colorful cartoon landscapes with minimal detail. "
    "Flat colors only. Clean vector-like digital line art. "
    "Inspired by animated explainer doodles and simple YouTube story animations. "
    "Consistent character proportions across every scene. "
    "Vertical 9:16 portrait framing — compose the full body and key action "
    "centered within the tall frame so nothing important is cropped at the edges."
)

NEGATIVE_BAN = (
    "NO realistic humans, "
    "NO cartoon human anatomy, "
    "NO rounded bodies, "
    "NO fat characters, "
    "NO muscular characters, "
    "NO realistic limbs, "
    "NO detailed fingers, "
    "NO Disney style, "
    "NO Pixar style, "
    "NO anime style, "
    "NO children's book illustration, "
    "NO painterly style, "
    "NO airbrush, "
    "NO gradients, "
    "NO realistic lighting, "
    "NO textured shading, "
    "NO photorealism, "
    "NO wide horizontal landscape composition with tiny characters, "
    "NO cropped-off heads or limbs at the frame edge."
)


def get_latest_project_folder() -> str | None:
    folders = [
        os.path.join(OUTPUTS_DIR, d)
        for d in os.listdir(OUTPUTS_DIR)
        if os.path.isdir(os.path.join(OUTPUTS_DIR, d))
    ]
    return max(folders, key=os.path.getmtime) if folders else None


def slugify(text: str) -> str:
    sanitized = re.sub(r"[^\w\s]", "", text)
    sanitized = re.sub(r"\s+", "_", sanitized.strip())
    return sanitized or "short"


def _build_brief_anchor(brief: dict) -> str:
    if not brief:
        return ""
    return (
        f"Main character: {brief.get('main_character', 'N/A')}. "
        f"Setting: {brief.get('setting', 'N/A')}. "
        f"Recurring props: {brief.get('key_props', 'N/A')}. "
        f"Mood/tone: {brief.get('tone', 'N/A')}."
    )


def select_intro_scenes(scenes: list[dict], max_duration: float = SHORT_MAX_DURATION) -> tuple[list[dict], float]:
    """
    Walks scenes in story order starting from sequence 1, accumulating
    duration until max_duration is reached. The final included scene is
    trimmed (its short_duration only — not its narrative/prompt) so the
    Short lands as close to max_duration as possible.
    """
    selected = []
    elapsed = 0.0

    for scene in scenes:
        if elapsed >= max_duration:
            break

        full_duration = round(scene["end_time"] - scene["start_time"], 2)
        if full_duration <= 0:
            continue

        short_duration = min(full_duration, max_duration - elapsed)
        selected.append({**scene, "short_duration": short_duration})
        elapsed += short_duration

    return selected, elapsed


def _build_enhance_prompt_short(selected_scenes: list[dict], story_concept: str,
                                 story_tone: str, brief: dict) -> str:
    lines = [f"  Scene {s['sequence']}: {s['visual_prompt']}" for s in selected_scenes]
    scenes_block = "\n".join(lines)
    sequence_ids = [s["sequence"] for s in selected_scenes]
    brief_anchor = _build_brief_anchor(brief)

    return f"""
You are a professional AI image prompt engineer specialising in minimalist stickman doodle animation for YouTube Shorts.

STORY CONTEXT
-------------
Concept : {story_concept}
Tone    : {story_tone}

STORY BRIEF (use this to keep character, setting, and props consistent)
-------------------------------------------------------------------------
{brief_anchor}

VISUAL STYLE (must be obeyed in every enhanced prompt)
-------------------------------------------------------
{AESTHETIC_ANCHOR}
{NEGATIVE_BAN}

YOUR TASK
---------
These are the OPENING scenes of the story, in order, being used to build a vertical
YouTube Short. Rewrite each scene below into a richer, more descriptive image
generation prompt.

Rules:
1. Keep the same narrative action as the original — do NOT invent new story events.
2. This will render in a vertical 9:16 (1080x1920) portrait frame — compose each
   scene as a tall portrait shot (full body or medium shot, centered), not a wide
   landscape composition. Nothing important should end up near the left/right edges.
3. Add specific visual detail: character pose, simple body language and gesture,
   composition framing, background elements. Do NOT describe skin tone, detailed
   facial features, muscle definition, or shading on the character's body — bodies
   stay plain white/flat per the style.
4. Reference the main character, setting, and props from the STORY BRIEF where relevant.
5. Dress every character in clothing, hairstyles, and gear appropriate to the time
   period and setting described in the STORY BRIEF — never modern clothing unless
   the STORY BRIEF is explicitly set in modern times.
6. If the scene includes animals, describe them in the same flat white/light body,
   bold black outline, minimal-feature doodle style — not realistic animals.
7. Keep each enhanced prompt under 80 words.
8. Return ONLY a JSON object with an "enhanced_prompts" array in the same order as
   the scenes below. No markdown, no preamble.

[SCENES]
{scenes_block}

Expected output format:
{{
  "enhanced_prompts": [
    "Enhanced prompt for scene {sequence_ids[0]}...",
    "Enhanced prompt for scene {sequence_ids[1] if len(sequence_ids) > 1 else sequence_ids[0]}..."
  ]
}}
"""


def _parse_json_block(raw: str) -> dict:
    clean = re.sub(r"^```json|^```|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    return json.loads(clean)


def _parse_enhanced(raw: str, selected_scenes: list[dict]) -> list[str]:
    result = _parse_json_block(raw)
    enhanced = result.get("enhanced_prompts", [])

    for i in range(len(enhanced), len(selected_scenes)):
        enhanced.append(selected_scenes[i]["visual_prompt"])
        print(f"[!] Provider returned fewer prompts than expected — using original for scene {selected_scenes[i]['sequence']}.")

    return enhanced


def enhance_intro_scenes(selected_scenes: list[dict], story_concept: str, story_tone: str,
                          brief: dict, gemini_client, groq_client: Groq) -> list[str]:
    prompt = _build_enhance_prompt_short(selected_scenes, story_concept, story_tone, brief)

    try:
        response = gemini_client.models.generate_content(
            model=os.environ["GEMINI_MODEL"],
            contents=prompt,
            config={"response_mime_type": "application/json"},
        )
        return _parse_enhanced(response.text, selected_scenes)

    except Exception as e:
        print(f"[!] Gemini failed for Short prompt enhancement: {e}")
        print(f"[*] Falling back to Groq …")

    try:
        response = groq_client.chat.completions.create(
            model=os.environ["GROQ_MODEL"],
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.7,
        )
        return _parse_enhanced(response.choices[0].message.content, selected_scenes)

    except Exception as e:
        print(f"[!] Groq also failed for Short prompt enhancement: {e}")
        print(f"[!] Using original visual_prompts.")
        return [s["visual_prompt"] for s in selected_scenes]


def _build_render_prompt(enhanced_prompt: str, brief: dict) -> str:
    brief_anchor = _build_brief_anchor(brief)
    full_prompt = (
        f"{AESTHETIC_ANCHOR} | "
        f"{brief_anchor} | "
        f"{NEGATIVE_BAN} | "
        f"Current Scene: {enhanced_prompt}"
    )
    if len(full_prompt) > MAX_PROMPT_CHARS:
        print(f"[!] Prompt too long ({len(full_prompt)} chars) — truncating to {MAX_PROMPT_CHARS}.")
        full_prompt = full_prompt[:MAX_PROMPT_CHARS]
    return full_prompt


def get_cloudflare_credentials():
    """Scans environment variables for pairs of API keys and Account IDs."""
    credentials = []
    i = 1
    while True:
        api_key = os.environ.get(f"CLOUDFLARE_API_KEY_{i}")
        acc_id = os.environ.get(f"CLOUDFLARE_ACC_ID_{i}")
        
        # Stop looking once we don't find the next numbered pair
        if not api_key or not acc_id:
            break
            
        credentials.append({"api_key": api_key, "acc_id": acc_id})
        i += 1
        
    # Fallback to the original single key variables if the numbered ones aren't used
    if not credentials:
        legacy_key = os.environ.get("CLOUDFLARE_API_KEY")
        legacy_id = os.environ.get("CLOUDFLARE_ACC_ID")
        if legacy_key and legacy_id:
            credentials.append({"api_key": legacy_key, "acc_id": legacy_id})
            
    return credentials

# Load the pool at script startup
CREDENTIALS_POOL = get_cloudflare_credentials()

def generate_image_cloudflare(enhanced_prompt: str, brief: dict, output_path: str, max_retries: int = 5) -> bool:
    if not CREDENTIALS_POOL:
        print("[X] No Cloudflare credentials found in .env (Check CLOUDFLARE_API_KEY_1 / CLOUDFLARE_ACC_ID_1 etc.)")
        return False

    # 1. Process prompt structure once outside the multi-account loop
    final_prompt = _build_render_prompt(enhanced_prompt, brief)

    # 2. Build the JSON payload data block required by the Leonardo Engine
    payload = {
        "prompt": final_prompt,
        "width": SHORT_W,
        "height": SHORT_H,
        "num_steps": ENHANCE_COUNT
    }

    # 3. Iterate sequentially through each available account in your pool
    for idx, creds in enumerate(list(CREDENTIALS_POOL), 1):
        api_key = creds["api_key"]
        acc_id = creds["acc_id"]

        # Dynamically build the endpoint URL matching the active fallback account context
        image_url = f"https://api.cloudflare.com/client/v4/accounts/{acc_id}/ai/run/{IMAGE_MODEL}"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        print(f"[*] [Account {idx}] Attempting scene render via Leonardo with key ending in ...{api_key[-4:]}")
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

                # Output daily neuron tracking metrics to terminal logs if request was valid
                ratelimit_header = response.headers.get("Ratelimit")
                if ratelimit_header and response.ok:
                    print(f"[*] [Account {idx}] Quota status: {ratelimit_header}")

                if response.status_code == 429:
                    # OPTIMIZED: Instantly flags key as depleted and moves to next pool entry
                    print(f"[!] Account {idx} is fully exhausted (Daily 10k Limit Reached). Swapping keys...")
                    account_failed = True
                    account_exhausted = True
                    break 

                if response.status_code in (401, 403):
                    print(f"[X] Account {idx} Auth error ({response.status_code}). Moving to next account.")
                    account_failed = True
                    break

                if response.status_code >= 500:
                    print(f"[!] Account {idx} server error ({response.status_code}). Retrying in {wait_time}s… (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                    wait_time *= 2
                    continue

                if not response.ok:
                    print(f"[X] Account {idx} error ({response.status_code}): {response.text[:200]}")
                    account_failed = True
                    break

                # data = response.json()

                # if not data.get("success", False):
                #     print(f"[X] Account {idx} reported failure: {data.get('errors')}")
                #     account_failed = True
                #     break

                # De-serialize the string metrics and dump image matrix straight to disk
                # b64_data = data["result"]["image"]
                # image_bytes = base64.b64decode(b64_data)

                image_bytes = response.content

                with open(output_path, "wb") as f:
                    f.write(image_bytes)
                print(f"[✓] Scene successfully generated using Account {idx}!")
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
            print(f"[!] Account {idx} failed or hit allocation roof. Checking failover channel status...")
            continue

    print("[X] Absolute Failure: All registered key channels in the pool are completely exhausted.")
    return False


def build_short_video(selected_scenes: list[dict], image_paths: list[str],
                       audio_path: str, total_duration: float, output_path: str):
    clips = []
    for scene, img_path in zip(selected_scenes, image_paths):
        clip = (
            ImageClip(img_path)
            .set_duration(scene["short_duration"])
            .resize((SHORT_W, SHORT_H))
        )
        clips.append(clip)

    print(f"\n[*] Stitching {len(clips)} vertical image clips…")
    base_video = concatenate_videoclips(clips, method="compose")

    audio_clip = AudioFileClip(audio_path)
    trimmed_audio = audio_clip.subclip(0, min(total_duration, audio_clip.duration))
    base_video.audio = trimmed_audio

    print(f"[*] Rendering vertical Short ({total_duration:.1f}s)…")
    base_video.write_videofile(
        output_path,
        fps=30,
        codec="libx264",
        audio_codec="aac",
        bitrate="10000k",
        audio_bitrate="192k",
        ffmpeg_params=[
            "-preset", "slow",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
        ],
        threads=4,
        logger="bar",
    )

    audio_clip.close()
    base_video.close()



def _build_copy_prompt(data: dict, selected_scenes: list[dict], brief: dict, short_duration: float) -> str:
    concept = data.get("concept", "")
    narrative_excerpt = " ".join(s.get("narrative", "") for s in selected_scenes)[:1500]
    seo_desc = data.get("metadata", {}).get("seo_description", "")
    brief_anchor = _build_brief_anchor(brief)

    return f"""
You are a YouTube Shorts copywriter. Write metadata for a {short_duration:.0f}-second
vertical Short built from the OPENING of a longer story video.

STORY CONCEPT
-------------
{concept}

STORY BRIEF
-----------
{brief_anchor}

LONG-VIDEO SEO DESCRIPTION (for tone reference only)
------------------------------------------------------
{seo_desc}

NARRATIVE COVERED IN THIS SHORT (the opening of the story only)
-------------------------------------------------------------------
{narrative_excerpt}

YOUR TASK
---------
Write punchy, scroll-stopping YouTube Shorts metadata. Rules:
1. "title": under 90 characters, hook-driven, must include the words "#Shorts" at the end.
2. "description": 2-4 short lines, written for mobile, end with 4-6 relevant hashtags
   (include #Shorts). Do NOT include any URLs or placeholder links.
3. "tags": an array of 8-15 single/short-phrase YouTube tags (no # symbol), relevant to
   the concept and characters.
4. Return ONLY a JSON object with keys: title, description, tags.
   No markdown, no preamble.
"""


def generate_short_copy(data: dict, selected_scenes: list[dict], brief: dict, short_duration: float,
                         gemini_client, groq_client: Groq) -> dict:
    prompt = _build_copy_prompt(data, selected_scenes, brief, short_duration)

    try:
        response = gemini_client.models.generate_content(
            model=os.environ["GEMINI_MODEL"],
            contents=prompt,
            config={"response_mime_type": "application/json"},
        )
        return _parse_json_block(response.text)
    except Exception as e:
        print(f"[!] Gemini failed for metadata generation: {e}")
        print(f"[*] Falling back to Groq for metadata generation …")

    try:
        response = groq_client.chat.completions.create(
            model=os.environ["GROQ_MODEL"],
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.8,
        )
        return _parse_json_block(response.choices[0].message.content)
    except Exception as e:
        print(f"[!] Groq also failed for metadata generation: {e}")
        print(f"[!] Using a basic template fallback for metadata.")
        concept = data.get("concept", "short_video")
        return {
            "title": f"{concept} #Shorts"[:90],
            "description": f"{concept}\n\n#Shorts #Story #Viral",
            "tags": [concept, "shorts", "story"],
        }


def write_metadata_txt(metadata_path: str, copy: dict, concept: str,
                        short_duration: float, num_scenes: int):
    tags = copy.get("tags", [])
    tags_line = ", ".join(tags) if isinstance(tags, list) else str(tags)

    lines = [
        f"SHORT METADATA — {concept}",
        "=" * 60,
        "",
        "TITLE",
        "-----",
        copy.get("title", ""),
        "",
        "DESCRIPTION",
        "-----------",
        copy.get("description", ""),
        "",
        "TAGS",
        "----",
        tags_line,
        "",
        "DETAILS",
        "-------",
        f"Duration            : {short_duration:.1f}s",
        f"Resolution          : {SHORT_W}x{SHORT_H} (9:16)",
        f"Source              : Opening {num_scenes} scene(s) of the story, freshly",
        f"                      rendered at {SHORT_W}x{SHORT_H} (not cropped from the long-form video).",
        "",
        "HOW TO LINK THIS SHORT BACK TO THE FULL VIDEO",
        "-----------------------------------------------",
        "Links pasted into a Short's description or comments are NOT clickable on",
        "YouTube. The only native, clickable way to send Short viewers to the full",
        "video is the 'Related video' field:",
        "",
        "  1. Upload both the long-form video and this Short to YouTube.",
        "  2. YouTube Studio > Content > select this Short > Edit.",
        "  3. Open the 'Related video' field on the details page.",
        "  4. Search for / select the long-form video. Save.",
        "",
        "This adds a clickable card on the Short pointing straight to the full video.",
        "Also helps: mention 'full story on my channel' on-screen/verbally, and keep",
        "this title close to the long video's title so viewers recognize it.",
        "",
        "Note: video_package.json was NOT modified — this metadata lives only here.",
    ]

    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    print("=" * 52)
    print("SHORTS COMPILER")
    print("=" * 52)

    user_input = input("[?] Enter project folder name (blank = most recent): ").strip()

    if user_input:
        project_path = os.path.join(OUTPUTS_DIR, user_input)
        if not os.path.exists(project_path):
            print(f"[X] Folder '{user_input}' not found at {project_path}")
            return
    else:
        project_path = get_latest_project_folder()
        if not project_path:
            print("[X] No project workspace found.")
            return

    json_path = os.path.join(project_path, "video_package.json")
    audio_path = os.path.join(project_path, "final_voiceover.wav")

    if not os.path.exists(json_path):
        print(f"[X] video_package.json not found at: {json_path}")
        return
    if not os.path.exists(audio_path):
        print(f"[X] final_voiceover.wav not found at: {audio_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    concept = data.get("concept", "output_video")
    story_tone = data.get("metadata", {}).get("seo_description", "")[:200]
    brief = data.get("brief", {})
    scenes = data.get("scenes", [])
    slug = slugify(concept)

    selected_scenes, total_duration = select_intro_scenes(scenes, SHORT_MAX_DURATION)
    if not selected_scenes:
        print("[X] No usable scenes found at the start of video_package.json. Aborting.")
        return

    short_dir = os.path.join(project_path, "Short")
    short_images_dir = os.path.join(short_dir, "short_scene_images")
    os.makedirs(short_images_dir, exist_ok=True)

    output_video_path = os.path.join(short_dir, f"{slug}_short.mp4")
    metadata_path = os.path.join(short_dir, "metadata.txt")

    print(f"\n[*] Project      : {os.path.basename(project_path)}")
    print(f"[*] Concept      : {concept}")
    print(f"[*] Scenes used  : {len(selected_scenes)} (from the start of the story)")
    print(f"[*] Short length : {total_duration:.1f}s")
    print(f"[*] Renderer     : Cloudflare Workers AI ({IMAGE_MODEL}, {SHORT_W}x{SHORT_H})")
    print(f"[*] Output       : {short_dir}\n")

    gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

    # 1. enhance the opening scenes' prompts for vertical framing
    print("[...] Enhancing opening scenes for vertical framing …")
    enhanced_prompts = enhance_intro_scenes(selected_scenes, concept, story_tone, brief,
                                             gemini_client, groq_client)

    # 2. render each scene as a NEW 1080x1920 image (not cropped from landscape)
    image_paths = []
    for scene, enhanced_prompt in zip(selected_scenes, enhanced_prompts):
        seq = scene["sequence"]
        start_time = scene["start_time"]

        if isinstance(enhanced_prompt, dict):
            enhanced_prompt = (
                enhanced_prompt.get("prompt")
                or enhanced_prompt.get("enhanced_prompt")
                or enhanced_prompt.get("text")
                or str(enhanced_prompt)
            )
        elif not isinstance(enhanced_prompt, str):
            enhanced_prompt = str(enhanced_prompt)
        
        prompt_hash = hash(enhanced_prompt) % 1000
        image_name = f"short_scene_{seq}_{start_time}_{prompt_hash}_image.jpg"
        image_path = os.path.join(short_images_dir, image_name)

        if os.path.exists(image_path) and os.path.getsize(image_path) > 1000:
            print(f"[=] Scene {seq} vertical image already exists — skipping.")
            image_paths.append(image_path)
            continue

        print(f"[...] Rendering vertical image for scene {seq}")
        print(f"    Enhanced : {enhanced_prompt[:70]}…")

        if generate_image_cloudflare(enhanced_prompt, brief, image_path):
            print(f"[✓] Saved → Short/short_scene_images/{image_name}")
            image_paths.append(image_path)
        else:
            print(f"[X] Failed to render scene {seq} — aborting Short build.")
            return

        time.sleep(1.5)

    # 3. build the vertical video (no subtitles)
    build_short_video(selected_scenes, image_paths, audio_path, total_duration, output_video_path)

    # 4. metadata (title/description/tags) — Gemini → Groq
    print("\n[*] Generating Short metadata (Gemini → Groq) …")
    copy = generate_short_copy(data, selected_scenes, brief, total_duration, gemini_client, groq_client)
    write_metadata_txt(metadata_path, copy, concept, total_duration, len(selected_scenes))

    print(f"\n[✓] Short complete!")
    print(f"[✓] Video    : {output_video_path}")
    print(f"[✓] Metadata : {metadata_path}")


if __name__ == "__main__":
    main()