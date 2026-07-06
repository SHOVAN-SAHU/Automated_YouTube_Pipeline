"""
Scene Image Generator — Gemini-Enhanced Prompts
  • Loads story brief from video_package.json for character/setting consistency
  • Gemini enhances visual_prompt in batches using sliding context window + brief
  • Groq fallback if Gemini fails for any batch
  • Pollinations renders the enhanced prompt into a 16:9 doodle image
  • Input/output structure of video_package.json stays identical
"""

import os
import json
import re
import requests
import time
import urllib.parse

from google import genai
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

BASE_DIR=r"D:\Automated_YouTube_Pipeline"
OUTPUTS_DIR=os.path.join(BASE_DIR, "outputs")

AESTHETIC_ANCHOR=(
    "A clean minimalist stickman doodle art style, professional webcomic illustration, "
    "smooth solid bold black outlines, clean digital ink work, flat simple primary colors, "
    "solid light beige background, high contrast drawing, charming simple character design, "
    "cute expressive dot eyes, perfectly drawn simple cartoon anatomy, "
    "8k resolution, native 16:9 widescreen composition."
)

NEGATIVE_BAN=(
    "NO photorealism, NO realistic skin, NO 3D rendering, NO creepy faces, NO realistic eyes, "
    "NO complex shading, NO messy crayon textures, NO shaky loose lines, NO extra limbs, NO distorted bodies."
)

BATCH_SIZE=5
WINDOW_BEFORE=2
WINDOW_AFTER=2


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
    print("-------------------------------------")

    user_input=input(f"\nSelect a folder name or number [Press Enter for '{latest_folder}']: ").strip()

    if not user_input:
        return os.path.join(OUTPUTS_DIR, latest_folder)
    if user_input.isdigit():
        idx=int(user_input) - 1
        if 0 <= idx < len(folders):
            return os.path.join(OUTPUTS_DIR, folders[idx])
    if user_input in folders:
        return os.path.join(OUTPUTS_DIR, user_input)

    print(f"[!] Invalid selection. Defaulting to most recent: {latest_folder}")
    return os.path.join(OUTPUTS_DIR, latest_folder)


def _build_brief_anchor(brief: dict) -> str:
    # builds a story-specific style anchor from the brief fields
    return (
        f"Main character: {brief['main_character']}. "
        f"Setting: {brief['setting']}. "
        f"Recurring props: {brief['key_props']}. "
        f"Mood/tone: {brief['tone']}."
    )


def _build_enhance_prompt(
    batch: list[dict],
    context_before: list[dict],
    context_after: list[dict],
    story_concept: str,
    story_tone: str,
    brief: dict,
) -> str:
    def fmt_scenes(scenes: list[dict], label: str) -> str:
        if not scenes:
            return ""
        lines=[f"\n[{label}]"]
        for s in scenes:
            lines.append(f"  Scene {s['sequence']}: {s['visual_prompt']}")
        return "\n".join(lines)

    context_block=fmt_scenes(context_before, "PREVIOUS SCENES — for visual continuity")
    target_block=fmt_scenes(batch, "TARGET SCENES — enhance these")
    upcoming_block=fmt_scenes(context_after, "UPCOMING SCENES — for narrative awareness")
    sequence_ids=[s["sequence"] for s in batch]
    brief_anchor=_build_brief_anchor(brief)

    return f"""
You are a professional AI image prompt engineer specialising in minimalist stickman doodle animation for YouTube.

STORY CONTEXT
-------------
Concept : {story_concept}
Tone    : {story_tone}

STORY BRIEF (use this to keep character, setting, and props consistent across all scenes)
-----------------------------------------------------------------------------------------
{brief_anchor}

VISUAL STYLE (must be obeyed in every enhanced prompt)
-------------------------------------------------------
{AESTHETIC_ANCHOR}
{NEGATIVE_BAN}

YOUR TASK
---------
Rewrite ONLY the TARGET SCENES below into richer, more descriptive image generation prompts.
Use the PREVIOUS and UPCOMING scenes purely for visual continuity — same background style,
same character proportions, consistent lighting direction, consistent emotional palette.

Rules:
1. Keep the same narrative action as the original — do NOT invent new story events.
2. Add specific visual detail: character pose, facial expression (dot eyes only),
   background elements, composition framing, mood lighting (flat colour only).
3. Reference the main character, setting, and props from the STORY BRIEF where relevant.
4. Every enhanced prompt must remain in the same minimalist stickman doodle style.
5. Keep each enhanced prompt under 80 words.
6. Return ONLY a JSON object with an "enhanced_prompts" array in the same order as the target scenes.
   No markdown, no preamble.
{context_block}
{target_block}
{upcoming_block}

Expected output format:
{{
  "enhanced_prompts": [
    "Enhanced prompt for scene {sequence_ids[0]}...",
    "Enhanced prompt for scene {sequence_ids[1] if len(sequence_ids) > 1 else sequence_ids[0]}..."
  ]
}}
"""


def _parse_enhanced(raw: str, batch: list[dict]) -> list[str]:
    clean=re.sub(r"^```json|^```|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    result=json.loads(clean)
    enhanced=result.get("enhanced_prompts", [])

    for i in range(len(enhanced), len(batch)):
        enhanced.append(batch[i]["visual_prompt"])
        print(f"[!] Provider returned fewer prompts than expected — using original for scene {batch[i]['sequence']}.")

    return enhanced


def enhance_batch(
    batch: list[dict],
    context_before: list[dict],
    context_after: list[dict],
    story_concept: str,
    story_tone: str,
    brief: dict,
    gemini_client,
    groq_client: Groq,
) -> list[str]:
    prompt=_build_enhance_prompt(batch, context_before, context_after, story_concept, story_tone, brief)

    try:
        response=gemini_client.models.generate_content(
            model=os.environ["GEMINI_MODEL"],
            contents=prompt,
            config={"response_mime_type": "application/json"},
        )
        return _parse_enhanced(response.text, batch)

    except Exception as e:
        print(f"[!] Gemini failed for this batch: {e}")
        print(f"[*] Falling back to Groq for prompt enhancement …")

    try:
        response=groq_client.chat.completions.create(
            model=os.environ["GROQ_MODEL"],
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.7,
        )
        return _parse_enhanced(response.choices[0].message.content, batch)

    except Exception as e:
        print(f"[!] Groq also failed for this batch: {e}")
        print(f"[!] Using original visual_prompts for this batch.")
        return [s["visual_prompt"] for s in batch]


def _build_render_prompt(enhanced_prompt: str, brief: dict) -> str:
    # injects brief anchor between style rules and scene description for consistent rendering
    brief_anchor=_build_brief_anchor(brief)
    return (
        f"{AESTHETIC_ANCHOR} | "
        f"{brief_anchor} | "
        f"{NEGATIVE_BAN} | "
        f"Current Scene: {enhanced_prompt}"
    )


def generate_image_pollinations(enhanced_prompt: str, brief: dict, output_path: str, max_retries: int=5) -> bool:
    final_prompt=_build_render_prompt(enhanced_prompt, brief)
    encoded_prompt=urllib.parse.quote(final_prompt)
    api_url=f"https://image.pollinations.ai/p/{encoded_prompt}?width=1920&height=1080&model=flux&nologo=true"
    wait_time=4.0

    for attempt in range(max_retries):
        try:
            response=requests.get(api_url, timeout=50)

            if response.status_code==429:
                print(f"[!] Rate limited (429). Retrying in {wait_time}s… (attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
                wait_time*=2
                continue

            response.raise_for_status()

            with open(output_path, "wb") as f:
                f.write(response.content)
            return True

        except requests.exceptions.RequestException as e:
            if attempt==max_retries - 1:
                print(f"[X] Network failure after {max_retries} retries: {e}")
                return False
            time.sleep(3)

    return False


def main():
    try:
        project_path=select_project_folder()
    except FileNotFoundError as e:
        print(f"[X] {e}")
        return

    json_path=os.path.join(project_path, "video_package.json")
    images_dir=os.path.join(project_path, "scene_images")
    os.makedirs(images_dir, exist_ok=True)

    if not os.path.exists(json_path):
        print(f"[X] video_package.json not found at: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data=json.load(f)

    scenes=data["scenes"]
    story_concept=data.get("concept", "Unknown concept")
    story_tone=data.get("metadata", {}).get("seo_description", "")[:200]
    brief=data.get("brief", {})
    total_scenes=len(scenes)

    if not brief:
        print("[!] Warning: no brief found in video_package.json — character/setting context will be missing.")

    print(f"\n[*] Project   : {os.path.basename(project_path)}")
    print(f"[*] Concept   : {story_concept}")
    print(f"[*] Character : {brief.get('main_character', 'N/A')}")
    print(f"[*] Setting   : {brief.get('setting', 'N/A')}")
    print(f"[*] Scenes    : {total_scenes}")
    print(f"[*] Batch size: {BATCH_SIZE} scenes per call")
    print(f"[*] Enhancer  : Gemini primary → Groq fallback\n")

    gemini_client=genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    groq_client=Groq(api_key=os.environ["GROQ_API_KEY"])
    success_count=0

    for batch_start in range(0, total_scenes, BATCH_SIZE):
        batch_end=min(batch_start + BATCH_SIZE, total_scenes)
        batch=scenes[batch_start:batch_end]
        ctx_before=scenes[max(0, batch_start - WINDOW_BEFORE):batch_start]
        ctx_after=scenes[batch_end:min(total_scenes, batch_end + WINDOW_AFTER)]

        print(f"[+] Enhancing scenes {batch[0]['sequence']}–{batch[-1]['sequence']} …")

        enhanced_prompts=enhance_batch(
            batch, ctx_before, ctx_after,
            story_concept, story_tone, brief,
            gemini_client, groq_client,
        )

        for scene, enhanced_prompt in zip(batch, enhanced_prompts):
            seq=scene["sequence"]
            start_time=scene["start_time"]

            prompt_hash=hash(enhanced_prompt) % 1000
            image_name=f"scene_{seq}_{start_time}_{prompt_hash}_image.png"
            image_path=os.path.join(images_dir, image_name)

            if os.path.exists(image_path) and os.path.getsize(image_path) > 1000:
                print(f"[=] Scene {seq}/{total_scenes} already exists — skipping.")
                success_count+=1
                continue

            print(f"\n[...] Rendering scene {seq}/{total_scenes}")
            print(f"    Original : {scene['visual_prompt'][:70]}…")
            print(f"    Enhanced : {enhanced_prompt[:70]}…")

            if generate_image_pollinations(enhanced_prompt, brief, image_path):
                print(f"[✓] Saved → scene_images/{image_name}")
                success_count+=1
            else:
                print(f"[X] Failed for scene {seq} — skipping.")

            time.sleep(1.5)

    print(f"\n{'='*60}")
    print(f"[✓] Done! {success_count}/{total_scenes} images rendered.")
    print(f"[*] Assets saved to: {images_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()