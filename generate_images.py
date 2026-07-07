import os
import json
import re
import glob
import requests
import time

from google import genai
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

BASE_DIR=r"D:\Automated_YouTube_Pipeline"
OUTPUTS_DIR=os.path.join(BASE_DIR, "outputs")

# Global Constants for Leonardo Engine
IMAGE_MODEL = os.environ.get("IMAGE_MODEL_TARGET", "@cf/leonardo/phoenix-1.0")
ENHANCE_COUNT = int(os.environ.get("IMAGE_ENHANCE_COUNT", 20))
MAX_PROMPT_CHARS=2048  # hard limit enforced by the model's input schema
IMAGE_WIDTH=1920        # YouTube 16:9 widescreen
IMAGE_HEIGHT=1080       # YouTube 16:9 widescreen

AESTHETIC_ANCHOR = (
    "Cinematic flat vector-art illustration style for a history documentary — crisp clean outlines, "
    "confident shapes, a stylized graphic-novel look, not photorealistic. Rich, vivid color grading that "
    "matches the ACTUAL time of day and lighting of each specific scene: warm saturated colors (blues, "
    "greens, ochres) for daylight scenes with fully visible, naturally colored figures, versus a dark "
    "palette lit by warm firelight or cool moonlight for night scenes with solid black silhouette figures. "
    "Never a flat gray, desaturated, or washed-out look regardless of time of day. Strong dramatic lighting "
    "and shadow work in every scene. Widescreen cinematic composition."
)

NEGATIVE_BAN = (
    "grayscale, black and white photo, sepia, desaturated, washed out colors, muddy colors, flat gray fog, "
    "flat lighting, photorealism, 3D render, low quality, blurry, distorted anatomy, distorted faces, "
    "messy lines, modern clothing, modern objects, text, watermark"
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
    return (
        f"Main character: {brief['main_character']}. "
        f"Setting: {brief['setting']}. "
        f"Recurring props: {brief['key_props']}. "
        f"Mood/tone: {brief['tone']}. "
        f"Overall color & lighting mood for this story: "
        f"{brief.get('color_mood', 'Naturally colored, matching the described setting and time of day.')}"
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
2. For EACH target scene, first judge its actual lighting from the narrative text and the STORY
   BRIEF's overall color mood (e.g. bright midday, dusk, deep night, firelit, cave interior).
3. If the scene's lighting is dark/low-light (night, deep shadow, cave interior, backlit only by
   fire or moonlight): render the human figures as solid black silhouettes with no visible color or
   surface detail on the body itself, set against a colorful, lit background — e.g. warm orange
   firelight glow, cool blue moonlight — never a flat gray/desaturated background.
4. If the scene's lighting is bright/well-lit (daylight, open sky, sunlit clearing): render the
   human figures with natural but stylized color — skin tone, hair, simple period-appropriate
   clothing — fully visible and colorful against a vividly colored daylight background.
5. State which lighting mode you chose at the start of the enhanced prompt (e.g. "Night scene,
   black silhouette figures..." or "Bright daylight scene, fully colored figures...") so it's
   unambiguous to the renderer.
6. Add specific visual detail: character pose, gesture, composition framing, background elements
   appropriate to the scene's specific moment — not generic.
7. Reference the main character, setting, and props from the STORY BRIEF where relevant, and dress
   characters in clothing/gear appropriate to the story's time period (never modern clothing unless
   the STORY BRIEF is explicitly set in modern times).
8. If the scene includes animals, render them in the same lighting mode (silhouette or colored) as
   the rest of that scene — not photorealistic.
9. Keep each enhanced prompt under 80 words.
10. Return ONLY a JSON object with an "enhanced_prompts" array in the same order as the target scenes.
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

    normalized=[]
    for item in enhanced:
        if isinstance(item, str):
            normalized.append(item)
        elif isinstance(item, dict):
            text=(
                item.get("prompt")
                or item.get("enhanced_prompt")
                or item.get("text")
                or item.get("visual_prompt")
            )
            if text:
                normalized.append(str(text))
            else:
                normalized.append(json.dumps(item))
                print(f"[!] Unexpected dict shape in enhanced_prompts, no known text key found: {item}")
        else:
            normalized.append(str(item))
    enhanced=normalized

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
    brief_anchor=_build_brief_anchor(brief)
    full_prompt=(
        f"{AESTHETIC_ANCHOR} | "
        f"{brief_anchor} | "
        f"{NEGATIVE_BAN} | "
        f"Current Scene: {enhanced_prompt}"
    )
    if len(full_prompt) > MAX_PROMPT_CHARS:
        print(f"[!] Prompt too long ({len(full_prompt)} chars) — truncating to {MAX_PROMPT_CHARS}.")
        full_prompt=full_prompt[:MAX_PROMPT_CHARS]
    return full_prompt


# ─── RESUME HELPERS ───
def find_existing_image(images_dir: str, sequence) -> str | None:
    pattern=os.path.join(images_dir, f"scene_{sequence}_*_image.jpg")
    for match in glob.glob(pattern):
        if os.path.getsize(match) > 1000:
            return match
    return None


# ─── CREDENTIAL POOL LOADER ───
def get_cloudflare_credentials():
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


def generate_image_cloudflare(enhanced_prompt: str, brief: dict, output_path: str, max_retries: int = 5) -> bool:
    if not CREDENTIALS_POOL:
        print("[X] No Cloudflare credentials found in .env (Check CLOUDFLARE_API_KEY_1 / CLOUDFLARE_ACC_ID_1 etc.)")
        return False

    print(f"Model: {IMAGE_MODEL}")

    final_prompt = _build_render_prompt(enhanced_prompt, brief)

    payload = {
        "prompt": final_prompt,
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
            print(f"[!] Account {idx} failed or exhausted limits. Falling back to the next credential layer...")
            continue

    print("[X] Critical pipeline error: All accounts in the available pool are fully exhausted.")
    return False


def run(project_path: str):
    """
    Core image-generation stage, reusable from the master pipeline.
    For every scene: enhances the visual_prompt in batches (Gemini -> Groq),
    then renders one image per scene via Cloudflare Workers AI.
    Resumable — scenes with an existing rendered image are skipped, which
    also skips their enhancement batch call if the whole batch is done.
    """
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

    completed_sequences={
        scene["sequence"]
        for scene in scenes
        if find_existing_image(images_dir, scene["sequence"])
    }
    remaining_count=total_scenes - len(completed_sequences)

    print(f"\n[*] Project   : {os.path.basename(project_path)}")
    print(f"[*] Concept   : {story_concept}")
    print(f"[*] Character : {brief.get('main_character', 'N/A')}")
    print(f"[*] Setting   : {brief.get('setting', 'N/A')}")
    print(f"[*] Scenes    : {total_scenes}")
    print(f"[*] Batch size: {BATCH_SIZE} scenes per call")
    print(f"[*] Enhancer  : Gemini primary → Groq fallback")
    print(f"[*] Renderer  : Cloudflare Workers AI ({IMAGE_MODEL}, {IMAGE_WIDTH}x{IMAGE_HEIGHT})")
    if completed_sequences:
        print(f"[*] Resume    : {len(completed_sequences)}/{total_scenes} scenes already rendered — {remaining_count} remaining.")
    print()

    if remaining_count == 0:
        print(f"[✓] All {total_scenes} scenes already rendered. Nothing to do.")
        print(f"[*] Assets saved to: {images_dir}")
        return

    gemini_client=genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    groq_client=Groq(api_key=os.environ["GROQ_API_KEY"])
    success_count=len(completed_sequences)

    for batch_start in range(0, total_scenes, BATCH_SIZE):
        batch_end=min(batch_start + BATCH_SIZE, total_scenes)
        batch=scenes[batch_start:batch_end]

        pending_in_batch=[s for s in batch if s["sequence"] not in completed_sequences]
        if not pending_in_batch:
            print(f"[=] Scenes {batch[0]['sequence']}–{batch[-1]['sequence']} already rendered — skipping batch.")
            continue

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

            if seq in completed_sequences:
                print(f"[=] Scene {seq}/{total_scenes} already exists — skipping.")
                continue

            if not isinstance(enhanced_prompt, str):
                print(f"[!] Scene {seq}: enhanced_prompt was not a string ({type(enhanced_prompt)}), falling back to original visual_prompt.")
                enhanced_prompt=str(scene.get("visual_prompt", ""))

            prompt_hash=hash(enhanced_prompt) % 1000
            image_name=f"scene_{seq}_{start_time}_{prompt_hash}_image.jpg"
            image_path=os.path.join(images_dir, image_name)

            print(f"\n[...] Rendering scene {seq}/{total_scenes}")
            print(f"    Original : {scene['visual_prompt'][:70]}…")
            print(f"    Enhanced : {enhanced_prompt[:70]}…")

            if generate_image_cloudflare(enhanced_prompt, brief, image_path):
                print(f"[✓] Saved → scene_images/{image_name}")
                success_count+=1
                completed_sequences.add(seq)
            else:
                print(f"[X] Failed for scene {seq} — skipping.")

            time.sleep(1.5)

    print(f"\n{'='*60}")
    print(f"[✓] Done! {success_count}/{total_scenes} images rendered.")
    print(f"[*] Assets saved to: {images_dir}")
    print(f"{'='*60}")


def main():
    try:
        project_path=select_project_folder()
    except FileNotFoundError as e:
        print(f"[X] {e}")
        return
    run(project_path)


if __name__ == "__main__":
    main()