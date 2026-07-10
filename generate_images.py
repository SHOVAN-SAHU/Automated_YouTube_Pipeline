import os
import json
import re
import glob
import base64
import requests
import time

from google import genai
from groq import Groq
from dotenv import load_dotenv

from pipeline_config import (
    CHARACTER_DESIGN, AESTHETIC_ANCHOR as STYLE_ANCHOR, NEGATIVE_BAN, RECOMMENDED_IMAGE_MODEL,
    CAMERA_SHOTS, MOOD_DAYLIGHT, MOOD_HARSH, MOOD_NIGHT, MOOD_DIAGRAM,
)

load_dotenv()

BASE_DIR=r"D:\Automated_YouTube_Pipeline"
OUTPUTS_DIR=os.path.join(BASE_DIR, "outputs")

# Global Constants for Render Engine
# Default now comes from pipeline_config (Leonardo Lucid Origin) — better
# prompt-adherence for a locked, simple flat-vector character than Phoenix,
# which is tuned more for painterly/photoreal prompt-following. Still
# overridable via env var if you want to A/B against flux-1-schnell etc.
IMAGE_MODEL = os.environ.get("IMAGE_MODEL_TARGET", RECOMMENDED_IMAGE_MODEL)
ENHANCE_COUNT = int(os.environ.get("IMAGE_ENHANCE_COUNT", 20))
# Cloudflare's documented schema for @cf/leonardo/lucid-origin only specifies
# minLength: 1 on `prompt` — no documented max. 2048 was a leftover guess from
# a different model and was silently chopping prompts mid-sentence (cutting off
# the negative-prompt list or scene action at random). With the CHARACTER_DESIGN
# duplication fixed, prompts should now land well under this anyway — this is
# a safety net, not an expected/normal limit.
MAX_PROMPT_CHARS=4000
IMAGE_WIDTH=1920        # YouTube 16:9 widescreen
IMAGE_HEIGHT=1080       # YouTube 16:9 widescreen

# The character design + base art style are now locked in pipeline_config.py
# and shared with generate_story.py, instead of being redefined here separately
# (that split was the main source of "character doesn't match" hallucination —
# the two scripts had no guarantee of agreeing on what the character looked like).
AESTHETIC_ANCHOR = f"{CHARACTER_DESIGN} {STYLE_ANCHOR}"

BATCH_SIZE=5
WINDOW_BEFORE=3
WINDOW_AFTER=3


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
    # character_role is just narrative flavor (e.g. "a Neanderthal hunting
    # party") — the actual visual design (CHARACTER_DESIGN) is injected once,
    # separately, via AESTHETIC_ANCHOR. Do NOT also put CHARACTER_DESIGN here:
    # that duplication is what blew the final render prompt up to ~3800 chars.
    return (
        f"Character's role in this story: {brief.get('character_role', 'an ancient human')}. "
        f"Setting: {brief.get('setting', '')}. "
        f"Recurring props: {brief.get('key_props', '')}. "
        f"Mood/tone: {brief.get('tone', '')}. "
        f"Overall color & lighting mood for this story: "
        f"{brief.get('color_mood', 'Naturally colored, matching the described setting and time of day.')}. "
        f"Night/low-light figure style for THIS story: {brief.get('night_figure_style', 'colored')}."
    )


def _night_figure_style(brief: dict) -> str:
    style=str(brief.get("night_figure_style", "colored")).strip().lower()
    return "silhouette" if style.startswith("silhouette") else "colored"


def _fmt_scenes(scenes: list[dict], label: str) -> str:
    if not scenes:
        return ""
    lines=[f"\n[{label}]"]
    for s in scenes:
        tags=[]
        if s.get("location"):
            tags.append(f"Location: {s['location']}")
        if s.get("lighting"):
            tags.append(f"Lighting: {s['lighting']}")
        if s.get("established_facts"):
            tags.append(f"Established facts: {s['established_facts']}")
        tag_str=f" [{' | '.join(tags)}]" if tags else ""
        lines.append(f"  Scene {s['sequence']}{tag_str}: {s['visual_prompt']}")
    return "\n".join(lines)


def _suggested_camera(sequence: int) -> str:
    # Rotates through CAMERA_SHOTS by sequence number so consecutive scenes
    # default to different framing instead of the same centered standing
    # shot every time. It's only a *suggestion* fed to the enhancer — rule 6
    # below tells it to override this when the scene's actual action calls
    # for something else.
    return CAMERA_SHOTS[(sequence - 1) % len(CAMERA_SHOTS)]


def _resolve_mood(lighting_text: str, night_style: str) -> str:
    """
    Turns a scene's free-text `lighting` tag (written by an LLM in
    generate_story.py, so wording varies) into exactly ONE of the four fixed
    mood snippets from pipeline_config, via simple keyword matching. Without
    this, MOOD_DAYLIGHT/HARSH/NIGHT/DIAGRAM were imported but never actually
    read anywhere — the enhancer was just left to freely reinterpret the
    lighting tag in prose each time, with no fixed vocabulary to land on.
    """
    text=(lighting_text or "").lower()
    if any(k in text for k in ("night", "moonlight", "firelight", "dark", "cave", "dusk", "torch")):
        return MOOD_NIGHT
    if any(k in text for k in ("diagram", "explainer", "cream background", "plain background")):
        return MOOD_DIAGRAM
    if any(k in text for k in ("harsh", "storm", "danger", "bleak", "grim", "freezing", "blizzard")):
        return MOOD_HARSH
    return MOOD_DAYLIGHT


def _fmt_target_scenes(scenes: list[dict], night_style: str) -> str:
    if not scenes:
        return ""
    lines=["\n[TARGET SCENES — enhance these]"]
    for s in scenes:
        tags=[]
        if s.get("location"):
            tags.append(f"Location: {s['location']}")
        if s.get("lighting"):
            tags.append(f"Lighting: {s['lighting']}")
        if s.get("established_facts"):
            tags.append(f"Established facts: {s['established_facts']}")
        tags.append(f"Suggested camera/framing: {_suggested_camera(s['sequence'])}")
        tags.append(f"Resolved mood/palette (use this wording): {_resolve_mood(s.get('lighting', ''), night_style)}")
        tag_str=f" [{' | '.join(tags)}]"
        lines.append(f"  Scene {s['sequence']}{tag_str}: {s['visual_prompt']}")
    return "\n".join(lines)


def _build_enhance_prompt(
    batch: list[dict],
    context_before: list[dict],
    context_after: list[dict],
    story_concept: str,
    story_tone: str,
    brief: dict,
) -> str:
    night_style=_night_figure_style(brief)
    context_block=_fmt_scenes(context_before, "PREVIOUS SCENES — for visual continuity")
    target_block=_fmt_target_scenes(batch, night_style)
    upcoming_block=_fmt_scenes(context_after, "UPCOMING SCENES — for narrative awareness")
    sequence_ids=[s["sequence"] for s in batch]
    brief_anchor=_build_brief_anchor(brief)

    return f"""
You are a professional AI image prompt engineer specialising in a locked, minimalist cartoon character design for a YouTube explainer channel.

STORY CONTEXT
-------------
Concept : {story_concept}
Tone    : {story_tone}

CHARACTER + STORY BRIEF (the character design is FIXED — never alter its described appearance,
only its pose/expression/action per scene; use the rest for setting, props, mood consistency)
-----------------------------------------------------------------------------------------
{brief_anchor}

VISUAL STYLE (must be obeyed in every enhanced prompt)
-------------------------------------------------------
{AESTHETIC_ANCHOR}
Avoid: {NEGATIVE_BAN}

CONTINUITY IS GROUND TRUTH
--------------------------
Every scene below (previous, target, and upcoming) is tagged with its authoritative
Location / Lighting / Established facts, computed sequentially from the full story.
Treat these tags as hard constraints, not suggestions:
- If a TARGET scene's tag says "Established facts: fire not yet discovered", the
  enhanced prompt must NOT contain fire, embers, torches, or any firelight — even if
  a nearby scene in the window does have fire. Show the alternative implied by the
  narrative (raw food, cold camp, moonlight/daylight only, etc.).
- If a TARGET scene's Location tag says an enclosed setting (e.g. "inside a cave"),
  keep that enclosed setting visible (rock walls, low ceiling, cave mouth) even if
  the scene's own narrative sentence doesn't repeat the word "cave" — the tag is the
  authority, not the sentence in isolation.
- Only change location/lighting/established facts within your enhanced prompt if the
  TARGET scene's own tag explicitly differs from the PREVIOUS scene's tag. Never
  invent a location or prop change that isn't reflected in the tags.

YOUR TASK
---------
Rewrite ONLY the TARGET SCENES below into richer, more descriptive image generation prompts.
Use the PREVIOUS and UPCOMING scenes purely for visual continuity — same background style,
same character proportions, consistent lighting direction, consistent emotional palette.

Rules:
1. Keep the same narrative action as the original — do NOT invent new story events.
2. Do NOT re-describe the character's fixed appearance (head shape, eyes, tunic, etc.) —
   that's already stated once above in VISUAL STYLE and will be included automatically at
   render time. Repeating it here would just waste your word budget. Focus entirely on THIS
   scene's specific pose, action, expression, and background.
3. Each TARGET scene lists a "Resolved mood/palette" tag — use that EXACT wording as the
   color/lighting basis of your enhanced prompt (it was already matched to the scene's
   Location/Lighting tags for you). Don't re-derive the mood yourself from scratch.
4. This story's NIGHT FIGURE STYLE is: "{night_style}". Apply it whenever the Resolved
   mood/palette tag is the night one (deep blue palette, firelight/moonlight accents):
   - If night_style is "silhouette": render the character as a solid black silhouette with no
     visible color or surface detail on the body itself, set against a colorful, lit flat-color
     background — e.g. warm orange firelight glow, cool blue moonlight.
   - If night_style is "colored": keep the character fully visible in flat stylized color, just
     shift the palette to a cooler, darker flat tone (deep blues/purples with warm firelight or
     moonlight accents) instead of the bright daylight palette. Do not desaturate to gray —
     darkness is a color choice, not gloom.
   Apply this same choice consistently across every dark scene in this story — don't mix modes.
5. For all other Resolved mood/palette tags (daylight/harsh/diagram), render the character
   fully visible and colorful in that palette — never desaturated or gray regardless of mood.
6. VARY THE STAGING. Each scene lists a "Suggested camera/framing" — use it unless the
   scene's own action clearly calls for something else, and NEVER give two consecutive
   scenes the same shot type or the same pose. Reflective/narration-only lines (no
   explicit physical action, e.g. "you are the last of your kind") are NOT an excuse to
   default to a plain centered standing shot — invent the camera angle, distance, and
   which part of the character is shown (close on the face, from behind looking at the
   landscape, low angle, aerial, etc.) even while the underlying pose stays simple. Two
   scenes can share the same narrative beat and still look completely different on screen.
7. Reference the setting and props from the STORY BRIEF where relevant.
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
    # NOTE: no "Avoid: {NEGATIVE_BAN}" here anymore. lucid-origin has no
    # negative_prompt parameter (confirmed against Cloudflare's published
    # schema), so that text was being read as positive prompt content —
    # almost certainly the cause of the partially-desaturated/grayscale
    # render, since "grayscale, sepia" was sitting right there as "content"
    # the model should include. NEGATIVE_BAN is still used, correctly, inside
    # _build_enhance_prompt where it's shown to a text-generating LLM that
    # actually understands negation.
    brief_anchor=_build_brief_anchor(brief)
    full_prompt=(
        f"{AESTHETIC_ANCHOR} | "
        f"{brief_anchor} | "
        f"Current Scene: {enhanced_prompt}"
    )
    if len(full_prompt) > MAX_PROMPT_CHARS:
        # Trim from the middle of "Current Scene" rather than blindly slicing
        # the whole assembled string — that used to risk cutting off the
        # scene action itself, whichever happened to fall past the cutoff.
        overflow=len(full_prompt) - MAX_PROMPT_CHARS
        print(f"[!] Prompt too long ({len(full_prompt)} chars) — trimming {overflow} chars from the scene description.")
        static_part=full_prompt[:full_prompt.rfind("Current Scene:") + len("Current Scene: ")]
        scene_part=full_prompt[len(static_part):]
        budget=MAX_PROMPT_CHARS - len(static_part)
        full_prompt=static_part + scene_part[:max(budget, 0)]
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


def _extract_image_bytes(response: requests.Response) -> bytes:
    """
    Cloudflare's REST endpoint for image models returns JSON —
    {"result": {"image": "<base64>"}, "success": true, ...} — NOT raw image
    bytes, even though the response "looked like" it could be saved directly.
    Writing response.content straight to a .jpg (the old behavior) silently
    produced a JSON text file with a .jpg extension: it "succeeded" and saved
    something, but nothing could actually open it as an image. This inspects
    the content-type and only treats the response as raw binary if it
    actually is one — otherwise it decodes the base64 payload.
    """
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

    # Some models/configurations do return raw binary directly — use as-is.
    return response.content


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

        print(f"[*] [Account {idx}] Attempting scene render with key ending in ...{api_key[-4:]}")
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
        print("[!] Warning: no brief found in video_package.json — setting/prop context will be missing "
              "(character design still applies from pipeline_config.py).")

    has_state_tags=any(s.get("location") or s.get("established_facts") for s in scenes)
    if not has_state_tags:
        print("[!] This video_package.json predates continuity tags (location/lighting/established_facts). "
              "Enhancement will fall back to the old ±window behaviour for this project — "
              "regenerate the story stage for full continuity guarantees.")

    completed_sequences={
        scene["sequence"]
        for scene in scenes
        if find_existing_image(images_dir, scene["sequence"])
    }
    remaining_count=total_scenes - len(completed_sequences)

    print(f"\n[*] Project   : {os.path.basename(project_path)}")
    print(f"[*] Concept   : {story_concept}")
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
            print(f"    Location : {scene.get('location', 'N/A')} | Lighting: {scene.get('lighting', 'N/A')}")
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