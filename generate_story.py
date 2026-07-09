import os
import re
import json
from datetime import datetime

from google import genai
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

BASE_DIR=r"D:\Automated_YouTube_Pipeline"
OUTPUTS_DIR=os.path.join(BASE_DIR, "outputs")
VISUAL_BATCH_SIZE=12


def create_project_folder(concept_name: str) -> str:
    date_str=datetime.now().strftime("%Y-%m-%d")
    sanitized=re.sub(r"[^a-z0-9]+", "_", concept_name.lower()).strip("_")
    path=os.path.join(OUTPUTS_DIR, f"{date_str}_{sanitized}")
    os.makedirs(path, exist_ok=True)
    return path


def _parse_json(raw: str) -> dict | list:
    clean=re.sub(r"^```json|^```|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    return json.loads(clean)


# ── Step 1: prose + brief ──────────────────────────────────────────────────

def _prose_prompt(topic: str, target_minutes: float) -> str:
    target_words = int(target_minutes * 140)
    min_words = int(target_words * 0.85)
    max_words = int(target_words * 1.15)

    return f"""
You are an elite documentary scriptwriter for high-retention history and evolutionary psychology YouTube essays.

TASK
----
Write a raw, gripping, analytical video essay about: "{topic}"
Then produce a compact story brief.

STORYTELLING RULES & FORMAT
---------------------------
- TONE: Visceral, intense, dark, academic yet deeply engaging. 
- PERSPECTIVE: Write predominantly in the second person ("You") to force the viewer into the environment immediately.
- STYLE: Use punchy, short, sharp sentences. Avoid flowery corporate transitions, marketing buzzwords, or introductory filler.
- DATA GROUNDING: Integrate concrete biological facts, anatomical mechanics, weapons, or archaeological milestones.
- CHARACTER CAP: Do not invent individual heroes, dialogue, or personal names. Speak of tribes, ancestors, or species.
- Retain strict informational uniqueness. Every sentence must move the focus forward without looping back to restate prior points.
- Total word count MUST be between {min_words} and {max_words} words.

OUTPUT FORMAT
-------------
Return ONLY a single valid JSON object — no markdown fences, no commentary, no text
outside the JSON. The "prose" field must contain the ENTIRE story as one plain text
string (use \\n\\n between paragraphs if you want breaks). Do not nest prose as JSON
or bullet points, just flowing narrative text inside the string value.

{{
  "prose": "Full story as one plain text string.",
  "brief": {{"main_character": "Short description of the main figure(s) in this story", "setting": "Where and when this story physically takes place", "visual_style": "Flat matte-paint / cutout animation style: clean thick outlines, solid flat color fills, simple flat-color backgrounds — think explainer-video paint style, not moody or photorealistic", "tone": "The emotional register of this story", "key_props": "Recurring objects/tools/weapons relevant to this story", "color_mood": "Decide the ACTUAL time of day and lighting for this specific story based on its content — do not default to night or dusk. If the story plausibly happens in daylight (e.g. daytime hunting, summer, midday activity), describe a bright, vividly colored daylight palette. If it genuinely happens at night, in a cave, or lit only by fire (e.g. this fire-discovery story), describe a palette lit by warm firelight or cool moonlight — but still flat and colorful, not desaturated or gloomy. Be specific and topic-appropriate, e.g. 'Bright warm midday sunlight, vivid greens and ochres, fully lit and colorful' or 'Deep blue night palette lit by warm orange firelight glow'.", "night_figure_style": "Decide, based on this specific story's tone and drama, how human figures should be rendered during night/cave/low-light scenes: either 'silhouette' (solid black silhouette figures against a colorful lit background — best for high-drama, primal, high-contrast moments) or 'colored' (figures stay fully visible in flat stylized color, just with a cooler/muted night palette instead of full daylight color — best for lighter, more explainer-style, less dramatic stories). Pick whichever suits THIS story, and use it consistently for every night/dark scene in the video. Respond with exactly the single word 'silhouette' or 'colored'."}}
}}
"""


def _parse_prose_response(raw: str) -> tuple[str, dict]:
    data=_parse_json(raw)
    if not isinstance(data, dict):
        raise ValueError("Response JSON was not an object with 'prose' and 'brief' keys.")
    prose=data.get("prose")
    brief=data.get("brief")
    if not prose or not brief:
        raise ValueError("Response JSON is missing 'prose' or 'brief' keys.")
    if not isinstance(prose, str):
        prose=str(prose)
    if not isinstance(brief, dict):
        raise ValueError("Response JSON's 'brief' field was not an object.")
    return prose.strip(), brief


def generate_prose(topic: str, target_minutes: float, groq_client: Groq, gemini_client) -> tuple[str, dict]:
    prompt=_prose_prompt(topic, target_minutes)
    target_words=int(target_minutes * 140)
    # Rough budget: ~1.4 tokens/word for the prose itself, plus headroom for the
    # brief object and JSON overhead, plus general safety margin so a long story
    # doesn't get silently truncated mid-JSON (which used to break parsing with
    # no clear signal as to why).
    max_tokens=min(16000, int(target_words * 2.2) + 1200)
    print(f"\n[+] Generating prose story (~{target_words} words) …")

    try:
        print("[*] Trying Groq for prose …")
        resp=groq_client.chat.completions.create(
            model=os.environ["GROQ_MODEL"],
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.5,
            max_tokens=max_tokens,
        )
        finish_reason=resp.choices[0].finish_reason
        if finish_reason == "length":
            print(f"[!] Groq prose response was truncated (finish_reason=length, max_tokens={max_tokens}).")
        prose, brief=_parse_prose_response(resp.choices[0].message.content)
        print(f"[✓] Groq prose done — {len(prose.split())} words.")
        return prose, brief

    except Exception as e:
        print(f"[!] Groq failed for prose: {e}")
        print("[*] Falling back to Gemini for prose …")

    try:
        resp=gemini_client.models.generate_content(
            model=os.environ["GEMINI_MODEL"],
            contents=prompt,
            config={"response_mime_type": "application/json", "max_output_tokens": max_tokens},
        )
        prose, brief=_parse_prose_response(resp.text)
        print(f"[✓] Gemini prose fallback done — {len(prose.split())} words.")
        return prose, brief

    except Exception as e:
        raise RuntimeError(f"Both Groq and Gemini failed for prose generation: {e}")


# ── Step 2: Python chunking (no LLM) ──────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    raw=re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in raw if s.strip()]


def chunk_prose(prose: str, sentences_per_scene: int=2) -> list[str]:
    sentences=_split_sentences(prose)
    chunks=[]
    for i in range(0, len(sentences), sentences_per_scene):
        group=sentences[i:i + sentences_per_scene]
        chunks.append(" ".join(group))
    print(f"[✓] Python chunker — {len(sentences)} sentences → {len(chunks)} chunks.")
    return chunks


# ── Step 3: batched visual prompts (WITH persistent world-state tracking) ─
#
# The single biggest source of context-drift bugs (fire appearing before
# it's discovered, characters silently teleporting out of a cave, etc.) is
# that each batch of chunks used to be enhanced in isolation. There was no
# hard record of "what is currently true about the physical world" that
# carried from one batch to the next — the model had to re-infer it from
# scratch every time, and LLMs are bad at silently inferring "no change
# happened" over a sliding window. So now every batch call:
#   1. Receives an explicit `world_state` block describing the CONFIRMED
#      state as of the end of the previous batch (location / lighting /
#      established facts like "fire not yet discovered").
#   2. Must return that same state for every chunk, unless the chunk's own
#      text explicitly narrates a change.
#   3. Whatever the LAST chunk in the batch ends up with becomes the seed
#      state for the NEXT batch. This makes state a running variable
#      instead of something re-guessed per batch.

def _visual_batch_prompt(
    brief: dict,
    prose: str,
    numbered_chunks: list[tuple[int, str]],
    world_state: dict,
) -> str:
    chunks_text="\n".join(f"{i}. \"{chunk}\"" for i, chunk in numbered_chunks)

    return f"""
You are a storyboard artist for a minimalist YouTube animation.

STORY BRIEF
-----------
Character : {brief['main_character']}
Setting   : {brief['setting']}
Style     : {brief['visual_style']}
Tone      : {brief['tone']}
Key props : {brief['key_props']}

FULL STORY (for context on low-context sentences)
-----------
{prose}

CONFIRMED WORLD STATE — as of the end of the previous batch
-------------------------------------------------------------
Location           : {world_state['location']}
Lighting            : {world_state['lighting']}
Established facts   : {world_state['established_facts']}

This state is GROUND TRUTH. Carry it forward UNCHANGED into every chunk below
unless that chunk's own text explicitly narrates a change (e.g. "they left the
cave", "the flame finally caught", "they now had spears"). Do not silently drop
or reverse an established fact just because a chunk doesn't repeat it — absence
of a mention is NOT evidence of change. Common failure to avoid: showing fire,
tools, or clothing that "established_facts" says do not exist yet; or moving
characters out of an enclosed setting (cave, shelter) into the open without the
text saying so.

TASK
----
For each numbered chunk below, produce ONE storyboard entry describing a single
minimalist stickman cartoon panel. For each entry:
- First decide whether this chunk changes the location, lighting, or any
  established fact. If not, repeat the CONFIRMED WORLD STATE values exactly.
- If it does change, state the NEW location/lighting/established_facts clearly
  and concisely — this becomes the new ground truth for future chunks.
- Then write a visual_prompt: one sentence, specific to THAT chunk's moment,
  referencing the character/setting/props from the brief where relevant, and
  consistent with the location/lighting you just decided.

CHUNKS TO ILLUSTRATE
--------------------
{chunks_text}

OUTPUT FORMAT
-------------
Return ONLY a valid JSON array, same order as the chunks, no extra keys, no markdown:

[
  {{
    "location": "concise current physical setting, e.g. 'inside a shallow rock cave'",
    "lighting": "concise lighting descriptor, e.g. 'pitch dark, only moonlight at the cave mouth'",
    "established_facts": "comma-separated list of story-critical facts still true right now, e.g. 'fire not yet discovered, only stone tools, tribe still nomadic'",
    "visual_prompt": "..."
  }}
]
"""


def _normalize_visual_items(result: list, batch: list[str], fallback_state: dict) -> list[dict]:
    """
    Defensively extracts a full {location, lighting, established_facts,
    visual_prompt} item from each entry, regardless of exact key names the
    LLM used, and regardless of whether it returned a plain string instead
    of a dict. Missing state fields fall back to the last confirmed state
    (fallback_state) rather than a blank string, so a malformed single
    entry can't silently reset/erase continuity for everything after it.
    """
    normalized=[]
    for item in result:
        if isinstance(item, dict):
            visual_prompt=(
                item.get("visual_prompt")
                or item.get("prompt")
                or item.get("description")
                or item.get("text")
                or "Stickman stands still in a minimalist scene."
            )
            normalized.append({
                "location": item.get("location") or fallback_state["location"],
                "lighting": item.get("lighting") or fallback_state["lighting"],
                "established_facts": item.get("established_facts") or fallback_state["established_facts"],
                "visual_prompt": str(visual_prompt),
            })
        elif isinstance(item, str):
            normalized.append({
                "location": fallback_state["location"],
                "lighting": fallback_state["lighting"],
                "established_facts": fallback_state["established_facts"],
                "visual_prompt": item,
            })
        else:
            print(f"[!] Unexpected item shape in visual prompt batch: {item}")
            normalized.append({
                "location": fallback_state["location"],
                "lighting": fallback_state["lighting"],
                "established_facts": fallback_state["established_facts"],
                "visual_prompt": "Stickman stands still in a minimalist scene.",
            })

    # pad if the model returned fewer entries than requested, carrying state forward
    for i in range(len(normalized), len(batch)):
        normalized.append({
            "location": fallback_state["location"],
            "lighting": fallback_state["lighting"],
            "established_facts": fallback_state["established_facts"],
            "visual_prompt": "Stickman stands still in a minimalist scene.",
        })

    return normalized[:len(batch)]


def generate_visual_prompts(
    chunks: list[str],
    brief: dict,
    prose: str,
    groq_client: Groq,
    gemini_client,
) -> list[dict]:
    print(f"\n[+] Generating visual prompts in batches of {VISUAL_BATCH_SIZE} (with world-state tracking) …")
    all_items: list[dict]=[]
    total_batches=((len(chunks) - 1) // VISUAL_BATCH_SIZE) + 1

    current_state={
        "location": brief.get("setting", "Unspecified setting"),
        "lighting": brief.get("color_mood", "Natural lighting appropriate to the scene"),
        "established_facts": "None established yet — infer sensibly from the opening of the story and the brief.",
    }

    for batch_idx in range(total_batches):
        start=batch_idx * VISUAL_BATCH_SIZE
        end=start + VISUAL_BATCH_SIZE
        batch=chunks[start:end]
        numbered=[(start + i + 1, chunk) for i, chunk in enumerate(batch)]
        prompt=_visual_batch_prompt(brief, prose, numbered, current_state)

        print(f"[*] Batch {batch_idx + 1}/{total_batches} — chunks {start + 1}–{start + len(batch)} …")
        print(f"    (entering with state: location='{current_state['location']}', "
              f"facts='{current_state['established_facts'][:80]}')")

        result=None

        try:
            resp=groq_client.chat.completions.create(
                model=os.environ["GROQ_MODEL"],
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.5,
            )
            raw=resp.choices[0].message.content
            parsed=_parse_json(raw)
            result=parsed if isinstance(parsed, list) else next(
                v for v in parsed.values() if isinstance(v, list)
            )

        except Exception as e:
            print(f"[!] Groq failed for batch {batch_idx + 1}: {e}")
            print(f"[*] Falling back to Gemini for batch {batch_idx + 1} …")

            try:
                resp=gemini_client.models.generate_content(
                    model=os.environ["GEMINI_MODEL"],
                    contents=prompt,
                    config={"response_mime_type": "application/json"},
                )
                parsed=_parse_json(resp.text)
                result=parsed if isinstance(parsed, list) else next(
                    v for v in parsed.values() if isinstance(v, list)
                )

            except Exception as e2:
                raise RuntimeError(f"Both failed for visual batch {batch_idx + 1}: {e2}")

        items=_normalize_visual_items(result, batch, current_state)
        all_items.extend(items)

        # carry the last chunk's confirmed state forward into the next batch
        current_state={
            "location": items[-1]["location"],
            "lighting": items[-1]["lighting"],
            "established_facts": items[-1]["established_facts"],
        }

        print(f"[✓] Batch {batch_idx + 1} done — {len(items)} prompts. "
              f"Exiting state: location='{current_state['location']}'")

    return all_items


# ── Step 4: merge + decorate ──────────────────────────────────────────────

def _jaccard(a: str, b: str) -> float:
    a_words=set(a.lower().split())
    b_words=set(b.lower().split())
    if not a_words or not b_words:
        return 0.0
    return len(a_words & b_words) / len(a_words | b_words)


def build_scenes(chunks: list[str], visual_items: list[dict], similarity_threshold: float=0.85) -> list[dict]:
    raw=[]
    for narrative, item in zip(chunks, visual_items):
        raw.append({
            "narrative": narrative,
            "visual_prompt": item.get("visual_prompt", ""),
            "location": item.get("location", ""),
            "lighting": item.get("lighting", ""),
            "established_facts": item.get("established_facts", ""),
        })

    decorated=[]
    removed=0

    for i, scene in enumerate(raw):
        is_dup=any(
            _jaccard(scene["narrative"], kept["narrative"]) >= similarity_threshold
            for kept in decorated
        )
        if is_dup:
            removed += 1
            print(f"[~] Duplicate dropped (index {i}): \"{scene['narrative'][:60]}…\"")
            continue

        decorated.append({
            "sequence": len(decorated) + 1,
            "narrative": scene["narrative"],
            "visual_prompt": scene["visual_prompt"],
            "location": scene["location"],
            "lighting": scene["lighting"],
            "established_facts": scene["established_facts"],
            "start_time": 0.0,
            "end_time": 0.0,
            "audio_file": "",
        })

    print(f"[✓] Scenes built — {len(decorated)} kept, {removed} duplicate(s) dropped.")
    return decorated


# ── Step 5: metadata ───────────────────────────────────────────────────────

def _metadata_prompt(concept: str, scenes: list) -> str:
    digest_scenes=scenes[:1] + scenes[4::5] + scenes[-1:]
    digest_text=" … ".join(s["narrative"] for s in digest_scenes)

    return f"""
You are a YouTube SEO and marketing expert.

Below is a brief digest of a YouTube animation story about "{concept}":

STORY DIGEST:
{digest_text}

Generate high-converting YouTube metadata for this video.
Return ONLY valid JSON — no markdown, no preamble.

{{
  "suggested_titles": [
    "High-CTR clickbait title option 1",
    "High-CTR conversational title option 2"
  ],
  "seo_description": "A comprehensive hook-first description (150-300 words) summarising the video and encouraging viewers to watch.",
  "tags": ["relevant", "seo", "tags", "storytime", "animation"],
  "thumbnail_concept": "A split-screen minimalist stickman cartoon scene capturing the most dramatic moment of the story."
}}
"""


def generate_metadata(concept: str, scenes: list, gemini_client, groq_client: Groq) -> dict:
    print(f"\n[+] Generating YouTube metadata …")
    prompt=_metadata_prompt(concept, scenes)

    try:
        print("[*] Trying Gemini for metadata …")
        resp=gemini_client.models.generate_content(
            model=os.environ["GEMINI_MODEL"],
            contents=prompt,
            config={"response_mime_type": "application/json"},
        )
        metadata=_parse_json(resp.text)
        print("[✓] Gemini metadata ready.")
        return metadata

    except Exception as e:
        print(f"[!] Gemini failed for metadata: {e}")
        print("[*] Falling back to Groq for metadata …")

    try:
        resp=groq_client.chat.completions.create(
            model=os.environ["GROQ_MODEL"],
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.7,
        )
        metadata=_parse_json(resp.choices[0].message.content)
        print("[✓] Groq fallback metadata ready.")
        return metadata

    except Exception as e:
        raise RuntimeError(f"Both Gemini and Groq failed for metadata generation: {e}")


# ── Main pipeline ──────────────────────────────────────────────────────────

def generate_pipeline(topic: str, target_minutes: float) -> dict:
    groq_client=Groq(api_key=os.environ["GROQ_API_KEY"])
    gemini_client=genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    prose, brief=generate_prose(topic, target_minutes, groq_client, gemini_client)
    chunks=chunk_prose(prose, sentences_per_scene=1)
    visual_items=generate_visual_prompts(chunks, brief, prose, groq_client, gemini_client)
    scenes=build_scenes(chunks, visual_items, similarity_threshold=0.92)
    metadata=generate_metadata(topic, scenes, gemini_client, groq_client)

    return {
        "concept": topic,
        "target_duration_minutes": target_minutes,
        "brief": brief,
        "metadata": metadata,
        "scenes": scenes,
    }


if __name__ == "__main__":
    print("=" * 52)
    print("DYNAMIC AUTOMATED YOUTUBE PIPELINE ENGINE")
    print("=" * 52)

    user_topic=input("[?] Enter your video concept/topic: ").strip()

    try:
        duration_input=input("[?] Target length in minutes (e.g. 1, 5, 8): ").strip()
        user_minutes=float(duration_input) if duration_input else 1.0
    except ValueError:
        print("[!] Invalid number — defaulting to 1.0 minute.")
        user_minutes=1.0

    try:
        package=generate_pipeline(user_topic, user_minutes)
        target_folder=create_project_folder(user_topic)
        save_path=os.path.join(target_folder, "video_package.json")

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(package, f, indent=2, ensure_ascii=False)

        print(f"\n[✓] Package saved successfully.")
        print(f"[*] Total scenes  : {len(package['scenes'])}")
        print(f"[*] Workspace root: {target_folder}")

    except Exception as e:
        print(f"\n[X] Pipeline error: {e}")
        raise