"""
Pipeline:
  Groq call #1  → plain prose story + story brief
  Python        → sentence-pair chunking (no LLM)
  Groq call #2  → visual_prompts in batches of 12 (brief + prose + chunks)
  Python        → merge into scenes[], decorate
  Gemini        → metadata (Groq fallback)
"""

# Next story on - Ancient humans fire invension and cooking meat

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
    target_words=int(target_minutes * 140)
    min_words=int(target_words * 0.85)
    max_words=int(target_words * 1.15)

    return f"""
You are a YouTube video scriptwriter.

TASK
----
Write a continuous story about: "{topic}"
Then produce a compact story brief.

STORY RULES
-----------
- Hook the viewer in the first sentence with internal conflict or dramatic irony.
- If the topic is historical: strict chronological order, no invented dialogue, accurate facts only.
- If fictional: real emotional stakes, wit, or tension throughout.
- Ultra-simple colloquial vocabulary. Short punchy sentences. No flowery prose.
- Every sentence must advance the story forward. Never repeat or rephrase a prior idea.
- Every sentence must be unique in meaning. No two sentences may describe the same moment, action, or setting even in different words.
- Total word count MUST be between {min_words} and {max_words} words.
- Clear arc: setup → rising tension → climax → resolution.

BRIEF RULES
-----------
The brief is used later to generate consistent stickman illustrations.
Keep it tight — one line per field.

OUTPUT FORMAT
-------------
Return your response using EXACTLY these two delimiters, nothing else outside them.

<PROSE>
Full story as plain flowing text. No JSON, no bullet points.
</PROSE>
<BRIEF>
{{"main_character": "Who the story follows (name, role, one descriptor).", "setting": "Where and when the story takes place.", "visual_style": "minimalist stickman doodle, black and white", "tone": "One or two words describing the emotional tone.", "key_props": "Comma-separated list of 4-6 recurring visual elements."}}
</BRIEF>
"""


def _parse_prose_response(raw: str) -> tuple[str, dict]:
    prose_match=re.search(r"<PROSE>(.*?)</PROSE>", raw, re.DOTALL)
    brief_match=re.search(r"<BRIEF>(.*?)</BRIEF>", raw, re.DOTALL)
    if not prose_match or not brief_match:
        raise ValueError("Response missing <PROSE> or <BRIEF> delimiters.")
    prose=prose_match.group(1).strip()
    brief=json.loads(brief_match.group(1).strip())
    return prose, brief


def generate_prose(topic: str, target_minutes: float, groq_client: Groq, gemini_client) -> tuple[str, dict]:
    prompt=_prose_prompt(topic, target_minutes)
    target_words=int(target_minutes * 140)
    print(f"\n[+] Generating prose story (~{target_words} words) …")

    try:
        print("[*] Trying Groq for prose …")
        resp=groq_client.chat.completions.create(
            model=os.environ["GROQ_MODEL"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
        )
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
        )
        prose, brief=_parse_prose_response(resp.text)
        print(f"[✓] Gemini prose fallback done — {len(prose.split())} words.")
        return prose, brief

    except Exception as e:
        raise RuntimeError(f"Both Groq and Gemini failed for prose generation: {e}")


# ── Step 2: Python chunking (no LLM) ──────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    # split on period/exclamation/question followed by whitespace or end
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


# ── Step 3: batched visual prompts ────────────────────────────────────────

def _visual_batch_prompt(brief: dict, prose: str, numbered_chunks: list[tuple[int, str]]) -> str:
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

TASK
----
For each numbered chunk below, write exactly one "visual_prompt".
Each visual_prompt must:
- Describe a single minimalist stickman cartoon panel.
- Be specific to THAT chunk's moment — not generic.
- Reference the character, setting, or props from the brief where relevant.
- Be one sentence only.

CHUNKS TO ILLUSTRATE
--------------------
{chunks_text}

OUTPUT FORMAT
-------------
Return ONLY a valid JSON array in the same order as the chunks. No extra keys.

[
  {{"visual_prompt": "..."}},
  {{"visual_prompt": "..."}}
]
"""


def generate_visual_prompts(
    chunks: list[str],
    brief: dict,
    prose: str,
    groq_client: Groq,
    gemini_client,
) -> list[str]:
    print(f"\n[+] Generating visual prompts in batches of {VISUAL_BATCH_SIZE} …")
    all_prompts=[]
    total_batches=((len(chunks) - 1) // VISUAL_BATCH_SIZE) + 1

    for batch_idx in range(total_batches):
        start=batch_idx * VISUAL_BATCH_SIZE
        end=start + VISUAL_BATCH_SIZE
        batch=chunks[start:end]
        numbered=[(start + i + 1, chunk) for i, chunk in enumerate(batch)]
        prompt=_visual_batch_prompt(brief, prose, numbered)

        print(f"[*] Batch {batch_idx + 1}/{total_batches} — chunks {start + 1}–{start + len(batch)} …")

        result=None

        try:
            resp=groq_client.chat.completions.create(
                model=os.environ["GROQ_MODEL"],
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.5,
            )
            # Groq returns json_object so wrap array responses need detection
            raw=resp.choices[0].message.content
            parsed=_parse_json(raw)
            # handle both {"items": [...]} and plain [...] responses
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

        prompts=[item["visual_prompt"] for item in result]

        # pad or trim to match batch size exactly
        if len(prompts) < len(batch):
            prompts += ["Stickman stands still in a minimalist scene."] * (len(batch) - len(prompts))
        prompts=prompts[:len(batch)]

        all_prompts.extend(prompts)
        print(f"[✓] Batch {batch_idx + 1} done — {len(prompts)} prompts.")

    return all_prompts


# ── Step 4: merge + decorate ──────────────────────────────────────────────

def _jaccard(a: str, b: str) -> float:
    a_words=set(a.lower().split())
    b_words=set(b.lower().split())
    if not a_words or not b_words:
        return 0.0
    return len(a_words & b_words) / len(a_words | b_words)


def build_scenes(chunks: list[str], visual_prompts: list[str], similarity_threshold: float=0.85) -> list[dict]:
    raw=[]
    for narrative, visual_prompt in zip(chunks, visual_prompts):
        raw.append({"narrative": narrative, "visual_prompt": visual_prompt})

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

    # 1. prose + brief from Groq (Gemini fallback)
    prose, brief=generate_prose(topic, target_minutes, groq_client, gemini_client)

    # 2. Python chunking — no LLM
    chunks=chunk_prose(prose, sentences_per_scene=1)

    # 3. visual prompts in batches — Groq primary, Gemini fallback per batch
    visual_prompts=generate_visual_prompts(chunks, brief, prose, groq_client, gemini_client)

    # 4. merge + deduplicate
    scenes=build_scenes(chunks, visual_prompts, similarity_threshold=0.92)

    # 5. metadata — Gemini primary, Groq fallback
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