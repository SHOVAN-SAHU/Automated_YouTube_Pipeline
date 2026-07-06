# build_video.py

import os
import json
import re
from moviepy.editor import (
    ImageClip, AudioFileClip, concatenate_videoclips,
    TextClip, CompositeVideoClip
)
from moviepy.config import change_settings

change_settings({"IMAGEMAGICK_BINARY": r"D:\ImageMagick-7.1.2-Q16-HDRI\magick.exe"})

BASE_DIR = r"D:\Automated_YouTube_Pipeline"
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

WORDS_PER_CHUNK = 6
FONT = "Arial-Bold"
FONT_SIZE = 40  # was 52 — much smaller now
FONT_COLOR = "white"
BG_COLOR = "rgba(0,0,0,0.6)"
STROKE_COLOR = "black"
STROKE_WIDTH = 0  # Text border
SUBTITLE_X = 80  # pixels from left
SUBTITLE_Y_OFFSET = 80  # pixels from bottom — moved up so nothing gets cut

# scene image files can come out of different generators with different
# extensions (.png, .jpg, .jpeg, .webp) — match any of them, not just .png
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")


def get_latest_project_folder() -> str | None:
    folders = [
        os.path.join(OUTPUTS_DIR, d)
        for d in os.listdir(OUTPUTS_DIR)
        if os.path.isdir(os.path.join(OUTPUTS_DIR, d))
    ]
    return max(folders, key=os.path.getmtime) if folders else None


def concept_to_filename(concept: str) -> str:
    sanitized = re.sub(r"[^\w\s]", "", concept)
    sanitized = re.sub(r"\s+", "_", sanitized.strip())
    return f"{sanitized}.mp4"


def build_word_clips(narrative: str, scene_start: float, scene_duration: float, video_height: int) -> list:
    """
    Splits narrative into chunks of exactly WORDS_PER_CHUNK whole words.
    Never splits mid-word. Each chunk gets an equal slice of scene duration.
    """
    # Split into whole words cleanly
    words = narrative.strip().split()

    if not words:
        return []

    # Build chunks of whole words only — no mid-word splits
    chunks = []
    for i in range(0, len(words), WORDS_PER_CHUNK):
        chunk = " ".join(words[i : i + WORDS_PER_CHUNK])
        chunks.append(chunk)

    chunk_duration = scene_duration / len(chunks)
    clips = []

    for idx, chunk in enumerate(chunks):
        chunk_start = scene_start + idx * chunk_duration

        txt_clip = (
            TextClip(
                chunk,
                fontsize=FONT_SIZE,
                font=FONT,
                color=FONT_COLOR,
                stroke_color=STROKE_COLOR,
                stroke_width=STROKE_WIDTH,
                bg_color=BG_COLOR,
                method="label",
                align="West",   # left-align text within the clip
            )
            .set_start(chunk_start)
            .set_duration(chunk_duration)
            .set_position((SUBTITLE_X, video_height - SUBTITLE_Y_OFFSET))
        )
        clips.append(txt_clip)

    return clips


def main():
    print("=" * 52)
    print("DYNAMIC MOVIEPY VIDEO COMPILER")
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

    json_path  = os.path.join(project_path, "video_package.json")
    audio_path = os.path.join(project_path, "final_voiceover.wav")
    images_dir = os.path.join(project_path, "scene_images")

    if not os.path.exists(images_dir):
        print(f"[X] 'scene_images' folder missing at: {images_dir}")
        return
    if not os.path.exists(json_path):
        print(f"[X] video_package.json not found at: {json_path}")
        return
    if not os.path.exists(audio_path):
        print(f"[X] final_voiceover.wav not found at: {audio_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    concept = data.get("concept", "output_video")
    output_name = concept_to_filename(concept)
    output_path = os.path.join(project_path, output_name)

    print(f"\n[*] Project  : {os.path.basename(project_path)}")
    print(f"[*] Concept  : {concept}")
    print(f"[*] Output   : {output_name}")
    print(f"[*] Scenes   : {len(data['scenes'])}")

    all_files   = os.listdir(images_dir)
    video_clips = []

    for scene in data["scenes"]:
        seq = scene["sequence"]
        start = scene["start_time"]
        end = scene["end_time"]
        duration = round(end - start, 2)

        if duration <= 0:
            continue

        prefix = f"scene_{seq}_{start}_"
        matched_image = next(
            (f for f in all_files
             if f.lower().startswith(prefix) and f.lower().endswith(IMAGE_EXTENSIONS)),
            None
        )

        if matched_image:
            img_path = os.path.join(images_dir, matched_image)
            clip = (
                ImageClip(img_path)
                .set_duration(duration)
                .resize((1920, 1080))
            )
            video_clips.append(clip)
            print(f"[+] Packed scene {seq} | {duration}s → {matched_image}")
        else:
            print(f"[!] Missing image for scene {seq} (expected prefix: {prefix})")

    if not video_clips:
        print("[X] No valid image clips found. Aborting.")
        return

    print(f"\n[*] Stitching {len(video_clips)} image clips…")
    base_video = concatenate_videoclips(video_clips, method="compose")
    video_w, video_h = base_video.size

    print(f"[*] Building subtitle track…")
    subtitle_clips = []
    elapsed = 0.0

    for scene in data["scenes"]:
        duration = round(scene["end_time"] - scene["start_time"], 2)
        if duration <= 0:
            continue

        narrative  = scene.get("narrative", "")
        word_clips = build_word_clips(narrative, elapsed, duration, video_h)
        subtitle_clips.extend(word_clips)
        elapsed += duration

    if subtitle_clips:
        final_video = CompositeVideoClip([base_video] + subtitle_clips)
    else:
        final_video = base_video

    print(f"[*] Attaching voiceover audio…")
    audio_clip = AudioFileClip(audio_path)
    final_video.audio = audio_clip

    print(f"[*] Encoding to HD MP4 — this may take a few minutes…")
    final_video.write_videofile(
        output_path,
        fps=24,
        codec="libx264",
        audio_codec="aac",
        bitrate="8000k",
        audio_bitrate="192k",
        ffmpeg_params=["-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",],
        threads=4,
        logger="bar",
    )

    audio_clip.close()
    final_video.close()

    print(f"\n[✓] Render complete!")
    print(f"[✓] Saved to: {output_path}")


if __name__ == "__main__":
    main()