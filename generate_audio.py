# generate_audio.py

import os
import json
import asyncio
import edge_tts
from pydub import AudioSegment
from pydub.effects import strip_silence

BASE_DIR = r"D:\Automated_YouTube_Pipeline"
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# --- VOICE CONFIGURATION ---
VOICE_MODEL = "en-US-BrianNeural"
VOICE_RATE = "-1%" 

# Explicitly link FFmpeg binaries for Pydub audio processing
FFMPEG_PATH = r"D:\ffmpeg\bin"
os.environ["PATH"] += os.pathsep + FFMPEG_PATH
AudioSegment.converter = os.path.join(FFMPEG_PATH, "ffmpeg.exe")
AudioSegment.ffprobe = os.path.join(FFMPEG_PATH, "ffprobe.exe")

def select_project_folder():
    """Lists available project folders and lets the user choose, defaulting to the latest."""
    if not os.path.exists(OUTPUTS_DIR):
        raise FileNotFoundError(f"Outputs directory not found at: {OUTPUTS_DIR}")
        
    folders = [d for d in os.listdir(OUTPUTS_DIR) if os.path.isdir(os.path.join(OUTPUTS_DIR, d))]
    if not folders:
        raise FileNotFoundError("No project workspace folders found inside outputs directory.")
    
    full_paths = [os.path.join(OUTPUTS_DIR, f) for f in folders]
    latest_folder = os.path.basename(max(full_paths, key=os.path.getmtime))
    
    print("\n--- Available Project Workspaces ---")
    for idx, folder in enumerate(folders, 1):
        suffix = " (Most Recent)" if folder == latest_folder else ""
        print(f"[{idx}] {folder}{suffix}")
    print("-------------------------------------")
    
    user_input = input(f"\nSelect a folder name or number [Press Enter for '{latest_folder}']: ").strip()
    
    if not user_input:
        return os.path.join(OUTPUTS_DIR, latest_folder)
    
    if user_input.isdigit():
        idx = int(user_input) - 1
        if 0 <= idx < len(folders):
            return os.path.join(OUTPUTS_DIR, folders[idx])
    
    if user_input in folders:
        return os.path.join(OUTPUTS_DIR, user_input)
        
    print(f"[!] Invalid selection. Defaulting to most recent: {latest_folder}")
    return os.path.join(OUTPUTS_DIR, latest_folder)

async def generate_scene_audio(text, temp_mp3_path):
    """Communicates with Edge-TTS to synthesize and save as an MP3 stream."""
    communicate = edge_tts.Communicate(text, VOICE_MODEL, rate=VOICE_RATE)
    await communicate.save(temp_mp3_path)

def process_and_convert_audio(temp_mp3_path, final_wav_path):
    """Loads the MP3, strips silences natively, saves a crisp WAV copy, and returns duration."""
    raw_audio = AudioSegment.from_mp3(temp_mp3_path)
    
    # Shave off the trailing dead air cleanly
    trimmed_audio = strip_silence(raw_audio, silence_thresh=-45, padding=30)
    trimmed_audio.export(final_wav_path, format="wav")
    
    duration_secs = round(len(trimmed_audio) / 1000.0, 2)
    
    if os.path.exists(temp_mp3_path):
        os.remove(temp_mp3_path)
        
    return duration_secs

async def main():
    try:
        project_path = select_project_folder()
    except FileNotFoundError as e:
        print(f"[X] Error: {e}")
        return

    json_path = os.path.join(project_path, "video_package.json")
    
    # Enforce output directly to the 'converted' folder to perfectly match stitch_audio.py
    converted_dir = os.path.join(project_path, "converted")
    os.makedirs(converted_dir, exist_ok=True)

    if not os.path.exists(json_path):
        print(f"[X] Error: 'video_package.json' not found at: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "scenes" not in data or not data["scenes"]:
        print("[X] Error: No scenes found in the JSON file.")
        return

    print(f"\n[*] Active Workspace: {os.path.basename(project_path)}")
    print(f"[*] Generating zero-silence continuous audio files inside 'converted' via {VOICE_MODEL}...")

    running_timeline_secs = 0.0

    for scene in data["scenes"]:
        seq = scene.get("sequence")
        text = scene.get("narrative")
        
        if not text:
            print(f"[!] Warning: Scene {seq} has no narrative text. Skipping.")
            continue
        
        temp_mp3 = os.path.join(converted_dir, f"scene_{seq}_temp.mp3")
        final_wav = os.path.join(converted_dir, f"scene_{seq}.wav")

        print(f"[+] Processing Scene {seq}...")
        try:
            await generate_scene_audio(text, temp_mp3)
            duration_secs = process_and_convert_audio(temp_mp3, final_wav)
            
            # Map clean, continuous timelines sequentially directly into the json file
            scene["start_time"] = round(running_timeline_secs, 2)
            scene["end_time"] = round(scene["start_time"] + duration_secs, 2)
            
            # Pointing path dynamically to the final folder structure
            scene["audio_file"] = f"converted/scene_{seq}.wav"
            
            running_timeline_secs = scene["end_time"]
            print(f"    -> Timeline Map: {scene['start_time']}s -> {scene['end_time']}s")
        except Exception as e:
            print(f" [!] Failed to process audio for Scene {seq}: {e}")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n[✓] Audio generation complete! 'video_package.json' has been updated with final timelines.")
    print(f"[*] Saved folder: {converted_dir}")
    print(f"[*] Total Voice Runtime Track: {round(running_timeline_secs, 2)} seconds")

if __name__ == "__main__":
    asyncio.run(main())