# This file add all the .wev files into a bigger file.

import os
import json
import random
from pydub import AudioSegment

BASE_DIR = r"D:\Automated_YouTube_Pipeline"
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# --- PACING CONTROLS (Change these numbers to adjust the silences) ---
MIN_GAP = 0.2   # Minimum silence in seconds
MAX_GAP = 0.35  # Maximum silence in seconds

# Explicitly link FFmpeg binaries
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
    
    # Get the latest folder based on modification time
    full_paths = [os.path.join(OUTPUTS_DIR, f) for f in folders]
    latest_folder = os.path.basename(max(full_paths, key=os.path.getmtime))
    
    print("\n--- Available Project Workspaces ---")
    for idx, folder in enumerate(folders, 1):
        suffix = " (Most Recent)" if folder == latest_folder else ""
        print(f"[{idx}] {folder}{suffix}")
    print("-------------------------------------")
    
    user_input = input(f"\nSelect a folder name or number [Press Enter for '{latest_folder}']: ").strip()
    
    # Handle blank input (Default to latest)
    if not user_input:
        return os.path.join(OUTPUTS_DIR, latest_folder)
    
    # Handle numeric selection
    if user_input.isdigit():
        idx = int(user_input) - 1
        if 0 <= idx < len(folders):
            return os.path.join(OUTPUTS_DIR, folders[idx])
    
    # Handle direct text input matching
    if user_input in folders:
        return os.path.join(OUTPUTS_DIR, user_input)
        
    print(f"[!] Invalid selection. Defaulting to most recent: {latest_folder}")
    return os.path.join(OUTPUTS_DIR, latest_folder)

def main():
    try:
        project_path = select_project_folder()
    except FileNotFoundError as e:
        print(f"[X] Error: {e}")
        return

    converted_dir = os.path.join(project_path, "converted")
    json_path = os.path.join(project_path, "video_package.json")
    
    if not os.path.exists(converted_dir):
        print(f"[X] Error: 'converted' folder missing at: {converted_dir}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"\n[*] Active Workspace: {os.path.basename(project_path)}")
    print(f"[*] Compiling narrative track with organic pacing...")
    
    master_track = AudioSegment.empty()
    running_timeline_secs = 0.0
    actual_files = os.listdir(converted_dir)

    for scene in data["scenes"]:
        seq = scene["sequence"]
        
        matched_file = None
        for f_name in actual_files:
            if f_name.lower().startswith(f"scene_{seq}.") or f_name.lower().startswith(f"scene_{seq}_"):
                matched_file = f_name
                break
        
        if matched_file:
            file_path = os.path.join(converted_dir, matched_file)
            scene_audio = AudioSegment.from_wav(file_path)
            
            # Lock in the exact start time for this scene
            scene["start_time"] = round(running_timeline_secs, 2)
            spoken_duration_secs = len(scene_audio) / 1000.0
            
            # GENERATE RANDOM SILENCE FOR THIS SPECIFIC SCENE
            random_gap_secs = round(random.uniform(MIN_GAP, MAX_GAP), 2)
            silence_padding_ms = int(random_gap_secs * 1000)
            silence_segment = AudioSegment.silent(duration=silence_padding_ms)
            
            # Glue vocal and the random silence segment together
            master_track += scene_audio + silence_segment
            
            # Calculate the final updated end time
            total_block_duration = spoken_duration_secs + random_gap_secs
            scene["end_time"] = round(scene["start_time"] + total_block_duration, 2)
            
            running_timeline_secs = scene["end_time"]
            print(f"[+] Synced {matched_file} | Voice: {round(spoken_duration_secs, 2)}s | Added Pause: {random_gap_secs}s | Timeline: {scene['start_time']}s -> {scene['end_time']}s")
        else:
            print(f"[!] Warning: Could not find file for scene_{seq}")

    # Export final master audio track
    output_audio_path = os.path.join(project_path, "final_voiceover.wav")
    master_track.export(output_audio_path, format="wav")
    
    # Clean up contiguous transitions for the video manifest
    for i in range(len(data["scenes"]) - 1):
        data["scenes"][i]["end_time"] = data["scenes"][i+1]["start_time"]

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        
    print(f"\n[✓] Master track compiled flawlessly with dynamic pauses: final_voiceover.wav")
    print(f"[*] Total Video Timeline Runtime: {round(len(master_track) / 1000.0, 2)} seconds")

if __name__ == "__main__":
    main()