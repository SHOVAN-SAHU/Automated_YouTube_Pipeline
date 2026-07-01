# Only use this if we're gonna record the oudio ourselvs.

import os
import json
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np

BASE_DIR = r"D:\Automated_YouTube_Pipeline"
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

def get_latest_project_folder():
    folders = [os.path.join(OUTPUTS_DIR, d) for d in os.listdir(OUTPUTS_DIR) if os.path.isdir(os.path.join(OUTPUTS_DIR, d))]
    if not folders:
        raise FileNotFoundError("No project folders found! Run Phase 1 first.")
    return max(folders, key=os.path.getmtime)

def record_scene_audio(text, output_path, sample_rate=16000):
    """Records a single line of dialogue cleanly without needing Ctrl+C traps."""
    print(f"\n👉 NEXT LINE TO READ:\n[ \" {text} \" ]")
    input("\n[▶] Press Enter to START recording...")
    
    print("🔴 RECORDING... (Press Enter again to STOP speaking)")
    audio_data = []
    
    def callback(indata, frames, time, status):
        audio_data.append(indata.copy())

    # Start a non-blocking recording stream
    stream = sd.InputStream(samplerate=sample_rate, channels=1, callback=callback)
    with stream:
        input() # Wait right here until the user hits Enter to stop
    
    print("⏹️ Stopped.")
    flat_audio = np.concatenate(audio_data, axis=0)
    wav.write(output_path, sample_rate, flat_audio)
    
    # Calculate exact duration based on total samples recorded
    duration = len(flat_audio) / sample_rate
    return round(duration, 2)

def main():
    project_path = get_latest_project_folder()
    json_path = os.path.join(project_path, "video_package.json")
    
    # AUTOMATION FIX: Define and create the raw_audio folder natively
    raw_audio_dir = os.path.join(project_path, "raw_audio")
    os.makedirs(raw_audio_dir, exist_ok=True)

    converted_dir = os.path.join(project_path, "converted")
    os.makedirs(converted_dir, exist_ok=True)
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    print(f"[*] Workspace Active: {os.path.basename(project_path)}")
    print(f"[*] Targeting Output Folder: {raw_audio_dir}")
    print("[*] Switching to Sequential Line-by-Line Recording Mode...")
    
    running_time = 0.0
    
    for scene in data["scenes"]:
        seq = scene["sequence"]
        scene_audio_name = f"scene_{seq}.wav"
        
        # AUTOMATION FIX: Direct file output path into the raw_audio folder
        scene_audio_path = os.path.join(raw_audio_dir, scene_audio_name)
        
        # Record the line and get the exact length of the file
        duration = record_scene_audio(scene["narrative"], scene_audio_path)
        
        # Flawless timeline math
        scene["start_time"] = round(running_time, 2)
        scene["end_time"] = round(running_time + duration, 2)
        
        # We save the relative folder path inside the json manifest for tracking
        scene["audio_file"] = f"raw_audio/{scene_audio_name}"
        
        running_time += duration
        print(f"[✓] Saved -> raw_audio/{scene_audio_name} | Duration: {duration}s | Timeline: {scene['start_time']}s -> {scene['end_time']}s")
        
    # Save the updated JSON manifest
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        
    print(f"\n[✓] Video package successfully built! Files organized directly for Applio batch conversion.")

if __name__ == "__main__":
    main()