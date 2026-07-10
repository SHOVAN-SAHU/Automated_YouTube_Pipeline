import os
import asyncio
import edge_tts

# --- CONFIGURATION ---
TEST_TEXT = "Hello! This is a test of the automated voice synthesis system. How does this narrator sound?"
OUTPUT_DIR = "voice_samples"

async def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("[*] Contacting Edge-TTS server to fetch available voice profiles...")
    try:
        # Fetch all raw voices globally
        all_voices = await edge_tts.list_voices()
    except Exception as e:
        print(f"[X] Failed to fetch voice list: {e}")
        return

    # Foolproof filter: Grab any voice that starts with "en-" (English variants)
    english_voices = [v for v in all_voices if v["Locale"].lower().startswith("en-")]
    # Sort them alphabetically by name
    english_voices = sorted(english_voices, key=lambda v: v["ShortName"])

    if not english_voices:
        print("[X] Truly no English voices could be parsed. Check your internet connection.")
        return

    print(f"\n[✓] Successfully detected {len(english_voices)} English voice profiles.")
    print(f"[*] Generating sample audio tracks inside the '{OUTPUT_DIR}' folder...\n")

    print("--- Voice Generation Progress ---")
    for idx, voice in enumerate(english_voices, 1):
        short_name = voice["ShortName"]     # e.g., 'en-US-BrianNeural'
        gender = voice["Gender"]             # Male / Female
        locale = voice["Locale"]             # e.g., 'en-US', 'en-GB'
        
        # Format a clean local filename
        filename = f"{short_name.replace('-', '_')}.mp3"
        file_path = os.path.join(OUTPUT_DIR, filename)
        
        print(f"[{idx}/{len(english_voices)}] Synthesizing: {short_name} [{locale}] ({gender})")
        
        try:
            communicate = edge_tts.Communicate(TEST_TEXT, short_name)
            await communicate.save(file_path)
        except Exception as e:
            print(f" [!] Error generating sample for {short_name}: {e}")

    print("\n------------------------------------------------")
    print(f"[✓] Finished generating all voice samples!")
    print(f"[*] Open the folder to listen: {os.path.abspath(OUTPUT_DIR)}")
    print("------------------------------------------------")

if __name__ == "__main__":
    asyncio.run(main())