# This is for after project clean, delete all the .wev and image folders.

import os
import shutil

BASE_DIR = r"D:\Automated_YouTube_Pipeline"
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

def get_latest_project_folder():
    """Finds the most recently modified project folder."""
    folders = [os.path.join(OUTPUTS_DIR, d) for d in os.listdir(OUTPUTS_DIR) if os.path.isdir(os.path.join(OUTPUTS_DIR, d))]
    if not folders:
        return None
    return max(folders, key=os.path.getmtime)

def main():
    print("====================================================")
    print("        AUTOMATED PIPELINE - STORAGE CLEANUP        ")
    print("====================================================")
    
    # 1. Ask the user for a specific folder or default to the latest one
    user_input = input("[?] Enter folder name to clean (Leave blank for most recent): ").strip()
    
    if user_input:
        # Target the specific folder the user typed
        project_path = os.path.join(OUTPUTS_DIR, user_input)
        if not os.path.exists(project_path):
            print(f"[X] Error: The folder '{user_input}' does not exist at {project_path}")
            return
    else:
        # Automatically detect the latest folder
        project_path = get_latest_project_folder()
        if not project_path:
            print("[X] Error: No project folders found inside the outputs directory.")
            return
            
    print(f"\n[*] Target Workspace Selected: {os.path.basename(project_path)}")
    
    # Define the 3 specific folders to wipe out
    folders_to_delete = ["converted", "raw_audio", "scene_images", "metadata", "Short"]
    
    # Confirmation safety prompt
    confirm = input(f"[!] Are you sure you want to delete {folders_to_delete} from this workspace? (y/n): ").strip().lower()
    if confirm != 'y':
        print("[X] Cleanup aborted by user.")
        return

    print("\n[*] Initializing storage purging...")
    
    deleted_count = 0
    for target in folders_to_delete:
        target_path = os.path.join(project_path, target)
        
        if os.path.exists(target_path):
            try:
                # shutil.rmtree deletes a folder and everything inside it cleanly
                shutil.rmtree(target_path)
                print(f"[✓] Deleted folder: {target}/")
                deleted_count += 1
            except Exception as e:
                print(f"[X] Failed to delete {target}/. Error: {e}")
        else:
            print(f"[-] Skipped: {target}/ (Folder did not exist or was already deleted)")

    print(f"\n[✓] Cleanup process complete! Wiped {deleted_count} asset cache directories.")
    print(f"[*] Kept safe: 'video_package.json' and 'final_output_video.mp4' remain in the root.")

if __name__ == "__main__":
    main()
