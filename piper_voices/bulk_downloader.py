import os
from huggingface_hub import snapshot_download

# --- Configuration ---
REPO_ID = "rhasspy/piper-voices"
LOCAL_DIR = os.path.expanduser("~/piper_all_en_voices")
INCLUDE_PATTERNS = [
    "en/**/*.onnx",   # Files in subdirectories of en
    "en/**/*.json",   # Files in subdirectories of en
]
# CRUCIAL FIX: Change "dataset" to "model"
REPO_TYPE = "model" 


# Create the local directory if it doesn't exist
os.makedirs(LOCAL_DIR, exist_ok=True)
print(f"Starting download to: {LOCAL_DIR}")

# --- Download Logic ---
try:
    snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE, # <-- FIX applied here
        allow_patterns=INCLUDE_PATTERNS,
        local_dir=LOCAL_DIR,
        local_dir_use_symlinks=False,
    )
    print("\n✅ Bulk download complete!")
    print(f"All English voices are saved in: {LOCAL_DIR}")

except Exception as e:
    print(f"\n❌ An error occurred during download: {e}")
