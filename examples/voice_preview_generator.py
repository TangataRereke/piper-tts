import os
import json
import subprocess
from pathlib import Path
import requests
import librosa
import numpy as np
from pydub import AudioSegment
import sys

# --- Configuration (Based on your system structure) ---
VOICES_ROOT_DIR = Path("/home/james/source/piper-tts/piper_voices/")
EXAMPLES_OUTPUT_DIR = Path("/home/james/source/piper-tts/examples/")
OUTPUT_INDEX_FILE = EXAMPLES_OUTPUT_DIR / "voice_portfolio_index.html" # Renamed to Index
SAMPLES_SUBDIR = "samples"
VOICES_JSON_URL = "https://huggingface.co/rhasspy/piper-voices/raw/main/v1.0.0/voices.json"

# !!! IMPORTANT: Uses the shell name 'piper' as requested !!!
PIPER_EXECUTABLE_PATH = "piper"

# --- Utility Dictionaries ---
QUALITY_ORDER = {"high": 3, "medium": 2, "low": 1, "x_low": 0}
SAMPLE_TEXT = "The quick brown fox jumps over the lazy dog."
SAMPLE_WORD_COUNT = len(SAMPLE_TEXT.split())

# --- Analysis, Conversion, and Execution Functions ---

def get_speaker_name_map():
    """Downloads and extracts the official Piper speaker name mapping for L2-ARCTIC."""
    try:
        response = requests.get(VOICES_JSON_URL, timeout=10)
        response.raise_for_status()
        voices_data = response.json()
        
        for key, voice in voices_data.items():
            if voice.get("key") == "en_US-l2arctic":
                speaker_names = voice.get("speakers", [])
                return {index: name for index, name in enumerate(speaker_names)}
        return {}
    except Exception:
        return {}

def analyze_audio_features(file_path):
    """Calculates Pitch, Gender/Age, Accent Variance, Speaking Rate, and Loudness (RMS)."""
    try:
        y, sr = librosa.load(file_path, sr=None)
        
        # 1. Pitch (F0) Analysis - FIX APPLIED HERE
        f0, voiced_flag, voiced_probs = librosa.pyin( 
            y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C6'), sr=sr
        )
        f0_clean = f0[voiced_flag]
        
        # 2. Loudness (RMS Energy) Calculation
        rms_energy = librosa.feature.rms(y=y)[0]
        loudness_db = librosa.amplitude_to_db(np.maximum(1e-10, rms_energy), ref=1.0)
        mean_loudness_db = np.mean(loudness_db)

        if f0_clean.size == 0:
            return "No Vocalization Detected", "Unknown", "N/A", f"{mean_loudness_db:.1f} dB"

        # Calculate Acoustic Metrics
        f0_mean = np.mean(f0_clean)
        f0_std = np.std(f0_clean)
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        centroid_std = np.std(spectral_centroids)
        
        # 3. Gender/Age Classification
        if f0_mean < 165:
            gender_age = "Male (Adult)"
        elif f0_mean < 255:
            gender_age = "Female (Adult)"
        else:
            gender_age = "Child/High Pitch"
            
        # 4. Speaking Rate (Rhythm)
        duration_seconds = librosa.get_duration(y=y, sr=sr)
        speaking_rate_wpm = (SAMPLE_WORD_COUNT / duration_seconds) * 60

        # Return a tuple: (Pitch Metric, Gender, Rate Metric, Loudness Metric)
        metric_string = f"Mean F0: {f0_mean:.0f}Hz | Var: {f0_std:.1f}"
        rate_string = f"Rate: {speaking_rate_wpm:.0f} WPM | Spec Var: {centroid_std:.1f}"
        loudness_string = f"{mean_loudness_db:.1f} dB"
        
        return metric_string, gender_age, rate_string, loudness_string

    except Exception as e:
        error_message = str(e).replace(' ', '_').replace(':', '')
        print(f"Analysis Failed for {file_path}: {e}")
        return f"ERROR: {error_message[:40]}...", "Unknown", "N/A", "N/A"

def convert_to_mp3(wav_path, mp3_path):
    """Converts a WAV file to a high-quality MP3 and deletes the original WAV."""
    try:
        audio = AudioSegment.from_wav(wav_path)
        audio.export(mp3_path, format="mp3", bitrate="128k") 
        os.remove(wav_path)
        return True
    except Exception:
        return False
        
def convert_mp3_to_wav_for_analysis(mp3_path):
    """Converts MP3 to a temporary WAV for reliable analysis."""
    temp_wav_path = mp3_path.with_suffix(".temp.wav")
    try:
        audio = AudioSegment.from_mp3(mp3_path)
        audio.export(temp_wav_path, format="wav")
        return temp_wav_path
    except Exception:
        return None

def run_piper_command(model_path, output_wav_path, speaker_id=None):
    """Executes the Piper command line tool to generate a sample WAV."""
    
    model_arg = f"-m \"{model_path}\""
    output_arg = f"-f \"{output_wav_path}\""
    speaker_arg = f"-s {speaker_id}" if speaker_id is not None else ""
    command_str = f"echo \"{SAMPLE_TEXT}\" | {str(PIPER_EXECUTABLE_PATH)} {model_arg} {output_arg} {speaker_arg}"

    try:
        subprocess.run(
            command_str,
            input=SAMPLE_TEXT.encode('utf-8'),
            capture_output=True,
            check=True,
            timeout=60,
            shell=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Command Failed: {command_str}")
        print(f"       Piper Stderr: {e.stderr.decode().strip()}")
        return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def generate_voice_portfolio():
    """Generates MP3 samples, analyzes them, and creates the four-page HTML portfolio."""
    
    voices_data = []
    SAMPLES_FULL_PATH = EXAMPLES_OUTPUT_DIR / SAMPLES_SUBDIR

    # 1. Setup Directories and Filtering Map
    os.makedirs(EXAMPLES_OUTPUT_DIR, exist_ok=True)
    os.makedirs(SAMPLES_FULL_PATH, exist_ok=True)
    
    best_quality_map = {}
    for onnx_path in VOICES_ROOT_DIR.rglob("*.onnx"):
        filename = onnx_path.stem
        parts = filename.split('-')
        if parts[-1] in QUALITY_ORDER:
            quality_suffix = parts[-1]
            base_model_name = '-'.join(parts[:-1])
            current_quality_rank = QUALITY_ORDER[quality_suffix]
            if current_quality_rank > best_quality_map.get(base_model_name, -1):
                best_quality_map[base_model_name] = current_quality_rank
    
    speaker_name_map = get_speaker_name_map()

    print(f"--- Starting Portfolio Creation (Processing {len(best_quality_map)} unique models) ---")

    # 2. Iterate, Filter by Quality, Generate MP3, Analyze, and Collect Data
    for onnx_path in VOICES_ROOT_DIR.rglob("*.onnx"):
        filename = onnx_path.stem
        
        # QUALITY FILTER CHECK
        is_quality_model = filename.split('-')[-1] in QUALITY_ORDER
        if is_quality_model:
            quality_suffix = filename.split('-')[-1]
            base_model_name = '-'.join(filename.split('-')[:-1])
            current_quality_rank = QUALITY_ORDER[quality_suffix]
            
            if current_quality_rank != best_quality_map.get(base_model_name):
                continue

        json_path = onnx_path.with_suffix(".onnx.json")
        if not json_path.exists():
            continue

        try:
            with open(json_path, 'r') as f:
                config = json.load(f)
            
            # Extract Speaker Map
            speaker_map = config.get("speaker_id_map")
            speaker_ids = list(speaker_map.values()) if speaker_map and isinstance(speaker_map, dict) else [0]
            model_name = onnx_path.stem.replace(".onnx", "")
            
            
            for index, speaker_id in enumerate(speaker_ids):
                
                # Get speaker name
                speaker_name = speaker_name_map.get(speaker_id)
                speaker_name = speaker_name if speaker_name else str(speaker_id)
                
                # Define WAV and MP3 paths
                wav_filename = f"sample_{model_name}_id_{speaker_id}.wav"
                mp3_filename = f"sample_{model_name}_id_{speaker_id}.mp3"
                
                output_wav_path = SAMPLES_FULL_PATH / wav_filename
                output_mp3_path = SAMPLES_FULL_PATH / mp3_filename
                
                # --- WAV/MP3 GENERATION & ANALYSIS ---
                
                is_newly_generated = False
                if not output_mp3_path.exists() or os.path.getsize(output_mp3_path) < 1000:
                    # 1. Generate new sample
                    print(f"  Generating new sample: {model_name} (ID: {speaker_id})...")
                    if not run_piper_command(onnx_path, output_wav_path, speaker_id):
                        continue
                    is_newly_generated = True

                
                if output_mp3_path.exists() and not is_newly_generated:
                    # 2. Analyze existing MP3 by temporarily converting to WAV
                    temp_wav_path = convert_mp3_to_wav_for_analysis(output_mp3_path)
                    
                    if temp_wav_path and temp_wav_path.exists():
                        accent_metric, gender_age, rate_metric, loudness_metric = analyze_audio_features(temp_wav_path)
                        os.remove(temp_wav_path)
                    else:
                        accent_metric, gender_age, rate_metric, loudness_metric = "ERROR: MP3_Conversion_Failed", "Unknown", "N/A", "N/A"
                
                elif is_newly_generated:
                    # 3. Analyze the high-quality WAV file (from generation)
                    accent_metric, gender_age, rate_metric, loudness_metric = analyze_audio_features(output_wav_path)
                    
                    # 4. Convert WAV to MP3 and delete WAV
                    if not convert_to_mp3(output_wav_path, output_mp3_path):
                        continue
                else:
                    accent_metric, gender_age, rate_metric, loudness_metric = "ERROR: Generation_Skipped_No_MP3", "Unknown", "N/A", "N/A"
                
                # Collect data for HTML
                model_info = {
                    "name": model_name,
                    "language": config.get("language", {}).get("code", "N/A"),
                    "speaker_id": speaker_id,
                    "speaker_name": speaker_name,
                    "quality": str(config.get("audio", {}).get("sample_rate", "N/A")),
                    "sample_file": str(Path(SAMPLES_SUBDIR) / mp3_filename),
                    "accent_data": accent_metric,
                    "gender_age": gender_age,
                    "rate_metric": rate_metric,
                    "loudness_metric": loudness_metric
                }
                voices_data.append(model_info)
            
        except json.JSONDecodeError:
            print(f"Skipping model: {onnx_path.name} (Invalid JSON)")
        except Exception as e:
            print(f"An unexpected error occurred during processing {onnx_path.name}: {e}")


    # 3. Categorize and Generate Pages
    male_voices = [v for v in voices_data if v['gender_age'] == 'Male (Adult)']
    female_voices = [v for v in voices_data if v['gender_age'] == 'Female (Adult)']
    child_voices = [v for v in voices_data if v['gender_age'] == 'Child/High Pitch']
    # Failed voices are those that start with ERROR or Analysis Failed (the index page)
    failed_voices = [v for v in voices_data if v['accent_data'].startswith('ERROR') or v['accent_data'].startswith('Analysis Failed')]
    
    
    # Generate Detail Pages
    detail_pages = [
        ('male_adults.html', male_voices, "Adult Male Voices"),
        ('female_adults.html', female_voices, "Adult Female Voices"),
        ('child_highpitch.html', child_voices, "Child/High Pitch Voices"),
        ('analysis_failed.html', failed_voices, "Analysis/Generation Failed Voices")
    ]
    
    for filename, data, title in detail_pages:
        html_content = build_html_page(data, title)
        with open(EXAMPLES_OUTPUT_DIR / filename, 'w', encoding='utf-8') as f:
            f.write(html_content)

    # Generate Index Page
    generate_index_page(detail_pages)

    print(f"\n✅ FULL Portfolio Process Complete!")
    print(f"Open this file in your browser: {OUTPUT_INDEX_FILE}")

# --- HTML Generators ---

def generate_index_page(detail_pages):
    """Generates the simple index page with links to categories."""
    links_html = ""
    for filename, data, title in detail_pages:
        links_html += f'<li><a href="{filename}">{title} ({len(data)} Samples)</a></li>\n'
    
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Piper Voice Portfolio Index</title>
        <style>body {{ font-family: sans-serif; margin: 40px; }}</style>
    </head>
    <body>
        <h1>Piper Voice Portfolio Index</h1>
        <p>Select a category to view the full list of generated and analyzed voice samples:</p>
        <ul>{links_html}</ul>
        <p>The **Analysis/Generation Failed Voices** page contains samples that could not be fully processed. Please check this page for error details.</p>
    </body>
    </html>
    """
    with open(OUTPUT_INDEX_FILE, 'w', encoding='utf-8') as f:
        f.write(html)


def build_html_page(voices_data, page_title):
    """Generates the full HTML content with a table of models and audio players."""
    
    table_rows = ""
    for idx, voice in enumerate(voices_data):
        name_display = f"ID: {voice['speaker_id']}<br>Name: {voice['speaker_name']}"
        
        audio_player = f"""
        <audio controls preload="none">
            <source src="{voice['sample_file']}" type="audio/mp3">
            Your browser does not support the audio element.
        </audio>
        """

        # Using explicit string concatenation for maximum compatibility
        row = (
            "<tr>"
            "<td>" + str(idx + 1) + "</td>"
            "<td><strong>" + voice['name'] + "</strong></td>"
            "<td>" + voice['language'] + "</td>"
            "<td>" + voice['quality'] + " Hz</td>"
            "<td>" + voice['gender_age'] + "</td>"
            "<td>" + voice['loudness_metric'] + "</td>"
            "<td>" + voice['accent_data'] + "</td>"
            "<td>" + voice['rate_metric'] + "</td>"
            "<td>" + name_display + "</td>"
            "<td>" + audio_player + "</td>"
            "</tr>"
        )
        table_rows += row
        
    # Full HTML structure
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>{page_title}</title>
        <style>
            body {{ font-family: sans-serif; margin: 20px; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            audio {{ width: 250px; }}
        </style>
    </head>
    <body>
        <h1>{page_title} ({len(voices_data)} Speakers/Samples)</h1>
        <p><a href="voice_portfolio_index.html">← Back to Index</a> | All metrics are calculated from a temporary high-quality WAV to ensure accuracy.</p>
        
        <table>
            <thead>
                <tr>
                    <th>#</th>
                    <th>Model Name</th>
                    <th>Language</th>
                    <th>Sample Rate</th>
                    <th>Gender/Age</th>
                    <th>Loudness (dB)</th>
                    <th>Pitch/F0 Metrics</th>
                    <th>Rhythm/Spec Metrics</th>
                    <th>Speaker ID/Name</th>
                    <th>Sample</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
    </body>
    </html>
    """
    return html

if __name__ == "__main__":
    generate_voice_portfolio()
