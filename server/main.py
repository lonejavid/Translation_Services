import os
import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# 1. Ensure the directory exists so the server doesn't crash on start
os.makedirs("translated_videos", exist_ok=True)

# 2. MOUNT THE DIRECTORY
# This makes your files available at http://127.0.0.1:8000/translated_videos/filename.mp4
app.mount("/translated_videos", StaticFiles(directory="translated_videos"), name="translated_videos")
# Force PyTorch to use the CPU for any operations not supported by MPS
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import asyncio
import uuid
import torch
import ffmpeg
import glob
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from TTS.api import TTS
import yt_dlp

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Project paths
BASE_DIR = "/Users/nassu/Downloads/Translation_Services-main/server"
VIDEO_DIR = os.path.join(BASE_DIR, "translated_videos")
DOWNLOADS_DIR = os.path.join(BASE_DIR, "downloads")

os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(DOWNLOADS_DIR, exist_ok=True)

# Device Configuration for Apple Silicon
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Models running on: {device}")

# 1. Load Models 
# Whisper for Speech-to-Text
transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device=device)

# NLLB-200 for High-Quality English to Hindi Translation
print("Loading NLLB-200 Translator...")
translation_model_name = "facebook/nllb-200-distilled-600M"
nllb_tokenizer = AutoTokenizer.from_pretrained(translation_model_name)
nllb_model = AutoModelForSeq2SeqLM.from_pretrained(translation_model_name).to(device)

# XTTS v2 for Voice Cloning - Forcing CPU to avoid MPS "Output channels" error
print("Loading XTTS v2 on CPU for stability...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cpu")

# Semaphore limits the system to 1 job at a time to prevent memory overload
process_semaphore = asyncio.Semaphore(1)

def translate_with_nllb(text):
    """High-quality translation using NLLB-200."""
    inputs = nllb_tokenizer(text, return_tensors="pt").to(device)
    translated_tokens = nllb_model.generate(
        **inputs, 
        forced_bos_token_id=nllb_tokenizer.lang_code_to_id["hin_Deva"], 
        max_length=512
    )
    return nllb_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

def chunk_text(text, max_chars=200):
    """Splits text into chunks to stay under the XTTS 250-character limit."""
    import re
    # Split by natural sentence pauses (English or Hindi punctuation)
    sentences = re.split(r'(?<=[.।?!])\s+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chars:
            current_chunk += " " + sentence
        else:
            if current_chunk: chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def process_video_locally(video_url: str, job_id: str):
    try:
        input_template = os.path.join(DOWNLOADS_DIR, f"{job_id}_input")
        original_audio = os.path.join(DOWNLOADS_DIR, f"{job_id}_audio.wav")
        cloned_audio = os.path.join(DOWNLOADS_DIR, f"{job_id}_cloned.wav")
        final_video = os.path.join(VIDEO_DIR, f"{job_id}.mp4")

        # Step 1: Download Video
        print(f"[{job_id}] Downloading...")
        ydl_opts = {
            'outtmpl': f"{input_template}.%(ext)s",
            'format': 'bestvideo+bestaudio/best',
            'merge_output_format': 'mp4',
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

        actual_input = glob.glob(f"{input_template}*")[0]

        # Step 2: Extract Audio
        ffmpeg.input(actual_input).output(original_audio, acodec='pcm_s16le', ac=1, ar='16k').run(overwrite_output=True)

        # Step 3: Transcribe
        print(f"[{job_id}] Transcribing...")
        original_text = transcriber(original_audio)['text']
        
        # Step 4: Translation
        print(f"[{job_id}] Translating...")
        translated_text = translate_with_nllb(original_text)

        # Step 5: Chunked Voice Cloning
        print(f"[{job_id}] Cloning voice...")
        text_chunks = chunk_text(translated_text)
        temp_audios = []
        
        for i, chunk in enumerate(text_chunks):
            if not chunk.strip(): continue
            chunk_path = f"{cloned_audio}_part_{i}.wav"
            tts.tts_to_file(
                text=chunk,
                speaker_wav=original_audio, 
                language="hi", 
                file_path=chunk_path
            )
            temp_audios.append(ffmpeg.input(chunk_path))
        
        # Concatenate audio chunks
        joined_audio = ffmpeg.concat(*temp_audios, v=0, a=1).node
        ffmpeg.output(joined_audio[1], cloned_audio, acodec='pcm_s16le').run(overwrite_output=True)

        # Step 6: Final Merge
        v = ffmpeg.input(actual_input).video
        a = ffmpeg.input(cloned_audio).audio
        ffmpeg.output(v, a, final_video, vcodec='copy', acodec='aac').run(overwrite_output=True)

        # Cleanup
        for i in range(len(temp_audios)):
            if os.path.exists(f"{cloned_audio}_part_{i}.wav"):
                os.remove(f"{cloned_audio}_part_{i}.wav")

        print(f"✅ Job {job_id} Complete!")

    except Exception as e:
        print(f"❌ Job {job_id} failed: {str(e)}")

async def run_translation_pipeline(video_url: str, job_id: str):
    async with process_semaphore:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, process_video_locally, video_url, job_id)

@app.post("/api/process-video")
async def process_video(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()
    video_url = data.get("youtube_url") or data.get("url")
    job_id = str(uuid.uuid4())[:8]
    background_tasks.add_task(run_translation_pipeline, video_url, job_id)
    return {"status": "processing", "job_id": job_id}

@app.get("/api/languages")
async def get_languages(request: Request):
    """Returns the list of supported languages."""
    return [{"code": "hi", "name": "Hindi"}]

@app.get("/api/download/{job_id}")
async def download_video(job_id: str):
    """Serves the final translated video file."""
    file_path = os.path.join(VIDEO_DIR, f"{job_id}.mp4")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found or still processing.")
    return FileResponse(file_path, media_type="video/mp4")

@app.get("/api/process-video/stream")
async def stream_updates(job_id: str):
    """Handles the status requests from the frontend."""
    file_path = os.path.join(VIDEO_DIR, f"{job_id}.mp4")
    if os.path.exists(file_path):
        return {"status": "completed", "job_id": job_id}
    return {"status": "processing", "job_id": job_id}

if __name__ == "__main__":
    import uvicorn
    # The uvicorn.run call should always be the very last thing in the file
    uvicorn.run(app, host="0.0.0.0", port=8000)

    from fastapi.staticfiles import StaticFiles
import os

# Ensure the folder exists
os.makedirs("translated_videos", exist_ok=True)

# This makes http://127.0.0.1:8000/translated_videos/540bf58f.mp4 a valid link
app.mount("/translated_videos", StaticFiles(directory="translated_videos"), name="translated_videos")