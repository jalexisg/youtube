import os
import shutil
import uuid
import asyncio
from pathlib import Path
from typing import Dict, Optional
from pydantic import BaseModel

# Force dependency import for the transcriber
os.environ['FORCE_IMPORT_DEPENDENCIES'] = '1'

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Import existing logic
# Assuming audio_transcriber_summarizer.py is in the same directory
try:
    from audio_transcriber_summarizer import AudioTranscriberSummarizer
except ImportError:
    # Fallback for when running from a different context or if dependencies fail
    print("Warning: Could not import AudioTranscriberSummarizer. Ensure dependencies are installed.")
    AudioTranscriberSummarizer = None

from yt_dlp import YoutubeDL

app = FastAPI(title="Audio Transcriber & Summarizer API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
BASE_DIR = Path(__file__).parent.absolute()
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "transcripciones"
WEB_DIR = BASE_DIR / "web"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Global state for tasks (in-memory for simplicity)
tasks: Dict[str, Dict] = {}

# Initialize Transcriber (Lazy load or global?)
# Global initialization might be slow on startup, but better for repeated requests.
# We'll initialize it lazily or on startup.
transcriber = None

def get_transcriber():
    global transcriber
    if transcriber is None:
        if AudioTranscriberSummarizer:
            transcriber = AudioTranscriberSummarizer(model_size="base", language="auto")
        else:
            raise RuntimeError("AudioTranscriberSummarizer not available")
    return transcriber

class URLRequest(BaseModel):
    url: str

def process_file_task(task_id: str, file_path: Path, original_filename: str):
    tasks[task_id]["status"] = "processing"
    tasks[task_id]["message"] = "Starting transcription..."
    
    try:
        ats = get_transcriber()
        
        # Process
        result = ats.process_media_file(
            file_path=str(file_path),
            output_dir=str(OUTPUT_DIR),
            keep_audio=True, # Keep audio for the user to download if they want
            summary_sentences=5
        )
        
        if result:
            # Add download URLs
            # For uploaded files, they are in UPLOAD_DIR
            filename = file_path.name
            result["media_url"] = f"/uploads/{filename}"
            result["title"] = original_filename
            
            # Check for audio file
            audio_path = OUTPUT_DIR / f"{file_path.stem}_audio.wav"
            if audio_path.exists():
                result["audio_url"] = f"/results/{audio_path.name}"

            tasks[task_id]["status"] = "completed"
            tasks[task_id]["result"] = result
            tasks[task_id]["message"] = "Processing complete"
        else:
            tasks[task_id]["status"] = "failed"
            tasks[task_id]["message"] = "Processing returned no result"
            
    except Exception as e:
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["message"] = f"Error: {str(e)}"
    finally:
        # Cleanup upload if needed, but maybe keep it for debugging?
        # os.remove(file_path) 
        pass

def process_url_task(task_id: str, url: str):
    tasks[task_id]["status"] = "downloading"
    tasks[task_id]["message"] = "Downloading video..."
    
    try:
        # Download with yt-dlp
        # Use task_id as filename to avoid special characters issues
        ydl_opts = {
            'outtmpl': str(UPLOAD_DIR / f'{task_id}.%(ext)s'),
            'format': 'bestvideo+bestaudio/best',
            'merge_output_format': 'mp4',
            'noplaylist': True,
            'quiet': True
        }
        
        # Add cookies if available
        cookies_path = BASE_DIR / "cookies.txt"
        if cookies_path.exists():
            ydl_opts['cookies'] = str(cookies_path)
            print("Using cookies.txt for authentication")
        
        downloaded_path = None
        video_title = "video"
        
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if 'entries' in info:
                info = info['entries'][0]
            
            video_title = info.get('title', 'video')
            
            filename = ydl.prepare_filename(info)
            # Check for merged file
            base = os.path.splitext(filename)[0]
            if os.path.exists(base + ".mp4"):
                filename = base + ".mp4"
            
            downloaded_path = Path(filename)

        if not downloaded_path or not downloaded_path.exists():
            raise Exception("Download failed or file not found")

        tasks[task_id]["status"] = "processing"
        tasks[task_id]["message"] = "Transcribing..."
        
        ats = get_transcriber()
        result = ats.process_media_file(
            file_path=str(downloaded_path),
            output_dir=str(OUTPUT_DIR),
            keep_audio=True,
            summary_sentences=5
        )
        
        if result:
            # Add download URLs
            filename = downloaded_path.name
            result["media_url"] = f"/uploads/{filename}"
            result["title"] = video_title
            
            audio_path = OUTPUT_DIR / f"{downloaded_path.stem}_audio.wav"
            if audio_path.exists():
                result["audio_url"] = f"/results/{audio_path.name}"

            tasks[task_id]["status"] = "completed"
            tasks[task_id]["result"] = result
            tasks[task_id]["message"] = "Processing complete"
        else:
            tasks[task_id]["status"] = "failed"
            tasks[task_id]["message"] = "Processing returned no result"

    except Exception as e:
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["message"] = f"Error: {str(e)}"

@app.post("/api/process-file")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    task_id = str(uuid.uuid4())
    # Use task_id for filename to avoid special chars
    ext = os.path.splitext(file.filename)[1]
    safe_filename = f"{task_id}{ext}"
    file_path = UPLOAD_DIR / safe_filename
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    tasks[task_id] = {
        "id": task_id,
        "status": "queued",
        "filename": file.filename,
        "type": "file",
        "message": "File uploaded, waiting for processing..."
    }
    
    background_tasks.add_task(process_file_task, task_id, file_path, file.filename)
    
    return {"task_id": task_id, "status": "queued"}

@app.post("/api/process-url")
async def process_url(background_tasks: BackgroundTasks, request: URLRequest):
    task_id = str(uuid.uuid4())
    
    tasks[task_id] = {
        "id": task_id,
        "status": "queued",
        "url": request.url,
        "type": "url",
        "message": "URL received, waiting for download..."
    }
    
    background_tasks.add_task(process_url_task, task_id, request.url)
    
    return {"task_id": task_id, "status": "queued"}

@app.get("/api/tasks/{task_id}")
async def get_task_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return tasks[task_id]

# Mount static files (Frontend)
# We mount this last so API routes take precedence
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.mount("/results", StaticFiles(directory=str(OUTPUT_DIR)), name="results")

if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="static")
else:
    print(f"Warning: Web directory {WEB_DIR} does not exist.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
