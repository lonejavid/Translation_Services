import os
import asyncio
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS so your frontend can communicate with the server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use the path from your server logs
BASE_DIR = "/Users/nassu/Downloads/Translation_Services-main/server"
VIDEO_DIR = os.path.join(BASE_DIR, "translated_videos")

# Ensure the output directory exists so FFmpeg doesn't fail later
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "downloads"), exist_ok=True)

@app.get("/api/process-video/stream")
async def stream_video(job_id: str):
    """
    Streaming endpoint that handles missing files 
    without crashing the server.
    """
    file_path = os.path.join(VIDEO_DIR, f"{job_id}.mp4")
    
    if not os.path.exists(file_path):
        print(f"Stream requested for {job_id}, but file is not ready yet.")
        # Proper exception handling now that HTTPException is imported
        raise HTTPException(
            status_code=404, 
            detail="Video processing in progress or job ID not found."
        )
    
    return FileResponse(file_path, media_type="video/mp4")

@app.post("/translate")
async def translate_video(request: Request):
    data = await request.json()
    video_url = data.get("url")
    # This is where your background processing logic is triggered
    return {"status": "processing", "job_id": "example_id"}

if __name__ == "__main__":
    import uvicorn
    # Startup on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)