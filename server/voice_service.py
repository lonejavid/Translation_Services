import os
import torch
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from TTS.api import TTS

app = FastAPI()

# Prevents the "4-way redundant processing" seen in your logs
processing_lock = asyncio.Lock()

# Auto-detect Apple Silicon (MPS) or fallback to CPU
if torch.backends.mps.is_available():
    device = "mps"
    print("Using Apple Silicon (MPS) for high-speed cloning")
else:
    device = "cpu"
    print("Using CPU (Warning: This will be slow for XTTS v2)")

# Initialize XTTS v2
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

class CloneRequest(BaseModel):
    text: str
    speaker_wav: str
    language: str
    output_path: str

@app.post("/clone")
async def clone_voice(request: CloneRequest):
    async with processing_lock:  # Process one cloning task at a time
        try:
            # Create downloads directory if missing
            os.makedirs(os.path.dirname(request.output_path), exist_ok=True)
            
            tts.tts_to_file(
                text=request.text,
                speaker_wav=request.speaker_wav,
                language=request.language,
                file_path=request.output_path
            )
            return {"status": "success", "path": request.output_path}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Port 5050 as used in your current setup
    uvicorn.run(app, host="0.0.0.0", port=5050)