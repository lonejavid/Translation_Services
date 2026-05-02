import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from TTS.api import TTS
import uvicorn

# Initialize FastAPI
app = FastAPI(title="Voice Cloning Service")

# Device configuration (Uses Mac GPU if available, else CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available():
    device = "mps"

print(f"Loading XTTS v2 on {device}...")
# Note: We use the local model path to ensure no cloud dependency
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

class CloneRequest(BaseModel):
    text: str
    language: str
    speaker_wav: str  # Path to the original audio sample
    output_path: str

@app.post("/clone")
async def clone_voice(request: CloneRequest):
    try:
        if not os.path.exists(request.speaker_wav):
            raise HTTPException(status_code=404, detail="Speaker WAV file not found")

        # Generate the cloned speech
        tts.tts_to_file(
            text=request.text,
            speaker_wav=request.speaker_wav,
            language=request.language,
            file_path=request.output_path
        )
        return {"status": "success", "output": request.output_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5050)