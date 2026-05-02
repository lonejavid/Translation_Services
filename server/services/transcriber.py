import whisper
import torch

class Transcriber:
    def __init__(self, model_size="base"):
        # Checks for local GPU, otherwise defaults to CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model(model_size, device=device)

    def transcribe(self, audio_path):
        """Transcribes local audio file to text with timestamps."""
        print(f"Transcribing {audio_path} locally...")
        result = self.model.transcribe(audio_path, verbose=False)
        return result # Returns dict with 'text' and 'segments'