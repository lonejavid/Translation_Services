import os
import torch
from TTS.api import TTS

class XTTSCloner:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # This will download the model once to your local machine
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    def clone_voice(self, text, speaker_wav, output_path, language="hi"):
        """Clones voice using a 5-10 second reference WAV file."""
        print(f"Cloning voice for text: {text[:20]}...")
        self.tts.tts_to_file(
            text=text,
            speaker_wav=speaker_wav,
            language=language,
            file_path=output_path
        )
        return output_path