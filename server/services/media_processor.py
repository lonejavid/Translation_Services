import subprocess
from pathlib import Path

class MediaProcessor:
    def replace_audio(self, video_path: str, audio_path: str, output_path: str):
        """
        Swaps original audio with cloned audio.
        Uses '-c:v copy' to avoid re-encoding the video, making it very fast.
        """
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-i', audio_path,
            '-map', '0:v:0', # Video from source
            '-map', '1:a:0', # Audio from clone
            '-c:v', 'copy',  # Stream copy video
            '-c:a', 'aac',   # Encode audio for compatibility
            '-shortest',
            output_path
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg Error: {e.stderr.decode()}")
            return False