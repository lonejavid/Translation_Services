#!/bin/bash
# Pin PyTorch stack + Coqui deps for XTTS voice cloning (CPU wheels by default).
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d "venv" ]; then
  echo "No venv found. Create it first (e.g. run ./run.sh once)."
  exit 1
fi

source venv/bin/activate

pip uninstall transformers -y 2>/dev/null || true
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers==4.38.0
pip install "TTS>=0.22.0"
pip install "librosa>=0.10.0" "soundfile>=0.12.0" "webrtcvad>=2.0.10" "noisereduce>=3.0.0"

export PYTHONPATH="$SCRIPT_DIR"
python check_xtts.py
echo "Installation complete!"
