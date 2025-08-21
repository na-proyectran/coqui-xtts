import os
from pathlib import Path

from TTS.api import TTS as _TTS

DATA_DIR = Path("data")
WAV_DIR = DATA_DIR / "wavs"
SAMPLES_DIR = Path("samples")
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
SPEAKER_NAME = "miguia"


def infer_sample():
    text = "Bienvenido al museo. ¿Sobre qué obra te gustaría saber más?"

    ref_wav = next(WAV_DIR.glob("*.wav"))
    tts = _TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    tts.tts_to_file(text=text, speaker_wav=str(ref_wav), language="es", file_path=str(SAMPLES_DIR / "zeroshot.wav"))

    # Ejemplo de uso del checkpoint finetune
    # tts_finetune = _TTS(model_path="checkpoints/xtts_finetune/best_model.pth")
    # tts_finetune.tts_to_file(text=text, speaker=SPEAKER_NAME, language="es", file_path=str(SAMPLES_DIR / "finetune.wav"))


if __name__ == "__main__" and os.environ.get("ENTRYPOINT") == "infer":
    infer_sample()
