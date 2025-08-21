import os, sys, csv, glob, subprocess
from pathlib import Path
from typing import List

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw_mp3"
WAV_DIR = DATA_DIR / "wavs"
METADATA = DATA_DIR / "metadata.csv"
PROMPTS = DATA_DIR / "prompts.txt"  # opcional si ya tienes transcripciones
SPEAKER_NAME = "miguia"              # cambia el id del locutor
TARGET_SR = 16000

FFMPEG = "ffmpeg"  # asume ffmpeg en PATH


def ffmpeg_convert_to_wav(src: Path, dst: Path, sr: int = TARGET_SR):
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Conversión: mono, 16 kHz, 16-bit PCM
    cmd = [
        FFMPEG, "-y", "-i", str(src),
        "-ac", "1", "-ar", str(sr), "-acodec", "pcm_s16le", str(dst)
    ]
    subprocess.run(cmd, check=True)


def load_prompts() -> List[str]:
    if PROMPTS.exists():
        with open(PROMPTS, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        return lines
    return []


def main_prepare():
    mp3s = sorted(glob.glob(str(RAW_DIR / "*.mp3")))
    if not mp3s:
        print(f"No hay MP3 en {RAW_DIR}. Coloca tus audios allí.")
        sys.exit(1)

    wav_paths = []
    for i, mp3 in enumerate(mp3s):
        src = Path(mp3)
        dst = WAV_DIR / f"utt_{i:04d}.wav"
        print(f"Convirtiendo {src.name} → {dst.name}")
        ffmpeg_convert_to_wav(src, dst)
        wav_paths.append(dst)

    prompts = load_prompts()
    if prompts and len(prompts) != len(wav_paths):
        print("Advertencia: prompts.txt y número de WAVs no coinciden. Se truncará al mínimo común.")
        n = min(len(prompts), len(wav_paths))
        prompts = prompts[:n]
        wav_paths = wav_paths[:n]

    # Si no hay prompts, asumimos que ejecutarás transcribe_whisper.py para generar metadata.csv
    if not prompts:
        print("No se encontró prompts.txt. Si no tienes transcripciones, ejecuta transcribe_whisper.py para generarlas y después crea metadata.csv.")
        print(f"WAVs generados en {WAV_DIR}.")
        return

    # Crear metadata.csv
    with open(METADATA, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter='|')
        for wav, text in zip(wav_paths, prompts):
            writer.writerow([str(wav), text, SPEAKER_NAME])
    print(f"metadata.csv creado en {METADATA} con {len(wav_paths)} filas.")


if __name__ == "__main__" and os.environ.get("ENTRYPOINT") == "prepare":
    main_prepare()
