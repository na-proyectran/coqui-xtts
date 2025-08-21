import os, glob, csv, subprocess
from pathlib import Path
from typing import List

from faster_whisper import WhisperModel

# Import common constants/functions from prepare_data if present
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw_mp3"
WAV_DIR = DATA_DIR / "wavs"
METADATA = DATA_DIR / "metadata.csv"
SPEAKER_NAME = "miguia"
TARGET_SR = 16000
FFMPEG = "ffmpeg"

CHUNK_SEC = 12   # longitud objetivo de segmento
WHISPER_SIZE = "medium"  # o "large-v3" si tienes VRAM


def ffmpeg_convert_to_wav(src: Path, dst: Path, sr: int = TARGET_SR):
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        FFMPEG, "-y", "-i", str(src),
        "-ac", "1", "-ar", str(sr), "-acodec", "pcm_s16le", str(dst)
    ]
    subprocess.run(cmd, check=True)


def transcribe_and_segment():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    WAV_DIR.mkdir(parents=True, exist_ok=True)

    mp3s = sorted(glob.glob(str(RAW_DIR / "*.mp3")))
    if not mp3s:
        print(f"No hay MP3 en {RAW_DIR}.")
        return

    model = WhisperModel(WHISPER_SIZE, device="auto", compute_type="float16")

    rows: List[List[str]] = []
    utt_id = 0

    for mp3 in mp3s:
        src = Path(mp3)
        tmp_wav = WAV_DIR / f"_tmp_{src.stem}.wav"
        ffmpeg_convert_to_wav(src, tmp_wav)

        segments, info = model.transcribe(str(tmp_wav), language="es", vad_filter=True, vad_parameters=dict(min_silence_duration_ms=300))

        for seg in segments:
            start = max(0.0, seg.start)
            end = seg.end
            dur = end - start
            if dur < 2.5:
                continue
            out_wav = WAV_DIR / f"utt_{utt_id:05d}.wav"
            cmd = [
                FFMPEG, "-y", "-i", str(tmp_wav),
                "-ss", str(start), "-to", str(end),
                "-ac", "1", "-ar", str(TARGET_SR), "-acodec", "pcm_s16le",
                str(out_wav)
            ]
            subprocess.run(cmd, check=True)
            text = seg.text.strip()
            rows.append([str(out_wav), text, SPEAKER_NAME])
            utt_id += 1

        try:
            tmp_wav.unlink()
        except Exception:
            pass

    with open(METADATA, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter='|')
        for r in rows:
            writer.writerow(r)
    print(f"Transcripción completada. metadata.csv con {len(rows)} filas en {METADATA}")


if __name__ == "__main__" and os.environ.get("ENTRYPOINT") == "transcribe":
    transcribe_and_segment()
