# Coqui XTTS-v2: Finetuning e inferencia en español

Este proyecto contiene scripts básicos para preparar datos, transcribir, entrenar y generar voz utilizando [Coqui XTTS-v2](https://coqui.ai/).

## Requisitos
- Python 3.10+
- `ffmpeg` instalado en el sistema
- GPU con al menos 8 GB de VRAM (recomendado)

### Instalación de dependencias
```bash
python -m venv .venv
. .venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install TTS==0.22.0
pip install soundfile pydub numpy librosa tqdm pandas
# Opcional para transcripciones automáticas
pip install faster-whisper==1.0.0
```

## Uso
1. **Preparar datos**: coloca tus MP3 en `data/raw_mp3/` y opcionalmente tus transcripciones en `data/prompts.txt`.
   ```bash
   ENTRYPOINT=prepare python prepare_data.py
   ```
2. **(Opcional) Transcribir con Whisper** si no tienes `prompts.txt`.
   ```bash
   ENTRYPOINT=transcribe python transcribe_whisper.py
   ```
3. **Entrenar** el modelo XTTS-v2 con tus datos.
   ```bash
   ENTRYPOINT=train python train_xtts.py
   ```
4. **Inferir** audio con tu voz.
   ```bash
   ENTRYPOINT=infer python infer_xtts.py
   ```

Los audios de prueba se guardarán en `samples/`.
