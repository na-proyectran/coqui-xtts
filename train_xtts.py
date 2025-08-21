import os
from pathlib import Path

import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

DATA_DIR = Path("data")
METADATA = DATA_DIR / "metadata.csv"
TARGET_SR = 16000
OUTPUT_DIR = Path("checkpoints/xtts_finetune")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def train_xtts():
    assert METADATA.exists(), f"No se encontró {METADATA}. Prepara datos primero."

    config = XttsConfig()
    config.model_args.update({
        "use_decoder_layerdrop": False,
        "dropout": 0.0,
        "decoder_input_dropout": 0.0,
        "use_spec_augment": False,
        "output_sample_rate": TARGET_SR,
        "phoneme_language": "es",
    })

    config.data.training_files = str(METADATA)
    config.data.validation_files = str(METADATA)
    config.data.path = str(DATA_DIR)

    config.batch_size = 4
    config.eval_batch_size = 4
    config.num_loader_workers = 2
    config.eval_steps = 500
    config.save_step = 1000
    config.max_steps = 5000
    config.learning_rate = 1e-4
    config.optimizer = "adamw"
    config.scheduler = "cosine"

    config.mixed_precision = True
    config.mixed_precision_type = "fp16"

    model = Xtts.init_from_config(config)
    model.load_checkpoint(restore_path=None, vocab_path=None)

    trainer = model.get_trainer(config=config, output_path=str(OUTPUT_DIR))
    trainer.fit(model)


if __name__ == "__main__" and os.environ.get("ENTRYPOINT") == "train":
    train_xtts()
