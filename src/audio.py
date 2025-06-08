"""Audio utilities."""

import numpy as np
import soundcard as sc
import soundfile as sf
from loguru import logger

from src.constants import OUTPUT_FILE_NAME, RECORD_SEC, SAMPLE_RATE

SPEAKER_ID = str(sc.default_speaker().name)   #спикер

#MIC_ID = str(sc.default_microphone().name) #микрофон

def record_batch(record_sec=5):
    speaker = sc.get_speaker(id=SPEAKER_ID)
    with speaker.recorder(samplerate=SAMPLE_RATE) as rec:
        audio = rec.record(numframes=SAMPLE_RATE * record_sec)
    return audio


def save_audio_file(audio_data: np.ndarray,
                    output_file_name: str = OUTPUT_FILE_NAME) -> None:
    """Сохраняет WAV-файл."""
    logger.debug(f"Saving audio file to {output_file_name} …")
    sf.write(file=output_file_name, data=audio_data, samplerate=SAMPLE_RATE)


