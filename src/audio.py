"""Audio utilities."""
import numpy as np
import soundcard as sc
import soundfile as sf
from loguru import logger
import librosa

from src.constants import OUTPUT_FILE_NAME, RECORD_SEC, SAMPLE_RATE

SPEAKER_ID = str(sc.default_speaker().name)


def record_batch(record_sec: int = RECORD_SEC) -> np.ndarray:
    """
    Records an audio batch for a specified duration.

    Args:
        record_sec (int): The duration of the recording in seconds. Defaults to the value of RECORD_SEC.

    Returns:
        np.ndarray: The recorded audio sample.

    Example:
        ```python
        audio_sample = record_batch(5)
        print(audio_sample)
        ```
    """
    logger.debug("Recording for {record_sec} second(s)...")
    with sc.get_microphone(
        id=SPEAKER_ID,
        include_loopback=True,
    ).recorder(samplerate=SAMPLE_RATE) as mic:
        audio_sample = mic.record(numframes=SAMPLE_RATE * record_sec)
    return audio_sample

def trim_silence(audio: np.ndarray, top_db: int = 30) -> np.ndarray:
    if audio.ndim > 1:
        audio = audio[:, 0]  # берём только 1 канал
    trimmed_audio, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed_audio


def save_audio_file(audio_data: np.ndarray, output_file_name: str = OUTPUT_FILE_NAME) -> None:
    logger.debug(f"Saving audio file to {output_file_name}...")
    trimmed = trim_silence(audio_data)
    sf.write(file=output_file_name, data=trimmed, samplerate=SAMPLE_RATE)