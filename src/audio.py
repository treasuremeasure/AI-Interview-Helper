"""Audio utilities."""
import numpy as np
import soundcard as sc
import soundfile as sf
from loguru import logger

from src.constants import OUTPUT_FILE_NAME, RECORD_SEC, SAMPLE_RATE

#SPEAKER_ID = str(sc.default_speaker().name)   #спикер

MIC_ID = str(sc.default_microphone().name) #микрофон

BLOCKSIZE = 2048 

def record_batch(record_sec: int = RECORD_SEC) -> np.ndarray:
    logger.debug(f"Recording {record_sec}s …")      # f-строка была без 'f'
    mic = sc.get_microphone(id=MIC_ID)
    sr  = mic.samplerate            # реальная частота устройства
    frames_total = sr * record_sec

    with mic.recorder(samplerate=sr, blocksize=BLOCKSIZE) as rec:
        chunks = []
        collected = 0
        while collected < frames_total:
            block = rec.record(numframes=BLOCKSIZE)
            chunks.append(block)
            collected += len(block)
    return np.concatenate(chunks, axis=0)


def save_audio_file(audio_data: np.ndarray, output_file_name: str = OUTPUT_FILE_NAME) -> None:
    """
    Saves an audio data array to a file.

    Args:
        audio_data (np.ndarray): The audio data to be saved.
        output_file_name (str): The name of the output file. Defaults to the value of OUTPUT_FILE_NAME.

    Returns:
        None

    Example:
        ```python
        audio_data = np.array([0.1, 0.2, 0.3])
        save_audio_file(audio_data, "output.wav")
        ```
    """
    logger.debug(f"Saving audio file to {output_file_name}...")
    sf.write(file=output_file_name, data=audio_data, samplerate=SAMPLE_RATE)
