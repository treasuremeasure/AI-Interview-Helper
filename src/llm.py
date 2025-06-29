import requests
from loguru import logger
import os
import soundfile as sf
import librosa
import numpy as np
import tempfile
import json

from src.constants import INTERVIEW_POSITION, OUTPUT_FILE_NAME, LLAMA_SERVER_URL

SYSTEM_PROMPT = (
    f"Ты проходишь собеседование на должность {INTERVIEW_POSITION} в России.\n"
    "Тебе будет передан текст вопроса. Он может быть неполным или немного искажённым — постарайся понять суть и дать развёрнутый ответ.\n"
    "Отвечай от первого лица, как будто ты кандидат. Каждый ответ сопровождай конкретными примерами из практики или гипотетических ситуаций.\n"
    "Ответ должен быть кратким — не более 150 слов."
)

project_id = "bf69751b-65af-4457-9a4c-a8d9453a6b06"
token = "87ce6187b84d0168781527c126b1769e"

FWS_URL = "http://87.228.81.70:8000/v1/audio/transcriptions"  # whisper REST
FWS_LANG = "ru"


def _resample_to_16k(path: str) -> str:
    """Любой wav/ogg/mp3 → временный 16-кГц mono-WAV, который понимает Whisper."""
    audio, sr = sf.read(path, always_2d=False)
    if audio.ndim == 2:
        audio = audio[:, 0]
    if sr != 16_000:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16_000)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(tmp_fd)
    sf.write(tmp_path, audio, 16_000, subtype="PCM_16")
    return tmp_path


def transcribe_audio(path_to_file: str = OUTPUT_FILE_NAME) -> str:
    """Транскрипция через faster-whisper-server (REST)."""
    wav16 = _resample_to_16k(path_to_file)
    with open(wav16, "rb") as f:
        files = {"file": ("audio.wav", f, "audio/wav")}
        data = {"language": FWS_LANG}
        logger.debug(f"POST {FWS_URL}")
        resp = requests.post(FWS_URL, files=files, data=data, timeout=120)
        resp.raise_for_status()
        text = resp.json()["text"]
    os.remove(wav16)
    return text


def stream_answer(transcript: str):
    """Генерация ответа с потоковым выводом через vLLM."""
    headers = {
        'Authorization': f'Bearer {token}',
        'x-project-id': project_id,
    }

    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": transcript}
        ],
        "model": "deepseek",
        "stream": True,
        "temperature": 0.65,
        "top_p": 0.9,
        "top_k": 40,
        "frequency_penalty": 0.1,
        "repetition_penalty": 1.05,
        "length_penalty": 1,
        "stop": ["math"]
    }

    url = f"{LLAMA_SERVER_URL.rstrip('/')}/v1/chat/completions"
    logger.debug(f"Streaming from llama-server at {url}")

    with requests.post(url, headers=headers, json=payload, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            if line.startswith(b"data: "):
                line = line.removeprefix(b"data: ")
            try:
                parsed = json.loads(line.decode("utf-8"))
                delta = parsed["choices"][0]["delta"]
                if "content" in delta:
                    yield delta["content"]
            except Exception as e:
                logger.warning(f"Stream parsing error: {e}")
                continue
