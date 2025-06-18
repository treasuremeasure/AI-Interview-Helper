import requests
from loguru import logger
import time
import asyncio
import numpy as np
import soundfile as sf
from wyoming.client import AsyncTcpClient   
from wyoming.audio import AudioStart, AudioChunk, AudioStop
from wyoming.event import Event

from src.constants import INTERVIEW_POSITION, OUTPUT_FILE_NAME, LLAMA_SERVER_URL

SYSTEM_PROMPT = (
    f"Ты проходишь собесеsдование на должность {INTERVIEW_POSITION} в России.\n"
    "Тебе будет передан текст вопроса. Он может быть неполным или немного искажённым — постарайся понять суть и дать развёрнутый ответ.\n"
    "Отвечай от первого лица, как будто ты кандидат. Каждый ответ сопровождай конкретными примерами из практики или гипотетических ситуаций.\n"
    "Ответ должен быть кратким — не более 150 слов."
) 

project_id = "bf69751b-65af-4457-9a4c-a8d9453a6b06"
token = "87ce6187b84d0168781527c126b1769e"

FAST_WHISPER_HOST = "87.228.102.104"   # вынесите в .env, если нужно
FAST_WHISPER_PORT = 10300

async def _wyoming_transcribe(path: str) -> str:
    """Отправляет wav/ogg/mp3 в faster-whisper через Wyoming."""
    # 1. читаем файл
    audio, sr = sf.read(path, always_2d=False)
    if audio.ndim == 2:                         # стерео → моно
        audio = audio[:, 0]
    if sr != 16_000:
        import librosa
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr,  target_sr=16_000)
    pcm_int16 = (audio * 32767).astype(np.int16).tobytes()

    # 2. шлём по Wyoming
    async with AsyncTcpClient(FAST_WHISPER_HOST, FAST_WHISPER_PORT) as client:

        # 0) запрос на транскрипцию (обязательно)
        await client.write_event(Event("transcribe", {"language": "ru"}))

        # 1) начало аудио  ### FIX: все три параметра
        await client.write_event(
            AudioStart(rate=16_000, width=2, channels=1).to_event()
        )

        # 2) кусок аудио   ### FIX: те же параметры + payload
        await client.write_event(
            AudioChunk(pcm_int16, rate=16_000, width=2, channels=1).to_event()
        )

        # 3) конец аудио
        await client.write_event(AudioStop().to_event())


        # 3. ждём ответ
        while True:
            evt = await asyncio.wait_for(client.read_event(), timeout=30)
            if evt.type == "transcript" and evt.data.get("is_final", True):
                return evt.data["text"]

def transcribe_audio(path_to_file: str = OUTPUT_FILE_NAME) -> str:
    """Транскрипция через faster-whisper (Wyoming)."""
    return asyncio.run(_wyoming_transcribe(path_to_file))


def generate_answer(transcript: str, temperature: float = 0.7) -> str:

    headers = {
        'Authorization': f'Bearer {token}',
        'x-project-id': project_id,
            }
   

    payload = {
        "messages": [
                     {"role": "system",   "content": SYSTEM_PROMPT},
                     {"role": "user", "content": transcript}
            ],
        "model": "model-run-jbq8l-famous",
        "frequency_penalty": 0.1,
        "stream": False,
        "temperature": 0.5,
        "top_p": 0.95,
        "top_k": 10,
        "repetition_penalty": 1.03,
        "length_penalty": 1,
        "max_tokens": 500
        }   

    url = f"{LLAMA_SERVER_URL.rstrip('/')}/v1/chat/completions"
    logger.debug(f"Calling llama-server at {url}")

    res = requests.post(url, headers=headers, json=payload, timeout=60)
    res.raise_for_status()
    data = res.json()

    # возвращаем текст из первого варианта
    return data["choices"][0]["message"]["content"]
