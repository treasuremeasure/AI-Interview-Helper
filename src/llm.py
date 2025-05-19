import requests
from loguru import logger

from src.constants import INTERVIEW_POSTION, OUTPUT_FILE_NAME, LLAMA_SERVER_URL

SYSTEM_PROMPT = (
    f"You are interviewing for a {INTERVIEW_POSTION} position.\n"
    "You will receive an audio transcription of the question. "
    "It may not be complete. You need to understand the question and write an answer to it.\n"
)
SHORTER_INSTRUCT = "Concisely respond, limiting your answer to 70 words."
LONGER_INSTRUCT  = "Before answering, take a deep breath and think step by step. Answer in no more than 150 words."

def transcribe_audio(path_to_file: str = OUTPUT_FILE_NAME) -> str:
    """
    (опционально) — оставляем OpenAI Whisper, если у вас есть ключ.
    Иначе сюда можно тоже в будущем зашить локальный whisper.cpp.
    """
    import openai
    openai.api_key = ""  # или берём из const.OPENAI_API_KEY
    with open(path_to_file, "rb") as audio_file:
        resp = openai.Audio.translate("whisper-1", audio_file)
    return resp["text"]

def generate_answer(transcript: str, short_answer: bool = True, temperature: float = 0.7) -> str:
    # собираем системную часть
    instr = SHORTER_INSTRUCT if short_answer else LONGER_INSTRUCT
    prompt = SYSTEM_PROMPT + instr

    payload = {
        "model": "chat",          # llama-server возьмёт любую модель по умолчанию
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user",   "content": transcript},
        ],
        "temperature": temperature,
    }

    logger.debug(f"Calling llama-server at {LLAMA_SERVER_URL} …")
    res = requests.post(f"{LLAMA_SERVER_URL}/v1/chat/completions", json=payload)
    res.raise_for_status()
    data = res.json()

    # возвращаем текст из первого варианта
    return data["choices"][0]["message"]["content"]
