import requests
import whisper
from loguru import logger
import json


from src.constants import INTERVIEW_POSITION, OUTPUT_FILE_NAME, LLAMA_SERVER_URL

SYSTEM_PROMPT = (
    f"Ты проходишь собеседование на должность {INTERVIEW_POSITION} в России.\n"
    "Тебе будет передан текст вопроса. Он может быть неполным или немного искажённым — постарайся понять суть и дать развёрнутый ответ.\n"
    "Отвечай от первого лица, как будто ты кандидат. Каждый ответ сопровождай конкретными примерами из практики или гипотетических ситуаций.\n"
    "Ответ должен быть кратким — не более 150 слов."
)

_WHISPER_MODEL = whisper.load_model("medium") 

project_id = "bf69751b-65af-4457-9a4c-a8d9453a6b06"
token = "87ce6187b84d0168781527c126b1769e"


def transcribe_audio(path_to_file: str = OUTPUT_FILE_NAME) -> str:
    """
    Локальная транскрипция с помощью библиотеки openai-whisper.

    Args:
        path_to_file (str): Путь до .wav-файла для транскрипции.

    Returns:
        str: Расшифрованный текст.
    """
    # вынужденно отключаем fp16 на CPU / WSL, если нет GPU
    result = _WHISPER_MODEL.transcribe(path_to_file, fp16=False, language = 'ru')
    return result["text"]

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
        "model": "model-run-we0hr-dust",
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
