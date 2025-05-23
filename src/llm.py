import requests
import whisper
from loguru import logger

from src.constants import INTERVIEW_POSTION, OUTPUT_FILE_NAME, LLAMA_SERVER_URL

SYSTEM_PROMPT = (
    f"У тебя берут интервью на позицию {INTERVIEW_POSTION}.\n"
    "Ты получишь транскрипцию вопроса. "
    "Она может быть не полной и не совсем правильной, но ты должен понять вопрос и ответить на него.\n"
)
SHORTER_INSTRUCT = "Отвечай кратко, не более 70 слов."
LONGER_INSTRUCT  = "Перед ответом надо немного подумать и ответить не более 150 слов."

_WHISPER_MODEL = whisper.load_model("small")  

def transcribe_audio(path_to_file: str = OUTPUT_FILE_NAME) -> str:
    """
    Локальная транскрипция с помощью библиотеки openai-whisper.

    Args:
        path_to_file (str): Путь до .wav-файла для транскрипции.

    Returns:
        str: Расшифрованный текст.
    """
    # вынужденно отключаем fp16 на CPU / WSL, если нет GPU
    result = _WHISPER_MODEL.transcribe(path_to_file, fp16=False)
    return result["text"]

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
