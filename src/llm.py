import requests
import whisper
from loguru import logger
import json

from src.constants import INTERVIEW_POSTION, OUTPUT_FILE_NAME, LLAMA_SERVER_URL

SYSTEM_PROMPT = (
    f"You are being interviewed for the position of {INTERVIEW_POSTION} in Russia.\n"
    "You will receive a transcription of a question. "
    "It might be incomplete or somewhat inaccurate, but you must understand the question and answer it.\n"
    "A systems analyst is a bridge between business goals and development: they formulate, clarify, and control requirements so that the team can quickly build the right product.\n"
    "They:\n"
    "1) Gather requirements → interviews, workshops, document analysis.\n"
    "2) Formalize → BPMN/UML diagrams, user stories, specifications.\n"
    "3) Coordinate → discuss requirements with stakeholders, architects, UX, QA.\n"
    "4) Support development → answer questions, manage the backlog, verify that the implementation meets requirements.\n"
    "5) Maintain the product → analyze metrics, prepare changes.\n"
)


ANSWER_INSTRUCT = "Каждый свой ответ подкрепляй примерами. Нужно отвечать не более 150 слов."

_WHISPER_MODEL = whisper.load_model("small") 

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
    result = _WHISPER_MODEL.transcribe(path_to_file, fp16=False)
    return result["text"]

def generate_answer(transcript: str, temperature: float = 0.7) -> str:

    headers = {
        'Authorization': f'Bearer {token}',
        'x-project-id': project_id,
            }
   
    prompt = SYSTEM_PROMPT + ANSWER_INSTRUCT

    payload = {
        "messages": [
                     {"role": "user",   "content": transcript + prompt}
            ],
        "model": "model-run-we0hr-dust",
        "frequency_penalty": 0.1,
        "stop": "math",
        "stream": False,
        "temperature": 0.5,
        "top_p": 0.95,
        "top_k": 10,
        "repetition_penalty": 1.03,
        "length_penalty": 1
        }   

    url = f"{LLAMA_SERVER_URL.rstrip('/')}/v1/chat/completions"
    logger.debug(f"Calling llama-server at {url}")

    res = requests.post(url, headers=headers, json=payload, timeout=60)
    res.raise_for_status()
    data = res.json()

    # возвращаем текст из первого варианта
    return data["choices"][0]["message"]["content"]
