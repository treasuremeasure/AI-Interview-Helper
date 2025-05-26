import requests
import whisper
from loguru import logger

from src.constants import INTERVIEW_POSTION, OUTPUT_FILE_NAME, LLAMA_SERVER_URL

SYSTEM_PROMPT = (
    f"У тебя берут интервью на позицию {INTERVIEW_POSTION} в России.\n"
    "Ты получишь транскрипцию вопроса. "
    "Она может быть не полной и не совсем правильной, но ты должен понять вопрос и ответить на него.\n"
    "Системный аналитик — мост между бизнес-целями и разработкой: формулирует, уточняет и контролирует требования так, чтобы команда могла быстро создать нужный продукт.\n"
    "Он\n:" 
    "1) Собирает требования → интервью, воркшопы, анализ документов.\n"
    "2) Формализует → BPMN/UML-диаграммы, user stories, спецификации (часто по ГОСТ 34).\n"
    "3) Согласовывает → обсуждает ТЗ с заказчиком, архитекторами, UX, QA.\n"
    "4) Сопровождает разработку → отвечает на вопросы, ведёт backlog, проверяет соответствие реализованного функций требованиям.\n"
    "5) Поддерживает продукт → анализирует метрики, готовит изменения.\n"

    "Его типичные стек технологий:\n"
    "Документация/управление: Confluence, Jira / YouTrack, Markdown, ГОСТ 34-формы.\n"
    "Моделирование: BPMN 2.0, UML, ER-диаграммы (Enterprise Architect, draw.io)\n"
    "Базы данных: PostgreSQL, MySQL, Oracle.\n"
    "API & интеграции: REST/JSON, SOAP/XML, OpenAPI/Swagger, Postman, Kafka, RabbitMQ.\n"

    "Тебе могут задать простые вопросы, например:\n"
    "1) В чем отличие BPMN от UML?\n"
    "2) Что такое REST API?\n"
    "3) Что такое Kafka?\n"
    "4) Что такое RabbitMQ?\n"

    "А также могут задать непростые задачи, например:\n"
    "1) Опиши как бы ты реализовал корпоративный чат с нуля. Как будешь масштабировать систему при росте нагрузки до 10к RPS.\n"
    
)

ANSWER_INSTRUCT = "Перед ответом надо немного подумать и ответить не более 150 слов."

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

def generate_answer(transcript: str, temperature: float = 0.7) -> str:
    # собираем системную часть
   
    prompt = SYSTEM_PROMPT + ANSWER_INSTRUCT

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
