from whispercpp import Whisper
import time

model_path = "D:/whisper.cpp/models/ggml-medium.bin"  # путь до твоей модели

def transcribe_with_cpp(path: str) -> str:
    print("⏳ Загружаем модель whisper.cpp…")
    start = time.time()
    w = Whisper(model_path)
    print(f"✅ Модель загружена за {time.time() - start:.2f} сек.")

    print("🧠 Начинаем транскрипцию…")
    start = time.time()
    w.transcribe(path)
    print(f"📄 Расшифровка завершена за {time.time() - start:.2f} сек.")

    return w.text
