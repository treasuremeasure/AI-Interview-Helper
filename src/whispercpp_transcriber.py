# src/whispercpp_transcriber.py
import subprocess, os, time, uuid, pathlib

WHISPER_CLI   = r"D:/whisper.cpp/build/bin/whisper-cli.exe"
WHISPER_MODEL = r"D:/whisper.cpp/models/ggml-medium.bin"

def transcribe_with_cpp(wav_path: str, lang: str = "ru") -> str:
    """
    Стартует whisper-cli.exe и возвращает текст.
    Файл .txt создаётся во временной папке рядом с .wav.
    """
    wav_path   = os.path.abspath(wav_path)
    assert os.path.isfile(wav_path), f"Файл {wav_path} не найден"

    # кладём расшифровку рядом с wav, но имя делаем уникальным
    txt_path = pathlib.Path(wav_path).with_suffix(f".{uuid.uuid4().hex}.txt")

    cmd = [
        WHISPER_CLI,
        "-m", WHISPER_MODEL,
        "-f", wav_path,
        "-l", lang,
        "-of", str(txt_path),     # <-- ЯВНО указываем итоговый .txt
        "--print-colors", "false"
    ]

    t0 = time.time()
    # вывод нам не важен → отправляем в DEVNULL, чтобы не ловить UnicodeDecodeError
    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        encoding="utf-8",        # ← пусть всё равно будет UTF-8
        errors="ignore",
    )

    elapsed = time.time() - t0
    print(f"🧠 whisper-cli отработал за {elapsed:.2f} сек")

    if result.returncode != 0:
        raise RuntimeError(f"whisper-cli вернул код {result.returncode}")

    if not txt_path.exists():
        raise FileNotFoundError(f"Файл транскрипции {txt_path} не найден")

    txt = txt_path.read_text(encoding="utf-8").strip()
    txt_path.unlink(missing_ok=True)      # чистим за собой
    return txt
