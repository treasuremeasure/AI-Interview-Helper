# src/whispercpp_transcriber.py
import subprocess, os, time, re

WHISPER_CLI   = r"D:/whisper.cpp/build/bin/whisper-cli.exe"
WHISPER_MODEL = r"D:/whisper.cpp/models/ggml-medium.bin"

ansi_re = re.compile(r"\x1B\[[0-9;]*m")

def transcribe_with_cpp(wav_path: str, lang: str = "ru") -> str:
    """Возвращает расшифровку сразу из stdout whisper-cli.exe."""
    wav_path = os.path.abspath(wav_path)
    assert os.path.isfile(wav_path), f"Файл {wav_path} не найден"
    assert os.path.isfile(WHISPER_CLI),   "whisper-cli.exe не найден"
    assert os.path.isfile(WHISPER_MODEL), "Файл модели не найден"

    cmd = [
        WHISPER_CLI,
        "-m", WHISPER_MODEL,
        "-f", wav_path,
        "-l", lang,
        "-pc", "0",
        # без -of / -otxt → CLI печатает чистый текст в stdout
    ]

    t0 = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True,     # ← перехватываем stdout/stderr
        text=True,
        encoding="utf-8",
        errors="ignore",
    )
    elapsed = time.time() - t0
    print(f"🧠 whisper-cli отработал за {elapsed:.2f} сек")

    if result.returncode != 0:
        raise RuntimeError(f"whisper-cli ошибка:\n{result.stderr}")
    
    clean = ansi_re.sub("", result.stdout)          # убираем цветовые коды
    clean = re.sub(r"\[.*?-->.*?\]", "", clean)     # убираем тайм-метки
    transcript = " ".join(line.strip()
                          for line in clean.splitlines()
                          if line.strip())


    return transcript
