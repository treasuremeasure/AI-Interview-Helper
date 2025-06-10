"""
Проверяем совпадение sample-rate устройства и WAV-файла.
"""

import sounddevice as sd
import soundfile as sf
from pathlib import Path

WAV_PATH = Path("out.wav")          # имя файла, который слушали

# ───────────────────────────────────────────────────────────────
print("\n=== Устройства, которые видит sounddevice ===")
for i, dev in enumerate(sd.query_devices()):
    if dev["max_input_channels"] > 0 or dev["max_output_channels"] > 0:
        kind = []
        if dev["max_input_channels"] > 0:
            kind.append("IN")
        if dev["max_output_channels"] > 0:
            kind.append("OUT")
        print(f"{i:2d}: {dev['name']:<40} {'/'.join(kind):<3} "
              f"default_sr={dev['default_samplerate']:.0f} Hz")

# ───────────────────────────────────────────────────────────────
speaker = sd.default.device[1]     # индекс устройства вывода по умолчанию
mic     = sd.default.device[0]     # индекс устройства ввода по умолчанию

spk_info = sd.query_devices(speaker)
mic_info = sd.query_devices(mic)

print("\n=== По умолчанию выбраны ===")
print(f"Speaker : {spk_info['name']}  → {spk_info['default_samplerate']:.0f} Hz")
print(f"Mic      : {mic_info['name']}  → {mic_info['default_samplerate']:.0f} Hz")

# ───────────────────────────────────────────────────────────────
if WAV_PATH.exists():
    info = sf.info(WAV_PATH)
    print(f"\n=== Файл {WAV_PATH.name} ===")
    print(f"Samplerate: {info.samplerate} Hz · Channels: {info.channels}")
    
    # Быстрое сравнение
    match_spk = abs(info.samplerate - spk_info['default_samplerate']) < 1
    match_mic = abs(info.samplerate - mic_info['default_samplerate']) < 1
    if match_spk or match_mic:
        print("✔️  Sample-rate файла совпадает с устройством.",
              "(speaker)" if match_spk else "(mic)")
    else:
        print("⚠️  Sample-rate файла не совпадает ни с динамиками,"
              " ни с микрофоном — возможен ресэмплинг/дропы.")
else:
    print(f"\nФайл {WAV_PATH} не найден — пропускаем проверку WAV.")
