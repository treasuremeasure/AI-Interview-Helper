import sounddevice as sd

devices = sd.query_devices()
for i, dev in enumerate(devices):
    print(f"{i}: {dev['name']}")
    print(f"   Max input channels: {dev['max_input_channels']}")
    print(f"   Default samplerate: {dev['default_samplerate']}")
