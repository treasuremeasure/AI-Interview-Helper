import numpy as np
import PySimpleGUI as sg
from loguru import logger

from src import audio, llm
from src.constants import APPLICATION_WIDTH, APPLICATION_HEIGHT, OFF_IMAGE, ON_IMAGE


def get_text_area(text: str, size: tuple) -> sg.Text:
    return sg.Text(
        text,
        size=size,
        background_color=sg.theme_background_color(),
        text_color="white",
    )


class BtnInfo:
    def __init__(self, state: bool = False):
        self.state = state


sg.theme("DarkAmber")

record_status_button = sg.Button(
    image_data=OFF_IMAGE,
    k="-TOGGLE1-",
    border_width=0,
    button_color=(sg.theme_background_color(), sg.theme_background_color()),
    disabled_button_color=(sg.theme_background_color(), sg.theme_background_color()),
    metadata=BtnInfo(),
)

analyzed_text_label = get_text_area("", size=(APPLICATION_WIDTH, 2))
answer               = get_text_area("", size=(APPLICATION_WIDTH, 120))

layout = [[
    sg.Column(
        [
            [sg.Text("Нажми R для записи", size=(int(APPLICATION_WIDTH * 0.8), 2)),
             record_status_button],
            [analyzed_text_label],
            [sg.Text("Ответ:")],
            [answer],
            [sg.Button("Отмена", key="Cancel")],          # ключ -- для закрытия
        ],
        scrollable=True,
        vertical_scroll_only=True,
        size=(APPLICATION_WIDTH, APPLICATION_HEIGHT),
        expand_x=True,
        expand_y=True,
    )
]]

WINDOW = sg.Window(
    "Keyboard Test",
    layout,
    return_keyboard_events=True,
    use_default_focus=False,
    resizable=True,
)


def background_recording_loop() -> None:
    audio_data = None
    while record_status_button.metadata.state:
        chunk = audio.record_batch()
        audio_data = chunk if audio_data is None else np.vstack((audio_data, chunk))
    audio.save_audio_file(audio_data)


while True:
    event, values = WINDOW.read()
    if event in ("Cancel", sg.WIN_CLOSED):
        logger.debug("Closing…")
        break

    # клавиша R ---------------------------------------------------------------
    if event in ("r", "R"):
        recording_now = record_status_button.metadata.state
        record_status_button.metadata.state = not recording_now

        if not recording_now:
            logger.debug("Starting recording…")
            WINDOW.perform_long_operation(background_recording_loop, "-RECORDING-")
            record_status_button.update(image_data=ON_IMAGE)
        else:
            logger.debug("Stopping recording…")
            record_status_button.update(image_data=OFF_IMAGE)

    # запись завершилась ------------------------------------------------------
    elif event == "-RECORDING-":
        logger.debug("Recording finished, start transcription…")
        analyzed_text_label.update("Start analyzing…")
        WINDOW.perform_long_operation(llm.transcribe_audio, "-WHISPER COMPLETED-")

    # whisper вернул текст ----------------------------------------------------
    elif event == "-WHISPER COMPLETED-":
        transcript = values["-WHISPER COMPLETED-"]
        analyzed_text_label.update(transcript)

        answer.update("Генерация ответа…")
        WINDOW.perform_long_operation(
            lambda: llm.generate_answer(transcript, temperature=0.7),
            "-ANSWER-",
        )

    # LLM вернул ответ --------------------------------------------------------
    elif event == "-ANSWER-":
        answer.update(values["-ANSWER-"])
