from pathlib import Path

from aiogram.types import (
    KeyboardButton,
    ReplyKeyboardMarkup,
)
from aiogram.utils.keyboard import InlineKeyboardBuilder

CHECKPOINTS_DIR = Path("checkpoints")

main_keyboard = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="Выбрать модель")],
        [KeyboardButton(text="Предсказать возраст")],
    ],
    resize_keyboard=True,
)


def models_keyboard():
    builder = InlineKeyboardBuilder()

    models = list(CHECKPOINTS_DIR.glob("*.ckpt"))

    for model in models:
        builder.button(
            text=model.stem,
            callback_data=f"model:{model.name}",
        )

    builder.adjust(1)

    return builder.as_markup()
