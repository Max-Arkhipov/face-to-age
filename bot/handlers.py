import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from aiogram import F, Router
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, FSInputFile, Message
from omegaconf import OmegaConf

from bot.keyboards import main_keyboard, models_keyboard
from bot.state import PredictState
from bot.storage import predictors_cache, user_models
from face_to_age.telegram_predictor import TelegramFacePredictor

logger = logging.getLogger(__name__)
cfg = OmegaConf.load("configs/config.yaml")
router = Router()


@router.message(F.text == "/start")
async def start_handler(message: Message):
    logger.info(f"User {message.from_user.id} started bot")
    await message.answer("Выберите действие", reply_markup=main_keyboard)


@router.message(F.text == "Выбрать модель")
async def choose_model_handler(message: Message):
    await message.answer("Выберите модель:", reply_markup=models_keyboard())


@router.callback_query(F.data.startswith("model:"))
async def model_selected_handler(callback: CallbackQuery):
    model_name = callback.data.split(":")[1]
    checkpoint_path = Path("checkpoints") / model_name

    if model_name not in predictors_cache:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        train_cfg = OmegaConf.create(checkpoint["hyper_parameters"]["cfg"])
        predictors_cache[model_name] = TelegramFacePredictor(
            cfg=train_cfg,
            checkpoint_path=str(checkpoint_path),
        )

    user_models[callback.from_user.id] = predictors_cache[model_name]
    logger.info(f"User {callback.from_user.id} selected model {model_name}")
    await callback.message.answer(f"Выбрана модель:\n{model_name}")
    await callback.answer()


@router.message(F.text == "Предсказать возраст")
async def predict_age_handler(message: Message, state: FSMContext):
    if message.from_user.id not in user_models:
        await message.answer("Сначала выберите модель")
        return
    await state.set_state(PredictState.waiting_for_photo)
    await message.answer("Отправьте фото")


@router.message(PredictState.waiting_for_photo, F.photo)
async def photo_handler(message: Message, state: FSMContext, bot):
    logger.info(f"User {message.from_user.id} sent photo")

    predictor = user_models[message.from_user.id]
    photo = message.photo[-1]
    file = await bot.get_file(photo.file_id)
    downloaded = await bot.download_file(file.file_path)

    image_np = np.frombuffer(downloaded.read(), np.uint8)
    img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    result = predictor.predict(img)

    if "error" in result:
        await message.answer(result["error"])
        await state.set_state(PredictState.waiting_for_photo)
        return

    age = result["predicted_age"]
    uncertainty = result["uncertainty"]
    aligned_img = result["aligned_image"]

    logger.info(f"Predicted age: {age}, uncertainty: {uncertainty}")

    if uncertainty is not None:
        caption = f"Предсказанный возраст: {age:.1f} ± {uncertainty:.1f} лет"
    else:
        caption = f"Предсказанный возраст: {age:.1f} лет"

    temp_path = "/tmp/result.jpg"
    cv2.imwrite(temp_path, aligned_img)

    await message.answer_photo(FSInputFile(temp_path), caption=caption)
    await state.set_state(PredictState.waiting_for_photo)
