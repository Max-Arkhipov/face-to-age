import logging
import os

from aiogram import Bot, Dispatcher
from aiogram.client.default import (
    DefaultBotProperties,
)
from aiogram.enums import ParseMode
from dotenv import load_dotenv

from bot.handlers import router

load_dotenv()

TOKEN = os.getenv("BOT_TOKEN")

logging.basicConfig(
    level=logging.INFO,
    format=("%(asctime)s | %(levelname)s | %(name)s | %(message)s"),
)

bot = Bot(
    token=TOKEN,
    default=DefaultBotProperties(parse_mode=ParseMode.HTML),
)

dp = Dispatcher()

dp.include_router(router)


if __name__ == "__main__":
    dp.run_polling(bot)
