from aiogram.fsm.state import State, StatesGroup


class PredictState(StatesGroup):
    waiting_for_photo = State()
