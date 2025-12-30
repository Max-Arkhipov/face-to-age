from pathlib import Path
import lightning as L
import torch
from face_to_age.data import UTKFaceDataModule
from face_to_age.model import SimpleRegressor, ConvRegressor
from face_to_age.lightning import AgeRegressionModule
from face_to_age.logger import build_logger


def build_model(cfg):
    if cfg.model.name == "simple_regressor":
        return SimpleRegressor(image_size=cfg.model.image_size)
    elif cfg.model.name == "conv_regressor":
        return ConvRegressor()
    else:
        raise ValueError(f"Unknown model: {cfg.model.name}")


def train(cfg):
    L.seed_everything(cfg.seed, workers=True)

    logger = build_logger(cfg)

    datamodule = UTKFaceDataModule(cfg)

    model = build_model(cfg)
    lit_module = AgeRegressionModule(model, cfg)

    trainer = L.Trainer(
        max_epochs=cfg.training.max_epochs,
        logger=logger,
    )

    trainer.fit(lit_module, datamodule=datamodule)