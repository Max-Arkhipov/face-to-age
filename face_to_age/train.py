from pathlib import Path

import lightning as L
import torch
from hydra import main
from omegaconf import DictConfig

from face_to_age.data import UTKFaceDataModule
from face_to_age.lightning import AgeRegressionModule
from face_to_age.logger import build_logger
from face_to_age.model import ConvRegressor, SimpleRegressor


@main(version_base=None, config_path="../configs", config_name="config")
def train(cfg: DictConfig):
    print("=" * 80)
    print("CONFIG:")
    print(cfg)
    print("=" * 80)

    # DataModule
    datamodule = UTKFaceDataModule(cfg)

    # Model
    if cfg.model.name == "simple_regressor":
        model = SimpleRegressor(cfg.model.image_size)
    elif cfg.model.name == "conv_regressor":
        model = ConvRegressor()
    else:
        raise ValueError(f"Unknown model: {cfg.model.name}")

    module = AgeRegressionModule(model, cfg)

    # Logger
    logger = build_logger(cfg)

    # Trainer
    trainer = L.Trainer(
        max_epochs=cfg.training.max_epochs,
        logger=logger,
    )

    # Train
    trainer.fit(module, datamodule=datamodule)

    # Test
    trainer.test(module, datamodule=datamodule)

    # Save model
    ckpt_dir = Path(cfg.paths.checkpoints_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = ckpt_dir / cfg.infer.checkpoint_name
    torch.save(module.model.state_dict(), ckpt_path)

    print(f"Model saved to {ckpt_path}")


if __name__ == "__main__":
    train()
