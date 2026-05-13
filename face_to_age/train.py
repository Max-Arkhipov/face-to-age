import lightning as L
from hydra import main
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig

from face_to_age.data import UTKFaceDataModule
from face_to_age.finetuning import BackboneFinetuning
from face_to_age.lightning import AgeRegressionModule
from face_to_age.logger import build_logger
from face_to_age.model import AgeModel, ConvRegressor, ConvRegressor_256, SimpleRegressor
from utils.dvc_utils import dvc_pull_if_needed


@main(version_base=None, config_path="../configs", config_name="config")
def train(cfg: DictConfig):
    print("=" * 80)
    print("CONFIG:")
    print(cfg)
    print("=" * 80)

    dvc_pull_if_needed(
        [
            cfg.dataset.train_data_dir,
            cfg.dataset.val_data_dir,
            cfg.dataset.test_data_dir,
        ]
    )

    # CallBack
    callbacks = []

    ckpt_callback = ModelCheckpoint(
        dirpath=cfg.paths.checkpoints_dir,
        filename=cfg.model.checkpoint_name.replace(".pth", ""),
        monitor="val_mae",
        mode="min",
        save_top_k=1,
        enable_version_counter=False,
    )

    callbacks.append(ckpt_callback)

    if cfg.model.get("finetune", {}).get("enabled", False):
        ft_cfg = cfg.model.finetune
        callbacks.append(
            BackboneFinetuning(
                unfreeze_epoch=ft_cfg.unfreeze_epoch,
                backbone_lr=ft_cfg.backbone_lr,
                backbone_name=cfg.model.backbone,
                unfreeze_layers=list(ft_cfg.unfreeze_layers)
                if ft_cfg.get("unfreeze_layers")
                else None,
            )
        )

    # DataModule
    datamodule = UTKFaceDataModule(cfg)

    # Model
    if cfg.model.name == "simple_regressor":
        model = SimpleRegressor(cfg.model.image_size)
    elif cfg.model.name == "conv_regressor":
        model = ConvRegressor()
    elif cfg.model.name == "conv_regressor_256":
        model = ConvRegressor_256()
    else:
        try:
            model = AgeModel(cfg)
        except Exception as e:
            raise ValueError(f"Unknown model: {cfg.model.name}. Error: {e}")

    module = AgeRegressionModule(model, cfg)

    # Logger
    logger = build_logger(cfg)

    # Trainer
    trainer = L.Trainer(
        max_epochs=cfg.training.max_epochs,
        logger=logger,
        callbacks=callbacks,
    )

    # Train
    trainer.fit(module, datamodule=datamodule)

    # Test
    trainer.test(module, datamodule=datamodule)

    print(f"Best checkpoint: {ckpt_callback.best_model_path}")


if __name__ == "__main__":
    train()
