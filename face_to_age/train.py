import subprocess
from pathlib import Path

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from face_to_age.dataset import UTKFaceDataModule
from face_to_age.model import AgeEstimationModel


def pull_data_with_dvc():
    """Pull data using DVC"""
    try:
        subprocess.run(["dvc", "pull"], check=True)
        print("Data pulled successfully with DVC")
    except subprocess.CalledProcessError as e:
        print(f"Warning: DVC pull failed: {e}")
        print("Continuing without DVC data...")


def get_git_commit_id() -> str:
    """Get current git commit ID"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown"


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def train(cfg: DictConfig):
    # Print config
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Set seed
    pl.seed_everything(cfg.training.seed, workers=True)
    
    # Pull data with DVC
    pull_data_with_dvc()
    
    # Initialize data module
    data_module = UTKFaceDataModule(
        data_dir=cfg.data.data_dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        image_size=cfg.data.image_size,
        train_split=cfg.data.train_split,
        val_split=cfg.data.val_split,
    )
    
    # Initialize model
    model = AgeEstimationModel(
        backbone=cfg.model.backbone,
        pretrained=cfg.model.pretrained,
        learning_rate=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=cfg.training.checkpoint_dir,
            filename="age-estimation-{epoch:02d}-{val_loss:.2f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=cfg.training.early_stopping_patience,
            mode="min",
            verbose=True,
        ),
    ]
    
    # Logger
    git_commit_id = get_git_commit_id()
    
    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.logging.experiment_name,
        tracking_uri=cfg.logging.mlflow_tracking_uri,
        run_name=cfg.logging.run_name,
    )
    
    # Log hyperparameters and git commit
    mlflow_logger.log_hyperparams({
        **OmegaConf.to_container(cfg, resolve=True),
        "git_commit_id": git_commit_id,
    })
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        logger=mlflow_logger,
        callbacks=callbacks,
        deterministic=cfg.training.deterministic,
        log_every_n_steps=cfg.training.log_every_n_steps,
        val_check_interval=cfg.training.val_check_interval,
    )
    
    # Train
    trainer.fit(model, data_module)
    
    # Test
    trainer.test(model, data_module)


if __name__ == "__main__":
    train()