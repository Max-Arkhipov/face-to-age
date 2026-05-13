from pathlib import Path

import lightning as L
import pandas as pd
import torch
from hydra import main
from omegaconf import DictConfig, OmegaConf

from face_to_age.data import UTKFaceDataModule
from face_to_age.lightning import AgeRegressionModule
from face_to_age.model import AgeModel
from utils.dvc_utils import dvc_pull_if_needed


@main(version_base=None, config_path="../configs", config_name="config")
def infer(cfg: DictConfig):
    print("=" * 80)
    print("CONFIG:")
    print(cfg)
    print("=" * 80)

    dvc_pull_if_needed([cfg.dataset.predict_data_dir])

    ckpt_path = Path(cfg.paths.checkpoints_dir) / cfg.infer.checkpoint_name

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    ckpt_path = f"{cfg.paths.checkpoints_dir}/{cfg.infer.checkpoint_name}"

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    train_cfg = OmegaConf.create(checkpoint["hyper_parameters"]["cfg"])

    train_cfg.dataset.predict_data_dir = cfg.dataset.predict_data_dir
    datamodule = UTKFaceDataModule(train_cfg)
    base_model = AgeModel(train_cfg)

    print(f"Loading model from {ckpt_path}")
    module = AgeRegressionModule.load_from_checkpoint(
        ckpt_path, model=base_model, cfg=train_cfg, strict=False, weights_only=False
    )

    trainer = L.Trainer()
    predictions = trainer.predict(module, datamodule=datamodule)

    print(f"Prediction batches: {len(predictions)}")

    all_preds, all_uncertainty, all_files = [], [], []

    for batch in predictions:
        all_preds.extend(batch["preds"].tolist())
        all_uncertainty.extend(batch["uncertainty"].tolist())
        all_files.extend(batch["filenames"])

    output_path = Path(cfg.paths.pred_dir) / cfg.infer.output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {"filename": all_files, "age_pred": all_preds, "uncertainty": all_uncertainty}
    )
    df.to_csv(output_path, index=False)

    print(f"Predictions with uncertainty saved to {output_path}")


if __name__ == "__main__":
    infer()
