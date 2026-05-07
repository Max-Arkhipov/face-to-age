from pathlib import Path

import lightning as L
import torch
from hydra import main
from omegaconf import DictConfig, OmegaConf

from face_to_age.data import UTKFaceDataModule
from face_to_age.lightning import AgeRegressionModule
from face_to_age.model import AgeModel
from utils.dvc_utils import dvc_pull_if_needed
from utils.predictions import save_predictions


@main(version_base=None, config_path="../configs", config_name="config")
def infer(cfg: DictConfig):
    print("=" * 80)
    print("CONFIG:")
    print(cfg)
    print("=" * 80)

    dvc_pull_if_needed([cfg.dataset.predict_data_dir])

    # Model
    ckpt_path = Path(cfg.paths.checkpoints_dir) / cfg.infer.checkpoint_name

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Путь к чекпоинту из конфига
    ckpt_path = f"{cfg.paths.checkpoints_dir}/{cfg.infer.checkpoint_name}"

    # Читаем конфиг прямо из чекпоинта, чтобы точно знать архитектуру
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # train_cfg = OmegaConf.create(checkpoint["hyper_parameters"])
    train_cfg = OmegaConf.create(checkpoint["hyper_parameters"]["cfg"])

    # DataModule
    datamodule = UTKFaceDataModule(train_cfg)

    # 3. Инициализируем AgeModel (тот самый backbone + head)
    # Используем train_cfg, чтобы архитектура совпала с весами
    base_model = AgeModel(train_cfg)

    print(f"Loading model from {ckpt_path}")
    module = AgeRegressionModule.load_from_checkpoint(
        ckpt_path, model=base_model, cfg=train_cfg, strict=False, weights_only=False
    )

    # module = AgeRegressionModule.load_from_checkpoint(ckpt_path, weights_only=False)

    # logger = build_logger(train_cfg)
    trainer = L.Trainer()

    predictions = trainer.predict(module, datamodule=datamodule)

    print(f"Prediction batches: {len(predictions)}")
    all_preds, all_files = [], []

    for batch in predictions:
        preds, filenames = batch
        all_preds.extend(preds.tolist())
        all_files.extend(filenames)

    # Save csv
    output_path = Path(cfg.paths.pred_dir) / cfg.infer.output_name

    save_predictions(
        all_preds,
        all_files,
        output_path,
        use_filenames=cfg.infer.use_filenames,
    )

    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    infer()
