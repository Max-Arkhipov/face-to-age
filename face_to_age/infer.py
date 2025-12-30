from pathlib import Path

import lightning as L
import torch
from hydra import main
from omegaconf import DictConfig

from face_to_age.data import UTKFaceDataModule
from face_to_age.lightning import AgeRegressionModule
from face_to_age.logger import build_logger
from face_to_age.model import ConvRegressor, SimpleRegressor
from utils.dvc_utils import dvc_pull_if_needed
from utils.predictions import save_predictions


@main(version_base=None, config_path="../configs", config_name="config")
def infer(cfg: DictConfig):
    print("=" * 80)
    print("CONFIG:")
    print(cfg)
    print("=" * 80)

    dvc_pull_if_needed([cfg.dataset.predict_data_dir])

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

    ckpt_path = Path(cfg.paths.checkpoints_dir) / cfg.infer.checkpoint_name

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading model from {ckpt_path}")
    state_dict = torch.load(ckpt_path)
    module.model.load_state_dict(state_dict)

    logger = build_logger(cfg)
    trainer = L.Trainer(logger=logger)

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
