import subprocess

from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig, OmegaConf


def get_git_commit_id():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
        return commit
    except Exception:
        return "unknown"


def build_logger(cfg: DictConfig):
    if cfg.logger.name != "mlflow":
        raise ValueError(f"Unknown logger: {cfg.logger.name}")

    # Сборка имени из параметров
    backbone = cfg.model.get("backbone", "no_backbone")
    head = cfg.model.get("head", "no_head")
    lr = cfg.model.optimizer.lr

    # Формируем красивую строку
    generated_run_name = f"{backbone}-{head}-lr:{lr}"

    logger = MLFlowLogger(
        experiment_name=cfg.logger.experiment_name,
        tracking_uri=cfg.logger.tracking_uri,
        run_name=generated_run_name,
    )

    # Конвертируем конфиг в dict и логируем как гиперпараметры
    params = OmegaConf.to_container(cfg, resolve=True)
    params["git_commit"] = get_git_commit_id()
    logger.log_hyperparams(params)

    return logger
