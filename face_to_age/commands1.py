from pathlib import Path
import fire
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from face_to_age.train import train


def load_cfg(overrides=None):
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    overrides = list(overrides) if overrides else []

    # configs лежит в корне проекта
    config_dir = Path(__file__).resolve().parents[1] / "configs"

    with initialize_config_dir(
        config_dir=str(config_dir),
        version_base=None,
    ):
        cfg = compose(
            config_name="config",
            overrides=overrides,
        )

    return cfg


def train_cmd(*overrides):
    cfg = load_cfg(overrides)
    train(cfg)


def main():
    fire.Fire({
        "train": train_cmd,
    })


if __name__ == "__main__":
    main()