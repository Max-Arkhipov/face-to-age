from typing import Any, Tuple
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import lightning as L
from omegaconf import DictConfig


def build_transforms(cfg: DictConfig, train: bool):
    transforms_list = []

    transforms_list.append(
        transforms.Resize(tuple(cfg.image.size))
    )

    if train and cfg.image.train.horizontal_flip_p > 0:
        transforms_list.append(
            transforms.RandomHorizontalFlip(
                p=cfg.image.train.horizontal_flip_p
            )
        )

    transforms_list.append(transforms.ToTensor())

    transforms_list.append(
        transforms.Normalize(
            mean=cfg.image.normalize.mean,
            std=cfg.image.normalize.std,
        )
    )

    return transforms.Compose(transforms_list)


class UTKFaceDataset(Dataset):
    """
    UTKFace age regression dataset.
    Label (age) is parsed from filename: age_gender_race_*.jpg
    """
    
    def __init__(self, data_dir: str, transform: transforms.Compose):
        self.paths = list(Path(data_dir).glob("*.jpg"))
        self.transform = transform

        if len(self.paths) == 0:
            raise FileNotFoundError(f"No .jpg images found in {data_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path = self.paths[idx]

        image = Image.open(path).convert("RGB")
        age = float(path.name.split("_")[0])

        image = self.transform(image)

        return image, torch.tensor(age, dtype=torch.float32)
    

class UTKFacePredictDataset(Dataset):
    def __init__(self, data_dir: str, transform):
        self.paths = list(Path(data_dir).glob("*.jpg"))
        self.transform = transform

        if len(self.paths) == 0:
            raise FileNotFoundError(f"No .jpg images found in {data_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]

        image = Image.open(path).convert("RGB")
        image = self.transform(image)

        return image, path.name
    

def init_dataloader(
    dataset: Any,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    persistent_workers: bool,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )


class UTKFaceDataModule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage=None):
        if stage in ("fit", None):
            self.train_dataset = UTKFaceDataset(
                self.cfg.dataset.train_data_dir,
                transform=build_transforms(
                    self.cfg.preprocessing, train=True
                ),
            )

            self.val_dataset = UTKFaceDataset(
                self.cfg.dataset.val_data_dir,
                transform=build_transforms(
                    self.cfg.preprocessing, train=False
                ),
            )

        if stage in ("test", None):
            self.test_dataset = UTKFaceDataset(
                self.cfg.dataset.test_data_dir,
                transform=build_transforms(
                    self.cfg.preprocessing, train=False
                ),
            )

        if stage == "predict":
            self.predict_dataset = UTKFacePredictDataset(
                self.cfg.dataset.predict_data_dir,
                transform=build_transforms(
                    self.cfg.preprocessing, train=False
                ),
            )

    def train_dataloader(self):
        return init_dataloader(
            self.train_dataset,
            batch_size=self.cfg.dataloader.train_batch_size,
            shuffle=True,
            num_workers=self.cfg.dataloader.num_workers,
            persistent_workers=self.cfg.dataloader.persistent_workers,
        )

    def val_dataloader(self):
        return init_dataloader(
            self.val_dataset,
            batch_size=self.cfg.dataloader.predict_batch_size,
            shuffle=False,
            num_workers=self.cfg.dataloader.num_workers,
            persistent_workers=self.cfg.dataloader.persistent_workers,
        )

    def test_dataloader(self):
        return init_dataloader(
            self.test_dataset,
            batch_size=self.cfg.dataloader.predict_batch_size,
            shuffle=False,
            num_workers=self.cfg.dataloader.num_workers,
            persistent_workers=self.cfg.dataloader.persistent_workers,
        )

    def predict_dataloader(self):
        return init_dataloader(
            self.predict_dataset,
            batch_size=self.cfg.dataloader.predict_batch_size,
            shuffle=False,
            num_workers=self.cfg.dataloader.num_workers,
            persistent_workers=self.cfg.dataloader.persistent_workers,
        )