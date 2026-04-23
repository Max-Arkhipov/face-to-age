from pathlib import Path
from typing import Any

import lightning as L
import torch
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def create_label_distribution(age, num_classes=117, sigma=2):
    x = torch.arange(num_classes).float()
    dist = torch.exp(-((x - age) ** 2) / (2 * sigma**2))
    dist = dist / dist.sum()
    return dist


def create_coral_target(age, num_classes):
    levels = torch.arange(num_classes - 1)
    return (age > levels).float()


def build_transforms(cfg: DictConfig, train: bool):
    transforms_list = []

    transforms_list.append(transforms.Resize(tuple(cfg.image.size)))

    if train and cfg.image.train.horizontal_flip_p > 0:
        transforms_list.append(transforms.RandomHorizontalFlip(p=cfg.image.train.horizontal_flip_p))

    transforms_list.append(transforms.ToTensor())

    transforms_list.append(
        transforms.Normalize(
            mean=cfg.image.normalize.mean,
            std=cfg.image.normalize.std,
        )
    )

    return transforms.Compose(transforms_list)


class UTKFaceDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        transform: transforms.Compose,
        use_dldl: bool = False,
        use_coral: bool = False,
        num_classes: int = 117,
        sigma: float = 2,
    ):
        self.paths = list(Path(data_dir).glob("*.jpg"))
        self.transform = transform

        self.use_coral = use_coral
        self.use_dldl = use_dldl
        self.num_classes = num_classes
        self.sigma = sigma

        if len(self.paths) == 0:
            raise FileNotFoundError(f"No .jpg images found in {data_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]

        image = Image.open(path).convert("RGB")
        age = float(path.name.split("_")[0])

        image = self.transform(image)
        age_tensor = torch.tensor(age, dtype=torch.float32)

        # -------- CORAL --------
        if self.use_coral:
            coral = create_coral_target(age, self.num_classes)
            return {
                "image": image,
                "age": age_tensor,
                "coral": coral,
            }

        # -------- DLDL --------
        if self.use_dldl:
            dist = create_label_distribution(
                age,
                num_classes=self.num_classes,
                sigma=self.sigma,
            ).float()

            return {
                "image": image,
                "age": age_tensor,
                "dist": dist,
            }

        # -------- REG --------
        return image, age_tensor


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
        use_dldl = self.cfg.model.loss.name in ["dldl", "dldl_hybrid"]
        use_coral = self.cfg.model.loss.name == "coral"
        sigma = self.cfg.model.loss.get("sigma", 2.0)

        if stage in ("fit", None):
            self.train_dataset = UTKFaceDataset(
                self.cfg.dataset.train_data_dir,
                transform=build_transforms(self.cfg.preprocessing, train=True),
                use_dldl=use_dldl,
                use_coral=use_coral,
                num_classes=self.cfg.model.num_classes,
                sigma=sigma,
            )

            self.val_dataset = UTKFaceDataset(
                self.cfg.dataset.val_data_dir,
                transform=build_transforms(self.cfg.preprocessing, train=False),
                use_dldl=use_dldl,
                use_coral=use_coral,
                num_classes=self.cfg.model.num_classes,
                sigma=sigma,
            )

        if stage in ("test", None):
            self.test_dataset = UTKFaceDataset(
                self.cfg.dataset.test_data_dir,
                transform=build_transforms(self.cfg.preprocessing, train=False),
                use_dldl=use_dldl,
                use_coral=use_coral,
                num_classes=self.cfg.model.num_classes,
                sigma=sigma,
            )

        if stage == "predict":
            self.predict_dataset = UTKFacePredictDataset(
                self.cfg.dataset.predict_data_dir,
                transform=build_transforms(self.cfg.preprocessing, train=False),
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
