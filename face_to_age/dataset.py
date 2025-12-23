from pathlib import Path
from typing import Optional, Tuple

import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms


class UTKFaceDataset(Dataset):
    def __init__(self, data_dir: Path, transform: Optional[transforms.Compose] = None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # UTKFace filename format: [age]_[gender]_[race]_[date&time].jpg
        self.image_paths = list(self.data_dir.glob("*.jpg"))
        
        # Filter valid images
        self.samples = []
        for img_path in self.image_paths:
            try:
                age = int(img_path.stem.split("_")[0])
                if 0 <= age <= 116:  # Valid age range
                    self.samples.append((img_path, age))
            except (ValueError, IndexError):
                continue
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, age = self.samples[idx]
        
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, age


class UTKFaceDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: int = 224,
        train_split: float = 0.8,
        val_split: float = 0.1,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.train_split = train_split
        self.val_split = val_split
        
        # Transforms
        self.train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            full_dataset = UTKFaceDataset(self.data_dir, transform=None)
            
            total_size = len(full_dataset)
            train_size = int(total_size * self.train_split)
            val_size = int(total_size * self.val_split)
            test_size = total_size - train_size - val_size
            
            train_dataset, val_dataset, test_dataset = random_split(
                full_dataset,
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(42),
            )
            
            # Apply transforms
            self.train_dataset = DatasetWithTransform(train_dataset, self.train_transform)
            self.val_dataset = DatasetWithTransform(val_dataset, self.val_transform)
            self.test_dataset = DatasetWithTransform(test_dataset, self.val_transform)
        
        if stage == "test" or stage is None:
            if not hasattr(self, "test_dataset"):
                full_dataset = UTKFaceDataset(self.data_dir, transform=None)
                self.test_dataset = DatasetWithTransform(full_dataset, self.val_transform)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class DatasetWithTransform(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    
    def __len__(self):
        return len(self.subset)
    
    def __getitem__(self, idx):
        image, age = self.subset[idx]
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, age