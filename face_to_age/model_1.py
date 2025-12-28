import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision import models
from torchmetrics import MeanAbsoluteError

class AgeRegressionModel(L.LightningModule):
    def __init__(
        self,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        pretrained: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Backbone
        self.model = models.resnet18(pretrained=pretrained)

        # Replace classifier
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, 1)

        self.lr = lr
        self.weight_decay = weight_decay


class AgeEstimationModel(pl.LightningModule):
    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Load backbone
        if backbone == "resnet18":
            self.backbone = models.resnet18(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == "resnet34":
            self.backbone = models.resnet34(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Regression head
        self.head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )
        
        # Loss and metrics
        self.criterion = nn.L1Loss()
        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()
        self.test_mae = MeanAbsoluteError()
    
    def forward(self, x):
        features = self.backbone(x)
        age = self.head(features)
        return age.squeeze(-1)
    
    def training_step(self, batch, batch_idx):
        images, ages = batch
        predictions = self(images)
        loss = self.criterion(predictions, ages.float())
        
        self.train_mae(predictions, ages)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_mae", self.train_mae, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, ages = batch
        predictions = self(images)
        loss = self.criterion(predictions, ages.float())
        
        self.val_mae(predictions, ages)
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_mae", self.val_mae, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        images, ages = batch
        predictions = self(images)
        loss = self.criterion(predictions, ages.float())
        
        self.test_mae(predictions, ages)
        
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_mae", self.test_mae, on_step=False, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            verbose=True,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }