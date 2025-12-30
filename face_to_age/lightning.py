import torch
import torch.nn
import torchmetrics
import lightning as L
from omegaconf import DictConfig


class AgeRegressionModule(L.LightningModule):
    """Lightning модуль для задачи регрессии возраста"""
    
    def __init__(self, model: torch.nn.Module, cfg: DictConfig):
        super().__init__()
        self.model = model
        self.cfg = cfg
        
        # Loss function
        if cfg.training.loss.name == "mse":
            self.criterion = torch.nn.MSELoss()
        elif cfg.training.loss.name == "mae":
            self.criterion = torch.nn.L1Loss()
        else:
            raise ValueError(f"Unknown loss: {cfg.training.loss.name}")
        
        # Metrics
        self.val_mae = torchmetrics.MeanAbsoluteError()
        self.test_mae = torchmetrics.MeanAbsoluteError()
        
        # Сохраняем гиперпараметры
        self.save_hyperparameters(ignore=['model'])
    
    def forward(self, inputs):
        return self.model(inputs).squeeze(1)
    
    def training_step(self, batch, batch_idx):
        inputs, target = batch
        preds = self.forward(inputs)
        loss = self.criterion(preds, target)
        
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        preds = self.forward(inputs)
        loss = self.criterion(preds, target)
        
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )
        
        self.val_mae(preds, target)
        self.log(
            "val_mae",
            self.val_mae,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )
    
    def test_step(self, batch, batch_idx):
        inputs, target = batch
        preds = self.forward(inputs)
        loss = self.criterion(preds, target)
        
        self.log(
            "test_loss",
            loss,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )
        
        self.test_mae(preds, target)
        self.log(
            "test_mae",
            self.test_mae,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )
    
    def predict_step(self, batch, batch_idx):
        """Шаг предсказания"""
        if isinstance(batch, tuple):
            inputs, filenames = batch
            preds = self.forward(inputs)
            return preds, filenames
        else:
            inputs = batch
            preds = self.forward(inputs)
            return preds
    
    def configure_optimizers(self):
        """Конфигурация оптимизатора из конфига"""
        opt_cfg = self.cfg.training.optimizer
        
        if opt_cfg.name == "adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                # lr=opt_cfg.lr,
                # betas=opt_cfg.betas,
                # eps=opt_cfg.eps,
                # weight_decay=opt_cfg.weight_decay,
            )
        elif opt_cfg.name == "sgd":
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=opt_cfg.lr,
                momentum=opt_cfg.get("momentum", 0.9),
                weight_decay=opt_cfg.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_cfg.name}")
        
        return optimizer