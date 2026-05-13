import lightning as L
import torch
import torch.nn.functional as F
import torchmetrics
from omegaconf import OmegaConf


class AgeRegressionModule(L.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        self.model = model
        self.save_hyperparameters(
            {"cfg": OmegaConf.to_container(cfg, resolve=True)}, ignore=["model"]
        )
        self.cfg = cfg
        loss_name = cfg.model.loss.name
        head = cfg.model.head
        self.is_reg = head == "regression"
        self.is_dldl = head == "dldl"
        self.is_hybrid = head == "hybrid"
        self.is_coral = head == "coral"
        self.is_cls = head == "classification"

        if self.is_dldl or self.is_hybrid:
            self.criterion_kl = torch.nn.KLDivLoss(reduction="batchmean")

        if loss_name == "mse":
            self.criterion_reg = torch.nn.MSELoss()

        elif loss_name == "mae":
            self.criterion_reg = torch.nn.L1Loss()

        if self.is_coral:
            self.criterion_coral = torch.nn.BCEWithLogitsLoss()

        if self.is_cls:
            self.criterion_ce = torch.nn.CrossEntropyLoss()

        self.val_mae = torchmetrics.MeanAbsoluteError()
        self.test_mae = torchmetrics.MeanAbsoluteError()

        self.register_buffer("age_range", torch.arange(cfg.model.num_classes).float())

    def mean_variance_loss(self, probs, target):
        """
        probs: [B, C]
        target: [B]
        """

        age_range = self.age_range.unsqueeze(0)
        mean = torch.sum(probs * age_range, dim=1)
        variance = torch.sum(
            probs * (age_range - mean.unsqueeze(1)) ** 2,
            dim=1,
        )
        loss_mean = F.mse_loss(mean, target)
        loss_var = variance.mean()

        return loss_mean, loss_var

    def forward(self, x):
        return self.model(x)

    # ================= TRAIN =================
    def training_step(self, batch, batch_idx):
        if self.is_reg:
            images, _ = batch
        else:
            images = batch["image"]

        outputs = self(images)

        # -------- REGRESSION --------
        if self.is_reg:
            images, target = batch
            preds = outputs.view(-1)
            loss = self.criterion_reg(preds, target)

        # -------- DLDL --------
        elif self.is_dldl:
            logits = outputs
            log_probs = F.log_softmax(logits, dim=1)
            loss = self.criterion_kl(log_probs, batch["dist"])

        # -------- HYBRID --------
        elif self.is_hybrid:
            logits, reg = outputs

            log_probs = F.log_softmax(logits, dim=1)

            loss_kl = self.criterion_kl(log_probs, batch["dist"])
            loss_reg = F.mse_loss(reg, batch["age"])

            lambda_reg = self.cfg.model.get("lambda_reg", 1.0)
            loss = loss_kl + lambda_reg * loss_reg

        # -------- CORAL --------
        elif self.is_coral:
            logits = outputs
            loss = self.criterion_coral(logits, batch["coral"])

        # -------- CLASSIFICATION --------
        elif self.is_cls:
            logits = outputs
            target = batch["age"].long()
            loss_ce = self.criterion_ce(logits, target)
            probs = F.softmax(logits, dim=1)
            loss_mean, loss_var = self.mean_variance_loss(
                probs,
                batch["age"],
            )
            lambda_mean = self.cfg.model.loss.get("lambda_mean", 0.2)
            lambda_var = self.cfg.model.loss.get("lambda_var", 0.05)
            loss = loss_ce + lambda_mean * loss_mean + lambda_var * loss_var
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)

        return loss

    # ================= COMMON PRED =================
    def _predict_age(self, outputs):
        # -------- REGRESSION ---------
        if self.is_reg:
            return outputs.view(-1)

        # -------- DLDL / HYBRID ------
        if self.is_dldl or self.is_hybrid:
            logits = outputs if self.is_dldl else outputs[0]
            probs = F.softmax(logits, dim=1)
            return torch.sum(probs * self.age_range, dim=1)

        # ----------- CORAL -----------
        if self.is_coral:
            probs = torch.sigmoid(outputs)
            return probs.sum(dim=1)

        # ----------- CEMV ------------
        if self.is_cls:
            probs = F.softmax(outputs, dim=1)
            return torch.sum(probs * self.age_range, dim=1)

    # ================= VALID =================
    def validation_step(self, batch, batch_idx):
        if self.is_reg:
            images, _ = batch
        else:
            images = batch["image"]

        outputs = self(images)

        if self.is_reg:
            images, target = batch
            preds = outputs.view(-1)
            loss = self.criterion_reg(preds, target)

        elif self.is_dldl:
            logits = outputs
            log_probs = F.log_softmax(logits, dim=1)
            loss = self.criterion_kl(log_probs, batch["dist"])
            preds = self._predict_age(logits)

        elif self.is_hybrid:
            logits, reg = outputs

            log_probs = F.log_softmax(logits, dim=1)

            loss_kl = self.criterion_kl(log_probs, batch["dist"])
            loss_reg = F.mse_loss(reg, batch["age"])

            lambda_reg = self.cfg.model.loss.get("lambda_reg", 1.0)
            loss = loss_kl + lambda_reg * loss_reg

            preds = self._predict_age((logits, reg))

        elif self.is_coral:
            logits = outputs
            loss = self.criterion_coral(logits, batch["coral"])
            preds = self._predict_age(logits)

        elif self.is_cls:
            logits = outputs

            target_cls = batch["age"].long()

            loss_ce = self.criterion_ce(logits, target_cls)

            probs = F.softmax(logits, dim=1)

            loss_mean, loss_var = self.mean_variance_loss(
                probs,
                batch["age"],
            )

            lambda_mean = self.cfg.model.loss.get("lambda_mean", 0.2)
            lambda_var = self.cfg.model.loss.get("lambda_var", 0.05)

            loss = loss_ce + lambda_mean * loss_mean + lambda_var * loss_var

            preds = self._predict_age(logits)

        target = batch["age"] if not self.is_reg else batch[1]

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.val_mae(preds, target)
        self.log("val_mae", self.val_mae, prog_bar=True, on_epoch=True)

    # ================= TEST =================
    def test_step(self, batch, batch_idx):
        if self.is_reg:
            images, _ = batch
        else:
            images = batch["image"]

        outputs = self(images)

        preds = self._predict_age(outputs)
        target = batch["age"] if not self.is_reg else batch[1]

        self.test_mae(preds, target)
        self.log("test_mae", self.test_mae, prog_bar=True, on_epoch=True)

    # ================= PREDICT =================
    def predict_step(self, batch, batch_idx):
        images, filenames = batch
        outputs = self(images)
        preds, uncertainty = self.predict_age_with_uncertainty(outputs)

        return {
            "preds": preds,
            "uncertainty": uncertainty,
            "filenames": filenames,
        }

    # ================= OPT =================
    def configure_optimizers(self):
        opt_cfg = self.cfg.model.optimizer

        head_params = [p for n, p in self.model.named_parameters() if "backbone" not in n]
        param_groups = [{"params": head_params, "lr": opt_cfg.lr}]

        if opt_cfg.name == "adam":
            optimizer = torch.optim.Adam(param_groups, lr=opt_cfg.lr)
        elif opt_cfg.name == "adamw":
            optimizer = torch.optim.AdamW(
                param_groups, lr=opt_cfg.lr, weight_decay=opt_cfg.weight_decay
            )
        elif opt_cfg.name == "sgd":
            optimizer = torch.optim.SGD(
                param_groups,
                lr=opt_cfg.lr,
                momentum=opt_cfg.get("momentum", 0.9),
                weight_decay=opt_cfg.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_cfg.name}")

        scheduler_cfg = self.cfg.model.get("scheduler", None)
        if not scheduler_cfg or scheduler_cfg.get("name", None) is None:
            return optimizer

        scheduler = self._build_scheduler(optimizer, scheduler_cfg)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": scheduler_cfg.get("interval", "epoch"),
                "frequency": scheduler_cfg.get("frequency", 1),
                "monitor": scheduler_cfg.get("monitor", "val_loss"),
            },
        }

    def _build_scheduler(self, optimizer, scheduler_cfg):
        name = scheduler_cfg.name

        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=scheduler_cfg.get("T_max", self.trainer.max_epochs),
                eta_min=scheduler_cfg.get("eta_min", 1e-6),
            )
        if name == "step":
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=scheduler_cfg.step_size,
                gamma=scheduler_cfg.get("gamma", 0.1),
            )
        if name == "reduce_on_plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=scheduler_cfg.get("mode", "min"),
                factor=scheduler_cfg.get("factor", 0.5),
                patience=scheduler_cfg.get("patience", 5),
            )
        if name == "warmup_cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

            warmup = LinearLR(
                optimizer,
                start_factor=scheduler_cfg.get("warmup_start_factor", 0.1),
                total_iters=scheduler_cfg.get("warmup_epochs", 3),
            )
            cosine = CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs - scheduler_cfg.get("warmup_epochs", 3),
                eta_min=scheduler_cfg.get("eta_min", 1e-6),
            )
            return SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[scheduler_cfg.get("warmup_epochs", 3)],
            )

        raise ValueError(f"Unknown scheduler: {name}")

    def predict_age_with_uncertainty(self, outputs):
        # ---------- REGRESSION -----------
        if self.is_reg:
            preds = outputs.view(-1)
            return preds, None

        # - DLDL / HYBRID / CLASSIFICATION -
        if self.is_dldl or self.is_hybrid or self.is_cls:
            logits = outputs if not self.is_hybrid else outputs[0]
            probs = F.softmax(logits, dim=1)
            mean = torch.sum(probs * self.age_range, dim=1)
            variance = torch.sum(
                probs * (self.age_range - mean.unsqueeze(1)) ** 2,
                dim=1,
            )
            std = torch.sqrt(variance)

            return mean, std

        # ----------- CORAL -------------
        if self.is_coral:
            probs = torch.sigmoid(outputs)  # [B, num_classes-1] — P(age > k)
            mean = probs.sum(dim=1)  # предсказанный возраст
            p_first = 1 - probs[:, :1]  # [B, 1]
            p_middle = probs[:, :-1] - probs[:, 1:]  # [B, num_classes-2]
            p_last = probs[:, -1:]  # [B, 1]
            age_probs = torch.cat([p_first, p_middle, p_last], dim=1)  # [B, num_classes]
            age_probs = age_probs.clamp(min=0)  # на случай численных артефактов

            # Дисперсия — как у classification/dldl
            age_range = self.age_range  # [num_classes]
            variance = torch.sum(
                age_probs * (age_range - mean.unsqueeze(1)) ** 2,
                dim=1,
            )
            uncertainty = torch.sqrt(variance)

            return mean, uncertainty
