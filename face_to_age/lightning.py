import lightning as L
import torch
import torch.nn.functional as F
import torchmetrics


class AgeRegressionModule(L.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        self.model = model
        self.cfg = cfg

        loss_name = cfg.model.loss.name

        head = cfg.model.head
        self.is_reg = head == "regression"
        self.is_dldl = head == "dldl"
        self.is_hybrid = head == "hybrid"
        self.is_coral = head == "coral"

        # Loss
        if self.is_dldl or self.is_hybrid:
            self.criterion_kl = torch.nn.KLDivLoss(reduction="batchmean")

        if loss_name == "mse":
            self.criterion_reg = torch.nn.MSELoss()
        elif loss_name == "mae":
            self.criterion_reg = torch.nn.L1Loss()

        if self.is_coral:
            self.criterion_coral = torch.nn.BCEWithLogitsLoss()

        # Metrics
        self.val_mae = torchmetrics.MeanAbsoluteError()
        self.test_mae = torchmetrics.MeanAbsoluteError()

        # Буфер для ожидания возраста
        self.register_buffer("age_range", torch.arange(cfg.model.num_classes).float())

        self.save_hyperparameters(ignore=["model"])

    def forward(self, x):
        return self.model(x)

    # ================= TRAIN =================
    def training_step(self, batch, batch_idx):
        outputs = self(batch["image"] if not self.is_reg else batch[0])

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

        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    # ================= COMMON PRED =================
    def _predict_age(self, outputs):
        # REG
        if self.is_reg:
            return outputs.view(-1)

        # DLDL / HYBRID
        if self.is_dldl or self.is_hybrid:
            logits = outputs if self.is_dldl else outputs[0]
            probs = F.softmax(logits, dim=1)
            return torch.sum(probs * self.age_range, dim=1)

        # CORAL
        if self.is_coral:
            probs = torch.sigmoid(outputs)
            return probs.sum(dim=1)

    # ================= VALID =================
    def validation_step(self, batch, batch_idx):
        outputs = self(batch["image"] if not self.is_reg else batch[0])

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

        target = batch["age"] if not self.is_reg else batch[1]

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.val_mae(preds, target)
        self.log("val_mae", self.val_mae, prog_bar=True, on_epoch=True)

    # ================= TEST =================
    def test_step(self, batch, batch_idx):
        outputs = self(batch["image"] if not self.is_reg else batch[0])

        preds = self._predict_age(outputs)
        target = batch["age"] if not self.is_reg else batch[1]

        self.test_mae(preds, target)
        self.log("test_mae", self.test_mae, prog_bar=True, on_epoch=True)

    # ================= PREDICT =================
    def predict_step(self, batch, batch_idx):
        images, filenames = batch
        outputs = self(images)
        preds = self._predict_age(outputs)
        return preds, filenames

    # ================= OPT =================
    def configure_optimizers(self):
        opt_cfg = self.cfg.model.optimizer

        # Разделяем параметры
        head_params = []
        backbone_params = []

        for name, param in self.model.named_parameters():
            if "backbone" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        param_groups = [
            {
                "params": head_params,
                "lr": opt_cfg.lr,
            }
        ]

        # backbone добавится позже через callback

        if opt_cfg.name == "adam":
            return torch.optim.Adam(param_groups, lr=opt_cfg.lr)

        if opt_cfg.name == "adamw":
            return torch.optim.AdamW(
                param_groups,
                lr=opt_cfg.lr,
                weight_decay=opt_cfg.weight_decay,
            )

        if opt_cfg.name == "sgd":
            return torch.optim.SGD(
                param_groups,
                lr=opt_cfg.lr,
                momentum=opt_cfg.get("momentum", 0.9),
                weight_decay=opt_cfg.weight_decay,
            )

        raise ValueError(f"Unknown optimizer: {opt_cfg.name}")
