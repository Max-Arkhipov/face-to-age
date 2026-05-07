from lightning.pytorch.callbacks import BaseFinetuning


class BackboneFinetuning(BaseFinetuning):
    def __init__(self, unfreeze_epoch=5, backbone_lr=1e-5):
        super().__init__()
        self.unfreeze_epoch = unfreeze_epoch
        self.backbone_lr = backbone_lr

    def freeze_before_training(self, pl_module):
        # Замораживаем backbone
        self.freeze(pl_module.model.backbone)

    def finetune_function(self, pl_module, current_epoch, optimizer):
        if current_epoch == self.unfreeze_epoch:
            print(f"\n>>> Unfreezing backbone at epoch {current_epoch}")

            self.unfreeze_and_add_param_group(
                modules=pl_module.model.backbone,
                optimizer=optimizer,
                lr=self.backbone_lr,
            )

    """def finetune_function(self, pl_module, current_epoch, optimizer):
        if current_epoch == self.unfreeze_epoch:
            # Размораживаем только четвертый (последний) блок ResNet
            # В torchvision.models.resnet18 это слой 'layer4'
            target_module = pl_module.model.backbone.layer4

            print(f"\n>>> Unfreezing layer4 at epoch {current_epoch}")

            self.unfreeze_and_add_param_group(
                modules=target_module,
                optimizer=optimizer,
                lr=self.backbone_lr,
            )"""
