from lightning.pytorch.callbacks import BaseFinetuning

# Маппинг архитектур на список слоёв в порядке от конца к началу
BACKBONE_LAYERS = {
    "resnet18": ["layer4", "layer3", "layer2", "layer1"],
    "resnet50": ["layer4", "layer3", "layer2", "layer1"],
    "efficientnet_b0": [
        "features.8",
        "features.7",
        "features.6",
        "features.5",
        "features.4",
        "features.3",
        "features.2",
        "features.1",
    ],
}


def get_module_by_path(module, path: str):
    """Получает вложенный модуль по пути вида 'features.8'."""
    for part in path.split("."):
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part, None)
        if module is None:
            return None
    return module


class BackboneFinetuning(BaseFinetuning):
    def __init__(
        self, unfreeze_epoch=5, backbone_lr=1e-5, unfreeze_layers=None, backbone_name=None
    ):
        super().__init__()
        self.unfreeze_epoch = unfreeze_epoch
        self.backbone_lr = backbone_lr
        self.backbone_name = backbone_name
        # None / "all" — весь backbone
        # integer N — последние N слоёв согласно BACKBONE_LAYERS
        # list ["layer4", "layer3"] — конкретные слои
        self.unfreeze_layers = unfreeze_layers

    def freeze_before_training(self, pl_module):
        self.freeze(pl_module.model.backbone)

    def finetune_function(self, pl_module, current_epoch, optimizer):
        if current_epoch != self.unfreeze_epoch:
            return

        backbone = pl_module.model.backbone

        # Весь backbone
        if not self.unfreeze_layers or self.unfreeze_layers == "all":
            print(f"\n>>> Unfreezing full backbone at epoch {current_epoch}")
            self.unfreeze_and_add_param_group(backbone, optimizer, lr=self.backbone_lr)
            return

        # Последние N слоёв по маппингу архитектуры
        if isinstance(self.unfreeze_layers, int):
            if self.backbone_name not in BACKBONE_LAYERS:
                raise ValueError(
                    f"backbone_name='{self.backbone_name}' not in BACKBONE_LAYERS. "
                    f"Known: {list(BACKBONE_LAYERS.keys())}"
                )
            layers_to_unfreeze = BACKBONE_LAYERS[self.backbone_name][: self.unfreeze_layers]
        else:
            layers_to_unfreeze = list(self.unfreeze_layers)

        for layer_path in layers_to_unfreeze:
            module = get_module_by_path(backbone, layer_path)
            if module is None:
                print(f">>> WARNING: '{layer_path}' not found in backbone, skipping")
                continue
            print(f"\n>>> Unfreezing {layer_path} at epoch {current_epoch}")
            self.unfreeze_and_add_param_group(module, optimizer, lr=self.backbone_lr)
