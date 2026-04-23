import torch
import torch.nn as nn
import torchvision.models as models

BACKBONES = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT, "fc"),
    "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT, "fc"),
    "efficientnet_b0": (
        models.efficientnet_b0,
        models.EfficientNet_B0_Weights.DEFAULT,
        "classifier",
    ),
}


def build_backbone(name: str, pretrained: bool):
    factory, weights, fc_attr = BACKBONES[name]
    backbone = factory(weights=weights if pretrained else None)

    # Узнаём in_features и убираем последний слой
    if fc_attr == "fc":
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
    elif fc_attr == "classifier":
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()

    return backbone, in_features


class AgeModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        model_cfg = cfg.model

        self.head_type = model_cfg.head  # regression | dldl | hybrid | coral
        num_classes = model_cfg.num_classes

        self.backbone, in_features = build_backbone(
            name=model_cfg.backbone,
            pretrained=model_cfg.get("pretrained", True),
        )

        if model_cfg.get("finetune", {}).get("enabled", False):
            for param in self.backbone.parameters():
                param.requires_grad = False

        # -------- HEAD --------
        if self.head_type == "regression":
            self.head = nn.Linear(in_features, 1)

        elif self.head_type == "dldl":
            self.head = nn.Linear(in_features, num_classes)

        elif self.head_type == "hybrid":
            self.shared = nn.Sequential(
                nn.Linear(in_features, 128),
                nn.ReLU(),
            )
            self.dist_head = nn.Linear(128, num_classes)
            self.reg_head = nn.Linear(128, 1)

        elif self.head_type == "coral":
            self.head = nn.Linear(in_features, num_classes - 1)

        else:
            raise ValueError(f"Unknown head type: {self.head_type}")

    def forward(self, x):
        features = self.backbone(x.float())

        if self.head_type == "regression":
            return self.head(features).squeeze(1)

        elif self.head_type == "dldl":
            return self.head(features)

        elif self.head_type == "hybrid":
            shared = self.shared(features)
            return self.dist_head(shared), self.reg_head(shared).squeeze(1)

        elif self.head_type == "coral":
            return self.head(features)


class SimpleRegressor(torch.nn.Module):
    """Linear model for age regression from images"""

    def __init__(self, image_size: int = 96):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(3 * image_size * image_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.model(x)


class ConvRegressor(torch.nn.Module):
    """Convolutional model for age regression from images"""

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, 3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(kernel_size=4),
            torch.nn.Flatten(),
            torch.nn.Linear(128 * 7 * 7, 96),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(96, 1),
        )

    def forward(self, x):
        return self.model(x.float())


class ConvRegressor_256(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 256, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((7, 7)),
            torch.nn.Flatten(),
            torch.nn.Linear(256 * 7 * 7, 96),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(96, 1),
        )

    def forward(self, x):
        return self.model(x.float())
