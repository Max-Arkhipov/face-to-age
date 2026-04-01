import torch
import torchvision.models as models


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


class ResNetRegressor(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        in_features = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(in_features, 1)

    def forward(self, x):
        return self.model(x.float())


class ResNetRegressor_last(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        in_features = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(in_features, 1)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.fc = torch.nn.Linear(in_features, 1)

    def forward(self, x):
        return self.model(x.float())


class ResNetRegressor_head(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        in_features = self.model.fc.in_features
        self.model.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.model(x.float())
