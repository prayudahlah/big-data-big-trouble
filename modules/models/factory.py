import torch
import timm
import torch.nn as nn


def get_encoder(name: str) -> nn.Module:
    model = timm.create_model(name, pretrained=True, num_classes=0)
    with torch.no_grad():
        dummy = torch.randn(1, 3, 224, 224)
        model.num_features = model(dummy).shape[-1]
    return model


def get_classifier(
    num_features: int, num_classes: int = 3, dropout: float = 0.3
) -> nn.Sequential:
    return nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(512, num_classes),
    )


class TrashClassifier(nn.Module):
    def __init__(self, encoder_name: str, num_classes: int = 3):
        super().__init__()
        self.encoder_name = encoder_name
        self.encoder = get_encoder(encoder_name)
        self.classifier = get_classifier(self.encoder.num_features, num_classes)

    def forward(self, x):
        return self.classifier(self.encoder(x))

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = True
