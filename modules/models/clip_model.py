import open_clip
import torch
import torch.nn as nn


class CLIPZeroShot(nn.Module):
    def __init__(self, model_name="ViT-B-32", class_prompts=None, pretrained="openai", device="cuda"):
        super().__init__()
        self.device = device
        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.tokenizer = open_clip.get_tokenizer(model_name)

        if class_prompts is None:
            class_prompts = [
                "a photo of recyclable waste",
                "a photo of electronic waste",
                "a photo of organic waste",
            ]
        text_tokens = self.tokenizer(class_prompts).to(device)
        with torch.no_grad():
            self.text_features = self.model.encode_text(text_tokens)
            self.text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)

    def forward(self, images):
        with torch.no_grad():
            image_features = self.model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = (100.0 * image_features @ self.text_features.T)
        return logits


class CLIPLinearProbe(nn.Module):
    def __init__(self, model_name="ViT-B-32", num_classes=3, pretrained="openai", device="cuda"):
        super().__init__()
        self.device = device
        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(device)
        self.model.eval()
        self.freeze_encoder()

        dummy = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            feat_dim = self.model.encode_image(dummy).shape[-1]

        self.classifier = nn.Linear(feat_dim, num_classes).to(device)

    def forward(self, images):
        with torch.no_grad():
            features = self.model.encode_image(images)
        return self.classifier(features.float())

    def freeze_encoder(self):
        for p in self.model.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):
        for p in self.model.parameters():
            p.requires_grad = True
