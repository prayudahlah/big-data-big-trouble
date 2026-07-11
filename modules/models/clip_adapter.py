import torch
import torch.nn as nn


CLIP_MODEL_NAMES = {
    "ViT-B/32": ("ViT-B-32", "laion2b_s34b_b79k"),
    "ViT-B/16": ("ViT-B-16", "laion2b_s34b_b88k"),
    "ViT-L/14": ("ViT-L-14", "laion2b_s32b_b82k"),
    "ViT-H/14": ("ViT-H-14", "laion2b_s32b_b79k"),
}


class CLIPAdapter(nn.Module):
    def __init__(self, clip_variant="ViT-B/32", num_classes=3, device="cuda"):
        super().__init__()
        try:
            import open_clip
        except ImportError:
            raise ImportError(
                "open_clip is required. Install with: pip install open-clip-torch"
            )

        arch, pretrained = CLIP_MODEL_NAMES.get(
            clip_variant, ("ViT-B-32", "laion2b_s34b_b79k")
        )
        self.clip, _, _ = open_clip.create_model_and_transforms(
            arch, pretrained=pretrained, device=device
        )
        self.clip.eval()
        self.tokenizer = open_clip.get_tokenizer(arch)
        self.visual_projection = nn.Linear(self.clip.visual.output_dim, num_classes)
        self.num_classes = num_classes
        self.device = device

    def encode_image(self, images):
        features = self.clip.encode_image(images)
        features = features / features.norm(dim=-1, keepdim=True)
        return features

    def zero_shot_classify(self, images, class_prompts):
        texts = self.tokenizer(class_prompts).to(self.device)
        text_features = self.clip.encode_text(texts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = self.encode_image(images)
        logits = image_features @ text_features.T * self.clip.logit_scale.exp()
        return logits

    def forward(self, images):
        features = self.encode_image(images)
        return self.visual_projection(features)
