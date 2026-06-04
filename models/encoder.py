"""
models/encoder.py — CNN Encoder based on a pretrained ResNet backbone.

Architecture
------------
backbone  : ResNet (conv layers + avgpool), initially FROZEN.
adaptation: Linear(feature_dim → embed_size) → BatchNorm1d → ReLU
            Always trained, even during the frozen phase.

Fine-tuning
-----------
Call encoder.unfreeze_backbone() at `finetune_epoch` (see train.py).
After unfreezing, pass encoder.parameters() to a new (lower-LR) optimiser.
"""

import torch
import torch.nn as nn
import torchvision.models as models


# Supported backbones: (constructor, default weights, feature dimension)
_BACKBONES = {
    "resnet50":  (models.resnet50,  models.ResNet50_Weights.DEFAULT,  2048),
    "resnet101": (models.resnet101, models.ResNet101_Weights.DEFAULT, 2048),
}


class EncoderCNN(nn.Module):

    def __init__(self, embed_size: int, backbone: str = "resnet50"):
        super().__init__()
        if backbone not in _BACKBONES:
            raise ValueError(f"backbone must be one of {list(_BACKBONES)}")

        model_fn, weights, feat_dim = _BACKBONES[backbone]
        cnn = model_fn(weights=weights)

        # Strip the FC classification head; keep conv stages + avgpool.
        # ResNet children order: conv1, bn1, relu, maxpool, layer1-4, avgpool, fc
        # [:-1] removes fc  →  output shape: (B, feat_dim, 1, 1)
        self.backbone = nn.Sequential(*list(cnn.children())[:-1])

        # Adaptation head: project to word-embedding space
        self.adaptation = nn.Sequential(
            nn.Linear(feat_dim, embed_size),
            nn.BatchNorm1d(embed_size),
            nn.ReLU(inplace=True),
        )

        # Start with backbone frozen
        self.freeze_backbone()

    # ── Freeze / Unfreeze ─────────────────────────────────────────────────
    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False
        print("[Encoder] Backbone FROZEN  — only adaptation head is trained.")

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True
        print("[Encoder] Backbone UNFROZEN — fine-tuning all layers.")

    # ── Forward ───────────────────────────────────────────────────────────
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        images : (B, 3, 224, 224)

        Returns
        -------
        features : (B, embed_size)
        """
        # Compute backbone gradients only when it is unfrozen
        backbone_requires_grad = any(
            p.requires_grad for p in self.backbone.parameters()
        )
        with torch.set_grad_enabled(backbone_requires_grad):
            x = self.backbone(images)   # (B, feat_dim, 1, 1)

        x = x.flatten(1)               # (B, feat_dim)
        return self.adaptation(x)       # (B, embed_size)
