from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models as tv_models

DROPOUT = 0.25
HIDDEN_DIM = 256
IMAGE_PROJ_DIM = 256
TAB_HIDDEN_DIM = 128
TAB_EMBED_DIM = 128
SAFE_MASK_FILL_VALUE = -1.0e4


class TabularMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = TAB_HIDDEN_DIM,
        out_dim: int = TAB_EMBED_DIM,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def create_encoder(backbone_name: str):
    if backbone_name != "efficientnet_b0":
        raise ValueError(f"Unsupported deployment backbone: {backbone_name}")
    model = tv_models.efficientnet_b0(weights=None)
    feature_dim = model.classifier[1].in_features
    model.classifier[1] = nn.Identity()
    return model, feature_dim


class MaviaV5Net(nn.Module):
    """Exact parameterized architecture used by the supplied V5.1 checkpoints."""

    def __init__(self, exp_cfg: dict, tab_dim: int = 0) -> None:
        super().__init__()
        self.exp_cfg = dict(exp_cfg)
        self.use_images = bool(exp_cfg.get("use_images", False))
        self.use_cbc = bool(exp_cfg.get("use_cbc", False))
        self.backbone_name = exp_cfg.get("backbone", "efficientnet_b0")
        self.fusion_type = exp_cfg.get("fusion_type", "attention")
        self.encoder_policy = exp_cfg.get("encoder_policy", "shared")
        self.num_views = int(exp_cfg.get("num_views", 0))
        self.task_mode = exp_cfg.get("task_mode", "multitask")

        visual_dim = 0
        if self.use_images:
            if self.encoder_policy == "view_specific" and self.num_views > 1:
                encoders = []
                dims = []
                for _ in range(self.num_views):
                    encoder, dim = create_encoder(self.backbone_name)
                    encoders.append(encoder)
                    dims.append(dim)
                self.encoders = nn.ModuleList(encoders)
                enc_dim = dims[0]
            else:
                self.encoder, enc_dim = create_encoder(self.backbone_name)

            self.image_projector = nn.Sequential(
                nn.Linear(enc_dim, IMAGE_PROJ_DIM),
                nn.LayerNorm(IMAGE_PROJ_DIM),
                nn.ReLU(inplace=True),
                nn.Dropout(DROPOUT),
            )
            visual_dim = IMAGE_PROJ_DIM

            if self.fusion_type == "attention":
                self.attn = nn.Sequential(
                    nn.Linear(IMAGE_PROJ_DIM, HIDDEN_DIM),
                    nn.Tanh(),
                    nn.Linear(HIDDEN_DIM, 1),
                )

        tab_out_dim = 0
        if self.use_cbc:
            if tab_dim <= 0:
                raise ValueError("use_cbc=True but tab_dim <= 0")
            self.tab_encoder = TabularMLP(tab_dim)
            tab_out_dim = TAB_EMBED_DIM

        fused_dim = visual_dim + tab_out_dim
        if fused_dim <= 0:
            raise ValueError("Model has neither image nor CBC input")

        self.shared = nn.Sequential(
            nn.Linear(fused_dim, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT),
        )
        self.binary_head = nn.Linear(HIDDEN_DIM, 1)
        self.severity_head = nn.Linear(HIDDEN_DIM, 4)

    def encode_images(self, images: torch.Tensor, mask: torch.Tensor):
        # images: B,V,C,H,W
        batch_size, views = images.shape[:2]

        if self.encoder_policy == "view_specific" and hasattr(self, "encoders"):
            feats = torch.stack(
                [self.encoders[j](images[:, j]) for j in range(views)],
                dim=1,
            )
        else:
            flat = images.view(batch_size * views, *images.shape[2:])
            feats = self.encoder(flat).view(batch_size, views, -1)

        proj = self.image_projector(feats)

        if views == 1:
            return proj[:, 0, :], mask

        mask = mask.float()
        if self.fusion_type == "attention":
            scores = self.attn(proj).squeeze(-1)
            scores = scores.masked_fill(mask <= 0, SAFE_MASK_FILL_VALUE)
            weights = torch.softmax(scores, dim=1)
            weights = weights * mask
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
            visual = torch.sum(proj * weights.unsqueeze(-1), dim=1)
        elif self.fusion_type == "mean":
            weights = mask / mask.sum(dim=1, keepdim=True).clamp_min(1e-6)
            visual = torch.sum(proj * weights.unsqueeze(-1), dim=1)
        elif self.fusion_type == "max":
            masked = proj.masked_fill(mask.unsqueeze(-1) <= 0, SAFE_MASK_FILL_VALUE)
            visual = masked.max(dim=1).values
            visual = torch.where(torch.isfinite(visual), visual, torch.zeros_like(visual))
            weights = mask / mask.sum(dim=1, keepdim=True).clamp_min(1e-6)
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")

        return visual, weights

    def forward(
        self,
        images: torch.Tensor,
        view_mask: torch.Tensor,
        tabular: torch.Tensor,
    ) -> dict[str, torch.Tensor | None]:
        parts = []
        attention_weights = None

        if self.use_images:
            visual, attention_weights = self.encode_images(images, view_mask)
            parts.append(visual)

        if self.use_cbc:
            parts.append(self.tab_encoder(tabular))

        fused = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        hidden = self.shared(fused)

        return {
            "binary_logit": self.binary_head(hidden).squeeze(1),
            "severity_logits": self.severity_head(hidden),
            "attention_weights": attention_weights,
        }
