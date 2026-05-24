"""ResNet-50 2.5D con attention pooling sobre cortes axiales.

Pesos pre-entrenados:
    - Por defecto ImageNet (vía torchvision).
    - Si se proveen pesos RadImageNet (Mei et al. 2022), cargar manualmente con `load_radimagenet_weights`.

Justificación en reports/02_methodology_crispmlq.md §Fase 3.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights

log = logging.getLogger(__name__)


class GatedAttentionPool(nn.Module):
    """Attention pooling (Ilse, Tomczak, Welling, ICML 2018).

    Recibe (N, D) features (N = #instancias de un paciente) y devuelve (D,) agregado.
    Para agregación dentro de un batch de cortes del mismo paciente.
    """

    def __init__(self, in_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.attn_v = nn.Linear(in_dim, hidden)
        self.attn_u = nn.Linear(in_dim, hidden)
        self.attn_w = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        v = torch.tanh(self.attn_v(x))
        u = torch.sigmoid(self.attn_u(x))
        logits = self.attn_w(v * u)
        weights = torch.softmax(logits, dim=0)
        pooled = (weights * x).sum(dim=0)
        return pooled, weights.squeeze(-1)


class ResNet25D(nn.Module):
    """ResNet-50 sobre stacks 2.5D (3 cortes = 3 canales). Output: logit binario o multiclase."""

    def __init__(
        self,
        num_classes: int = 1,
        in_channels: int = 3,
        pretrained: bool = True,
        dropout: float = 0.3,
        attention_pool: str = "gated",
    ) -> None:
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet50(weights=weights)
        if in_channels != 3:
            # Reemplazamos la primera conv si los canales son distintos.
            old = backbone.conv1
            new = nn.Conv2d(in_channels, old.out_channels, kernel_size=old.kernel_size,
                            stride=old.stride, padding=old.padding, bias=False)
            if pretrained and in_channels == 1:
                with torch.no_grad():
                    new.weight.copy_(old.weight.mean(dim=1, keepdim=True))
            backbone.conv1 = new
        self.feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.attention_pool_name = attention_pool
        self.attn_pool = GatedAttentionPool(self.feat_dim) if attention_pool == "gated" else None
        self.head = nn.Linear(self.feat_dim, num_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) — devuelve logits (B, num_classes).
        f = self.encode(x)
        f = self.dropout(f)
        return self.head(f)

    @torch.no_grad()
    def aggregate_patient(self, slice_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Agrega features de los cortes de UN paciente. (n_slices, feat_dim) -> (num_classes,)."""
        if self.attn_pool is not None:
            pooled, weights = self.attn_pool(slice_features)
            logits = self.head(pooled)
            return logits, weights
        pooled = slice_features.mean(dim=0)
        logits = self.head(pooled)
        return logits, None


def build_resnet25d(
    num_classes: int = 1,
    in_channels: int = 3,
    pretrained: bool = True,
    dropout: float = 0.3,
    attention_pool: str = "gated",
    radimagenet_ckpt: Path | None = None,
) -> ResNet25D:
    model = ResNet25D(
        num_classes=num_classes,
        in_channels=in_channels,
        pretrained=pretrained,
        dropout=dropout,
        attention_pool=attention_pool,
    )
    if radimagenet_ckpt is not None:
        load_radimagenet_weights(model, radimagenet_ckpt)
    return model


def load_radimagenet_weights(model: ResNet25D, ckpt_path: Path) -> None:
    """Carga pesos RadImageNet (Mei et al., 2022) sobre el backbone ResNet-50.

    El formato esperado son state_dicts publicados por los autores; si la versión local difiere,
    se ignoran capas no coincidentes y se reporta cuántas se cargaron.
    """
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    own = model.backbone.state_dict()
    loaded = 0
    for k, v in state.items():
        kk = k.replace("module.", "").replace("backbone.", "")
        if kk in own and own[kk].shape == v.shape:
            own[kk] = v
            loaded += 1
    model.backbone.load_state_dict(own)
    log.info("RadImageNet: %d/%d tensores cargados.", loaded, len(own))
