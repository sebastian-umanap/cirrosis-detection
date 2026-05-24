"""Grad-CAM sobre ResNet-50 2.5D para interpretabilidad (Objetivo O5).

Usa la implementación de `pytorch-grad-cam` (Gildenblat 2021) sobre la última capa convolucional.
También calcula `pointing game accuracy`: porcentaje de casos donde el píxel de máxima activación
cae dentro de la máscara hepática (validez como métrica cuantitativa de interpretabilidad).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass
class GradCAMOutput:
    cam: np.ndarray            # (H, W) en [0, 1]
    target_class: int
    score: float
    inside_mask: bool          # pointing game para este caso


def make_gradcam(model: nn.Module, target_layer: nn.Module) -> "GradCAM":
    return GradCAM(model=model, target_layer=target_layer)


class GradCAM:
    """Mínima implementación propia para no depender estrictamente del paquete externo.

    Si `pytorch-grad-cam` está instalado, ese suele ser preferible; este módulo provee
    un fallback que produce el mismo mapa para ResNet-50.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        target_layer.register_forward_hook(self._fwd_hook)
        target_layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, _m: nn.Module, _i: tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
        self._activations = out.detach()

    def _bwd_hook(self, _m: nn.Module, _gi: tuple[torch.Tensor, ...], go: tuple[torch.Tensor, ...]) -> None:
        self._gradients = go[0].detach()

    @torch.no_grad()
    def _normalize(self, cam: torch.Tensor) -> torch.Tensor:
        cam = torch.clamp(cam, min=0)
        cam = cam - cam.amin(dim=(-2, -1), keepdim=True)
        denom = cam.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
        return cam / denom

    def __call__(self, x: torch.Tensor, target_class: int | None = None) -> GradCAMOutput:
        self.model.zero_grad(set_to_none=True)
        x = x.requires_grad_(True)
        out = self.model(x)
        if out.dim() == 1 or out.shape[-1] == 1:
            score = out.squeeze()
            target_class = 1 if target_class is None else target_class
        else:
            target_class = int(out.argmax(dim=-1).item()) if target_class is None else target_class
            score = out[..., target_class]
        score.sum().backward(retain_graph=False)
        assert self._activations is not None and self._gradients is not None
        # (1, C, H, W)
        weights = self._gradients.mean(dim=(-2, -1), keepdim=True)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)
        cam = self._normalize(cam)
        cam_np = cam.squeeze().cpu().numpy()
        return GradCAMOutput(cam=cam_np, target_class=int(target_class), score=float(score.detach().item()),
                             inside_mask=False)


def upsample_to(cam: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    import torch.nn.functional as F
    t = torch.from_numpy(cam).float().unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=hw, mode="bilinear", align_corners=False)
    return t.squeeze().numpy()


def pointing_game(cam: np.ndarray, mask: np.ndarray) -> bool:
    """True si el píxel de máxima activación cae dentro de la máscara."""
    if mask.sum() == 0:
        return False
    if cam.shape != mask.shape:
        cam = upsample_to(cam, mask.shape)  # type: ignore[arg-type]
    y, x = np.unravel_index(int(cam.argmax()), cam.shape)
    return bool(mask[y, x] > 0)


def mass_inside_mask(cam: np.ndarray, mask: np.ndarray, threshold: float = 0.5) -> float:
    """Fracción de la masa del CAM (> threshold) que cae dentro de la máscara."""
    if mask.sum() == 0:
        return float("nan")
    if cam.shape != mask.shape:
        cam = upsample_to(cam, mask.shape)  # type: ignore[arg-type]
    binary_cam = cam > threshold
    total = float(binary_cam.sum())
    if total == 0:
        return float("nan")
    inside = float((binary_cam & (mask > 0)).sum())
    return inside / total
