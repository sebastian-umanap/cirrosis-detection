"""Loaders, transforms y splits para CirrMRI600+.

Los símbolos pesados (Dataset/Transform que requieren torch/MONAI) se exponen via
funciones `get_*` con importación lazy para que `splits.py` y otros utilitarios livianos
puedan importarse sin instalar todo el stack de DL.
"""

from src.data.inventory import build_inventory, Inventory, PatientRecord
from src.data.splits import make_stratified_group_kfold

__all__ = [
    "build_inventory",
    "Inventory",
    "PatientRecord",
    "make_stratified_group_kfold",
    "get_transforms",
    "get_dataset_module",
]


def get_transforms():
    """Importación lazy de build_preprocess / build_train_augment."""
    from src.data.transforms import build_preprocess, build_train_augment
    return build_preprocess, build_train_augment


def get_dataset_module():
    """Importación lazy del módulo Dataset/DataModule (requiere torch)."""
    from src.data import dataset
    return dataset
