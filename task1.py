import argparse
import pickle
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image

# -----------------------------
# Global speed knobs
# -----------------------------
torch.backends.cudnn.benchmark = True  # good when image size is fixed (224x224)

# -----------------------------
# Config & utilities
# -----------------------------

@dataclass
class RunConfig:
    train_root: str = "./dataset/part_one_dataset/train_data"
    eval_root: str = "./dataset/part_one_dataset/eval_data"
    num_sets: int = 10                     # D1..D10
    batch_size: int = 128
    num_workers: int = 4
    use_dataparallel: bool = False         # use all visible GPUs if True
    save_prototypes_path: str = "new_best_prototypes.pkl"
    seed: int = 42


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Safe torch.load for PyTorch >= 2.6
# -----------------------------

def safe_torch_load(path: str):
    """
    Works with PyTorch 2.6+ default weights_only=True while allow-listing
    NumPy reconstruct, and falls back to weights_only=False if needed.
    ONLY use the fallback if you trust the file.
    """
    try:
        from torch.serialization import safe_globals
        with safe_globals([np.core.multiarray._reconstruct]):
            return torch.load(path, map_location="cpu")  # weights_only=True by default in 2.6
    except Exception:
        # Trusted files only:
        return torch.load(path, map_location="cpu", weights_only=False)


# -----------------------------
# Dataset
# -----------------------------

def to_pil_if_needed(x):
    # Accepts torch.Tensor (C,H,W) or numpy (H,W,C)/(H,W) or PIL already
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        if x.dtype != torch.uint8:
            x = (x.clamp(0, 1) * 255).to(torch.uint8)
        if x.ndim == 3 and x.shape[0] in (1, 3):
            x = x.permute(1, 2, 0)
        return TF.to_pil_image(x)
    elif isinstance(x, np.ndarray):
        if x.dtype != np.uint8:
            x = np.clip(x, 0, 1)
            x = (x * 255).astype(np.uint8)
        return TF.to_pil_image(x)
    else:
        return x  # assume PIL.Image.Image


class PthImageDataset(Dataset):
    """
    Expects .pth with keys:
      - 'data': list/array/tensors of images (uint8 or float [0,1])
      - 'targets': (optional) labels (int)
    """
    def __init__(self, file_path: str, transform: transforms.Compose, labeled: bool = True):
        self.raw = safe_torch_load(file_path)
        self.data = self.raw["data"]
        self.labeled = labeled
        self.targets = self.raw.get("targets", None) if labeled else None
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        img = self.data[idx]
        img = to_pil_if_needed(img)
        img = self.transform(img)
        if self.labeled:
            label = self.targets[idx]
            if isinstance(label, torch.Tensor):
                label = label.item()
            return img, int(label)
        return img


# -----------------------------
# Backbone / Feature Extractor
# -----------------------------

def build_backbone(device: torch.device, dataparallel: bool = False) -> Tuple[nn.Module, transforms.Compose]:
    weights = EfficientNet_B0_Weights.DEFAULT
    backbone = models.efficientnet_b0(weights=weights)
    backbone.classifier = nn.Identity()
    backbone.eval()
    backbone.to(device)
    # channels-last for convnets
    if device.type == "cuda":
        backbone = backbone.to(memory_format=torch.channels_last)
    return backbone, weights.transforms()


@torch.inference_mode()
def extract_features_dl(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device
) -> np.ndarray:
    feats: List[np.ndarray] = []
    use_cuda = device.type == "cuda"
    for batch in loader:
        imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
        if use_cuda:
            imgs = imgs.to(memory_format=torch.channels_last)
        imgs = imgs.to(device, non_blocking=True)
        if use_cuda:
            from torch.amp import autocast  # Changed for PyTorch >= 2.6
            with autocast('cuda', dtype=torch.float16):
                out = model(imgs)
        else:
            out = model(imgs)
        if isinstance(out, (list, tuple)):
            out = out[0]
        feats.append(out.detach().cpu().numpy())
    return np.concatenate(feats, axis=0)


# -----------------------------
# LwP: Prototypes & Prediction
# -----------------------------

def compute_class_prototypes(features: np.ndarray, labels: np.ndarray) -> Dict[int, np.ndarray]:
    """Mean vector per class."""
    prototypes: Dict[int, np.ndarray] = {}
    classes = np.unique(labels)
    for c in classes:
        mask = (labels == c)
        if mask.any():
            prototypes[int(c)] = features[mask].mean(axis=0)
    return prototypes


def predict_labels(features: np.ndarray, prototypes: Dict[int, np.ndarray]) -> np.ndarray:
    """Vectorized nearest-prototype (Euclidean) using torch.cdist."""
    if not prototypes:
        raise ValueError("Empty prototypes provided.")
    labels = np.array(list(prototypes.keys()), dtype=np.int64)
    proto_mat = np.stack([prototypes[k] for k in labels], axis=0)  # (C, D)
    with torch.inference_mode():
        X = torch.from_numpy(features)           # (N, D)
        P = torch.from_numpy(proto_mat)          # (C, D)
        dists = torch.cdist(X, P, p=2.0)         # (N, C)
        idx = torch.argmin(dists, dim=1).cpu().numpy()
    return labels[idx]


def ema_update_prototypes(old: Dict[int, np.ndarray], new: Dict[int, np.ndarray], alpha: float) -> Dict[int, np.ndarray]:
    """EMA: old <- (1-alpha)*old + alpha*new; also initializes unseen classes."""
    updated = dict(old)
    for k, v in new.items():
        if k in updated:
            updated[k] = (1.0 - alpha) * updated[k] + alpha * v
        else:
            updated[k] = v.copy()
    return updated


# -----------------------------
# Loader helpers
# -----------------------------

def build_dataset_and_loader(file_path: str, transform, labeled: bool, cfg: RunConfig) -> Tuple[PthImageDataset, DataLoader]:
    ds = PthImageDataset(file_path=file_path, transform=transform, labeled=labeled)
    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
        drop_last=False,
    )
    return ds, dl


# -----------------------------
# Training / Evaluation Loop
# -----------------------------

def incremental_training_and_evaluation(cfg: RunConfig) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = build_backbone(device, cfg.use_dataparallel)

    # ----- Initialize with D1 (labeled) -----
    d1_train_path = f"{cfg.train_root}/1_train_data.tar.pth"
    d1_eval_path  = f"{cfg.eval_root}/1_eval_data.tar.pth"

    d1_train_ds, d1_train_loader = build_dataset_and_loader(d1_train_path, preprocess, labeled=True, cfg=cfg)
    d1_eval_ds,  d1_eval_loader  = build_dataset_and_loader(d1_eval_path,  preprocess, labeled=True, cfg=cfg)

    d1_train_feats = extract_features_dl(model, d1_train_loader, device)
    # labels from dataset directly (faster and cleaner)
    d1_train_labels = np.array(d1_train_ds.targets, dtype=np.int64)

    prototypes = compute_class_prototypes(d1_train_feats, d1_train_labels)

    # Accuracy matrix (Fi vs ^Dj)
    accuracy_matrix = np.zeros((cfg.num_sets, cfg.num_sets), dtype=np.float32)

    # Evaluate F1 on ^D1
    d1_eval_feats = extract_features_dl(model, d1_eval_loader, device)
    d1_eval_labels = np.array(d1_eval_ds.targets, dtype=np.int64)
    preds = predict_labels(d1_eval_feats, prototypes)
    accuracy_matrix[0, 0] = accuracy_score(d1_eval_labels, preds) * 100.0
    print(f"[INFO] F1 accuracy on ^D1: {accuracy_matrix[0, 0]:.2f}%")

    # ----- Sequentially adapt on D2..Dk (unlabeled) -----
    for i in range(2, cfg.num_sets + 1):
        print(f"[INFO] Training F{i} with D{i} (unlabeled)")
        di_train_path = f"{cfg.train_root}/{i}_train_data.tar.pth"
        di_train_ds, di_train_loader = build_dataset_and_loader(di_train_path, preprocess, labeled=False, cfg=cfg)
        di_feats = extract_features_dl(model, di_train_loader, device)

        # pseudo-labels under current prototypes
        di_pseudo = predict_labels(di_feats, prototypes)
        di_new_protos = compute_class_prototypes(di_feats, di_pseudo)

        # EMA with alpha = 1/i (moving-average spirit)
        alpha = 1.0 / float(i)
        prototypes = ema_update_prototypes(prototypes, di_new_protos, alpha=alpha)

        # Evaluate F_i on ^D1..^Di
        for j in range(1, i + 1):
            dj_eval_path = f"{cfg.eval_root}/{j}_eval_data.tar.pth"
            dj_eval_ds, dj_eval_loader = build_dataset_and_loader(dj_eval_path, preprocess, labeled=True, cfg=cfg)
            dj_feats = extract_features_dl(model, dj_eval_loader, device)
            dj_labels = np.array(dj_eval_ds.targets, dtype=np.int64)
            dj_preds = predict_labels(dj_feats, prototypes)
            acc = accuracy_score(dj_labels, dj_preds) * 100.0
            accuracy_matrix[i - 1, j - 1] = acc
            print(f"[INFO] F{i} accuracy on ^D{j}: {acc:.2f}%")

    return accuracy_matrix, prototypes


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="LwP continual learning with EfficientNet-B0 features")
    parser.add_argument("--train_root", type=str, default="./dataset/part_one_dataset/train_data")
    parser.add_argument("--eval_root", type=str, default="./dataset/part_one_dataset/eval_data")
    parser.add_argument("--num_sets", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--dataparallel", action="store_true")
    parser.add_argument("--save_prototypes", type=str, default="new_best_prototypes.pkl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = RunConfig(
        train_root=args.train_root,
        eval_root=args.eval_root,
        num_sets=args.num_sets,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_dataparallel=args.dataparallel,
        save_prototypes_path=args.save_prototypes,
        seed=args.seed,
    )

    set_seed(cfg.seed)
    acc, protos = incremental_training_and_evaluation(cfg)
    print("Accuracy Matrix:\n", acc)

    with open(cfg.save_prototypes_path, "wb") as f:
        pickle.dump(protos, f)
    print(f"[INFO] Saved prototypes to: {cfg.save_prototypes_path}")


if __name__ == "__main__":
    main()
