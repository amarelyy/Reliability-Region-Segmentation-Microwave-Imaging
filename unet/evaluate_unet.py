"""
unet/evaluate_unet.py

Evaluation of the trained UNet2Ch (unet/checkpoints/unet_reliability.pth) on HELD-OUT phantoms only,
with exactly the preprocessing used in training (MicrowaveImagingDataset: per-channel min-max,
128x128 resize, nearest-neighbour mask). Metrics: Dice, IoU, accuracy, precision, recall, F1.

The held-out phantom list is written by unet/training/train.py to unet/checkpoints/val_phantoms.npy.
The model already ends in a Sigmoid, so its output is used as a probability (no second sigmoid).

  python -m unet.evaluate_unet
"""

import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO))

from unet.models.unet_2ch import UNet2Ch
from src.data_loading import load_all_data

# ==============================================================================
# CONFIG
# ==============================================================================
MODEL_PATH = str(REPO / "unet" / "checkpoints" / "unet_reliability.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD = 0.5  # probability -> binary mask


# ==============================================================================
# METRICS
# ==============================================================================
def compute_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
    """
    Binary segmentation metrics.
    pred_mask & gt_mask: 2D numpy arrays with values 0 or 1.
    """
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    tp = np.sum(pred & gt)
    fp = np.sum(pred & ~gt)
    fn = np.sum(~pred & gt)
    tn = np.sum(~pred & ~gt)

    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = (2 * precision * recall) / (precision + recall + 1e-8)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


# ==============================================================================
# EVALUATION
# ==============================================================================
@torch.no_grad()
def evaluate_unet_heldout() -> pd.DataFrame:
    """Evaluate on held-out phantoms only, with EXACTLY the training preprocessing."""
    from unet.dataset.microwave_dataset import MicrowaveImagingDataset

    model_path = Path(MODEL_PATH)
    val_path = model_path.parent / "val_phantoms.npy"
    for p in (model_path, val_path):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found. Train first: python -m unet.training.train")

    model = UNet2Ch(n_channels=2, n_classes=1).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()

    data = load_all_data()
    tm = data["tumor_model"]
    val_phantoms = np.load(val_path, allow_pickle=True)
    tm_val = tm[tm["phant_id"].isin(val_phantoms)].reset_index(drop=True)
    print(f"Held-out: {len(tm_val)} scans from {tm_val['phant_id'].nunique()} phantoms")

    ds = MicrowaveImagingDataset(data["s21"], tm_val, data["id_to_original_idx"], data["freq_axis"])
    ds.prepare()
    rows = []
    for i in range(len(ds)):
        x, y = ds[i]
        prob = model(x.unsqueeze(0).to(DEVICE)).squeeze().cpu().numpy()   # already sigmoid
        m = compute_metrics((prob > THRESHOLD).astype(np.float32), y.squeeze().numpy())
        rows.append({"phant_id": tm_val.at[i, "phant_id"], **m})
    df = pd.DataFrame(rows)
    print(df[["dice", "iou", "precision", "recall"]].describe().loc[["mean", "std"]])
    return df


if __name__ == "__main__":
    evaluate_unet_heldout()
