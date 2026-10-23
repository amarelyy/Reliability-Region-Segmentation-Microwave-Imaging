"""
rrs_mbi/evaluate_unet.py
Evaluasi akurasi model UNet2Ch yang sudah di-train (unet_reliability.pth).
Menghitung: Dice, IoU, Accuracy, Precision, Recall, F1
"""

import sys
from pathlib import Path
import numpy as np
import torch
import pandas as pd
from typing import Dict, List, Any, Tuple

sys.path.append(str(Path(__file__).resolve().parent.parent))

from unet.models.unet_2ch import UNet2Ch
from src.data_loading import load_all_data
from src.pipeline import reconstruct_scan


# ==============================================================================
# KONFIGURASI
# ==============================================================================
MODEL_PATH = "unet/checkpoints/unet_reliability.pth"  # Sesuaikan path jika berbeda
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
THRESHOLD = 0.5  # Threshold untuk konversi probabilitas → binary mask


# ==============================================================================
# METRIK EVALUASI
# ==============================================================================
def compute_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
    """
    Menghitung metrik segmentasi biner.
    pred_mask & gt_mask: numpy array 2D dengan nilai 0 atau 1.
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


def create_gt_mask(img_shape: Tuple[int, int], gt_x_mm: float, gt_y_mm: float,
                   gt_r_mm: float, axis_mm: np.ndarray) -> np.ndarray:
    """
    Membuat ground truth binary mask dari koordinat dan radius tumor (dalam mm).
    Mengonversi koordinat mm ke pixel index berdasarkan axis_mm.
    """
    if np.isnan(gt_x_mm) or np.isnan(gt_y_mm) or np.isnan(gt_r_mm):
        return np.zeros(img_shape, dtype=np.float32)

    # Cari index pixel terdekat dari koordinat mm
    ix = int(np.argmin(np.abs(axis_mm - gt_x_mm)))
    iy = int(np.argmin(np.abs(axis_mm - gt_y_mm)))

    # Hitung radius dalam pixel
    pixel_size_mm = abs(axis_mm[1] - axis_mm[0]) if len(axis_mm) > 1 else 1.0
    r_px = gt_r_mm / pixel_size_mm

    # Buat circular mask
    y_grid, x_grid = np.ogrid[:img_shape[0], :img_shape[1]]
    mask = ((x_grid - ix) ** 2 + (y_grid - iy) ** 2) <= r_px ** 2
    return mask.astype(np.float32)


# ==============================================================================
# FUNGSI UTAMA EVALUASI
# ==============================================================================
@torch.no_grad()
def evaluate_unet() -> pd.DataFrame:
    print("=" * 70)
    print("UNET ACCURACY EVALUATION")
    print("=" * 70)

    # 1. Load Model
    print(f"\nLoading model from: {MODEL_PATH}")
    print(f"Device: {DEVICE}")

    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}\n"
            f"Pastikan path benar. Cek folder unet/checkpoints/"
        )

    model = UNet2Ch(n_channels=2, n_classes=1).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("✅ Model loaded successfully.")

    # 2. Load Data
    print("\nLoading dataset...")
    data = load_all_data()  # type: ignore
    s21 = data["s21"]
    tumor_model = data["tumor_model"]
    id_to_original_idx = data["id_to_original_idx"]

    total_scans = len(tumor_model)
    print(f"Total scans: {total_scans}")

    # 3. Evaluasi Loop
    results: List[Dict[str, Any]] = []
    skipped = 0

    print(f"\nEvaluating {total_scans} scans...")
    for i in range(total_scans):
        if i % 50 == 0:
            print(f"  Processing scan {i + 1}/{total_scans}...")

        try:
            # Reconstruct dengan diagnostics untuk mendapatkan gambar + axis_mm
            res = reconstruct_scan(
                i, s21, tumor_model, id_to_original_idx,
                return_diagnostics=True
            )  # type: ignore

            img = res["diagnostics"]["image"]
            axis_mm = res["diagnostics"]["axis_mm"]

            # Siapkan input untuk U-Net (2 channel: image + image/CF map)
            # Sesuaikan ini dengan cara training kamu menyiapkan input!
            # Asumsi: channel 1 = reconstructed image, channel 2 = CF map atau duplicate
            cf_map = res["diagnostics"].get("cf_map", None)
            if cf_map is not None:
                input_tensor = np.stack([img, cf_map], axis=0)  # Shape: (2, H, W)
            else:
                input_tensor = np.stack([img, img], axis=0)  # Fallback: duplicate

            # Normalize ke [0, 1]
            input_min = input_tensor.min()
            input_max = input_tensor.max()
            if input_max - input_min > 1e-8:
                input_tensor = (input_tensor - input_min) / (input_max - input_min)
            else:
                input_tensor = np.zeros_like(input_tensor)

            # Inference
            x = torch.from_numpy(input_tensor).unsqueeze(0).float().to(DEVICE)  # (1, 2, H, W)
            output = model(x)  # (1, 1, H, W)
            pred_prob = torch.sigmoid(output).squeeze().cpu().numpy()  # (H, W)
            pred_mask = (pred_prob > THRESHOLD).astype(np.float32)

            # Buat Ground Truth Mask
            gt_x = float(res.get("gt_x_mm", np.nan))
            gt_y = float(res.get("gt_y_mm", np.nan))
            gt_r = float(res.get("gt_r_mm", np.nan))
            gt_mask = create_gt_mask(img.shape, gt_x, gt_y, gt_r, axis_mm)

            # Skip jika tidak ada tumor (healthy scan)
            if gt_mask.sum() == 0:
                skipped += 1
                continue

            # Hitung Metrik
            metrics = compute_metrics(pred_mask, gt_mask)

            scan_id = res.get("phant_id", tumor_model.iloc[i].get("id", i))
            results.append({
                "scan_id": str(scan_id),
                "scan_idx": i,
                "gt_x_mm": gt_x,
                "gt_y_mm": gt_y,
                "gt_r_mm": gt_r,
                **metrics,
            })

        except Exception as e:
            skipped += 1
            continue

    # 4. Summary
    df = pd.DataFrame(results)

    print(f"\n{'=' * 70}")
    print(f"EVALUATION COMPLETE")
    print(f"{'=' * 70}")
    print(f"Scans evaluated: {len(df)}")
    print(f"Scans skipped (no tumor / error): {skipped}")
    print(f"{'=' * 70}")

    if not df.empty:
        metric_cols = ["dice", "iou", "accuracy", "precision", "recall", "f1"]
        print(f"\n{'Metric':<15} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
        print("-" * 60)
        for col in metric_cols:
            vals = df[col]
            print(f"{col:<15} {vals.mean():>10.4f} {vals.std():>10.4f} "
                  f"{vals.min():>10.4f} {vals.max():>10.4f}")
        print("-" * 60)

        # Save CSV
        out_path = Path("rrs_mbi/results/unet_evaluation.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"\n✅ Results saved to: {out_path}")
    else:
        print("\n⚠️ No valid results. Check model path and data loading.")

    return df


if __name__ == "__main__":
    evaluate_unet()