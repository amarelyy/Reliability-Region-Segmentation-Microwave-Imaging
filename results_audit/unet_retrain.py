"""Step 5: cache U-Net inputs once, retrain with the phantom-grouped split, evaluate held-out.

Same model, loss, optimiser, batch size, epochs and split as unet/training/train.py (patched);
the only difference is that the 2-channel inputs come from a disk cache built with
MicrowaveImagingDataset.__getitem__ instead of being re-reconstructed every epoch.
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupShuffleSplit

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.data_loading import load_all_data
from unet.dataset.microwave_dataset import MicrowaveImagingDataset
from unet.models.unet_2ch import UNet2Ch
from unet.training.losses import CombinedLoss
import unet.evaluate_unet as ev

OUT = REPO / "results_audit/fixed/unet"
OUT.mkdir(parents=True, exist_ok=True)
CKPT = REPO / "unet/checkpoints/unet_reliability.pth"
CACHE = REPO / "results_audit/fixed/cache/unet_inputs.npz"
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

backup = REPO / "results_audit/baseline/unet_reliability_ORIGINAL.pth"
if CKPT.exists() and not backup.exists():
    shutil.copy2(CKPT, backup)

d = load_all_data()
tm = d["tumor_model"]
if CACHE.exists():
    z = np.load(CACHE, allow_pickle=True)
    X, Y = z["X"], z["Y"]
else:
    ds = MicrowaveImagingDataset(d["s21"], tm, d["id_to_original_idx"], d["freq_axis"])
    X, Y = [], []
    for i in range(len(ds)):
        if i % 50 == 0:
            print(f"  caching {i + 1}/{len(ds)}", flush=True)
        x, y = ds[i]
        X.append(x.numpy()); Y.append(y.numpy())
    X, Y = np.stack(X), np.stack(Y)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, X=X, Y=Y, phant_id=tm["phant_id"].to_numpy())

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr, va = next(gss.split(tm, groups=tm["phant_id"]))
val_ph = tm["phant_id"].iloc[va].unique()
np.save(CKPT.parent / "val_phantoms.npy", val_ph)
assert not set(tm["phant_id"].iloc[tr]) & set(val_ph)
print(f"train {len(tr)} scans / {tm.phant_id.iloc[tr].nunique()} phantoms | "
      f"val {len(va)} scans / {len(val_ph)} phantoms")

mk = lambda idx, sh: DataLoader(TensorDataset(torch.from_numpy(X[idx]), torch.from_numpy(Y[idx])),
                                batch_size=4, shuffle=sh)
train_loader, val_loader = mk(tr, True), mk(va, False)
model = UNet2Ch(n_channels=2, n_classes=1).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
crit = CombinedLoss()
log = []
for ep in range(20):
    model.train(); tl = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step(); tl += loss.item()
    model.eval(); vl = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            vl += crit(model(xb.to(device)), yb.to(device)).item()
    log.append(dict(epoch=ep + 1, train_loss=tl / len(train_loader), val_loss=vl / len(val_loader)))
    print(f"Epoch [{ep + 1}/20] - Train Loss: {log[-1]['train_loss']:.4f} | Val Loss: {log[-1]['val_loss']:.4f}", flush=True)
pd.DataFrame(log).to_csv(OUT / "training_log.csv", index=False)
torch.save(model.state_dict(), CKPT)

# Official held-out evaluation (re-runs the dataset pipeline on held-out phantoms only)
ev.MODEL_PATH = str(CKPT)
df = ev.evaluate_unet_heldout()
df.to_csv(OUT / "unet_heldout_metrics.csv", index=False)

# Extra: localisation from the predicted mask, and trivial reference predictors (same masks)
model.eval()
rows = []
with torch.no_grad():
    for k, i in enumerate(va):
        prob = model(torch.from_numpy(X[i:i + 1]).to(device)).squeeze().cpu().numpy()
        gt = Y[i, 0]
        R = float(tm.breast_radius_mm.iloc[i])
        ax = np.linspace(-1.5 * R, 1.5 * R, prob.shape[0])  # grid spans +-1.5 R (approx. after resize)
        gx, gy = np.meshgrid(ax, ax)
        pm = prob > ev.THRESHOLD
        w = prob * pm
        if w.sum() > 0:
            px, py = (w * gx).sum() / w.sum(), (w * gy).sum() / w.sum()
        else:
            iy, ix = np.unravel_index(np.argmax(prob), prob.shape); px, py = ax[ix], ax[iy]
        le = np.hypot(px - tm.tumor_x_mm.iloc[i], py - tm.tumor_y_mm.iloc[i])
        m_all = ev.compute_metrics(np.ones_like(gt), gt)
        rows.append(dict(phant_id=tm.phant_id.iloc[i], pred_pixels=int(pm.sum()), unet_LE_mm=le,
                         **{f"unet_{k2}": v for k2, v in ev.compute_metrics(pm.astype(np.float32), gt).items()},
                         allones_dice=m_all["dice"], allones_precision=m_all["precision"]))
extra = pd.DataFrame(rows)
extra.to_csv(OUT / "unet_heldout_extra.csv", index=False)
print(extra.drop(columns="phant_id").mean().round(4).to_string())
print("det@15mm from U-Net mask centroid: %.1f%%" % ((extra.unet_LE_mm <= 15).mean() * 100))
print("scans with empty predicted mask:", int((extra.pred_pixels == 0).sum()))
