"""U-Net re-run on the FINAL pipeline (REAL data; no synthetic data).

1. Cache the 2-channel inputs + masks once (current src/ pipeline) and the per-scan image axis.
2. Train from scratch with a phantom-grouped split, for several seeds / different splits.
3. Evaluate on held-out phantoms only: Dice, precision, recall, mask-centroid LE (mm), det@15.
4. Reference predictors on the SAME held-out scans: all-ones mask, disc at the chamber centre,
   and the final DAS+CF pipeline (results_audit/final/sat20 metrics_raw.csv).
Seed 42 is the official checkpoint (unet/checkpoints/unet_reliability.pth + val_phantoms.npy),
cross-checked with evaluate_unet_heldout().
"""
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupShuffleSplit

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.data_loading import load_all_data
import unet.dataset.microwave_dataset as md
from unet.models.unet_2ch import UNet2Ch
from unet.training.losses import CombinedLoss
import unet.evaluate_unet as ev

OUT = REPO / "results_audit/final/unet"
OUT.mkdir(parents=True, exist_ok=True)
CKPT = REPO / "unet/checkpoints/unet_reliability.pth"
CACHE = REPO / "results_audit/final/cache/unet_inputs_final.npz"
CACHE.parent.mkdir(parents=True, exist_ok=True)
SEEDS = [42, 43, 44, 45, 46]
EPOCHS = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# keep the ORIGINAL (bugged) checkpoint safe
backup = REPO / "results_audit/baseline/unet_reliability_ORIGINAL.pth"
if CKPT.exists() and not backup.exists():
    shutil.copy2(CKPT, backup)

d = load_all_data()
tm = d["tumor_model"].reset_index(drop=True)
N = len(tm)

# ---------------------------------------------------------------- 1. cache
if CACHE.exists():
    z = np.load(CACHE, allow_pickle=True)
    X, Y, AX, ok = z["X"], z["Y"], z["AX"], z["ok"]
    print("cache loaded", X.shape, "usable", int(ok.sum()), flush=True)
else:
    last = {}
    orig = md.reconstruct_scan

    def wrap(*a, **k):
        r = orig(*a, **k)
        last["axis"] = np.asarray(r["diagnostics"]["axis_mm"], float)
        return r

    md.reconstruct_scan = wrap
    ds = md.MicrowaveImagingDataset(d["s21"], tm, d["id_to_original_idx"], d["freq_axis"])
    X = np.zeros((N, 2, 128, 128), np.float32)
    Y = np.zeros((N, 1, 128, 128), np.float32)
    AX = np.full((N, 2), np.nan)
    ok = np.zeros(N, bool)
    skipped = []
    t0 = time.time()
    for i in range(N):
        if i % 50 == 0:
            print(f"  caching {i}/{N}  {time.time() - t0:.0f}s", flush=True)
        try:
            x, y = ds[i]
            X[i], Y[i] = x.numpy(), y.numpy()
            AX[i] = [last["axis"][0], last["axis"][-1]]
            ok[i] = True
        except Exception as e:  # report, never skip silently
            skipped.append((i, tm.phant_id.iloc[i], repr(e)[:120]))
    md.reconstruct_scan = orig
    pd.DataFrame(skipped, columns=["scan_idx", "phant_id", "error"]).to_csv(OUT / "skipped_samples.csv", index=False)
    print(f"cached {int(ok.sum())}/{N} scans; skipped {len(skipped)}", flush=True)
    np.savez_compressed(CACHE, X=X, Y=Y, AX=AX, ok=ok)
print("samples skipped:", int((~ok).sum()), flush=True)

pipe = pd.read_csv(REPO / "results_audit/final/sat20/rrs_mbi/results/metrics_raw.csv")
pipe = pipe.set_index("scan_idx")
med_r = float(tm.tumor_radius_mm.median())


def dice_prec_rec(pm, gt):
    m = ev.compute_metrics(pm.astype(np.float32), gt)
    return m["dice"], m["precision"], m["recall"]


def run(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    idx_ok = np.where(ok)[0]
    sub = tm.iloc[idx_ok]
    tr_, va_ = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed).split(sub, groups=sub.phant_id))
    tr, va = idx_ok[tr_], idx_ok[va_]
    val_ph = tm.phant_id.iloc[va].unique()
    assert not set(tm.phant_id.iloc[tr]) & set(val_ph)
    mk = lambda ii, sh: DataLoader(TensorDataset(torch.from_numpy(X[ii]), torch.from_numpy(Y[ii])), batch_size=4, shuffle=sh)
    trl, val = mk(tr, True), mk(va, False)
    model = UNet2Ch(n_channels=2, n_classes=1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = CombinedLoss()
    log = []
    for ep in range(EPOCHS):  # fixed epoch count: NO checkpoint selection on the held-out phantoms
        model.train(); tl = 0
        for xb, yb in trl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step(); tl += loss.item()
        model.eval(); vl = 0
        with torch.no_grad():
            for xb, yb in val:
                vl += crit(model(xb.to(device)), yb.to(device)).item()
        log.append(dict(seed=seed, epoch=ep + 1, train_loss=tl / len(trl), val_loss=vl / len(val)))
    print(f"seed {seed}: train {len(tr)} scans/{tm.phant_id.iloc[tr].nunique()} ph | held-out {len(va)} scans/{len(val_ph)} ph | "
          f"final train {log[-1]['train_loss']:.3f} val {log[-1]['val_loss']:.3f}", flush=True)
    if seed == SEEDS[0]:
        torch.save(model.state_dict(), CKPT)
        np.save(CKPT.parent / "val_phantoms.npy", val_ph)
    else:
        torch.save(model.state_dict(), OUT / f"unet_seed{seed}.pth")

    model.eval(); rows = []
    with torch.no_grad():
        for i in va:
            prob = model(torch.from_numpy(X[i:i + 1]).to(device)).squeeze().cpu().numpy()
            gt = Y[i, 0]
            ax = np.linspace(AX[i, 0], AX[i, 1], 128)
            gx, gy = np.meshgrid(ax, ax)
            pm = prob > ev.THRESHOLD
            if pm.any():
                w = prob * pm; px, py = (w * gx).sum() / w.sum(), (w * gy).sum() / w.sum()
            else:
                iy, ix = np.unravel_index(np.argmax(prob), prob.shape); px, py = ax[ix], ax[iy]
            tx, ty = float(tm.tumor_x_mm.iloc[i]), float(tm.tumor_y_mm.iloc[i])
            dsc, pr, rc = dice_prec_rec(pm, gt)
            cd = (np.hypot(gx, gy) <= med_r)  # reference: disc at chamber centre, median tumour radius
            cdsc, cpr, crc = dice_prec_rec(cd, gt)
            adsc, apr, _ = dice_prec_rec(np.ones_like(gt, bool), gt)
            rows.append(dict(seed=seed, scan_idx=int(i), phant_id=tm.phant_id.iloc[i], pred_pixels=int(pm.sum()),
                             dice=dsc, precision=pr, recall=rc, unet_LE_mm=float(np.hypot(px - tx, py - ty)),
                             centre_disc_dice=cdsc, centre_disc_LE_mm=float(np.hypot(tx, ty)),
                             allones_dice=adsc, allones_precision=apr,
                             pipeline_LE_mm=float(pipe.localization_error.get(i, np.nan))))
    return pd.DataFrame(rows), pd.DataFrame(log), model, val_ph


all_rows, all_logs, official = [], [], None
for s in SEEDS:
    r, lg, m, vp = run(s)
    all_rows.append(r); all_logs.append(lg)
per = pd.concat(all_rows, ignore_index=True)
pd.concat(all_logs).to_csv(OUT / "training_log_all_seeds.csv", index=False)
per.to_csv(OUT / "heldout_per_scan_all_seeds.csv", index=False)

# per-seed and overall summary
def summ(g):
    return pd.Series(dict(
        n_scans=len(g), n_phantoms=g.phant_id.nunique(),
        dice=g.dice.mean(), precision=g.precision.mean(), recall=g.recall.mean(),
        empty_mask_pct=100 * (g.pred_pixels == 0).mean(),
        unet_LE_mm=g.unet_LE_mm.mean(), unet_det15_pct=100 * (g.unet_LE_mm <= 15).mean(),
        centre_disc_dice=g.centre_disc_dice.mean(), allones_dice=g.allones_dice.mean(),
        centre_LE_mm=g.centre_disc_LE_mm.mean(), centre_det15_pct=100 * (g.centre_disc_LE_mm <= 15).mean(),
        pipeline_LE_mm=g.pipeline_LE_mm.mean(), pipeline_det15_pct=100 * (g.pipeline_LE_mm <= 15).mean()))
by_seed = per.groupby("seed").apply(summ)
by_seed.to_csv(OUT / "summary_by_seed.csv")
overall = by_seed.drop(columns=["n_scans", "n_phantoms"]).agg(["mean", "std"])
overall.to_csv(OUT / "summary_over_seeds.csv")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print("\n== per seed ==\n", by_seed.round(3).to_string())
print("\n== mean / std over seeds ==\n", overall.round(3).to_string(), flush=True)

# Official cross-check: evaluate_unet_heldout() on the seed-42 checkpoint (re-runs dataset pipeline)
ev.MODEL_PATH = str(CKPT)
df = ev.evaluate_unet_heldout()
df.to_csv(OUT / "official_heldout_seed42.csv", index=False)
print("\nofficial evaluate_unet_heldout (seed 42): dice %.3f precision %.3f recall %.3f n=%d"
      % (df.dice.mean(), df.precision.mean(), df.recall.mean(), len(df)), flush=True)
