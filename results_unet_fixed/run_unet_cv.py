"""U-Net 5-fold GroupKFold (grouped by phantom id) on cached REAL inputs.
Inputs : results_audit/final/cache/unet_inputs_final.npz (read only; built by results_audit/unet_final.py)
Outputs: results_unet_fixed/ only.  Fixed settings, nothing tuned on held-out folds.
"""
import json, subprocess, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupKFold

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.data_loading import load_all_data
from unet.models.unet_2ch import UNet2Ch
from unet.training.losses import CombinedLoss
import unet.evaluate_unet as ev

OUT = REPO / "results_unet_fixed"
CFG = dict(seed=42, n_splits=5, epochs=20, batch_size=4, lr=1e-3, loss="0.5*BCE+0.5*Dice (CombinedLoss)",
           threshold=ev.THRESHOLD, checkpoint_selection="none (last epoch)", overfit_scans=8, overfit_epochs=100,
           inputs="results_audit/final/cache/unet_inputs_final.npz")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
git = lambda *a: subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()
CFG.update(git_commit=git("rev-parse", "HEAD"), git_branch=git("branch", "--show-current"),
           git_dirty_files=git("status", "--short"), device=str(dev), torch=torch.__version__)
(OUT / "config.json").write_text(json.dumps(CFG, indent=2))

d = load_all_data(); tm = d["tumor_model"].reset_index(drop=True); N = len(tm)
z = np.load(REPO / CFG["inputs"], allow_pickle=True)
X, Y, ok = z["X"], z["Y"], z["ok"]
assert len(X) == N
skips = [dict(scan_idx=int(i), phant_id=tm.phant_id.iloc[i], reason="input not cached (reconstruction failed; see results_audit/final/unet/skipped_samples.csv)")
         for i in np.where(~ok)[0]]
# scans with an empty GT mask would have undefined metrics: log them too
empty = [i for i in np.where(ok)[0] if Y[i].sum() == 0]
skips += [dict(scan_idx=int(i), phant_id=tm.phant_id.iloc[i], reason="empty ground-truth mask") for i in empty]
bad = {s["scan_idx"] for s in skips}
idx = np.array([i for i in range(N) if i not in bad])
pd.DataFrame(skips, columns=["scan_idx", "phant_id", "reason"]).to_csv(OUT / "skipped_scans.csv", index=False)
print(f"scans total {N}, usable {len(idx)}, skipped {len(skips)}", flush=True)


def make_model():
    torch.manual_seed(CFG["seed"]); np.random.seed(CFG["seed"])
    return UNet2Ch(n_channels=2, n_classes=1).to(dev)


def train(model, ii, epochs):
    dl = DataLoader(TensorDataset(torch.from_numpy(X[ii]), torch.from_numpy(Y[ii])), batch_size=CFG["batch_size"], shuffle=True,
                    generator=torch.Generator().manual_seed(CFG["seed"]))
    opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"]); crit = CombinedLoss()
    for _ in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()


@torch.no_grad()
def predict(model, ii):
    model.eval()
    return np.concatenate([model(torch.from_numpy(X[ii[k:k + 16]]).to(dev)).squeeze(1).cpu().numpy()
                           for k in range(0, len(ii), 16)])


def metr(pm, gt):
    m = ev.compute_metrics(pm.astype(np.float32), gt); return m["dice"], m["iou"], m["precision"], m["recall"]


# ---- E. sanity: overfit 8 training scans of fold 0 (training Dice only; says nothing about generalisation)
groups = tm.phant_id.values[idx]
folds = list(GroupKFold(n_splits=CFG["n_splits"]).split(idx, groups=groups))
tr0 = idx[folds[0][0]]
sub = tr0[np.random.RandomState(CFG["seed"]).choice(len(tr0), CFG["overfit_scans"], replace=False)]
m = make_model(); t0 = time.time(); train(m, sub, CFG["overfit_epochs"])
pr = predict(m, sub); od = [metr(pr[k] > ev.THRESHOLD, Y[i, 0])[0] for k, i in enumerate(sub)]
pd.DataFrame(dict(scan_idx=sub, train_dice=od)).to_csv(OUT / "overfit_check.csv", index=False)
print(f"OVERFIT 8 scans x {CFG['overfit_epochs']} ep: train Dice mean {np.mean(od):.3f} (min {np.min(od):.3f}, max {np.max(od):.3f}) [{time.time()-t0:.0f}s]", flush=True)

# ---- B/C/D. 5-fold CV
rows = []
for f, (tr_, te_) in enumerate(folds):
    tr, te = idx[tr_], idx[te_]
    ptr, pte = set(tm.phant_id.iloc[tr]), set(tm.phant_id.iloc[te])
    assert not (ptr & pte), f"fold {f}: phantom leakage {ptr & pte}"
    assert len(set(tr) & set(te)) == 0
    model = make_model(); train(model, tr, CFG["epochs"])
    pr = predict(model, te)
    print(f"fold {f}: train {len(tr)} scans/{len(ptr)} ph | held-out {len(te)} scans/{len(pte)} ph (disjoint OK)", flush=True)
    for k, i in enumerate(te):
        gt = Y[i, 0]; pm = pr[k] > ev.THRESHOLD
        r = dict(fold=f, scan_idx=int(i), phant_id=tm.phant_id.iloc[i], tumour_px_frac=float(gt.mean()), pred_px=int(pm.sum()))
        for name, mask in [("unet", pm), ("allpos", np.ones_like(gt, bool)), ("allbg", np.zeros_like(gt, bool))]:
            for mn, v in zip(["dice", "iou", "precision", "recall"], metr(mask, gt)):
                r[f"{name}_{mn}"] = v
        rows.append(r)
per = pd.DataFrame(rows); per.to_csv(OUT / "heldout_per_scan.csv", index=False)
assert per.scan_idx.is_unique and len(per) == len(idx)
cols = [c for c in per.columns if c.split("_")[0] in ("unet", "allpos", "allbg")] + ["tumour_px_frac"]
bf = per.groupby("fold").agg(n_scans=("scan_idx", "size"), n_phantoms=("phant_id", "nunique"), **{c: (c, "mean") for c in cols},
                             empty_mask_pct=("pred_px", lambda s: 100 * (s == 0).mean()))
pooled = per[cols].mean().to_frame("pooled_mean").T
pooled["n_scans"] = len(per); pooled["empty_mask_pct"] = 100 * (per.pred_px == 0).mean()
bf.to_csv(OUT / "metrics_per_fold.csv"); pooled.to_csv(OUT / "metrics_pooled.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print(bf.round(3).to_string()); print(pooled.round(3).to_string())
print("per-fold sd of unet_dice:", round(bf.unet_dice.std(), 3))
