"""
svm_experiment/build_candidates.py

Candidate generator for the SVM peak-verification experiment, on the CURRENT pipeline.

Image  : src.pipeline.reconstruct_scan with the same call as rrs_mbi/evaluator and
         reconstruction/reconstruct_images.py (adipose ref, TVSVD, RMS norm, full band, gate 0.7 ns,
         DAS x CF, breast-disc mask). Nothing in the pipeline is changed.
Candidates: every connected component of the pipeline's own blob stage (LoG sigma 2-5 -> Otsu
         threshold). The pipeline keeps only the LARGEST component; here all of them are candidates,
         so "pipeline pick" = largest blob = exactly the pipeline peak (asserted below).
Position of a candidate = intensity-weighted centroid of its component (same as the pipeline).

Outputs (git-ignored *.csv):
    svm_experiment/svm_candidates_v2.csv   one row per candidate
    svm_experiment/svm_scans_v2.csv        one row per scan (baselines: pipeline peak, image argmax, ...)

Run from the repo root:  python svm_experiment/build_candidates.py --n-jobs 6
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import label, binary_erosion

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from src.data_loading import load_all_data          # noqa: E402
from src.pipeline import reconstruct_scan           # noqa: E402
from src import blob_detection as bd                # noqa: E402

LABEL_THRESHOLD_MM = 20.0   # a candidate is "real" if its centroid is within this distance of the GT centre


def _compactness(mask):
    area = int(mask.sum())
    if area == 0:
        return 0.0
    perim = max(int((mask & ~binary_erosion(mask)).sum()), 1)
    return float(4 * np.pi * area / perim ** 2)


def scan_candidates(i, s21, tm, idm, freq_axis):
    r = reconstruct_scan(i, s21, tm, idm, freq_axis=freq_axis, use_tvsvd=True, use_rms_norm=True,
                         bandpass_mode="full", gate_ns=0.70, return_diagnostics=True)
    g = r["diagnostics"]
    img, cf, axis, pipe_mask = g["image"], g["cf_map"], g["axis_mm"], g["tumor_mask"]
    sx, sy, R = float(r["shell_cx_mm"]), float(r["shell_cy_mm"]), float(r["breast_radius_mm"])
    gx, gy = float(r["gt_x_mm"]), float(r["gt_y_mm"])
    xx, yy = np.meshgrid(axis, axis)
    region = np.hypot(xx - sx, yy - sy) <= R
    img_max = float(img[region].max()) + 1e-12
    breast_px = int(region.sum())

    _, response, thresh = bd.select_blob_mask_log(img)
    lab, n = label(response >= thresh)
    sizes = [int((lab == k).sum()) for k in range(1, n + 1)]
    pipe_k = (int(np.argmax(sizes)) + 1) if n else 0
    if n:   # the pipeline's own choice must be reproduced exactly
        assert np.array_equal(lab == pipe_k, pipe_mask), f"scan {i}: largest blob != pipeline mask"

    rows = []
    comps = range(1, n + 1) if n else []
    for k in comps:
        m = lab == k
        ys, xs = np.where(m)
        w = img[ys, xs]
        cx, cy = float(np.average(axis[xs], weights=w)), float(np.average(axis[ys], weights=w))
        rows.append(dict(
            scan_i=i, phant_id=r["phant_id"], cand_k=k, is_pipeline=int(k == pipe_k),
            cx_mm=cx, cy_mm=cy, le_mm=float(np.hypot(cx - gx, cy - gy)),
            area_px=sizes[k - 1], compactness=_compactness(m),
            # SVM features (all from the current image / geometry)
            f_cf_mean=float(cf[ys, xs].mean()),
            f_peak_rel=float(img[ys, xs].max() / img_max),
            f_dist_ratio=float(np.hypot(cx - sx, cy - sy) / R),
            f_area_frac=sizes[k - 1] / breast_px))
    iy, ix = np.unravel_index(np.argmax(img), img.shape)
    am_x, am_y = float(axis[ix]), float(axis[iy])
    scan = dict(scan_i=i, phant_id=r["phant_id"], n_cand=n, gt_x=gx, gt_y=gy, shell_cx=sx, shell_cy=sy,
                breast_R=R, pipe_x=float(r["peak_x_mm"]), pipe_y=float(r["peak_y_mm"]),
                pipe_le=float(r["localization_error_mm"]), argmax_x=am_x, argmax_y=am_y,
                argmax_le=float(np.hypot(am_x - gx, am_y - gy)),
                centre_guess_le=float(np.hypot(sx - gx, sy - gy)))
    return rows, scan


def _chunk(idx, s21, tm, idm, freq_axis):
    out_c, out_s, err = [], [], []
    for i in idx:
        try:
            c, s = scan_candidates(int(i), s21, tm, idm, freq_axis)
            out_c += c
            out_s.append(s)
        except Exception as e:
            err.append((int(i), repr(e)))
    return out_c, out_s, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="debug: first N scans only")
    args = ap.parse_args()

    from joblib import Parallel, delayed
    data = load_all_data()
    tm = data["tumor_model"]
    order = tm.sort_values("phant_id").index.to_numpy()          # group phantoms -> delay-cache hits
    if args.limit:
        order = order[: args.limit]
    chunks = [c for c in np.array_split(order, args.n_jobs * 4) if len(c)]
    res = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(_chunk)(c, data["s21"], tm, data["id_to_original_idx"], data["freq_axis"]) for c in chunks)

    cand = pd.DataFrame([r for c, _, _ in res for r in c])
    scans = pd.DataFrame([s for _, s, _ in res for s in s]).sort_values("scan_i").reset_index(drop=True)
    errs = [e for _, _, e in res for e in e]
    cand["label"] = (cand["le_mm"] <= LABEL_THRESHOLD_MM).astype(int)
    cand.to_csv(HERE / "svm_candidates_v2.csv", index=False)
    scans.to_csv(HERE / "svm_scans_v2.csv", index=False)
    print(f"scans: {len(scans)}  phantoms: {scans['phant_id'].nunique()}  candidates: {len(cand)}  "
          f"errors: {len(errs)}  scans without blob: {int((scans['n_cand'] == 0).sum())}")
    for e in errs:
        print("ERROR", e)
    print(f"pipeline LE mean {scans['pipe_le'].mean():.2f}  argmax LE mean {scans['argmax_le'].mean():.2f}  "
          f"centre-guess {scans['centre_guess_le'].mean():.2f}")


if __name__ == "__main__":
    main()
