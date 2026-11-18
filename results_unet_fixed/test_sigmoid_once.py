"""Unit test for bug 1: sigmoid must be applied exactly once.
Run: python results_unet_fixed/test_sigmoid_once.py
"""
import re, sys
from pathlib import Path
import torch
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from unet.models.unet_2ch import UNet2Ch

torch.manual_seed(0)
m = UNet2Ch(n_channels=2, n_classes=1).eval()
x = torch.rand(4, 2, 128, 128)
with torch.no_grad():
    out = m(x)
frac = (out > 0.5).float().mean().item()
print(f"model output in [{out.min():.3f}, {out.max():.3f}], fraction > 0.5 = {frac:.3f}")
assert 0.0 <= out.min() and out.max() <= 1.0, "model output is not a probability"
assert frac < 1.0, "all pixels > 0.5 (double-sigmoid symptom)"
# second sigmoid on a probability can never go below 0.5: this is the original bug
assert (torch.sigmoid(out) > 0.5).all(), "sanity: sigmoid(prob) must be > 0.5 everywhere"
# static check: no torch.sigmoid( call in the evaluation code
src = (REPO / "unet/evaluate_unet.py").read_text(encoding="utf-8")
code = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))
assert not re.search(r"torch\.sigmoid\(|\.sigmoid\(", code), "evaluate_unet.py still applies sigmoid"
print("PASS: sigmoid applied once (inside the model only)")
