#!/usr/bin/env python3
"""v9: диагностика — печатает полный инвентарь 4-мерных весов pkl
(ключ + форма) и завершается без экспорта. По списку проектируем инкремент 3."""
import os, glob
from collections import Counter
import torch


def _pref(p):
    s = 0
    b = os.path.basename(p).lower()
    if "4.26" in b or "426" in b: s += 4
    if "v4" in b: s += 1
    if "flownet" in b: s -= 3
    return s


pkls = [p for p in
        glob.glob(os.path.join("model_src", "**", "*.pkl"), recursive=True) +
        glob.glob(os.path.join("model_src", "**", "*.pth"), recursive=True) +
        glob.glob(os.path.join("model_src", "**", "*.pt"), recursive=True)]
pkls.sort(key=_pref, reverse=True)
print("pkl:", pkls[:3])
sd_raw = torch.load(pkls[0], map_location="cpu", weights_only=False)
if isinstance(sd_raw, dict) and "state_dict" in sd_raw:
    sd_raw = sd_raw["state_dict"]
sd = {}
for k, v in sd_raw.items():
    k2 = k
    while k2.startswith("module."):
        k2 = k2[len("module."):]
    sd[k2] = v

print("total keys:", len(sd))
w4 = [(k, tuple(v.shape)) for k, v in sd.items() if v.dim() == 4]
print("4-dim weights:", len(w4))
hist = Counter(sh for _, sh in w4)
print("=== shape histogram ===")
for sh, n in sorted(hist.items(), key=lambda x: (-x[1], x[0])):
    print("  %-18s x%d" % (str(sh), n))
print("=== full list (key -> shape) ===")
for k, sh in w4:
    print("  %-40s %s" % (k, sh))
print("=== 1-dim (bias/beta) sample ===")
b1 = [(k, tuple(v.shape)) for k, v in sd.items() if v.dim() == 1]
print("1-dim count:", len(b1))
for k, sh in b1[:20]:
    print("  %-40s %s" % (k, sh))
print("DIAG DONE")
