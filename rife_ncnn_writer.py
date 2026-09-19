#!/usr/bin/env python3
"""Инкремент 1: минимальный рукописный ncnn (Crop->Conv) для проверки
формата bin и загрузки весов. Вес первой 3->16 свёртки ищется по форме,
имя ключа печатается. Ожидание бисекта: Crop c=3, Conv c=16, ALL LAYERS OK."""
import struct, os, glob
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

W = B = None
wkey = None
for k, v in sd.items():
    if v.dim() == 4 and tuple(v.shape) == (16, 3, 3, 3):
        W = v
        wkey = k
        B = sd.get(k.replace(".weight", ".bias"))
        break
print("found conv weight key:", wkey, "W", tuple(W.shape), "B",
      tuple(B.shape) if B is not None else None)
assert W is not None, "no (16,3,3,3) conv weight found"
assert B is not None and tuple(B.shape) == (16,), "bias mismatch"


def wblob(f, t):
    a = t.detach().cpu().contiguous()
    f.write(struct.pack("<i", 0))          # flag 0 = fp32 raw
    f.write(a.numpy().astype("<f4").tobytes())


with open("rife_hand.ncnn.param", "w") as f:
    f.write("7767577\n")
    f.write("3 2\n")
    f.write("Input in0 0 1 in0\n")
    f.write("Crop crop0 1 1 in0 c0 2=0 5=3\n")
    f.write("Convolution conv0 1 1 c0 out0 0=16 1=3 5=1 6=%d\n" % W.numel())

with open("rife_hand.ncnn.bin", "wb") as f:
    wblob(f, W)
    wblob(f, B)

print("wrote rife_hand.ncnn.param / .bin (%d bytes)" % os.path.getsize("rife_hand.ncnn.bin"))
