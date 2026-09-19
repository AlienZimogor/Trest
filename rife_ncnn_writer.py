#!/usr/bin/env python3
"""Инкремент 2: scale0 conv-цепочка (3->16->16->16). Ожидание бисекта:
layers=5, Crop c=3, conv0/1/2 c=16, ALL LAYERS OK."""
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
sd_raw = torch.load(pkls[0], map_location="cpu", weights_only=False)
if isinstance(sd_raw, dict) and "state_dict" in sd_raw:
    sd_raw = sd_raw["state_dict"]
sd = {}
for k, v in sd_raw.items():
    k2 = k
    while k2.startswith("module."):
        k2 = k2[len("module."):]
    sd[k2] = v

convs = []   # (key, W, B) в порядке ключей
for k, v in sd.items():
    if v.dim() == 4 and tuple(v.shape) in ((16, 3, 3, 3), (16, 16, 3, 3)):
        b = sd.get(k.replace(".weight", ".bias"))
        if b is not None and tuple(b.shape) == (16,):
            convs.append((k, v, b))
    if len(convs) == 3:
        break
print("picked convs:", [(k, tuple(w.shape)) for k, w, _ in convs])
assert len(convs) == 3 and tuple(convs[0][1].shape) == (16, 3, 3, 3)


def wblob(f, t):
    a = t.detach().cpu().contiguous()
    f.write(struct.pack("<i", 0))
    f.write(a.numpy().astype("<f4").tobytes())


with open("rife_hand.ncnn.param", "w") as f:
    f.write("7767577\n")
    f.write("5 5\n")
    f.write("Input in0 0 1 in0\n")
    f.write("Crop crop0 1 1 in0 c0 2=0 5=3\n")
    prev = "c0"
    for i, (k, w, b) in enumerate(convs):
        out = "f%d" % i
        f.write("Convolution conv%d 1 1 %s %s 0=16 1=3 5=1 6=%d\n"
                % (i, prev, out, w.numel()))
        prev = out

with open("rife_hand.ncnn.bin", "wb") as f:
    for k, w, b in convs:
        wblob(f, w)
        wblob(f, b)

print("wrote rife_hand.ncnn.param / .bin (%d bytes)"
      % os.path.getsize("rife_hand.ncnn.bin"))
