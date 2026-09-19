#!/usr/bin/env python3
"""Инкремент 2 (v4): scale0 conv-цепочка (3->16->16->16) с корректным
заголовком param и самопроверкой. Ожидание бисекта: layers=5,
Crop c=3, conv0/1/2 c=16, ALL LAYERS OK."""
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

convs = []
for k, v in sd.items():
    if v.dim() == 4 and tuple(v.shape) in ((16, 3, 3, 3), (16, 16, 3, 3)):
        b = sd.get(k.replace(".weight", ".bias"))
        if b is not None and tuple(b.shape) == (16,):
            convs.append((k, v, b))
    if len(convs) == 3:
        break
print("picked convs:", [(k, tuple(w.shape)) for k, w, _ in convs])
assert len(convs) == 3 and tuple(convs[0][1].shape) == (16, 3, 3, 3)

# список слоёв: (type, name, bottoms, tops, params)
layers = [
    ("Input", "in0", [], ["in0"], ""),
    ("Crop", "crop0", ["in0"], ["c0"], "2=0 5=3"),
]
prev = "c0"
blobs = ["in0", "c0"]
for i, (k, w, b) in enumerate(convs):
    out = "f%d" % i
    layers.append(("Convolution", "conv%d" % i, [prev], [out],
                   "0=16 1=3 5=1 6=%d" % w.numel()))
    prev = out
    blobs.append(out)

param_path = "rife_hand.ncnn.param"
bin_path = "rife_hand.ncnn.bin"
with open(param_path, "w") as f:
    f.write("7767577\n")
    f.write("%d %d\n" % (len(layers), len(blobs)))
    for typ, name, bottoms, tops, params in layers:
        f.write("%s %s %d %d %s %s%s\n" % (
            typ, name, len(bottoms), len(tops),
            " ".join(bottoms + tops),
            (" " + params) if params else "",
            ""))


def wblob(f, t):
    a = t.detach().cpu().contiguous()
    f.write(struct.pack("<i", 0))
    f.write(a.numpy().astype("<f4").tobytes())


with open(bin_path, "wb") as f:
    for k, w, b in convs:
        wblob(f, w)
        wblob(f, b)

# самопроверка: magic и счётчики
with open(param_path, "rb") as f:
    head = f.read(64)
print("param head bytes:", head[:24])
first = head.split(b"\n", 1)[0].strip()
assert first == b"7767577", "BAD MAGIC: %r" % first
print("param lines head:")
for ln in head.decode("utf-8", "replace").splitlines()[:4]:
    print("   |", ln)
print("wrote %s (%d B) and %s (%d B)" % (
    param_path, os.path.getsize(param_path),
    bin_path, os.path.getsize(bin_path)))
