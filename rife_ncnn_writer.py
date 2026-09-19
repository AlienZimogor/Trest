#!/usr/bin/env python3
"""Инкремент 3a: scale1 = Concat(15ch) + Conv(15->96).
Ожидание бисекта: layers=7, Input(3)+Input(3)+Input(4)+Input(16)+Input(16)
-> Concat c=15 -> Conv c=96, ALL LAYERS OK."""
import os, glob
import torch
import torch.nn as nn


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

# Ищем первую conv scale1: (96, 15, 3, 3) или (96, 15, 1, 1)
w96_15 = b96_15 = key96_15 = None
for k, v in sd.items():
    if v.dim() == 4 and v.shape[0] == 96 and v.shape[1] == 15:
        w96_15 = v
        key96_15 = k
        b96_15 = sd.get(k.replace(".weight", ".bias"))
        break
print("scale1 conv key:", key96_15, tuple(w96_15.shape) if w96_15 is not None else None)
assert w96_15 is not None, "scale1 conv (96,15,*,*) not found"
print("bias:", tuple(b96_15.shape) if b96_15 is not None else None)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # 5 входов: img0_down(3), img1_down(3), flow0_up(4), feat0(16), feat1(16)
        self.conv = nn.Conv2d(15, 96, w96_15.shape[2:], 1, w96_15.shape[2]//2,
                              bias=(b96_15 is not None))
        with torch.no_grad():
            self.conv.weight.copy_(w96_15)
            if b96_15 is not None:
                self.conv.bias.copy_(b96_15)

    def forward(self, i0d, i1d, f0u, f0, f1):
        cat = torch.cat([i0d, i1d, f0u, f0, f1], dim=1)
        return self.conv(cat)


net = Net().eval()
i0d = torch.rand(1, 3, 192, 256)
i1d = torch.rand(1, 3, 192, 256)
f0u = torch.rand(1, 4, 192, 256)
f0 = torch.rand(1, 16, 192, 256)
f1 = torch.rand(1, 16, 192, 256)
with torch.no_grad():
    y = net(i0d, i1d, f0u, f0, f1)
print("eager out:", tuple(y.shape))
assert y.shape[1] == 96
torch.onnx.export(net, (i0d, i1d, f0u, f0, f1), "rife_hand.onnx", opset_version=13,
                  input_names=["i0d", "i1d", "f0u", "f0", "f1"], output_names=["out0"],
                  dynamo=False, do_constant_folding=True)
print("wrote rife_hand.onnx (%d B)" % os.path.getsize("rife_hand.onnx"))
