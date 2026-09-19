#!/usr/bin/env python3
"""Инкремент 1 (v5): строим СВОЙ ONNX из весов pkl (scale0: 3 свёртки 16ch),
затем pnnx даёт param/bin, которые устройство читает гарантированно.
Первая свёртка расширена 3->7 каналов нулями под 7-канальный вход бисекта.
Ожидание бисекта: LOAD ok, слои Conv, финал c=16, ALL LAYERS OK."""
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

c0, b0 = sd["encode.cnn0.weight"], sd["encode.cnn0.bias"]
c1, b1 = sd["encode.cnn1.weight"], sd["encode.cnn1.bias"]
c2, b2 = sd["encode.cnn2.weight"], sd["encode.cnn2.bias"]
print("picked:", tuple(c0.shape), tuple(c1.shape), tuple(c2.shape))
assert tuple(c0.shape) == (16, 3, 3, 3)

W0 = torch.zeros(16, 7, 3, 3)
with torch.no_grad():
    W0[:, 0:3] = c0


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.c0 = nn.Conv2d(7, 16, 3, 1, 1)
        self.c1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.c2 = nn.Conv2d(16, 16, 3, 1, 1)
        with torch.no_grad():
            self.c0.weight.copy_(W0); self.c0.bias.copy_(b0)
            self.c1.weight.copy_(c1); self.c1.bias.copy_(b1)
            self.c2.weight.copy_(c2); self.c2.bias.copy_(b2)

    def forward(self, x):
        return self.c2(self.c1(self.c0(x)))


net = Net().eval()
x = torch.rand(1, 7, 384, 512)
with torch.no_grad():
    y = net(x)
print("eager out:", tuple(y.shape), "mean=%.4f" % float(y.mean()))
torch.onnx.export(net, x, "rife_hand.onnx", opset_version=13,
                  input_names=["in0"], output_names=["out0"],
                  dynamo=False, do_constant_folding=True)
print("wrote rife_hand.onnx (%d B)" % os.path.getsize("rife_hand.onnx"))
