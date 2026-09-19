#!/usr/bin/env python3
"""Инкремент 2: scale0 = Conv(7->16) x3 + ConvTranspose(16->4) = flow0.
Ожидание бисекта: layers=5, conv_0/1/2 c=16 @512x384,
Deconvolution c=4 @1024x768, ALL LAYERS OK."""
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

dk = db = dkey = None
for k, v in sd.items():
    if (v.dim() == 4 and v.shape[0] == 16 and v.shape[1] == 4
            and v.shape[2] == v.shape[3] and v.shape[2] in (2, 4)):
        dk, dkey = v, k
        db = sd.get(k.replace(".weight", ".bias"))
        break
print("deconv key:", dkey, tuple(dk.shape) if dk is not None else None)
assert dk is not None, "deconv (16,4,k,k) not found"
kk = dk.shape[2]
pad = 0 if kk == 2 else 1
print("deconv kernel=%d stride=2 pad=%d bias=%s" % (kk, pad, db is not None))

W0 = torch.zeros(16, 7, 3, 3)
with torch.no_grad():
    W0[:, 0:3] = c0


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.c0 = nn.Conv2d(7, 16, 3, 1, 1)
        self.c1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.c2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.d0 = nn.ConvTranspose2d(16, 4, kk, 2, pad, bias=(db is not None))
        with torch.no_grad():
            self.c0.weight.copy_(W0); self.c0.bias.copy_(b0)
            self.c1.weight.copy_(c1); self.c1.bias.copy_(b1)
            self.c2.weight.copy_(c2); self.c2.bias.copy_(b2)
            self.d0.weight.copy_(dk)
            if db is not None:
                self.d0.bias.copy_(db)

    def forward(self, x):
        return self.d0(self.c2(self.c1(self.c0(x))))


net = Net().eval()
x = torch.rand(1, 7, 384, 512)
with torch.no_grad():
    y = net(x)
print("eager out:", tuple(y.shape))
assert y.shape[2] == 768 and y.shape[3] == 1024, "deconv must double resolution"
torch.onnx.export(net, x, "rife_hand.onnx", opset_version=13,
                  input_names=["in0"], output_names=["out0"],
                  dynamo=False, do_constant_folding=True)
print("wrote rife_hand.onnx (%d B)" % os.path.getsize("rife_hand.onnx"))
