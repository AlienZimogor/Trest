#!/usr/bin/env python3
"""Инкремент 3: single-input граф =
scale0 encode(7->16->16->16) + deconv flow0(16->4)
+ feat-ветка(16->16->16->16 + deconv 16->4)
+ cat_1 = img0(3)+img1(3)+flow0(4)+feat0(4)+ts(1) = 15
+ conv scale1 (15->96, stride2).
Ожидание бисекта: c=16,16,16,4,16,16,16,4,15(cat),96; ALL LAYERS OK."""
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
sd_raw = torch.load(pkls[0], map_location="cpu", weights_only=False)
if isinstance(sd_raw, dict) and "state_dict" in sd_raw:
    sd_raw = sd_raw["state_dict"]
sd = {}
for k, v in sd_raw.items():
    k2 = k
    while k2.startswith("module."):
        k2 = k2[len("module."):]
    sd[k2] = v


def find_all(shape):
    out = []
    for k, v in sd.items():
        if v.dim() == 4 and tuple(v.shape) == shape:
            out.append((k, v, sd.get(k.replace(".weight", ".bias"))))
    return out


enc0 = find_all((16, 3, 3, 3))
c16 = find_all((16, 16, 3, 3))
d16_4 = find_all((16, 4, 4, 4))
c96_15 = find_all((96, 15, 3, 3))
print("enc0=%d c16=%d d16_4=%d c96_15=%d" % (len(enc0), len(c16), len(d16_4), len(c96_15)))
assert len(enc0) >= 1 and len(c16) >= 5 and len(d16_4) >= 2 and len(c96_15) >= 1

e0w, e0b = enc0[0][1], enc0[0][2]
e1w, e1b = c16[0][1], c16[0][2]
e2w, e2b = c16[1][1], c16[1][2]
f1w, f1b = c16[2][1], c16[2][2]
f2w, f2b = c16[3][1], c16[3][2]
f3w, f3b = c16[4][1], c16[4][2]
dflow_w, dflow_b = d16_4[0][1], d16_4[0][2]
dfeat_w, dfeat_b = d16_4[1][1], d16_4[1][2]
s1w, s1b = c96_15[0][1], c96_15[0][2]

W0 = torch.zeros(16, 7, 3, 3)
with torch.no_grad():
    W0[:, 0:3] = e0w


def mk_conv(w, b, stride=1):
    c = nn.Conv2d(w.shape[1], w.shape[0], 3, stride, 1, bias=(b is not None))
    with torch.no_grad():
        c.weight.copy_(w)
        if b is not None:
            c.bias.copy_(b)
    return c


def mk_deconv(w, b):
    c = nn.ConvTranspose2d(w.shape[0], w.shape[1], 4, 2, 1, bias=(b is not None))
    with torch.no_grad():
        c.weight.copy_(w)
        if b is not None:
            c.bias.copy_(b)
    return c


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.e0 = mk_conv(W0, e0b)
        self.e1 = mk_conv(e1w, e1b)
        self.e2 = mk_conv(e2w, e2b)
        self.dflow = mk_deconv(dflow_w, dflow_b)
        self.f1 = mk_conv(f1w, f1b)
        self.f2 = mk_conv(f2w, f2b)
        self.f3 = mk_conv(f3w, f3b)
        self.dfeat = mk_deconv(dfeat_w, dfeat_b)
        self.s1 = mk_conv(s1w, s1b, stride=2)

    def forward(self, x):
        img0 = x[:, 0:3]
        img1 = x[:, 3:6]
        ts = x[:, 6:7]
        e = self.e2(self.e1(self.e0(x)))
        flow0 = self.dflow(e)
        feat0 = self.dfeat(self.f3(self.f2(self.f1(e))))
        cat = torch.cat([img0, img1, flow0, feat0, ts], dim=1)
        return self.s1(cat)


net = Net().eval()
x = torch.rand(1, 7, 384, 512)
with torch.no_grad():
    y = net(x)
print("eager out:", tuple(y.shape))
assert y.shape[1] == 96
torch.onnx.export(net, x, "rife_hand.onnx", opset_version=13,
                  input_names=["in0"], output_names=["out0"],
                  dynamo=False, do_constant_folding=True)
print("wrote rife_hand.onnx (%d B)" % os.path.getsize("rife_hand.onnx"))
