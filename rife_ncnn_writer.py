#!/usr/bin/env python3
"""Инкремент 3: scale0 (encode x2, общие веса) + cat_1(15) + block0 полный
(conv96/conv192/8xconvblock+beta/residual/LeakyReLU/lastconv/PixelShuffle/upsample)
+ split 13 -> flow(4)/mask(1)/feat(8). Single-input 7ch.
Ожидание бисекта: 16,16,16,4,16,16,16,4,15,96,192,192x8,52,13,13,4/1/8."""
import os, glob
import torch
import torch.nn as nn
import torch.nn.functional as F


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


def W(key):
    return sd[key]


def B(key):
    return sd.get(key.replace(".weight", ".bias"))


def conv(key, stride=1):
    w = W(key)
    c = nn.Conv2d(w.shape[1], w.shape[0], w.shape[2], stride, w.shape[2] // 2,
                  bias=(B(key) is not None))
    with torch.no_grad():
        c.weight.copy_(w)
        if B(key) is not None:
            c.bias.copy_(B(key))
    return c


def deconv(key):
    w = W(key)
    c = nn.ConvTranspose2d(w.shape[0], w.shape[1], w.shape[2], 2, 1,
                           bias=(B(key) is not None))
    with torch.no_grad():
        c.weight.copy_(w)
        if B(key) is not None:
            c.bias.copy_(B(key))
    return c


class ConvBlock(nn.Module):
    def __init__(self, prefix):
        super().__init__()
        self.c = conv(prefix + ".conv.weight")
        self.register_buffer("beta", W(prefix + ".beta"))
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.act(x + self.c(x) * self.beta)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.e0 = conv("encode.cnn0.weight")
        self.e1 = conv("encode.cnn1.weight")
        self.e2 = conv("encode.cnn2.weight")
        self.ed = deconv("encode.cnn3.weight")
        self.b0c0 = conv("block0.conv0.0.0.weight", stride=2)
        self.b0c1 = conv("block0.conv0.1.0.weight", stride=2)
        self.cb = nn.ModuleList(
            [ConvBlock("block0.convblock.%d" % i) for i in range(8)])
        self.b0d = deconv("block0.lastconv.0.weight")
        self.ps = nn.PixelShuffle(2)
        self.relu = nn.ReLU()

    def forward(self, x):
        img0 = x[:, 0:3]
        img1 = x[:, 3:6]
        ts = x[:, 6:7]
        e0 = self.e2(self.e1(self.e0(img0)))
        d0 = self.ed(e0)
        e1 = self.e2(self.e1(self.e0(img1)))
        d1 = self.ed(e1)
        cat = torch.cat([img0, img1, d0, d1, ts], dim=1)
        y = F.interpolate(cat, scale_factor=0.0625,
                          mode="bilinear", align_corners=False)
        y = self.relu(self.b0c0(y))
        y = self.relu(self.b0c1(y))
        for cb in self.cb:
            y = cb(y)
        y = self.ps(self.b0d(y))
        y = F.interpolate(y, scale_factor=16.0,
                          mode="bilinear", align_corners=False)
        return y


net = Net().eval()
x = torch.rand(1, 7, 384, 512)
with torch.no_grad():
    y = net(x)
print("eager out:", tuple(y.shape))
assert y.shape[1] == 13
torch.onnx.export(net, x, "rife_hand.onnx", opset_version=13,
                  input_names=["in0"], output_names=["out0"],
                  dynamo=False, do_constant_folding=True)
print("wrote rife_hand.onnx (%d B)" % os.path.getsize("rife_hand.onnx"))
