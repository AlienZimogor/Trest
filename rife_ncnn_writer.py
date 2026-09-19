#!/usr/bin/env python3
"""Инкремент 3: scale0 полностью (flow0 + flow1) + cat_1(15ch) + scale1 convrelu_6/7.
Веса берутся из упорядоченных encode.cnnN с проверкой форм.
Ожидание бисекта: финал Convolution c=192 w=8 h=6, ALL LAYERS OK."""
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

enc = [(k, v) for k, v in sd.items() if k.startswith("encode.") and v.dim() == 4]
print("encode conv/deconv order:")
for k, v in enc[:12]:
    print("  ", k, tuple(v.shape))

EXPECT = [
    (16, 3, 3, 3),   # 0 conv0
    (16, 16, 3, 3),  # 1 conv1
    (16, 16, 3, 3),  # 2 conv2
    (16, 4, 4, 4),   # 3 deconv0 (ConvTranspose)
    (16, 4, 3, 3),   # 4 conv3
    (16, 16, 3, 3),  # 5 conv4
    (16, 16, 3, 3),  # 6 conv5
    (16, 4, 4, 4),   # 7 deconv1 (ConvTranspose)
    (96, 15, 3, 3),  # 8 conv6
    (192, 96, 3, 3), # 9 conv7
]
assert len(enc) >= len(EXPECT), "not enough encode weights: %d" % len(enc)
W = []
for i, (k, v) in enumerate(enc[:len(EXPECT)]):
    assert tuple(v.shape) == EXPECT[i], "shape mismatch at %d (%s): %s vs %s" % (
        i, k, tuple(v.shape), EXPECT[i])
    W.append(v)
B = [sd["encode.cnn%d.bias" % i] for i in range(len(EXPECT))]


def conv(i, inch, outch):
    m = nn.Conv2d(inch, outch, 3, 1, 1)
    with torch.no_grad():
        m.weight.copy_(W[i]); m.bias.copy_(B[i])
    return m


def deconv(i):
    m = nn.ConvTranspose2d(16, 4, 4, 2, 1)
    with torch.no_grad():
        m.weight.copy_(W[i]); m.bias.copy_(B[i])
    return m


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.c0 = conv(0, 3, 16)   # input img0 only; we pad 7->3 by slicing
        self.c1 = conv(1, 16, 16)
        self.c2 = conv(2, 16, 16)
        self.d0 = deconv(3)
        self.c3 = conv(4, 4, 16)
        self.c4 = conv(5, 16, 16)
        self.c5 = conv(6, 16, 16)
        self.d1 = deconv(7)
        self.c6 = conv(8, 15, 96)
        self.c7 = conv(9, 96, 192)

    def forward(self, x):
        i0 = x[:, 0:3]
        i1 = x[:, 3:6]
        t = x[:, 6:7]
        f = self.c2(self.c1(self.c0(i0)))
        flow0 = self.d0(f)
        g = self.c5(self.c4(self.c3(flow0)))
        flow1 = self.d1(g)
        cat1 = torch.cat([i0, i1, flow0, flow1, t], 1)
        h = F.interpolate(cat1, scale_factor=0.0625, mode="bilinear",
                          align_corners=False)
        h = self.c7(self.c6(h))
        return h


# conv0 weights are (16,3,3,3) but we feed 7ch input; slice to img0 inside forward (done).
net = Net().eval()
x = torch.rand(1, 7, 384, 512)
with torch.no_grad():
    y = net(x)
print("eager out:", tuple(y.shape))
assert tuple(y.shape) == (1, 192, 6, 8), y.shape
torch.onnx.export(net, x, "rife_hand.onnx", opset_version=13,
                  input_names=["in0"], output_names=["out0"],
                  dynamo=False, do_constant_folding=True)
print("wrote rife_hand.onnx (%d B)" % os.path.getsize("rife_hand.onnx"))
