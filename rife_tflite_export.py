#!/usr/bin/env python3
"""RIFE v4.26 -> ONNX(opset16) -> onnx2tf -> TFLite + числовой гейт torch vs tflite."""
import os, glob, shutil, subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf

H, W = 384, 512

def _pref(p):
    s = 0; b = os.path.basename(p).lower()
    if "4.26" in b or "426" in b: s += 4
    if "v4" in b: s += 1
    if "flownet" in b: s -= 3
    return s

pkls = sorted(glob.glob(os.path.join("model_src", "**", "*.pkl"), recursive=True),
              key=_pref, reverse=True)
raw = torch.load(pkls[0], map_location="cpu", weights_only=False)
if isinstance(raw, dict) and "state_dict" in raw: raw = raw["state_dict"]
sd = {}
for k, v in raw.items():
    k2 = k
    while k2.startswith("module."): k2 = k2[7:]
    sd[k2] = v
print("pkl keys:", len(sd))

def conv(key, stride=1):
    w = sd[key + ".weight"]; b = sd.get(key + ".bias")
    c = nn.Conv2d(w.shape[1], w.shape[0], w.shape[2], stride, w.shape[2] // 2,
                  bias=(b is not None))
    with torch.no_grad():
        c.weight.copy_(w)
        if b is not None: c.bias.copy_(b)
    return c

def deconv(key):
    w = sd[key + ".weight"]; b = sd.get(key + ".bias")
    c = nn.ConvTranspose2d(w.shape[0], w.shape[1], w.shape[2], 2, 1,
                           bias=(b is not None))
    with torch.no_grad():
        c.weight.copy_(w)
        if b is not None: c.bias.copy_(b)
    return c

def warp(x, flow):
    N, C, HH, WW = x.shape
    ys = torch.arange(HH, dtype=x.dtype, device=x.device)
    xs = torch.arange(WW, dtype=x.dtype, device=x.device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    base = torch.stack([gx, gy], -1).unsqueeze(0)
    vg = base + flow.permute(0, 2, 3, 1)
    vg = torch.stack([2 * vg[..., 0] / (WW - 1) - 1,
                      2 * vg[..., 1] / (HH - 1) - 1], -1)
    return F.grid_sample(x, vg, align_corners=True, padding_mode="zeros")

class ConvBlock(nn.Module):
    def __init__(self, prefix):
        super().__init__()
        self.c = conv(prefix + ".conv")
        self.register_buffer("beta", sd[prefix + ".beta"])   # buffer, БЕЗ .weight
        self.act = nn.LeakyReLU(0.2)
    def forward(self, x):
        return self.act(x + self.c(x) * self.beta)

class Block(nn.Module):
    def __init__(self, i, factor):
        super().__init__()
        self.factor = factor
        self.c0 = conv("block%d.conv0.0.0" % i, 2)
        self.c1 = conv("block%d.conv0.1.0" % i, 2)
        self.cb = nn.ModuleList([ConvBlock("block%d.convblock.%d" % (i, n)) for n in range(8)])
        self.last = deconv("block%d.lastconv.0" % i)
        self.ps = nn.PixelShuffle(2)
        self.relu = nn.ReLU()
    def forward(self, x):
        if self.factor > 1:
            x = F.interpolate(x, scale_factor=1.0 / self.factor,
                              mode="bilinear", align_corners=False)
        y = self.relu(self.c0(x))
        y = self.relu(self.c1(y))
        for cb in self.cb: y = cb(y)
        y = self.ps(self.last(y))
        if self.factor > 1:
            y = F.interpolate(y, scale_factor=float(self.factor),
                              mode="bilinear", align_corners=False)
        return y

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.e0 = conv("encode.cnn0", 2)
        self.e1 = conv("encode.cnn1")
        self.e2 = conv("encode.cnn2")
        self.ed = deconv("encode.cnn3")
        self.b0 = Block(0, 16); self.b1 = Block(1, 8); self.b2 = Block(2, 4)
        self.b3 = Block(3, 2);  self.b4 = Block(4, 1)
    def enc(self, img):
        return self.ed(self.e2(self.e1(self.e0(img))))
    def forward(self, x):
        img0 = x[:, 0:3]; img1 = x[:, 3:6]; ts = x[:, 6:7]
        d0 = self.enc(img0); d1 = self.enc(img1)
        y = self.b0(torch.cat([img0, img1, d0, d1, ts], 1))
        flow = y[:, :4]; mask = torch.sigmoid(y[:, 4:5]); feat = y[:, 5:13]
        for blk in (self.b1, self.b2, self.b3, self.b4):
            f0 = flow[:, :2]; f1 = flow[:, 2:4]
            w0 = warp(img0, f0); w1 = warp(img1, f1)
            wd0 = warp(d0, f0); wd1 = warp(d1, f1)
            x2 = torch.cat([w0, w1, wd0, wd1, ts, mask, feat, flow], 1)
            y = blk(x2)
            flow = y[:, :4]; mask = torch.sigmoid(y[:, 4:5]); feat = y[:, 5:13]
        f0 = flow[:, :2]; f1 = flow[:, 2:4]
        w0 = warp(img0, f0); w1 = warp(img1, f1)
        return w0 * mask + w1 * (1 - mask)

net = Net().eval()
x = torch.rand(1, 7, H, W)
with torch.no_grad():
    ref = net(x)
print("torch ref mean=%.4f min=%.4f max=%.4f"
      % (ref.mean().item(), ref.min().item(), ref.max().item()))

torch.onnx.export(net, x, "rife_v426.onnx", opset_version=16,
                  input_names=["in0"], output_names=["out0"],
                  do_constant_folding=True)
print("wrote rife_v426.onnx", os.path.getsize("rife_v426.onnx"))

subprocess.run(["onnx2tf", "-i", "rife_v426.onnx", "-o", "tfl_out", "-b", "1"],
               check=True)
cand = "tfl_out/rife_v426_sim_float32.tflite"
if not os.path.exists(cand):
    cand = sorted(glob.glob("tfl_out/*_float32.tflite"))[0]
print("onnx2tf produced:", cand)

interp = tf.lite.Interpreter(model_path=cand)
interp.allocate_tensors()
din = interp.get_input_details()[0]; dout = interp.get_output_details()[0]
print("tflite input :", din["shape"], din["dtype"])
print("tflite output:", dout["shape"], dout["dtype"])
xin = x.numpy().transpose(0, 2, 3, 1).astype(np.float32)
interp.set_tensor(din["index"], xin)
interp.invoke()
got = interp.get_tensor(dout["index"])
refn = ref.numpy().transpose(0, 2, 3, 1)
diff = float(np.max(np.abs(got - refn)))
print("GATE max|diff| = %.6f" % diff)
assert diff < 1e-3, "torch/tflite mismatch too large"
shutil.copy(cand, "rife_v426.tflite")
print("wrote rife_v426.tflite", os.path.getsize("rife_v426.tflite"))
