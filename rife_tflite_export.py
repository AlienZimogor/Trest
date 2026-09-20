#!/usr/bin/env python3
"""RIFE v4.26 -> LiteRT (.tflite) без ncnn/pnnx.
Цепочка: pkl -> PyTorch -> ONNX(opset16, нативный GridSample) -> onnxsim -> onnx2tf -> tflite.
Числовой гейт: tflite vs PyTorch на одном входе, max|diff| < 1e-3.
beta в pkl — сам тензор (block{i}.convblock.{n}.beta), суффикса .weight у него НЕТ."""
import os, glob, shutil, subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf

H, W = 384, 512

# ---------- загрузка весов ----------
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
pkl = {}
for k, v in raw.items():
    k2 = k
    while k2.startswith("module."): k2 = k2[7:]
    pkl[k2] = v
print("pkl keys:", len(pkl))

# ---------- внутренние имена -> оригинальные ключи pkl ----------
sd = {}

def _cp(internal, orig):
    sd[internal + ".weight"] = pkl[orig + ".weight"]
    if (orig + ".bias") in pkl:
        sd[internal + ".bias"] = pkl[orig + ".bias"]

CONV_MAP = [
    ("e0", "encode.cnn0"), ("e1", "encode.cnn1"),
    ("e2", "encode.cnn2"), ("ed", "encode.cnn3"),
]
for i in range(5):
    CONV_MAP += [
        ("b%d.c0" % i, "block%d.conv0.0.0" % i),
        ("b%d.c1" % i, "block%d.conv0.1.0" % i),
        ("b%d.last" % i, "block%d.lastconv.0" % i),
    ]
    for n in range(8):
        CONV_MAP.append(("b%d.cb.%d.c" % (i, n), "block%d.convblock.%d.conv" % (i, n)))
for a, b in CONV_MAP:
    _cp(a, b)
# beta — отдельный тензор, копируется БЕЗ суффикса .weight
for i in range(5):
    for n in range(8):
        sd["b%d.cb.%d.beta" % (i, n)] = pkl["block%d.convblock.%d.beta" % (i, n)]

# ---------- модель ----------
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
    c = nn.ConvTranspose2d(w.shape[0], w.shape[1], w.shape[2], 2, 1, bias=(b is not None))
    with torch.no_grad():
        c.weight.copy_(w)
        if b is not None: c.bias.copy_(b)
    return c

def warp(x, flow):
    HH, WW = x.shape[2], x.shape[3]
    gy, gx = torch.meshgrid(torch.arange(HH, dtype=x.dtype),
                            torch.arange(WW, dtype=x.dtype), indexing="ij")
    base = torch.stack([gx, gy], -1).unsqueeze(0)
    vgrid = base + flow.permute(0, 2, 3, 1)
    vgrid = torch.stack([2 * vgrid[..., 0] / (WW - 1) - 1,
                         2 * vgrid[..., 1] / (HH - 1) - 1], -1)
    return F.grid_sample(x, vgrid, align_corners=True, padding_mode="zeros")

class ConvBlock(nn.Module):
    def __init__(self, key):
        super().__init__()
        self.c = conv(key)
        self.register_buffer("beta", sd[key + ".beta"].clone())
        self.act = nn.LeakyReLU(0.2)
    def forward(self, x):
        return self.act(x + self.c(x) * self.beta)

class Block(nn.Module):
    def __init__(self, i, factor):
        super().__init__()
        self.factor = factor
        self.c0 = conv("b%d.c0" % i, 2)
        self.c1 = conv("b%d.c1" % i, 2)
        self.cb = nn.ModuleList([ConvBlock("b%d.cb.%d" % (i, n)) for n in range(8)])
        self.last = deconv("b%d.last" % i)
        self.ps = nn.PixelShuffle(2)
    def forward(self, x):
        if self.factor > 1:
            x = F.interpolate(x, scale_factor=1.0 / self.factor,
                              mode="bilinear", align_corners=False)
        y = F.relu(self.c0(x))
        y = F.relu(self.c1(y))
        for cb in self.cb:
            y = cb(y)
        y = self.ps(self.last(y))
        if self.factor > 1:
            y = F.interpolate(y, scale_factor=float(self.factor),
                              mode="bilinear", align_corners=False)
        return y  # 13 = flow4 + mask1 + feat8

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.e0 = conv("e0", 2)
        self.e1 = conv("e1")
        self.e2 = conv("e2")
        self.ed = deconv("ed")
        self.b = nn.ModuleList([Block(i, f) for i, f in enumerate([16, 8, 4, 2, 1])])
    def enc(self, img):
        return self.ed(self.e2(self.e1(self.e0(img))))
    def forward(self, x):
        img0 = x[:, 0:3]; img1 = x[:, 3:6]; ts = x[:, 6:7]
        d0 = self.enc(img0); d1 = self.enc(img1)
        y = self.b[0](torch.cat([img0, img1, d0, d1, ts], 1))
        flow = y[:, :4]; mask = torch.sigmoid(y[:, 4:5]); ft = y[:, 5:13]
        for blk in self.b[1:]:
            f0 = flow[:, :2]; f1 = flow[:, 2:4]
            y = blk(torch.cat([warp(img0, f0), warp(img1, f1),
                               warp(d0, f0), warp(d1, f1), ts, mask, ft, flow], 1))
            flow = y[:, :4]; mask = torch.sigmoid(y[:, 4:5]); ft = y[:, 5:13]
        f0 = flow[:, :2]; f1 = flow[:, 2:4]
        return warp(img0, f0) * mask + warp(img1, f1) * (1.0 - mask)

net = Net().eval()

# ---------- эталон ----------
torch.manual_seed(0)
x = torch.rand(1, 7, H, W)
with torch.no_grad():
    ref = net(x).numpy()

# ---------- ONNX (opset 16: нативный GridSample) ----------
torch.onnx.export(net, x, "rife_v426.onnx", opset_version=16,
                  input_names=["in0"], output_names=["out0"],
                  do_constant_folding=True)
print("wrote rife_v426.onnx %d B" % os.path.getsize("rife_v426.onnx"))

# ---------- onnxsim (без него onnx2tf падает на Expand) ----------
import onnx as onnx_lib
from onnxsim import simplify
m = onnx_lib.load("rife_v426.onnx")
m_sim, ok = simplify(m)
onnx_lib.save(m_sim, "rife_v426_sim.onnx")
print("onnxsim ok=%s, wrote rife_v426_sim.onnx %d B" % (ok, os.path.getsize("rife_v426_sim.onnx")))

# ---------- onnx2tf -> tflite ----------
subprocess.run(["onnx2tf", "-i", "rife_v426_sim.onnx", "-o", "tfl_out"], check=True)
cand = sorted(glob.glob("tfl_out/*_float32.tflite"))
assert cand, "no float32 tflite produced"
src = cand[0]
print("onnx2tf produced:", src)

# ---------- гейт: tflite vs PyTorch ----------
interp = tf.lite.Interpreter(model_path=src)
interp.allocate_tensors()
din = interp.get_input_details()[0]
dout = interp.get_output_details()[0]
print("tflite input :", din["shape"], din["dtype"])
print("tflite output:", dout["shape"], dout["dtype"])
xin = np.ascontiguousarray(x.numpy().transpose(0, 2, 3, 1), dtype=np.float32)
interp.set_tensor(din["index"], xin)
interp.invoke()
got = interp.get_tensor(dout["index"])
diff = float(np.max(np.abs(got - ref.transpose(0, 2, 3, 1))))
print("GATE max|diff| = %.6f" % diff)
assert diff < 1e-3, "torch/tflite mismatch too large"

# ---------- артефакты ----------
shutil.copy(src, "rife_v426.tflite")
print("wrote rife_v426.tflite %d B" % os.path.getsize("rife_v426.tflite"))
f16 = sorted(glob.glob("tfl_out/*_float16.tflite"))
if f16:
    shutil.copy(f16[0], "rife_v426_f16.tflite")
    print("wrote rife_v426_f16.tflite %d B" % os.path.getsize("rife_v426_f16.tflite"))
