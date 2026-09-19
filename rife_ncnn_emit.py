#!/usr/bin/env python3
"""Прямой эмиттер ncnn param/bin для RIFE v4.26 из rife426.pth (без onnx/pnnx).
Warp = кастомный слой rife.Warp (CPU в vfi_mini). Плюс эталонный eager-прогон
на том же детерминированном входе, что использует бенч устройства, для сравнения."""
import os, struct
import numpy as np
import torch
import torch.nn.functional as F

W_PATH = "rife426.pth"
raw = torch.load(W_PATH, map_location="cpu", weights_only=False)
if isinstance(raw, dict) and "state_dict" in raw:
    raw = raw["state_dict"]
sd = {}
for k, v in raw.items():
    k2 = k
    while k2.startswith("module."):
        k2 = k2[len("module."):]
    sd[k2] = v
print("weights:", len(sd))

param_lines = ["7767577"]
bin_blobs = []
layers = []


def L(line, blobs=()):
    layers.append(line)
    for arr in blobs:
        a = np.ascontiguousarray(arr.detach().cpu().numpy().reshape(-1), dtype="<f4")
        bin_blobs.append(a)


def conv(name, bottom, top, key, stride):
    w = sd[key + ".weight"]; b = sd.get(key + ".bias")
    o, i, k, _ = w.shape
    p = (k - 1) // 2
    L("Convolution %s 1 1 %s %s 0=%d 1=%d 2=1 3=%d 4=%d 5=1 6=%d"
      % (name, bottom, top, o, k, stride, p, o * i * k * k), (w, b) if b is not None else (w,))


def deconv(name, bottom, top, key):
    w = sd[key + ".weight"]; b = sd.get(key + ".bias")
    i, o, k, _ = w.shape
    L("Deconvolution %s 1 1 %s %s 0=%d 1=%d 2=1 3=2 4=1 5=1 6=%d"
      % (name, bottom, top, o, k, i * o * k * k), (w, b) if b is not None else (w,))


L("Input in0 0 1 in0")
L("Slice split_in 1 3 in0 img0 img1 ts -23300=3,3,1 1=0")

conv("eA0", "img0", "eA0o", "encode.cnn0", 2)
conv("eA1", "eA0o", "eA1o", "encode.cnn1", 1)
conv("eA2", "eA1o", "eA2o", "encode.cnn2", 1)
deconv("eAd", "eA2o", "d0", "encode.cnn3")
conv("eB0", "img1", "eB0o", "encode.cnn0", 2)
conv("eB1", "eB0o", "eB1o", "encode.cnn1", 1)
conv("eB2", "eB1o", "eB2o", "encode.cnn2", 1)
deconv("eBd", "eB2o", "d1", "encode.cnn3")
L("Concat cat0 5 1 img0 img1 d0 d1 ts cat0in 0=0")

FACT = [(16, 96, 192), (8, 64, 128), (4, 48, 96), (2, 32, 64), (1, 16, 32)]
pf = pm = pft = None
for i, (f, c0c, c1c) in enumerate(FACT):
    pre = "b%d" % i
    if i == 0:
        src = "cat0in"
    else:
        L("Slice %s_fs 1 2 %s %s_fa %s_fb -23300=2,2 1=0" % (pre, pf, pre, pre))
        L("rife.Warp %s_w0 2 1 img0 %s_fa %s_wi0" % (pre, pre, pre))
        L("rife.Warp %s_w1 2 1 img1 %s_fb %s_wi1" % (pre, pre, pre))
        L("rife.Warp %s_w2 2 1 d0 %s_fa %s_wd0" % (pre, pre, pre))
        L("rife.Warp %s_w3 2 1 d1 %s_fb %s_wd1" % (pre, pre, pre))
        L("Concat %s_cat 8 1 %s_wi0 %s_wi1 %s_wd0 %s_wd1 ts %s %s %s %s_catin 0=0"
          % (pre, pre, pre, pre, pre, pm, pft, pf, pre))
        src = "%s_catin" % pre
    if f > 1:
        L("Interp %s_down 1 1 %s %s_bd 0=2 1=%f 2=%f 3=-233 4=-233 5=0"
          % (pre, src, pre, 1.0 / f, 1.0 / f))
        bd = "%s_bd" % pre
    else:
        bd = src
    conv("%s_c0" % pre, bd, "%s_c0o" % pre, "block%d.conv0.0.0" % i, 2)
    L("ReLU %s_r0 1 1 %s_c0o %s_c0r" % (pre, pre, pre))
    conv("%s_c1" % pre, "%s_c0r" % pre, "%s_c1o" % pre, "block%d.conv0.1.0" % i, 2)
    L("ReLU %s_r1 1 1 %s_c1o %s_x" % (pre, pre, pre))
    x = "%s_x" % pre
    for n in range(8):
        ck = "block%d.convblock.%d.conv" % (i, n)
        bk = "block%d.convblock.%d.beta" % (i, n)
        conv("%s_cb%d" % (pre, n), x, "%s_cb%do" % (pre, n), ck, 1)
        beta = sd[bk].reshape(-1)
        L("MemoryData %s_cb%db 0 1 %s_cb%db 0=1 1=1 2=%d" % (pre, n, pre, n, beta.numel()), (sd[bk],))
        L("BinaryOp %s_cb%dm 2 1 %s_cb%do %s_cb%db %s_cb%dm 0=2 1=0 2=0" % (pre, n, pre, n, pre, n, pre, n))
        L("BinaryOp %s_cb%da 2 1 %s_cb%dm %s %s_cb%da 0=0 1=0 2=0" % (pre, n, pre, n, x, pre, n))
        L("ReLU %s_cb%dl 1 1 %s_cb%da %s_cb%dl 0=0.2" % (pre, n, pre, n, pre, n))
        x = "%s_cb%dl" % (pre, n)
    deconv("%s_last" % pre, x, "%s_lo" % pre, "block%d.lastconv.0" % i)
    L("PixelShuffle %s_ps 1 1 %s_lo %s_pso 0=2" % (pre, pre, pre))
    if f > 1:
        L("Interp %s_up 1 1 %s_pso %s_f13 0=2 1=%f 2=%f 3=-233 4=-233 5=0"
          % (pre, pre, pre, float(f), float(f)))
        f13 = "%s_f13" % pre
    else:
        f13 = "%s_pso" % pre
    L("Slice %s_sp 1 3 %s %s_flow %s_mraw %s_feat -23300=4,1,8 1=0" % (pre, f13, pre, pre, pre))
    L("Sigmoid %s_sig 1 1 %s_mraw %s_mask" % (pre, pre, pre))
    pf, pm, pft = "%s_flow" % pre, "%s_mask" % pre, "%s_feat" % pre

L("Slice fin_fs 1 2 %s fin_fa fin_fb -23300=2,2 1=0" % pf)
L("rife.Warp fin_w0 2 1 img0 fin_fa fw0")
L("rife.Warp fin_w1 2 1 img1 fin_fb fw1")
L("BinaryOp fin_m0 2 1 fw0 %s fm0 0=2 1=0 2=0" % pm)
L("BinaryOp fin_inv 1 1 %s finv 0=7 1=1 2=1.0" % pm)
L("BinaryOp fin_m1 2 1 fw1 finv fm1 0=2 1=0 2=0" )
L("BinaryOp fin_out 2 1 fm0 fm1 out 0=0 1=0 2=0")

param_lines.append("%d %d" % (len(layers), len(set(sum([l.split()[4:4 + int(l.split()[2]) + int(l.split()[3])] for l in layers], [])))))
param_lines += layers
open("rife_hand.ncnn.param", "w").write("\n".join(param_lines) + "\n")
with open("rife_hand.ncnn.bin", "wb") as fbin:
    for a in bin_blobs:
        fbin.write(struct.pack("<i", 0))
        fbin.write(a.tobytes())
print("wrote param layers=%d bin=%d B" % (len(layers), os.path.getsize("rife_hand.ncnn.bin")))

# ---- эталон на том же входе, что у бенча устройства ----
W, H = 512, 384
xs = torch.arange(W, dtype=torch.float32)
ys = torch.arange(H, dtype=torch.float32)
a = torch.stack([torch.frac((xs[None, :] + c * 37) / 255).expand(H, W) for c in range(3)])
b = torch.stack([torch.frac((xs[None, :] + ys[:, None] + c * 37) / 255).expand(H, W) for c in range(3)])
tin = torch.full((1, 1, H, W), 0.5)
xin = torch.cat([a.unsqueeze(0), b.unsqueeze(0), tin], 1)


def cw(x, w, b, s):
    return F.conv2d(x, w, b, stride=s, padding=(w.shape[2] - 1) // 2)


def dw(x, w, b):
    return F.conv_transpose2d(x, w, b, stride=2, padding=1)


def warp(x, fl):
    N, C, HH, WW = x.shape
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    base = torch.stack([gx, gy], -1).unsqueeze(0)
    vg = base + fl.permute(0, 2, 3, 1)
    vg = torch.stack([2 * vg[..., 0] / (WW - 1) - 1, 2 * vg[..., 1] / (HH - 1) - 1], -1)
    return F.grid_sample(x, vg, align_corners=True, padding_mode="zeros")


img0, img1, ts = xin[:, :3], xin[:, 3:6], xin[:, 6:7]
d0 = dw(cw(cw(cw(img0, sd["encode.cnn0.weight"], sd["encode.cnn0.bias"], 2),
              sd["encode.cnn1.weight"], sd["encode.cnn1.bias"], 1),
           sd["encode.cnn2.weight"], sd["encode.cnn2.bias"], 1), sd["encode.cnn3.weight"], sd["encode.cnn3.bias"])
d1 = dw(cw(cw(cw(img1, sd["encode.cnn0.weight"], sd["encode.cnn0.bias"], 2),
              sd["encode.cnn1.weight"], sd["encode.cnn1.bias"], 1),
           sd["encode.cnn2.weight"], sd["encode.cnn2.bias"], 1), sd["encode.cnn3.weight"], sd["encode.cnn3.bias"])
cur = torch.cat([img0, img1, d0, d1, ts], 1)
pf = pm = pft = None
for i, (f, c0c, c1c) in enumerate(FACT):
    if i > 0:
        fa, fb = pf[:, :2], pf[:, 2:4]
        cur = torch.cat([warp(img0, fa), warp(img1, fb), warp(d0, fa), warp(d1, fb),
                         ts, pm, pft, pf], 1)
    if f > 1:
        cur = F.interpolate(cur, scale_factor=1.0 / f, mode="bilinear", align_corners=False)
    cur = F.relu(cw(cur, sd["block%d.conv0.0.0.weight" % i], sd["block%d.conv0.0.0.bias" % i], 2))
    cur = F.relu(cw(cur, sd["block%d.conv0.1.0.weight" % i], sd["block%d.conv0.1.0.bias" % i], 2))
    for n in range(8):
        c = cw(cur, sd["block%d.convblock.%d.conv.weight" % (i, n)],
               sd["block%d.convblock.%d.conv.bias" % (i, n)], 1)
        c = c * sd["block%d.convblock.%d.beta" % (i, n)]
        cur = F.leaky_relu(cur + c, 0.2)
    cur = dw(cur, sd["block%d.lastconv.0.weight" % i], sd["block%d.lastconv.0.bias" % i])
    cur = F.pixel_shuffle(cur, 2)
    if f > 1:
        cur = F.interpolate(cur, scale_factor=float(f), mode="bilinear", align_corners=False)
    pf, pm, pft = cur[:, :4], torch.sigmoid(cur[:, 4:5]), cur[:, 5:13]
fa, fb = pf[:, :2], pf[:, 2:4]
out = warp(img0, fa) * pm + warp(img1, fb) * (1 - pm)
print("REF mean=%.4f min=%.4f max=%.4f" % (out.mean(), out.min(), out.max()))
