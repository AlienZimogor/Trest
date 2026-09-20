#!/usr/bin/env python3
"""RIFE v4.26 -> LiteRT (.tflite), NHWC [1,384,512,7] -> [1,384,512,3].
Warp = TFLite-builtin gather_nd (без grid_sample/Flex).
ГЕЙТ: постадийный max|diff| torch vs tf (d0/d1, block0 flow/mask/feat, финал);
конвертация только если финальный diff < 1e-3."""
import os, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf

H, W = 384, 512
FACT = [16, 8, 4, 2, 1]


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


def npw(k): return sd[k].detach().cpu().numpy()


def npb(k):
    b = sd.get(k + ".bias")
    return None if b is None else b.detach().cpu().numpy().astype(np.float32)


# ---------------- PyTorch-эталон ----------------
def tconv(key, stride=1):
    w = sd[key + ".weight"]; b = sd.get(key + ".bias")
    c = nn.Conv2d(w.shape[1], w.shape[0], w.shape[2], stride, w.shape[2] // 2,
                  bias=(b is not None))
    with torch.no_grad():
        c.weight.copy_(w)
        if b is not None: c.bias.copy_(b)
    return c


def tdeconv(key):
    w = sd[key + ".weight"]; b = sd.get(key + ".bias")
    c = nn.ConvTranspose2d(w.shape[0], w.shape[1], w.shape[2], 2, 1, bias=(b is not None))
    with torch.no_grad():
        c.weight.copy_(w)
        if b is not None: c.bias.copy_(b)
    return c


def twarp(x, flow):
    HH, WW = x.shape[2], x.shape[3]
    ys = torch.arange(HH, dtype=torch.float32); xs = torch.arange(WW, dtype=torch.float32)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    base = torch.stack([gx, gy], -1).unsqueeze(0)
    vg = base + flow.permute(0, 2, 3, 1)
    vg = torch.stack([2.0 * vg[..., 0] / (WW - 1) - 1.0,
                      2.0 * vg[..., 1] / (HH - 1) - 1.0], -1)
    return F.grid_sample(x, vg, align_corners=True, padding_mode="zeros")


class TConvBlock(nn.Module):
    def __init__(s, pre):
        super().__init__()
        s.c = tconv(pre + ".conv")
        s.register_buffer("beta", sd[pre + ".beta"])
        s.act = nn.LeakyReLU(0.2)

    def forward(s, x):
        return s.act(x + s.c(x) * s.beta)


class TBlock(nn.Module):
    def __init__(s, i, f):
        super().__init__()
        s.f = f
        s.c0 = tconv("block%d.conv0.0.0" % i, 2)
        s.c1 = tconv("block%d.conv0.1.0" % i, 2)
        s.cb = nn.ModuleList([TConvBlock("block%d.convblock.%d" % (i, n)) for n in range(8)])
        s.last = tdeconv("block%d.lastconv.0" % i)
        s.ps = nn.PixelShuffle(2)
        s.relu = nn.ReLU()

    def forward(s, x):
        if s.f > 1:
            x = F.interpolate(x, scale_factor=1.0 / s.f, mode="bilinear", align_corners=False)
        y = s.relu(s.c0(x)); y = s.relu(s.c1(y))
        for cb in s.cb: y = cb(y)
        y = s.ps(s.last(y))
        if s.f > 1:
            y = F.interpolate(y, scale_factor=float(s.f), mode="bilinear", align_corners=False)
        return y[:, :4], torch.sigmoid(y[:, 4:5]), y[:, 5:13]


class TNet(nn.Module):
    def __init__(s):
        super().__init__()
        s.e0 = tconv("encode.cnn0", 2); s.e1 = tconv("encode.cnn1")
        s.e2 = tconv("encode.cnn2"); s.ed = tdeconv("encode.cnn3")
        s.b = nn.ModuleList([TBlock(i, FACT[i]) for i in range(5)])
        s.debug = False
        s.dbg = {}

    def enc(s, img):
        return s.ed(s.e2(s.e1(s.e0(img))))

    def forward(s, x):
        img0 = x[:, :3]; img1 = x[:, 3:6]; ts = x[:, 6:7]
        d0 = s.enc(img0); d1 = s.enc(img1)
        if s.debug: s.dbg["d0"] = d0; s.dbg["d1"] = d1
        flow, mask, feat = s.b[0](torch.cat([img0, img1, d0, d1, ts], 1))
        if s.debug:
            s.dbg["f0"] = flow; s.dbg["m0"] = mask; s.dbg["fe0"] = feat
        for i in range(1, 5):
            fa = flow[:, :2]; fb = flow[:, 2:4]
            cat = torch.cat([twarp(img0, fa), twarp(img1, fb), twarp(d0, fa), twarp(d1, fb),
                             ts, mask, feat, flow], 1)
            flow, mask, feat = s.b[i](cat)
        fa = flow[:, :2]; fb = flow[:, 2:4]
        out = twarp(img0, fa) * mask + twarp(img1, fb) * (1 - mask)
        return out


# ---------------- TF-зеркало ----------------
CK = {}; BK = {}; DK = {}; BV = {}
CONV_KEYS = ["encode.cnn0", "encode.cnn1", "encode.cnn2"] + \
            ["block%d.conv0.0.0" % i for i in range(5)] + \
            ["block%d.conv0.1.0" % i for i in range(5)] + \
            ["block%d.convblock.%d.conv" % (i, n) for i in range(5) for n in range(8)]
for pre in CONV_KEYS:
    CK[pre] = tf.constant(npw(pre + ".weight").transpose(2, 3, 1, 0).astype(np.float32))
    b = npb(pre)
    BK[pre] = None if b is None else tf.constant(b)
for pre in ["encode.cnn3"] + ["block%d.lastconv.0" % i for i in range(5)]:
    DK[pre] = tf.constant(npw(pre + ".weight").transpose(2, 3, 1, 0).astype(np.float32))
    b = npb(pre)
    BK[pre] = None if b is None else tf.constant(b)
for i in range(5):
    for n in range(8):
        kp = "block%d.convblock.%d.conv" % (i, n)
        BV[kp] = tf.constant(npw("block%d.convblock.%d.beta" % (i, n)).reshape(-1).astype(np.float32))


def conv2d(x, pre, stride):
    y = tf.nn.conv2d(x, CK[pre], strides=[1, stride, stride, 1], padding="SAME")
    return y if BK[pre] is None else tf.nn.bias_add(y, BK[pre])


def deconv2d(x, pre):
    s = tf.shape(x)
    y = tf.nn.conv2d_transpose(x, DK[pre],
                               output_shape=[s[0], s[1] * 2, s[2] * 2, DK[pre].shape[2]],
                               strides=[1, 2, 2, 1], padding="SAME")
    return y if BK[pre] is None else tf.nn.bias_add(y, BK[pre])


def fwarp(x, flow):
    xx = x[0]; ff = flow[0]
    Hh = tf.shape(xx)[0]; Ww = tf.shape(xx)[1]
    ys = tf.cast(tf.range(Hh), tf.float32)[:, None] * tf.ones([Hh, Ww])
    xs = tf.cast(tf.range(Ww), tf.float32)[None, :] * tf.ones([Hh, Ww])
    gx = xs + ff[:, :, 0]; gy = ys + ff[:, :, 1]
    x0 = tf.floor(gx); y0 = tf.floor(gy)
    valid = (gx >= 0) & (gy >= 0) & (gx <= tf.cast(Ww - 1, tf.float32)) & (gy <= tf.cast(Hh - 1, tf.float32))
    cx0 = tf.clip_by_value(tf.cast(x0, tf.int32), 0, Ww - 1)
    cy0 = tf.clip_by_value(tf.cast(y0, tf.int32), 0, Hh - 1)
    cx1 = tf.clip_by_value(cx0 + 1, 0, Ww - 1)
    cy1 = tf.clip_by_value(cy0 + 1, 0, Hh - 1)
    idx = lambda cy, cx: tf.gather_nd(xx, tf.stack([cy, cx], -1))
    v00 = idx(cy0, cx0); v01 = idx(cy0, cx1); v10 = idx(cy1, cx0); v11 = idx(cy1, cx1)
    ax = (gx - x0)[:, :, None]; ay = (gy - y0)[:, :, None]
    out = (v00 * (1 - ax) + v01 * ax) * (1 - ay) + (v10 * (1 - ax) + v11 * ax) * ay
    out = out * tf.cast(valid[:, :, None], tf.float32)
    return out[None]


def fblock(x, i):
    f = FACT[i]
    if f > 1:
        x = tf.image.resize(x, [H // f, W // f], method="bilinear")
    y = tf.nn.relu(conv2d(x, "block%d.conv0.0.0" % i, 2))
    y = tf.nn.relu(conv2d(y, "block%d.conv0.1.0" % i, 2))
    for n in range(8):
        cp = "block%d.convblock.%d.conv" % (i, n)
        y = tf.nn.leaky_relu(y + conv2d(y, cp, 1) * BV[cp], 0.2)
    y = tf.nn.depth_to_space(deconv2d(y, "block%d.lastconv.0" % i), 2)
    if f > 1:
        y = tf.image.resize(y, [H, W], method="bilinear")
    return y[:, :, :, :4], tf.sigmoid(y[:, :, :, 4:5]), y[:, :, :, 5:13]


@tf.function(input_signature=[tf.TensorSpec([1, H, W, 7], tf.float32)])
def fnet(x):
    img0 = x[:, :, :, :3]; img1 = x[:, :, :, 3:6]; ts = x[:, :, :, 6:7]
    d0 = deconv2d(conv2d(conv2d(conv2d(img0, "encode.cnn0", 2), "encode.cnn1", 1), "encode.cnn2", 1), "encode.cnn3")
    d1 = deconv2d(conv2d(conv2d(conv2d(img1, "encode.cnn0", 2), "encode.cnn1", 1), "encode.cnn2", 1), "encode.cnn3")
    flow, mask, feat = fblock(tf.concat([img0, img1, d0, d1, ts], -1), 0)
    for i in range(1, 5):
        fa = flow[:, :, :, :2]; fb = flow[:, :, :, 2:4]
        cat = tf.concat([fwarp(img0, fa), fwarp(img1, fb), fwarp(d0, fa), fwarp(d1, fb),
                         ts, mask, feat, flow], -1)
        flow, mask, feat = fblock(cat, i)
    fa = flow[:, :, :, :2]; fb = flow[:, :, :, 2:4]
    return fwarp(img0, fa) * mask + fwarp(img1, fb) * (1 - mask)


@tf.function(input_signature=[tf.TensorSpec([1, H, W, 7], tf.float32)])
def fnet_stage0(x):
    img0 = x[:, :, :, :3]; img1 = x[:, :, :, 3:6]; ts = x[:, :, :, 6:7]
    d0 = deconv2d(conv2d(conv2d(conv2d(img0, "encode.cnn0", 2), "encode.cnn1", 1), "encode.cnn2", 1), "encode.cnn3")
    d1 = deconv2d(conv2d(conv2d(conv2d(img1, "encode.cnn0", 2), "encode.cnn1", 1), "encode.cnn2", 1), "encode.cnn3")
    flow, mask, feat = fblock(tf.concat([img0, img1, d0, d1, ts], -1), 0)
    return d0, d1, flow, mask, feat


# ---------------- гейт ----------------
def md(a, b): return float(np.max(np.abs(a - b)))


xr = np.random.RandomState(0).rand(1, H, W, 7).astype(np.float32)
xt = torch.tensor(xr).permute(0, 3, 1, 2)
tnet = TNet().eval(); tnet.debug = True
with torch.no_grad():
    ref = tnet(xt)
    dbg = {k: v.permute(0, 2, 3, 1).numpy() for k, v in tnet.dbg.items()}
sd0 = fnet_stage0(tf.constant(xr))
print("STAGE diff d0=%.6f d1=%.6f flow0=%.6f mask0=%.6f feat0=%.6f"
      % (md(sd0[0].numpy(), dbg["d0"]), md(sd0[1].numpy(), dbg["d1"]),
         md(sd0[2].numpy(), dbg["f0"]), md(sd0[3].numpy(), dbg["m0"]),
         md(sd0[4].numpy(), dbg["fe0"])))
got = fnet(tf.constant(xr)).numpy()
diff = md(got, ref.permute(0, 2, 3, 1).numpy())
print("GATE max|diff| = %.6f" % diff)
assert diff < 1e-3, "torch/tf mismatch too large"

conv = tf.lite.TFLiteConverter.from_concrete_functions(
    [fnet.get_concrete_function()], fnet)
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
conv.allow_custom_ops = False
tfl = conv.convert()
open("rife_v426.tflite", "wb").write(tfl)
print("wrote rife_v426.tflite %d B" % os.path.getsize("rife_v426.tflite"))
