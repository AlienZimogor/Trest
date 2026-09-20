#!/usr/bin/env python3
"""RIFE v4.26 -> LiteRT (.tflite), NHWC [1,384,512,7] -> [1,384,512,3].
Warp = gather_nd-билинейка (TFLite-builtin), без grid_sample/Flex.
CI-гейт: постадийный max|diff| torch vs tf (block0..block4 + отдельный warp),
конвертация только при финальном diff < 1e-3."""
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
        s.dbg = {}

    def enc(s, img):
        return s.ed(s.e2(s.e1(s.e0(img))))

    def forward(s, x):
        img0 = x[:, :3]; img1 = x[:, 3:6]; ts = x[:, 6:7]
        d0 = s.enc(img0); d1 = s.enc(img1)
        flow, mask, feat = s.b[0](torch.cat([img0, img1, d0, d1, ts], 1))
        s.dbg["b0"] = (flow, mask, feat)
        for i in range(1, 5):
            fa = flow[:, :2]; fb = flow[:, 2:4]
            if i == 1:
                s.dbg["w0"] = twarp(img0, fa)
            cat = torch.cat([twarp(img0, fa), twarp(img1, fb), twarp(d0, fa), twarp(d1, fb),
                             ts, mask, feat, flow], 1)
            flow, mask, feat = s.b[i](cat)
            s.dbg["b%d" % i] = (flow, mask, feat)
        fa = flow[:, :2]; fb = flow[:, 2:4]
        return twarp(img0, fa) * mask + twarp(img1, fb) * (1 - mask)


# ---------------- TF-зеркало ----------------
CK = {}; BK = {}; DK = {}; BV = {}
for k in (["encode.cnn0", "encode.cnn1", "encode.cnn2"]
          + ["block%d.conv0.0.0" % i for i in range(5)]
          + ["block%d.conv0.1.0" % i for i in range(5)]
          + ["block%d.convblock.%d.conv" % (i, n) for i in range(5) for n in range(8)]):
    CK[k] = tf.constant(npw(k + ".weight").transpose(2, 3, 1, 0).astype(np.float32))
    b = npb(k)
    BK[k] = None if b is None else tf.constant(b)
for k in ["encode.cnn3"] + ["block%d.lastconv.0" % i for i in range(5)]:
    DK[k] = tf.constant(npw(k + ".weight").transpose(2, 3, 1, 0).astype(np.float32))
    b = npb(k)
    BK[k] = None if b is None else tf.constant(b)
for i in range(5):
    for n in range(8):
        BV["block%d.convblock.%d.conv" % (i, n)] = tf.constant(
            npw("block%d.convblock.%d.beta" % (i, n)).reshape(-1).astype(np.float32))


def conv2d(x, key, stride):
    if stride == 1:
        y = tf.nn.conv2d(x, CK[key], strides=1, padding="SAME")
    else:
        xp = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
        y = tf.nn.conv2d(xp, CK[key], strides=stride, padding="VALID")
    b = BK[key]
    return y if b is None else y + b


def deconv2d(x, key):
    w = DK[key]
    s = tf.shape(x)
    y = tf.nn.conv2d_transpose(x, w,
                               output_shape=[s[0], s[1] * 2 + 2, s[2] * 2 + 2, w.shape[2]],
                               strides=2, padding="VALID")
    y = y[:, 1:-1, 1:-1, :]
    b = BK[key]
    return y if b is None else y + b


def resize_tf(x, size):
    return tf.image.resize(x, size, method="bilinear")


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
        x = resize_tf(x, [H // f, W // f])
    y = tf.nn.relu(conv2d(x, "block%d.conv0.0.0" % i, 2))
    y = tf.nn.relu(conv2d(y, "block%d.conv0.1.0" % i, 2))
    for n in range(8):
        cp = "block%d.convblock.%d.conv" % (i, n)
        y = tf.nn.leaky_relu(y + conv2d(y, cp, 1) * BV[cp], 0.2)
    y = tf.nn.depth_to_space(deconv2d(y, "block%d.lastconv.0" % i), 2)
    if f > 1:
        y = resize_tf(y, [H, W])
    return y[..., :4], tf.sigmoid(y[..., 4:5]), y[..., 5:13]


@tf.function(input_signature=[tf.TensorSpec([1, H, W, 7], tf.float32)])
def fnet(x):
    img0 = x[..., :3]; img1 = x[..., 3:6]; ts = x[..., 6:7]
    d0 = deconv2d(conv2d(conv2d(conv2d(img0, "encode.cnn0", 2), "encode.cnn1", 1), "encode.cnn2", 1), "encode.cnn3")
    d1 = deconv2d(conv2d(conv2d(conv2d(img1, "encode.cnn0", 2), "encode.cnn1", 1), "encode.cnn2", 1), "encode.cnn3")
    flow, mask, feat = fblock(tf.concat([img0, img1, d0, d1, ts], -1), 0)
    for i in range(1, 5):
        fa = flow[..., :2]; fb = flow[..., 2:4]
        cat = tf.concat([fwarp(img0, fa), fwarp(img1, fb), fwarp(d0, fa), fwarp(d1, fb),
                         ts, mask, feat, flow], -1)
        flow, mask, feat = fblock(cat, i)
    fa = flow[..., :2]; fb = flow[..., 2:4]
    return fwarp(img0, fa) * mask + fwarp(img1, fb) * (1 - mask)


# ---------------- гейт с локализацией ----------------
def md(a, b): return float(np.max(np.abs(a - b)))


xr = np.random.RandomState(0).rand(1, H, W, 7).astype(np.float32)
xt = torch.tensor(xr).permute(0, 3, 1, 2)
tnet = TNet().eval()
with torch.no_grad():
    ref = tnet(xt)
    dbg = {k: tuple(t.permute(0, 2, 3, 1).numpy() for t in v) if isinstance(v, tuple)
           else v.permute(0, 2, 3, 1).numpy() for k, v in tnet.dbg.items()}

xi = tf.constant(xr)
img0 = xi[..., :3]; img1 = xi[..., 3:6]; ts = xi[..., 6:7]
d0 = deconv2d(conv2d(conv2d(conv2d(img0, "encode.cnn0", 2), "encode.cnn1", 1), "encode.cnn2", 1), "encode.cnn3")
d1 = deconv2d(conv2d(conv2d(conv2d(img1, "encode.cnn0", 2), "encode.cnn1", 1), "encode.cnn2", 1), "encode.cnn3")
tflow, tmask, tfeat = fblock(tf.concat([img0, img1, d0, d1, ts], -1), 0)
print("STAGE b0 flow=%.6f mask=%.6f feat=%.6f"
      % (md(tflow.numpy(), dbg["b0"][0]), md(tmask.numpy(), dbg["b0"][1]), md(tfeat.numpy(), dbg["b0"][2])))
print("WARP1 diff=%.6f" % md(fwarp(img0, tflow[..., :2]).numpy(), dbg["w0"]))
for i in range(1, 5):
    fa = tflow[..., :2]; fb = tflow[..., 2:4]
    cat = tf.concat([fwarp(img0, fa), fwarp(img1, fb), fwarp(d0, fa), fwarp(d1, fb),
                     ts, tmask, tfeat, tflow], -1)
    tflow, tmask, tfeat = fblock(cat, i)
    g = dbg["b%d" % i]
    print("STAGE b%d flow=%.6f mask=%.6f feat=%.6f"
          % (i, md(tflow.numpy(), g[0]), md(tmask.numpy(), g[1]), md(tfeat.numpy(), g[2])))

got = fnet(xi).numpy()
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
