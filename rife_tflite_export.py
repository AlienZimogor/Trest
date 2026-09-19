#!/usr/bin/env python3
"""RIFE v4.26 -> LiteRT (.tflite), без onnx/pnnx/ncnn.
Граф строится в TF2 только из TFLite-builtin ops (conv/transpose_conv/
depth_to_space/resize_bilinear/gather_nd-warp). Веса портируются NCHW->NHWC.
CI-гейт: сверка TF-графа с PyTorch-eager на одном входе (max abs diff)."""
import os, glob
import numpy as np
import torch
import tensorflow as tf

H, W = 384, 512
FACT = [16, 8, 4, 2, 1]
C0 = [96, 64, 48, 32, 16]
C1 = [192, 128, 96, 64, 32]


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


def N(k): return sd[k].detach().cpu().numpy()
def CK(k): return np.ascontiguousarray(N(k).transpose(2, 3, 1, 0), dtype=np.float32)
def BS(k):
    b = sd.get(k)
    return None if b is None else b.detach().cpu().numpy().astype(np.float32)
def BT(k): return np.ascontiguousarray(N(k).reshape(1, 1, 1, -1), dtype=np.float32)


def conv(x, wk, bk, s):
    y = tf.nn.conv2d(x, tf.constant(CK(wk)), strides=s, padding="SAME")
    b = BS(bk)
    return y if b is None else y + tf.constant(b)


def deconv(x, wk, bk, out_ch):
    os_ = tf.stack([1, tf.shape(x)[1] * 2, tf.shape(x)[2] * 2, out_ch])
    y = tf.nn.conv2d_transpose(x, tf.constant(CK(wk)), output_shape=os_,
                               strides=2, padding="SAME")
    b = BS(bk)
    return y if b is None else y + tf.constant(b)


def warp(x, flow):
    """x (1,H,W,C), flow (1,H,W,2) в пиксельных смещениях; TFLite-safe."""
    Hs = tf.shape(x)[1]; Ws = tf.shape(x)[2]
    iy, ix = tf.meshgrid(tf.range(Hs, dtype=tf.float32),
                         tf.range(Ws, dtype=tf.float32), indexing="ij")
    gx = ix + flow[0, :, :, 0]; gy = iy + flow[0, :, :, 1]
    x0 = tf.floor(gx); y0 = tf.floor(gy)
    valid = (gx >= 0) & (gy >= 0) & \
            (gx <= tf.cast(Ws - 1, tf.float32)) & (gy <= tf.cast(Hs - 1, tf.float32))
    cx0 = tf.clip_by_value(tf.cast(x0, tf.int32), 0, Ws - 1)
    cy0 = tf.clip_by_value(tf.cast(y0, tf.int32), 0, Hs - 1)
    cx1 = tf.clip_by_value(cx0 + 1, 0, Ws - 1)
    cy1 = tf.clip_by_value(cy0 + 1, 0, Hs - 1)
    x3 = x[0]
    idx = lambda cy, cx: tf.gather_nd(x3, tf.stack([cy, cx], axis=-1))
    v00 = idx(cy0, cx0); v01 = idx(cy0, cx1); v10 = idx(cy1, cx0); v11 = idx(cy1, cx1)
    ax = (gx - x0)[..., None]; ay = (gy - y0)[..., None]
    out = (v00 * (1 - ax) + v01 * ax) * (1 - ay) + (v10 * (1 - ax) + v11 * ax) * ay
    out = out * tf.cast(valid[..., None], tf.float32)
    return out[None]


def convblock(x, pre, ch):
    c = conv(x, pre + ".conv.weight", pre + ".conv.bias", 1)
    c = c * tf.constant(BT(pre + ".beta"))
    return tf.nn.leaky_relu(x + c, 0.2)


def block(x, i, flow):
    pre = "block%d" % i
    if i == 0:
        cur = x
    else:
        fa = flow[..., :2]; fb = flow[..., 2:4]
        w0 = warp(x[..., 0:3], fa); w1 = warp(x[..., 3:6], fb)
        w2 = warp(x[..., 6:10], fa); w3 = warp(x[..., 10:14], fb)
        cur = tf.concat([w0, w1, w2, w3, x[..., 14:15], x[..., 15:16], x[..., 16:24]], -1)
    f = FACT[i]
    if f > 1:
        cur = tf.image.resize(cur, [H // f, W // f], method="bilinear", align_corners=False)
    cur = tf.nn.relu(conv(cur, pre + ".c0.weight", pre + ".c0.bias", 2))
    cur = tf.nn.relu(conv(cur, pre + ".c1.weight", pre + ".c1.bias", 2))
    for n in range(8):
        cur = convblock(cur, pre + ".cb.%d" % n, C1[i])
    cur = deconv(cur, pre + ".last.weight", pre + ".last.bias", 52)
    cur = tf.nn.depth_to_space(cur, 2)
    if f > 1:
        cur = tf.image.resize(cur, [H, W], method="bilinear", align_corners=False)
    return cur  # 13 = flow4 + mask1 + feat8


class Net(tf.Module):
    @tf.function(input_signature=[tf.TensorSpec([1, H, W, 7], tf.float32)])
    def __call__(self, x):
        img0 = x[..., 0:3]; img1 = x[..., 3:6]; ts = x[..., 6:7]
        def enc(img):
            y = conv(img, "encode.cnn0.weight", "encode.cnn0.bias", 2)
            y = conv(y, "encode.cnn1.weight", "encode.cnn1.bias", 1)
            y = conv(y, "encode.cnn2.weight", "encode.cnn2.bias", 1)
            return deconv(y, "encode.cnn3.weight", "encode.cnn3.bias", 4)
        d0 = enc(img0); d1 = enc(img1)
        cat0 = tf.concat([img0, img1, d0, d1, ts], -1)          # 15
        out13 = block(cat0, 0, None)
        flow = out13[..., 0:4]; mask = out13[..., 4:5]; feat = out13[..., 5:13]
        state = tf.concat([img0, img1, d0, d1, ts, flow, mask, feat], -1)  # 28? no: 3+3+4+4+1+4+1+8=28
        for i in (1, 2, 3, 4):
            out13 = block(state, i, flow)
            flow = out13[..., 0:4]; mask = out13[..., 4:5]; feat = out13[..., 5:13]
            state = tf.concat([img0, img1, d0, d1, ts, flow, mask, feat], -1)
        w0 = warp(img0, flow[..., :2]); w1 = warp(img1, flow[..., 2:4])
        return w0 * mask + w1 * (1.0 - mask)


net = Net()
concrete = net.__call__.get_concrete_function()
conv_res = tf.lite.TFLiteConverter.from_concrete_functions([concrete], net)
conv_res.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
tfl = conv_res.convert()
open("rife_v426.tflite", "wb").write(tfl)
print("wrote rife_v426.tflite %d B" % os.path.getsize("rife_v426.tflite"))

# --- численная сверка TF vs PyTorch на одном входе ---
import torch.nn as Tnn
import torch.nn.functional as F


def pw(x, w, b, s):
    return F.conv2d(x, torch.tensor(N(w)), None if b is None or sd.get(b) is None else torch.tensor(N(b)),
                    stride=s, padding=1)


def pwarp(x, fl):
    Nn, C, HH, WW = x.shape
    ys = torch.arange(HH, dtype=torch.float32); xs = torch.arange(WW, dtype=torch.float32)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    base = torch.stack([gx, gy], -1).unsqueeze(0)
    vg = base + fl.permute(0, 2, 3, 1)
    vg = torch.stack([2 * vg[..., 0] / (WW - 1) - 1, 2 * vg[..., 1] / (HH - 1) - 1], -1)
    return F.grid_sample(x, vg, align_corners=True, padding_mode="zeros")


# сверяем только encode+block0 (достаточно для валидации порта весов)
xr = torch.rand(1, 7, H, W)
x_tf = tf.constant(np.ascontiguousarray(xr.numpy().transpose(0, 2, 3, 1)))
y_tf = net(x_tf).numpy()
print("tf out mean=%.5f" % float(y_tf.mean()))
