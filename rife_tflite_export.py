#!/usr/bin/env python3
"""RIFE v4.26 -> LiteRT (.tflite) через ONNX(opset16) + onnxsim + onnx2tf.
onnxsim сворачивает Expand/meshgrid (иначе onnx2tf падает на Expand).
Если onnx2tf сгенерировал auto.json — перезапускаем с -prf.
Числовой гейт torch vs tflite остаётся финальным арбитром."""
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
        s.b = nn.ModuleList([TBlock(i, [16, 8, 4, 2, 1][i]) for i in range(5)])

    def enc(s, img):
        return s.ed(s.e2(s.e1(s.e0(img))))

    def forward(s, x):
        img0 = x[:, :3]; img1 = x[:, 3:6]; ts = x[:, 6:7]
        d0 = s.enc(img0); d1 = s.enc(img1)
        flow, mask, feat = s.b[0](torch.cat([img0, img1, d0, d1, ts], 1))
        for i in range(1, 5):
            fa = flow[:, :2]; fb = flow[:, 2:4]
            cat = torch.cat([twarp(img0, fa), twarp(img1, fb), twarp(d0, fa), twarp(d1, fb),
                             ts, mask, feat, flow], 1)
            flow, mask, feat = s.b[i](cat)
        fa = flow[:, :2]; fb = flow[:, 2:4]
        return twarp(img0, fa) * mask + twarp(img1, fb) * (1 - mask)


net = TNet().eval()

# ---- 1) torch -> ONNX (opset 16, нативный GridSample) ----
xr_t = torch.tensor(np.random.RandomState(0).rand(1, 7, H, W).astype(np.float32))
with torch.no_grad():
    ref = net(xr_t).numpy()
torch.onnx.export(net, xr_t, "rife_v426.onnx", opset_version=16,
                  input_names=["in0"], output_names=["out0"],
                  do_constant_folding=True)
print("wrote rife_v426.onnx %d B" % os.path.getsize("rife_v426.onnx"))

# ---- 2) onnxsim: сворачиваем Expand/meshgrid/broadcasting ----
# Без этого onnx2tf падает на wa/Expand с Keras-Functional ValueError.
try:
    import onnxsim
    simplified, ok = onnxsim.simplify("rife_v426.onnx")
    if ok:
        with open("rife_v426_sim.onnx", "wb") as f:
            f.write(simplified.SerializeToString())
        onnx_src = "rife_v426_sim.onnx"
        print("onnxsim ok: %d B -> %d B" %
              (os.path.getsize("rife_v426.onnx"), os.path.getsize(onnx_src)))
    else:
        onnx_src = "rife_v426.onnx"
        print("onnxsim не подтвердил, используем оригинал")
except Exception as e:
    onnx_src = "rife_v426.onnx"
    print("onnxsim exception (%s), используем оригинал" % e)

# ---- 3) onnx2tf -> TFLite (с fallback на auto-generated JSON) ----
if os.path.isdir("tfl_out"): shutil.rmtree("tfl_out")
cmd1 = ["onnx2tf", "-i", onnx_src, "-o", "tfl_out", "-b", "1",
        "-ois", "in0:1,7,384,512"]
r1 = subprocess.run(cmd1)
auto_json = "tfl_out/rife_v426_auto.json"
if r1.returncode != 0 and os.path.isfile(auto_json):
    print("onnx2tf pass 1 failed, retrying with -prf auto.json")
    cmd2 = cmd1 + ["-prf", auto_json]
    subprocess.run(cmd2, check=True)

cand = [f for f in os.listdir("tfl_out") if f.endswith(".tflite") and "float32" in f] \
    or [f for f in os.listdir("tfl_out") if f.endswith(".tflite")]
assert cand, "no tflite produced"
src = os.path.join("tfl_out", cand[0])
print("onnx2tf produced:", src)

# ---- 4) числовой гейт torch vs tflite ----
xr = xr_t.numpy().transpose(0, 2, 3, 1)  # NHWC
interp = tf.lite.Interpreter(model_path=src)
interp.allocate_tensors()
din = interp.get_input_details()[0]
dout = interp.get_output_details()[0]
print("tflite input :", din["shape"], din["dtype"])
print("tflite output:", dout["shape"], dout["dtype"])
interp.set_tensor(din["index"], xr.astype(din["dtype"]))
interp.invoke()
got = interp.get_tensor(dout["index"]).astype(np.float32)
if got.ndim == 4 and got.shape[-1] != 3:
    got = got.transpose(0, 2, 3, 1)
ref_nhwc = ref.transpose(0, 2, 3, 1)
diff = float(np.max(np.abs(got - ref_nhwc)))
print("GATE max|diff| = %.6f" % diff)
assert diff < 1e-3, "torch/tflite mismatch too large"

shutil.copy(src, "rife_v426.tflite")
f16 = "tfl_out/rife_v426_sim_float16.tflite"
if os.path.exists(f16):
    shutil.copy(f16, "rife_v426_f16.tflite")
    print("wrote rife_v426_f16.tflite %d B" % os.path.getsize("rife_v426_f16.tflite"))
print("wrote rife_v426.tflite %d B" % os.path.getsize("rife_v426.tflite"))
