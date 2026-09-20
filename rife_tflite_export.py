#!/usr/bin/env python3
"""RIFE v4.26 -> ONNX(opset16) -> onnx2tf -> TFLite, гейт torch-vs-tflite.
Используем ОФИЦИАЛЬНЫЙ IFNet_HDv3.py из zip, но подменяем его warp (в zip-warplayer
нет permute перед grid_sample) на корректный monkey-patch."""
import os, sys, glob, shutil, subprocess
import numpy as np
import torch
import torch.nn.functional as F

H, W = 384, 512

ROOT = os.path.abspath("model_src")
srcdir = None
for d in [ROOT] + [p for p in glob.glob(os.path.join(ROOT, "**"), recursive=True) if os.path.isdir(p)]:
    if os.path.exists(os.path.join(d, "IFNet_HDv3.py")):
        srcdir = d
        break
assert srcdir, "IFNet_HDv3.py not found under model_src"
sys.path.insert(0, srcdir)

pkls = glob.glob(os.path.join(srcdir, "**", "*.pkl"), recursive=True)
assert pkls, "pkl not found"
raw = torch.load(pkls[0], map_location="cpu", weights_only=False)
if isinstance(raw, dict) and "state_dict" in raw:
    raw = raw["state_dict"]
sd = {}
for k, v in raw.items():
    sd[k[7:] if k.startswith("module.") else k] = v
print("pkl keys:", len(sd))

import IFNet_HDv3
import model.warplayer as _wp

def _warp(x, flow):
    """Корректный warp (NCHW in -> NCHW out): grid + flow, permute перед grid_sample."""
    N, C, Hh, Ww = x.size()
    xs = torch.linspace(-1.0, 1.0, Ww, device=x.device, dtype=x.dtype).view(1, 1, 1, Ww).expand(N, -1, Hh, -1)
    ys = torch.linspace(-1.0, 1.0, Hh, device=x.device, dtype=x.dtype).view(1, 1, Hh, 1).expand(N, -1, -1, Ww)
    grid = torch.cat([xs, ys], 1)
    f = torch.cat([flow[:, 0:1, :, :] / ((Ww - 1.0) / 2.0),
                   flow[:, 1:2, :, :] / ((Hh - 1.0) / 2.0)], 1)
    return F.grid_sample(x, (grid + f).permute(0, 2, 3, 1),
                         mode="bilinear", padding_mode="zeros", align_corners=True)

_wp.warp = _warp
IFNet_HDv3.warp = _warp

net = IFNet_HDv3.IFNet()
missing, unexpected = net.load_state_dict(sd, strict=False)
print("missing:", list(missing)[:10])
print("unexpected:", list(unexpected)[:10])
assert not missing, "state_dict missing keys -> architecture mismatch"
net.eval()

x = torch.rand(1, 7, H, W)
with torch.no_grad():
    ref = net(x)
print("torch ref mean=%.4f min=%.4f max=%.4f" % (ref.mean().item(), ref.min().item(), ref.max().item()))

torch.onnx.export(net, x, "rife_v426.onnx", opset_version=16,
                  input_names=["in0"], output_names=["out0"], do_constant_folding=True)
print("wrote rife_v426.onnx", os.path.getsize("rife_v426.onnx"))

onnx_in = "rife_v426.onnx"
try:
    import onnx
    from onnxsim import simplify
    m = onnx.load(onnx_in)
    ms, ok = simplify(m)
    if ok:
        onnx.save(ms, "rife_v426_sim.onnx")
        onnx_in = "rife_v426_sim.onnx"
        print("onnxsim ok ->", onnx_in)
except Exception as e:
    print("onnxsim skip:", e)

subprocess.run(["onnx2tf", "-i", onnx_in, "-o", "tfl_out", "-b", "1"], check=True)
cand = sorted(glob.glob("tfl_out/*_float32.tflite"))
assert cand, "no float32 tflite produced"
f32 = cand[0]
print("onnx2tf produced:", f32)

import tensorflow as tf
interp = tf.lite.Interpreter(model_path=f32)
interp.allocate_tensors()
di = interp.get_input_details()[0]; do = interp.get_output_details()[0]
print("tflite input :", di["shape"], di["dtype"])
print("tflite output:", do["shape"], do["dtype"])
xin = x.numpy().transpose(0, 2, 3, 1).astype(np.float32)
interp.set_tensor(di["index"], xin)
interp.invoke()
got = interp.get_tensor(do["index"]).astype(np.float32)
refn = ref.numpy().transpose(0, 2, 3, 1)
diff = float(np.abs(got - refn).max())
print("GATE max|diff| = %.6f" % diff)
assert diff < 1e-3, "torch/tflite mismatch too large"
shutil.copy(f32, "rife_v426.tflite")
print("wrote rife_v426.tflite", os.path.getsize("rife_v426.tflite"))
