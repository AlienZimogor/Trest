#!/usr/bin/env python3
"""RIFE v4.26 -> LiteRT (.tflite) через АВТОРСКИЙ IFNet_HDv3.py + onnx2tf.
Сначала синтезируем пакет model/ (warplayer/refine/loss/laplacian), ПОТОМ импорт.
Гейт: tflite vs torch(авторский net) на одном входе, max|diff| < 1e-3."""
import os, sys, glob, shutil, subprocess
import numpy as np
import torch
import tensorflow as tf

H, W = 384, 512

ROOT = os.path.abspath("model_src")
srcdir = None
for d in [ROOT] + [p for p in glob.glob(os.path.join(ROOT, "**"), recursive=True) if os.path.isdir(p)]:
    if os.path.exists(os.path.join(d, "IFNet_HDv3.py")):
        srcdir = os.path.abspath(d)
        break
assert srcdir, "IFNet_HDv3.py not found under model_src"

# ---- синтез пакета model/ ДО импорта IFNet_HDv3 ----
pkg = os.path.join(srcdir, "model")
os.makedirs(pkg, exist_ok=True)
open(os.path.join(pkg, "__init__.py"), "w").close()

WARP_SRC = '''
import torch
import torch.nn.functional as F

backwarp_tenGrid = {}

def warp(tenInput, tenFlow):
    k = (str(tenFlow.device), str(tenFlow.size()))
    if k not in backwarp_tenGrid:
        tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=tenFlow.device).view(1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=tenFlow.device).view(1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
        backwarp_tenGrid[k] = torch.cat([tenHorizontal, tenVertical], 1).to(tenFlow.device)
    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
                         tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)
    return F.grid_sample(input=tenInput, grid=(backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1),
                         mode='bilinear', padding_mode='zeros', align_corners=True)
'''
open(os.path.join(pkg, "warplayer.py"), "w").write(WARP_SRC)
for name in ("loss.py", "laplacian.py"):
    p = os.path.join(pkg, name)
    if not os.path.exists(p):
        open(p, "w").write("# stub\n")
ref_src = os.path.join(srcdir, "refine.py")
ref_dst = os.path.join(pkg, "refine.py")
if os.path.exists(ref_src) and not os.path.exists(ref_dst):
    shutil.copy(ref_src, ref_dst)

sys.path.insert(0, srcdir)
import IFNet_HDv3

# ---- веса ----
pkls = glob.glob(os.path.join(srcdir, "**", "*.pkl"), recursive=True)
assert pkls, "pkl not found"
raw = torch.load(pkls[0], map_location="cpu", weights_only=False)
if isinstance(raw, dict) and "state_dict" in raw:
    raw = raw["state_dict"]
sd = {}
for k, v in raw.items():
    sd[k[7:] if k.startswith("module.") else k] = v
print("pkl keys:", len(sd))

net = IFNet_HDv3.IFNet()
missing, unexpected = net.load_state_dict(sd, strict=False)
missing, unexpected = list(missing), list(unexpected)
print("missing:", missing[:10])
print("unexpected:", unexpected[:10])
assert not missing, "state_dict missing keys -> wrong architecture"
net.eval()

x = torch.rand(1, 7, H, W)
with torch.no_grad():
    ref = net(x)
print("torch ref mean=%.4f min=%.4f max=%.4f" % (ref.mean().item(), ref.min().item(), ref.max().item()))

torch.onnx.export(net, x, "rife_v426.onnx", opset_version=16,
                  input_names=["in0"], output_names=["out0"], do_constant_folding=True)
print("wrote rife_v426.onnx", os.path.getsize("rife_v426.onnx"))

subprocess.run(["onnx2tf", "-i", "rife_v426.onnx", "-o", "tfl_out", "-b", "1"], check=True)
cand = sorted(glob.glob("tfl_out/*_float32.tflite"))
assert cand, "no float32 tflite produced"
f32 = cand[0]
print("onnx2tf produced:", f32)

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
