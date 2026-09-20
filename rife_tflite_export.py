#!/usr/bin/env python3
"""RIFE v4.26 -> ONNX(opset16) -> onnx2tf -> TFLite.
Эталон = АВТОРСКИЙ IFNet_HDv3.py из release-zip (не реконструкция!).
Недостающий model/warplayer.py синтезируем (стандартный warp).
Гейты: (a) tflite == torch-эталон; (b) анти-дегенерат: mid != 0.5*(A+B)."""
import os, sys, glob, shutil, subprocess
import numpy as np
import torch
import tensorflow as tf

H, W = 384, 512

# ---------- 1) авторская архитектура из release-zip ----------
ROOT = os.path.abspath("model_src")
src = None
for d in [ROOT] + [p for p in glob.glob(os.path.join(ROOT, "**"), recursive=True) if os.path.isdir(p)]:
    if os.path.exists(os.path.join(d, "IFNet_HDv3.py")):
        src = os.path.abspath(d); break
assert src, "IFNet_HDv3.py not found under model_src"

pkg = os.path.join(src, "model")
os.makedirs(pkg, exist_ok=True)
open(os.path.join(pkg, "__init__.py"), "w").close()
WARP = (
    "import torch\n"
    "import torch.nn.functional as F\n"
    "\n"
    "def warp(x, flow):\n"
    "    N, C, H, W = x.shape\n"
    "    xx = torch.arange(0, W, device=x.device, dtype=torch.float32).view(1, -1).repeat(H, 1)\n"
    "    yy = torch.arange(0, H, device=x.device, dtype=torch.float32).view(-1, 1).repeat(1, W)\n"
    "    xx = xx.view(1, 1, H, W).repeat(N, 1, 1, 1)\n"
    "    yy = yy.view(1, 1, H, W).repeat(N, 1, 1, 1)\n"
    "    grid = torch.cat((xx, yy), 1)\n"
    "    vgrid = grid + flow\n"
    "    vgrid[:, 0] = 2.0 * vgrid[:, 0] / max(W - 1, 1) - 1.0\n"
    "    vgrid[:, 1] = 2.0 * vgrid[:, 1] / max(H - 1, 1) - 1.0\n"
    "    return F.grid_sample(x, vgrid, align_corners=True, padding_mode='zeros')\n"
)
if not os.path.exists(os.path.join(pkg, "warplayer.py")):
    open(os.path.join(pkg, "warplayer.py"), "w").write(WARP)
for fname in ["refine.py"]:
    a = os.path.join(src, fname); b = os.path.join(pkg, fname)
    if os.path.exists(a) and not os.path.exists(b):
        shutil.copy(a, b)
sys.path.insert(0, src)
import IFNet_HDv3

# ---------- 2) веса ----------
pkls = glob.glob(os.path.join(src, "**", "*.pkl"), recursive=True)
assert pkls, "pkl not found"
raw = torch.load(pkls[0], map_location="cpu", weights_only=False)
if isinstance(raw, dict) and "state_dict" in raw: raw = raw["state_dict"]
sd = {}
for k, v in raw.items():
    sd[k[7:] if k.startswith("module.") else k] = v
print("pkl keys:", len(sd))
net = IFNet_HDv3.IFNet()
missing, unexpected = net.load_state_dict(sd, strict=False)
missing, unexpected = list(missing), list(unexpected)
print("missing:", missing[:20])
print("unexpected:", unexpected[:20])
assert not missing, "state_dict missing keys -> architecture mismatch"
net.eval()

# ---------- 3) эталон + анти-дегенерат гейт ----------
def square_frames():
    ys = torch.linspace(0, 1, H).view(1, 1, H, 1)
    xs = torch.linspace(0, 1, W).view(1, 1, 1, W)
    bg = torch.cat([xs.expand(1, 1, H, W), ys.expand(1, 1, H, W),
                      torch.full((1, 1, H, W), 0.3)], 1)
    a = bg.clone(); b = bg.clone()
    s = W // 8; x0 = int(0.30 * W); x1 = int(0.62 * W); y0 = H // 2 - s // 2
    a[:, :, y0:y0 + s, x0:x0 + s] = 1.0
    b[:, :, y0:y0 + s, x1:x1 + s] = 1.0
    return a, b

with torch.no_grad():
    x = torch.rand(1, 7, H, W)
    ref = net(x)
    a, b = square_frames()
    t = torch.full((1, 1, H, W), 0.5)
    mid = net(torch.cat([a, b, t], 1))
    motion = (mid - 0.5 * (a + b)).abs().max().item()
print("MOTION torch max|mid - crossdissolve| = %.4f" % motion)
assert motion > 0.1, "DEGENERATE torch reference: flow~0 (cross-dissolve)"

# ---------- 4) ONNX -> onnx2tf ----------
torch.onnx.export(net, x, "rife_v426.onnx", opset_version=16,
                  input_names=["in0"], output_names=["out0"],
                  do_constant_folding=True)
print("wrote rife_v426.onnx", os.path.getsize("rife_v426.onnx"))
subprocess.run(["onnx2tf", "-i", "rife_v426.onnx", "-o", "tfl_out", "-b", "1"], check=True)
cand = sorted(glob.glob("tfl_out/*_float32.tflite"))
assert cand, "no float32 tflite produced"
f32 = cand[0]
print("onnx2tf produced:", f32)

# ---------- 5) гейты tflite ----------
def tflite_run(path, arr_nhwc):
    it = tf.lite.Interpreter(model_path=path)
    it.allocate_tensors()
    di = it.get_input_details()[0]; do = it.get_output_details()[0]
    it.set_tensor(di["index"], arr_nhwc)
    it.invoke()
    return it.get_tensor(do["index"])

got = tflite_run(f32, x.numpy().transpose(0, 2, 3, 1).astype(np.float32))
diff = float(np.abs(got - ref.numpy().transpose(0, 2, 3, 1)).max())
print("GATE max|diff| = %.6f" % diff)
assert diff < 1e-3, "torch/tflite mismatch"

sq = torch.cat([a, b, t], 1).numpy().transpose(0, 2, 3, 1).astype(np.float32)
diss = (0.5 * (a + b)).numpy().transpose(0, 2, 3, 1)
m32 = tflite_run(f32, sq)
motion32 = float(np.abs(m32 - diss).max())
print("MOTION tflite-f32 max|mid - crossdissolve| = %.4f" % motion32)
assert motion32 > 0.1, "DEGENERATE tflite: flow~0"
shutil.copy(f32, "rife_v426.tflite")
print("wrote rife_v426.tflite", os.path.getsize("rife_v426.tflite"))

f16s = sorted(glob.glob("tfl_out/*_float16.tflite"))
if f16s:
    ok = False
    try:
        m16 = tflite_run(f16s[0], sq)
        m16v = float(np.abs(m16 - diss).max())
        print("MOTION tflite-f16 = %.4f" % m16v)
        ok = m16v > 0.1
    except Exception as e:
        print("f16 invalid:", e)
    if ok:
        shutil.copy(f16s[0], "rife_v426_f16.tflite")
        print("wrote rife_v426_f16.tflite", os.path.getsize("rife_v426_f16.tflite"))
    else:
        print("SKIP f16 (invalid or degenerate)")
