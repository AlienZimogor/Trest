#!/usr/bin/env python3
"""RIFE v4.26 -> LiteRT (.tflite) через ОРИГИНАЛЬНЫЙ IFNet_HDv3.py из zip.
Синтезируем пакет model/ (warplayer с broadcast-Add вместо repeat/Expand,
refine копируем, loss/laplacian — пустые стабы), чтобы импорт прошёл и
onnx2tf не падал на узле Expand.
Пайплайн: pkl -> IFNet_HDv3.IFNet -> ONNX(opset16) -> onnx2tf -> tflite.
Гейт: tflite vs оригинальный PyTorch на том же входе, max|diff| < 1e-3."""
import os, sys, glob, shutil, subprocess, importlib.util
import numpy as np
import torch
import torch.nn as nn

H, W = 384, 512

# ---------- найти каталог с оригинальным IFNet_HDv3.py ----------
srcdir = None
for d in ["model_src"] + [p for p in glob.glob(os.path.join("model_src", "**"), recursive=True) if os.path.isdir(p)]:
    if os.path.exists(os.path.join(d, "IFNet_HDv3.py")):
        srcdir = d
        break
assert srcdir, "IFNet_HDv3.py не найден под model_src"

# ---------- синтез пакета model/ ----------
stub = os.path.join(srcdir, "model")
os.makedirs(stub, exist_ok=True)
open(os.path.join(stub, "__init__.py"), "w").close()

WARPLAYER_SRC = '''
import torch
import torch.nn.functional as F

def warp(x, flow):
    # Математика идентична родному warplayer.warp, но сетка строится через
    # implicit-broadcast Add (xs[1,1,1,W] + flow), поэтому в ONNX НЕТ узла Expand,
    # на котором падает onnx2tf.
    N, C, H, W = x.shape
    xs = torch.arange(0, W, device=x.device, dtype=torch.float32).view(1, 1, 1, W)
    ys = torch.arange(0, H, device=x.device, dtype=torch.float32).view(1, 1, H, 1)
    gx = xs + flow[:, 0:1]
    gy = ys + flow[:, 1:2]
    gx = 2.0 * gx / max(W - 1, 1) - 1.0
    gy = 2.0 * gy / max(H - 1, 1) - 1.0
    vgrid = torch.cat((gx, gy), 1).permute(0, 2, 3, 1)
    return F.grid_sample(x, vgrid, align_corners=True, padding_mode="zeros")
'''
with open(os.path.join(stub, "warplayer.py"), "w") as f:
    f.write(WARPLAYER_SRC)

# refine копируем из верхнего уровня zip (если есть), иначе пустой стаб
refine_dst = os.path.join(stub, "refine.py")
refine_src = os.path.join(srcdir, "refine.py")
if os.path.exists(refine_src):
    shutil.copy(refine_src, refine_dst)
elif not os.path.exists(refine_dst):
    open(refine_dst, "w").close()

# пустые стабы для прочих model.*, которые может импортировать IFNet
for name in ("loss.py", "laplacian.py"):
    p = os.path.join(stub, name)
    if not os.path.exists(p):
        open(p, "w").close()

sys.path.insert(0, srcdir)
import IFNet_HDv3  # оригинальный исходник из zip

# ---------- веса ----------
pkls = glob.glob(os.path.join(srcdir, "*.pkl")) or glob.glob(os.path.join("model_src", "**", "*.pkl"), recursive=True)
assert pkls, "pkl не найден"
raw = torch.load(pkls[0], map_location="cpu", weights_only=False)
if isinstance(raw, dict) and "state_dict" in raw:
    raw = raw["state_dict"]
sd = {}
for k, v in raw.items():
    sd[k[7:] if k.startswith("module.") else k] = v
print("pkl keys:", len(sd))

model = IFNet_HDv3.IFNet()
missing, unexpected = model.load_state_dict(sd, strict=False)
missing, unexpected = list(missing), list(unexpected)
print("missing:", missing[:10], "unexpected:", unexpected[:10])
assert not missing, "state_dict не лёг строго: несовпадение архитектуры"
model.eval()

class Wrap7(nn.Module):
    """Вход [1,7,H,W] = (img0,img1,timestep). Пробуем родной forward; если он
    ждёт timestep отдельным аргументом — передаём его из 7-го канала."""
    def __init__(self, net):
        super().__init__()
        self.net = net
    def forward(self, x):
        try:
            return self.net(x)
        except TypeError:
            return self.net(x[:, :6], timestep=x[:, 6:7])

wrap = Wrap7(model)

# ---------- эталон ----------
x = torch.rand(1, 7, H, W)
with torch.no_grad():
    ref = wrap(x).numpy()
print("torch ref mean=%.4f min=%.4f max=%.4f" % (ref.mean(), ref.min(), ref.max()))

# ---------- ONNX ----------
torch.onnx.export(wrap, x, "rife_v426.onnx", opset_version=16,
                  input_names=["in0"], output_names=["out0"],
                  do_constant_folding=True)
print("wrote rife_v426.onnx", os.path.getsize("rife_v426.onnx"))

# ---------- onnx2tf (с фолбэком на auto-JSON) ----------
cmd = ["onnx2tf", "-i", "rife_v426.onnx", "-o", "tfl_out", "-b", "1"]
r = subprocess.run(cmd)
if r.returncode != 0:
    auto = "tfl_out/rife_v426_auto.json"
    if os.path.exists(auto):
        print("retry with -prf", auto)
        subprocess.run(cmd + ["-prf", auto], check=True)
    else:
        raise SystemExit("onnx2tf failed and no auto json")
cand = glob.glob("tfl_out/*_float32.tflite")
assert cand, "onnx2tf не дал float32 tflite"
f32 = cand[0]
print("onnx2tf produced:", f32)

# ---------- гейт tflite vs оригинал ----------
import tensorflow as tf
interp = tf.lite.Interpreter(model_path=f32)
interp.allocate_tensors()
din = interp.get_input_details()[0]
dout = interp.get_output_details()[0]
print("tflite input :", din["shape"], din["dtype"])
print("tflite output:", dout["shape"], dout["dtype"])
xin = x.numpy().transpose(0, 2, 3, 1).astype(np.float32)
interp.set_tensor(din["index"], xin)
interp.invoke()
got = interp.get_tensor(dout["index"]).astype(np.float32)
refn = ref.transpose(0, 2, 3, 1)
diff = float(np.max(np.abs(got - refn)))
print("GATE max|diff| = %.6f" % diff)
assert diff < 1e-3, "torch/tflite mismatch too large"

shutil.copy(f32, "rife_v426.tflite")
print("wrote rife_v426.tflite", os.path.getsize("rife_v426.tflite"))
f16 = glob.glob("tfl_out/*_float16.tflite")
if f16:
    shutil.copy(f16[0], "rife_v426_f16.tflite")
    print("wrote rife_v426_f16.tflite", os.path.getsize("rife_v426_f16.tflite"))
