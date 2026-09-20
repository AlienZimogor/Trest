#!/usr/bin/env python3
"""RIFE v4.26 -> LiteRT (.tflite) через ОРИГИНАЛЬНЫЙ IFNet_HDv3.py из zip.
- Синтезируем пакет model/ (warplayer с broadcast-Add вместо repeat/Expand,
  refine копируем, loss/laplacian — пустые стабы), чтобы импорт прошёл и
  onnx2tf не падал на узле Expand.
- teacher.* ключи pkl игнорируем (strict=False): это distillation-teacher,
  которого в инференс-сети нет.
- forward вызываем через probe стилей вызова (scale_list=[16,8,4,2,1] и т.п.),
  чтобы не словить IndexError на scale_list[i].
Пайплайн: pkl -> IFNet_HDv3.IFNet -> ONNX(opset16) -> onnx2tf -> tflite.
Гейт: tflite vs оригинальный PyTorch на том же входе, max|diff| < 1e-3."""
import os, sys, glob, shutil, subprocess
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

refine_dst = os.path.join(stub, "refine.py")
refine_src = os.path.join(srcdir, "refine.py")
if os.path.exists(refine_src):
    shutil.copy(refine_src, refine_dst)
elif not os.path.exists(refine_dst):
    open(refine_dst, "w").close()

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

# ---------- probe стилей вызова forward (scale_list coarse->fine) ----------
SCALES = [16, 8, 4, 2, 1]
styles = [
    ("scale_list", lambda x: model(x, scale_list=SCALES)),
    ("ts+scale_list", lambda x: model(x, 0.5, SCALES)),
    ("scale", lambda x: model(x, scale=1.0)),
    ("ts", lambda x: model(x, 0.5)),
    ("bare", lambda x: model(x)),
]

def _pick_tensor(o):
    if torch.is_tensor(o):
        return o
    if isinstance(o, (list, tuple)):
        for t in reversed(o):
            if torch.is_tensor(t):
                return t
    return None

chosen_name, chosen_fn = None, None
with torch.no_grad():
    for name, fn in styles:
        try:
            t = _pick_tensor(fn(torch.rand(1, 7, H, W)))
            if t is not None and t.dim() == 4 and (t.shape[1] == 3 or t.shape[-1] == 3):
                chosen_name, chosen_fn = name, fn
                break
        except Exception:
            continue
assert chosen_fn, "не подошёл ни один стиль вызова forward"
print("forward style:", chosen_name)

class Wrap7(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return self.fn(x)

wrap = Wrap7(chosen_fn)

# ---------- эталон ----------
x = torch.rand(1, 7, H, W)
with torch.no_grad():
    ref = _pick_tensor(wrap(x)).numpy()
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
if got.ndim == 4 and got.shape[1] == 3:
    got = got.transpose(0, 2, 3, 1)
refn = ref.transpose(0, 2, 3, 1) if (ref.ndim == 4 and ref.shape[1] == 3) else ref
diff = float(np.max(np.abs(got - refn)))
print("GATE max|diff| = %.6f" % diff)
assert diff < 1e-3, "torch/tflite mismatch too large"

shutil.copy(f32, "rife_v426.tflite")
print("wrote rife_v426.tflite", os.path.getsize("rife_v426.tflite"))
f16 = glob.glob("tfl_out/*_float16.tflite")
if f16:
    shutil.copy(f16[0], "rife_v426_f16.tflite")
    print("wrote rife_v426_f16.tflite", os.path.getsize("rife_v426_f16.tflite"))
