#!/usr/bin/env python3
"""RIFE v4.26 -> LiteRT (.tflite) через ОРИГИНАЛЬНЫЙ исходник IFNet_HDv3.py из zip.
Архитектуру берём из исходника (там правильный flow*scale), не из ручной копии.
Пайплайн: pkl -> IFNet_HDv3.IFNet -> ONNX(opset16, GridSample) -> onnx2tf -> tflite.
Гейт: tflite vs оригинальная PyTorch-модель на том же входе, max|diff| < 1e-3."""
import os, sys, glob, shutil, subprocess
import numpy as np
import torch
import torch.nn as nn

H, W = 384, 512

# ---------- найти каталог с оригинальным IFNet_HDv3.py ----------
srcdir = None
for d in ["model_src"] + [d for d in glob.glob(os.path.join("model_src", "**"), recursive=True) if os.path.isdir(d)]:
    if os.path.exists(os.path.join(d, "IFNet_HDv3.py")):
        srcdir = d
        break
assert srcdir, "IFNet_HDv3.py не найден под model_src"
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
missing = [m for m in missing]
unexpected = [u for u in unexpected]
print("missing:", missing[:10], "unexpected:", unexpected[:10])
assert not missing, "state_dict не лёг строго: несовпадение архитектуры"
model.eval()

class Wrap7(nn.Module):
    """Обёртка: вход [1,7,H,W] = (img0,img1,timestep). Оригинальный forward сам
    нарежет img0=x[:,:3], img1=x[:,3:6], timestep=x[:,6:7] (или проигнорирует 7-й канал)."""
    def __init__(self, net):
        super().__init__()
        self.net = net
    def forward(self, x):
        return self.net(x)

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

# ---------- onnx2tf ----------
subprocess.run(["onnx2tf", "-i", "rife_v426.onnx", "-o", "tfl_out", "-b", "1"], check=True)
f32 = "tfl_out/rife_v426_sim_float32.tflite"
f16 = "tfl_out/rife_v426_sim_float16.tflite"
assert os.path.exists(f32), "onnx2tf не дал float32 tflite"
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
if os.path.exists(f16):
    shutil.copy(f16, "rife_v426_f16.tflite")
    print("wrote rife_v426_f16.tflite", os.path.getsize("rife_v426_f16.tflite"))
