#!/usr/bin/env python3
"""RIFE v4.26 -> ONNX(opset16) -> onnx2tf -> TFLite. FINAL.
Используем ОФИЦИАЛЬНЫЙ IFNet_HDv3.py из zip (correct flow), передаём scale_list явно.
GATE: tflite vs official-torch на одном входе, max|diff| < 1e-3."""
import os, sys, glob, subprocess, shutil
import numpy as np
import torch
import tensorflow as tf

H, W = 384, 512
ROOT = os.path.abspath("model_src")
srcdir = None
for d in [ROOT] + [p for p in glob.glob(os.path.join(ROOT, "**"), recursive=True) if os.path.isdir(p)]:
    if os.path.exists(os.path.join(d, "IFNet_HDv3.py")):
        srcdir = os.path.abspath(d); break
assert srcdir, "IFNet_HDv3.py not found under model_src"
sys.path.insert(0, srcdir)
import IFNet_HDv3

pkls = glob.glob(os.path.join(srcdir, "**", "*.pkl"), recursive=True)
assert pkls, "pkl not found"
raw = torch.load(pkls[0], map_location="cpu", weights_only=False)
if isinstance(raw, dict) and "state_dict" in raw: raw = raw["state_dict"]
sd = {}
for k, v in raw.items():
    k2 = k
    while k2.startswith("module."): k2 = k2[7:]
    sd[k2] = v
print("pkl keys:", len(sd))

net = IFNet_HDv3.IFNet()
missing, unexpected = net.load_state_dict(sd, strict=False)
print("missing:", list(missing)[:5], "unexpected:", list(unexpected)[:5])
assert not missing, "state_dict missing keys"
net.eval()

x = torch.rand(1, 7, H, W)
ref = None; used = None
for sl in ([16, 8, 4, 2, 1], [1, 2, 4, 8, 16], [1, 2, 4, 8], [16, 8, 4, 2]):
    try:
        with torch.no_grad():
            ref = net(x, scale_list=sl)
        used = sl; break
    except Exception as e:
        print("scale_list", sl, "failed:", type(e).__name__, e)
if ref is None:
    try:
        with torch.no_grad():
            ref = net(x)
        used = "default"
    except Exception as e:
        print("default also failed:", e)
assert ref is not None, "ALL scale_list attempts failed -> pivot to on-device conversion"
print("used scale_list:", used)

torch.onnx.export(net, x, "rife_v426.onnx", opset_version=16,
                  input_names=["in0"], output_names=["out0"], do_constant_folding=True)
print("wrote rife_v426.onnx", os.path.getsize("rife_v426.onnx"))

subprocess.run(["onnx2tf", "-i", "rife_v426.onnx", "-o", "tfl_out", "-b", "1"], check=True)
cand = sorted(glob.glob("tfl_out/*_float32.tflite"))
assert cand, "no float32 tflite"
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
diff = float(np.max(np.abs(got - refn)))
print("GATE max|diff| = %.6f" % diff)
assert diff < 1e-3, "torch/tflite mismatch too large"
shutil.copy(f32, "rife_v426.tflite")
print("wrote rife_v426.tflite", os.path.getsize("rife_v426.tflite"))
