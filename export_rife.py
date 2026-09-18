import sys, os, glob, types, importlib.util
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = "model_src"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _warp(x, flow):
    N, C, H, W = x.shape
    ys = torch.arange(H, device=x.device, dtype=x.dtype)
    xs = torch.arange(W, device=x.device, dtype=x.dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    base = torch.stack([gx, gy], 0).unsqueeze(0)
    vgrid = base + flow
    vgrid = torch.stack([
        2.0 * vgrid[:, 0] / max(W - 1, 1) - 1.0,
        2.0 * vgrid[:, 1] / max(H - 1, 1) - 1.0], 1)
    return F.grid_sample(x, vgrid, align_corners=True)


model_pkg = types.ModuleType("model")
model_pkg.__path__ = []
sys.modules.setdefault("model", model_pkg)
wp_stub = types.ModuleType("model.warplayer")
wp_stub.warp = _warp
sys.modules.setdefault("model.warplayer", wp_stub)
setattr(model_pkg, "warplayer", wp_stub)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def _pref(p):
    s = 0
    b = os.path.basename(p).lower()
    if "4.26" in b or "426" in b:
        s += 4
    if "v4" in b:
        s += 1
    if "flownet" in b:
        s -= 3
    if "__pycache__" in p:
        s -= 10
    return s


pys = [p for p in glob.glob(os.path.join(ROOT, "**", "*.py"), recursive=True)
       if "__pycache__" not in p
       and "class IFNet" in open(p, encoding="utf-8", errors="ignore").read(200000)]
pkls = [p for p in
        glob.glob(os.path.join(ROOT, "**", "*.pkl"), recursive=True) +
        glob.glob(os.path.join(ROOT, "**", "*.pth"), recursive=True) +
        glob.glob(os.path.join(ROOT, "**", "*.pt"), recursive=True)]
pys.sort(key=_pref, reverse=True)
pkls.sort(key=_pref, reverse=True)
print("py candidates:", pys)
print("pkl candidates:", pkls)
if not pys or not pkls:
    raise SystemExit("no IFNet py or weights found under " + ROOT)


def strip_prefix(sd):
    out = {}
    for k, v in sd.items():
        k2 = k
        while k2.startswith("module."):
            k2 = k2[len("module."):]
        out[k2] = v
    return out


best = None
for pi, py in enumerate(pys):
    d = os.path.dirname(py)
    ref = os.path.join(d, "refine.py")
    if os.path.isfile(ref):
        try:
            rm = _load(ref, "model.refine")
            setattr(model_pkg, "refine", rm)
        except Exception as e:
            print("refine load fail:", e)
    wpr = os.path.join(d, "warplayer.py")
    if os.path.isfile(wpr):
        try:
            wm = _load(wpr, "model.warplayer")
            if not hasattr(wm, "warp"):
                wm.warp = _warp
            setattr(model_pkg, "warplayer", wm)
        except Exception as e:
            print("warplayer load fail, keep stub:", e)
            sys.modules["model.warplayer"] = wp_stub
            setattr(model_pkg, "warplayer", wp_stub)
    try:
        mod = _load(py, "prov_ifnet_%d" % pi)
    except Exception as e:
        print("skip py:", py, "->", e)
        continue
    cls = getattr(mod, "IFNet", None)
    if cls is None:
        print("no class IFNet in", py)
        continue
    for pkl in pkls:
        try:
            sd = torch.load(pkl, map_location="cpu", weights_only=False)
        except Exception as e:
            print("skip pkl:", pkl, "->", e)
            continue
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        sd = strip_prefix(sd)
        try:
            probe = cls()
        except Exception as e:
            print("init fail for", py, "->", e)
            break
        miss, unexp = probe.load_state_dict(sd, strict=False)
        bad = len(miss) + len(unexp)
        print("pair %s + %s -> missing=%d unexpected=%d" %
              (os.path.basename(py), os.path.basename(pkl), len(miss), len(unexp)))
        if best is None or bad < best[0]:
            best = (bad, py, pkl, cls, sd)
        if bad == 0:
            break
    if best is not None and best[0] == 0:
        break

if best is None:
    raise SystemExit("no usable arch/weights pair")
bad, py, pkl, cls, sd = best
print("CHOSEN arch=%s weights=%s mismatch=%d" % (py, pkl, bad))
if bad != 0:
    print("WARNING: mismatch>0, output may be garbage")

net = cls()
net.load_state_dict(sd, strict=False)
net.eval()


class W(nn.Module):
    def __init__(s, n):
        super().__init__()
        s.n = n

    def forward(s, x):
        t = x[:, 6:7].mean(dim=(2, 3), keepdim=True)
        try:
            return s.n(x[:, 0:3], x[:, 3:6], timestep=t)
        except TypeError:
            pass
        try:
            return s.n(x[:, 0:3], x[:, 3:6], t)
        except TypeError:
            pass
        return s.n(x)


w = W(net)
dummy = torch.randn(1, 7, 288, 512)
with torch.no_grad():
    ref = w(dummy)
print("REF out %s mean=%.4f min=%.4f max=%.4f" %
      (tuple(ref.shape), ref.mean(), ref.min(), ref.max()))
torch.onnx.export(w, dummy, "rife_clean.onnx", opset_version=13,
                  input_names=["in0"], output_names=["out0"],
                  do_constant_folding=True)
print("EXPORT OK")
