import sys, os, glob, types, inspect, importlib.util, subprocess, shutil, zipfile
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
    base = torch.stack([gx, gy], dim=-1).unsqueeze(0)
    f = flow.permute(0, 2, 3, 1).float()
    vgrid = base + f
    vgrid = torch.stack([
        2.0 * vgrid[..., 0] / max(W - 1, 1) - 1.0,
        2.0 * vgrid[..., 1] / max(H - 1, 1) - 1.0], dim=-1)
    return F.grid_sample(x, vgrid, align_corners=True)


model_pkg = types.ModuleType("model")
model_pkg.__path__ = []
sys.modules.setdefault("model", model_pkg)
wp = types.ModuleType("model.warplayer")
wp.warp = _warp
sys.modules.setdefault("model.warplayer", wp)
setattr(model_pkg, "warplayer", wp)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


for d in sorted(glob.glob(os.path.join(ROOT, "**", "refine.py"), recursive=True)):
    try:
        rm = _load(d, "model.refine")
        setattr(model_pkg, "refine", rm)
        print("refine loaded from", d)
    except Exception as e:
        print("refine load fail:", e)


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
       if "__pycache__" not in p]
pkls = [p for p in
        glob.glob(os.path.join(ROOT, "**", "*.pkl"), recursive=True) +
        glob.glob(os.path.join(ROOT, "**", "*.pth"), recursive=True) +
        glob.glob(os.path.join(ROOT, "**", "*.pt"), recursive=True)]
pkls.sort(key=_pref, reverse=True)
print("pkl candidates:", pkls)
if not pkls:
    raise SystemExit("no weights found")

sd_raw = torch.load(pkls[0], map_location="cpu", weights_only=False)
if isinstance(sd_raw, dict) and "state_dict" in sd_raw:
    sd_raw = sd_raw["state_dict"]


def strip_mod(sd):
    out = {}
    for k, v in sd.items():
        k2 = k
        while k2.startswith("module."):
            k2 = k2[len("module."):]
        out[k2] = v
    return out


sd_base = strip_mod(sd_raw)

TARGETS = [
    ("ifnet", "IFNet_HDv3.py", ["IFNet"]),
    ("rifehdv3", "RIFE_HDv3.py", ["Model", "RIFE", "RIFE_HDv3"]),
]


def find_py(base):
    for p in pys:
        if os.path.basename(p) == base:
            return p
    return None


def make_wrapper(net):
    fparams = list(inspect.signature(net.forward).parameters.keys())
    nblk = len(getattr(net, "block", []) or []) or 5
    scale_list = [float(2 ** (nblk - j)) for j in range(nblk)]
    print("blocks=%d scale_list=%s" % (nblk, scale_list))

    class W(nn.Module):
        def __init__(s):
            super().__init__()
            s.n = net

        def forward(s, x):
            i0 = x[:, 0:3]
            i1 = x[:, 3:6]
            tm = x[:, 6:7]
            ts = tm.mean(dim=(2, 3), keepdim=True)
            c6 = torch.cat([i0, i1], 1)
            if fparams and fparams[0] in ("x", "inputs", "inp", "frame"):
                if "scale_list" in fparams and "timestep" in fparams:
                    return s.n(c6, ts, scale_list)
                if "timestep" in fparams:
                    return s.n(c6, ts)
                return s.n(c6)
            if fparams and fparams[0] in ("img0", "x0", "image0"):
                if "scale_list" in fparams and "timestep" in fparams:
                    return s.n(i0, i1, ts, scale_list)
                if "timestep" in fparams:
                    return s.n(i0, i1, ts)
                return s.n(i0, i1)
            last = None
            for fn in (lambda: s.n(c6, ts, scale_list), lambda: s.n(c6, ts), lambda: s.n(c6),
                       lambda: s.n(x, ts, scale_list), lambda: s.n(x, ts), lambda: s.n(x),
                       lambda: s.n(i0, i1, ts, scale_list), lambda: s.n(i0, i1, ts), lambda: s.n(i0, i1)):
                try:
                    return fn()
                except Exception as e:
                    last = e
            raise last
    return W(), fparams


produced = []
for set_name, py_base, class_names in TARGETS:
    py = find_py(py_base)
    if py is None:
        print("SET %s: SKIP (py %s not found)" % (set_name, py_base))
        continue
    try:
        mod = _load(py, "prov_%s" % set_name)
    except Exception as e:
        print("SET %s: SKIP (import fail %s)" % (set_name, e))
        continue
    cls = None
    for cn in class_names:
        if hasattr(mod, cn):
            cls = getattr(mod, cn)
            break
    if cls is None:
        print("SET %s: SKIP (no class %s)" % (set_name, class_names))
        continue
    net = cls()
    best_map, best_miss = None, None
    for map_name, sd in (("direct", sd_base),
                         ("flownet.", {"flownet." + k: v for k, v in sd_base.items()})):
        miss, unexp = net.load_state_dict(sd, strict=False)
        print("SET %s map=%s missing=%d unexpected=%d" %
              (set_name, map_name, len(miss), len(unexp)))
        if best_miss is None or len(miss) < best_miss:
            best_map, best_miss = map_name, len(miss)
        if len(miss) == 0:
            break
    if best_miss != 0:
        print("SET %s: SKIP (missing=%d with map=%s; weights incomplete)" %
              (set_name, best_miss, best_map))
        continue
    net.load_state_dict(
        sd_base if best_map == "direct"
        else {"flownet." + k: v for k, v in sd_base.items()},
        strict=False)
    net.eval()
    w, fparams = make_wrapper(net)
    print("SET %s: forward params %s" % (set_name, fparams))
    onnx_path = "rife_%s.onnx" % set_name
    dummy = torch.randn(1, 7, 384, 512)
    with torch.no_grad():
        ref = w(dummy)
    print("SET %s REF out %s mean=%.4f min=%.4f max=%.4f" %
          (set_name, tuple(ref.shape), ref.mean(), ref.min(), ref.max()))
    torch.onnx.export(w, dummy, onnx_path, opset_version=13,
                      input_names=["in0"], output_names=["out0"],
                      do_constant_folding=True)
    pnnx = shutil.which("pnnx") or "pnnx"
    r = subprocess.run([pnnx, onnx_path, "inputshape=[1,7,384,512]"],
                       capture_output=True, text=True)
    print("SET %s pnnx rc=%d" % (set_name, r.returncode))
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])
        continue
    produced.append(set_name)

print("PRODUCED SETS:", produced)
if not produced:
    raise SystemExit("no sets produced")
with zipfile.ZipFile("rife_models.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for s in produced:
        for ext in ("param", "bin"):
            f = "rife_%s.ncnn.%s" % (s, ext)
            if os.path.isfile(f):
                z.write(f, os.path.join(s, os.path.basename(f)))
print("ZIP OK")
