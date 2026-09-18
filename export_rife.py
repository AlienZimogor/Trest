#!/usr/bin/env python3
import sys, os, re, glob, types, inspect, importlib.util, importlib.machinery
import subprocess, shutil, zipfile, traceback
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
    return F.grid_sample(x, vgrid, align_corners=True, mode="bilinear", padding_mode="zeros")


def _noop(*a, **k):
    return None


class _DummyObj:
    def __init__(self, *a, **k):
        pass
    def __call__(self, *a, **k):
        return torch.zeros(1)
    def __getattr__(self, n):
        return _noop


def _finish_module(mod, name, is_package=True):
    spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=is_package)
    spec.origin = "<stub>"
    mod.__spec__ = spec
    mod.__file__ = "<stub:%s>" % name
    mod.__loader__ = None
    if is_package:
        mod.__path__ = []
    mod.__all__ = []
    return mod


def _make_stub(name):
    mod = types.ModuleType(name)
    _finish_module(mod, name, is_package=True)
    def _getattr(attr):
        if attr.startswith("_"):
            raise AttributeError(attr)
        return _noop
    mod.__getattr__ = _getattr
    sys.modules[name] = mod
    return mod


def _missing_name(e):
    n = getattr(e, "name", None)
    if n:
        return n
    m = re.search(r"name '(\w+)' is not defined", str(e))
    return m.group(1) if m else None


def _call_with_injection(mod, fn, max_inject=16):
    for _ in range(max_inject):
        try:
            return fn()
        except NameError as e:
            name = _missing_name(e)
            if not name or name in mod.__dict__:
                raise
            mod.__dict__[name] = _DummyObj
            print("  injected dummy global:", name)
    return fn()


model_pkg = types.ModuleType("model")
_finish_module(model_pkg, "model")
sys.modules.setdefault("model", model_pkg)

wp_stub = types.ModuleType("model.warplayer")
_finish_module(wp_stub, "model.warplayer", is_package=False)
wp_stub.warp = _warp
sys.modules.setdefault("model.warplayer", wp_stub)
setattr(model_pkg, "warplayer", wp_stub)

tl_pkg = types.ModuleType("train_log")
_finish_module(tl_pkg, "train_log")
tl_pkg.__path__ = sorted(set(
    os.path.dirname(p) for p in glob.glob(os.path.join(ROOT, "**", "*.py"), recursive=True)
))
sys.modules.setdefault("train_log", tl_pkg)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def _load_with_stubs(path, name, max_stub=10):
    for _ in range(max_stub):
        try:
            return _load(path, name)
        except ModuleNotFoundError as e:
            missing = e.name
            print("  auto-stub module:", missing)
            _make_stub(missing)
            parent = missing.rsplit(".", 1)[0]
            if parent in sys.modules:
                setattr(sys.modules[parent], missing.rsplit(".", 1)[1], sys.modules[missing])
    raise RuntimeError("too many auto-stubs for %s" % path)


for d in sorted(glob.glob(os.path.join(ROOT, "**", "refine.py"), recursive=True)):
    if "__pycache__" in d:
        continue
    try:
        rm = _load_with_stubs(d, "model.refine")
        setattr(model_pkg, "refine", rm)
        print("refine loaded from", d)
        break
    except Exception as e:
        print("refine load fail at", d, "->", e)


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
print("pkl candidates:")
for p in pkls:
    print("  ", p, "(%.2f MB, pref=%d)" % (os.path.getsize(p)/1e6, _pref(p)))
if not pkls:
    raise SystemExit("no weights found under " + ROOT)

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
sd_ifnet = {k: v for k, v in sd_base.items() if not k.startswith("refine.")}
sd_refine = {k[len("refine."):]: v for k, v in sd_base.items() if k.startswith("refine.")}
print("weights keys sample:", list(sd_base.keys())[:6])
print("ifnet keys=%d refine keys=%d" % (len(sd_ifnet), len(sd_refine)))

TARGETS = [
    ("rifehdv3", "RIFE_HDv3.py", ["Model", "RIFE", "RIFE_HDv3"]),
    ("ifnet",    "IFNet_HDv3.py", ["IFNet"]),
]


def find_py(base):
    for p in pys:
        if os.path.basename(p) == base:
            return p
    return None


def _pick(out):
    if isinstance(out, dict):
        for k in ("merged", "img_pred", "pred", "output"):
            if k in out and torch.is_tensor(out[k]):
                return _pick(out[k])
        out = [v for v in out.values() if torch.is_tensor(v)]
    if isinstance(out, (tuple, list)):
        shapes = [tuple(t.shape) for t in out if torch.is_tensor(t)]
        print("  output tuple shapes:", shapes)
        c3 = [t for t in out if torch.is_tensor(t) and t.dim() == 4 and t.shape[1] == 3]
        if c3:
            return c3[-1]
        raise RuntimeError("no 3-channel frame in output tuple: %s" % (shapes,))
    if torch.is_tensor(out) and out.dim() == 4 and out.shape[1] == 3:
        return out
    raise RuntimeError("output is not a 3-channel frame: %s" %
                       (tuple(out.shape) if torch.is_tensor(out) else type(out),))


DUMMY = torch.rand(1, 7, 384, 512)
SCALE = [32.0, 16.0, 8.0, 4.0, 2.0]


def make_wrapper_model(net, mod):
    class W(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = net
            self.mod = mod

        def forward(self, x):
            i0 = x[:, 0:3]
            i1 = x[:, 3:6]
            errs = []
            for idx, fn in enumerate((
                lambda: self.net.inference(i0, i1, 0.5),
                lambda: self.net.inference(i0, i1, timestep=0.5),
                lambda: self.net.inference(i0, i1),
                lambda: self.net(i0, i1, 0.5),
            )):
                try:
                    return _pick(fn())
                except NameError:
                    raise
                except Exception as e:
                    errs.append("#%d %s: %s" % (idx, type(e).__name__, e))
            print("  Model inference attempts failed:")
            for e in errs:
                print("   ", e)
            raise RuntimeError("Model inference failed")
    return W()


def make_wrapper_ifnet(net):
    fparams = list(inspect.signature(net.forward).parameters.keys())
    print("  forward params:", fparams)

    class W(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = net

        def forward(self, x):
            ts = x[:, 6:7].mean(dim=(2, 3), keepdim=True)
            i0 = x[:, 0:3]; i1 = x[:, 3:6]
            attempts = [
                lambda: self.net(x, ts, SCALE, False, True, False),
                lambda: self.net(x, ts, SCALE, False, False, False),
                lambda: self.net(x, ts, SCALE, False, True),
                lambda: self.net(x, ts, SCALE, False, False),
                lambda: self.net(x, ts, SCALE),
                lambda: self.net(x, ts),
                lambda: self.net(x),
            ]
            errs = []
            for idx, fn in enumerate(attempts):
                try:
                    return _pick(fn())
                except NameError:
                    raise
                except Exception as e:
                    errs.append("#%d %s: %s" % (idx, type(e).__name__, e))
            print("  IFNet forward attempts failed:")
            for e in errs:
                print("   ", e)
            raise RuntimeError("IFNet forward failed")
    return W()


produced = []
for set_name, py_base, class_names in TARGETS:
    print("\n========== SET %s ==========" % set_name)
    py = find_py(py_base)
    if py is None:
        print("SKIP: py %s not found" % py_base)
        continue
    print("py:", py)
    try:
        mod = _load_with_stubs(py, "prov_%s" % set_name)
    except Exception as e:
        print("SKIP: import fail ->", e)
        traceback.print_exc()
        continue
    cls = None
    for cn in class_names:
        if hasattr(mod, cn):
            cls = getattr(mod, cn)
            print("class found:", cn)
            break
    if cls is None:
        print("SKIP: no class %s" % (class_names,))
        continue

    net = None
    for ctor in (lambda: cls(), lambda: cls(-1), lambda: cls(local_rank=-1), lambda: cls(None)):
        try:
            net = _call_with_injection(mod, ctor)
            break
        except TypeError as e:
            print("  ctor TypeError:", e)
            continue
        except Exception as e:
            print("  ctor fail:", type(e).__name__, e)
            continue
    if net is None:
        print("SKIP: all constructors failed")
        continue

    target = getattr(net, "flownet", net)
    miss, unexp = target.load_state_dict(sd_ifnet, strict=False)
    print("  flownet/IFNet load missing=%d unexpected=%d" % (len(miss), len(unexp)))
    if hasattr(net, "refine") and sd_refine:
        m2, u2 = net.refine.load_state_dict(sd_refine, strict=False)
        print("  refine load missing=%d unexpected=%d" % (len(m2), len(u2)))
    if len(miss) != 0:
        print("SKIP: missing=%d > 0" % len(miss))
        continue
    net.eval()

    convs_28 = [m for m in target.modules() if isinstance(m, nn.Conv2d) and m.in_channels == 28]
    print("  Conv2d in_channels=28 count=%d" % len(convs_28))
    seen = {}
    if convs_28:
        def hk(m, inp, out):
            seen.setdefault("ch", []).append(inp[0].shape[1])
        for c in convs_28:
            c.register_forward_hook(hk)

    w = make_wrapper_model(net, mod) if set_name == "rifehdv3" else make_wrapper_ifnet(net)
    w.eval()
    try:
        with torch.no_grad():
            ref = _call_with_injection(mod, lambda: w(DUMMY))
    except Exception as e:
        print("SKIP: forward fail ->", e)
        traceback.print_exc()
        continue
    print("  REF out shape=%s mean=%.4f min=%.4f max=%.4f" %
          (tuple(ref.shape), float(ref.mean()), float(ref.min()), float(ref.max())))
    if convs_28:
        print("  hook in_channels unique:", sorted(set(seen.get("ch", []))))
    if tuple(ref.shape) != (1, 3, 384, 512):
        print("SKIP: unexpected ref shape (wanted (1,3,384,512))")
        continue
    if not (0.0 <= float(ref.mean()) <= 1.0):
        print("SKIP: mean out of [0,1]")
        continue

    onnx_path = "rife_%s.onnx" % set_name
    try:
        torch.onnx.export(w, DUMMY, onnx_path, opset_version=13,
                          input_names=["in0"], output_names=["out0"],
                          do_constant_folding=True)
        print("  ONNX exported:", onnx_path, "(%.2f MB)" % (os.path.getsize(onnx_path)/1e6))
    except Exception as e:
        print("SKIP: onnx export fail ->", e)
        traceback.print_exc()
        continue
    produced.append(set_name)

print("\n========== PACKING ==========")
print("PRODUCED SETS:", produced)
if not produced:
    raise SystemExit("no sets produced — check logs above")
with zipfile.ZipFile("rife_models.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for s in produced:
        for ext in ("param", "bin"):
            f = "rife_%s.ncnn.%s" % (s, ext)
            if os.path.isfile(f):
                z.write(f, os.path.join(s, os.path.basename(f)))
                print("  packed:", os.path.join(s, os.path.basename(f)))
print("ZIP OK: rife_models.zip (%.2f MB)" % (os.path.getsize("rife_models.zip")/1e6))
