#!/usr/bin/env python3
"""v14: диагностика рекурренсии RIFE v4.26 — печатает фактические формы
входов/выходов block0..block4 при одном прогоне Model.inference.
Даёт каналы cat, пространственные размеры (Interp-факторы) и выходы lastconv."""
import os, sys, glob, types, re, importlib.util, importlib.machinery
import torch


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
    return torch.nn.functional.grid_sample(x, vgrid, align_corners=True)


def _noop(*a, **k):
    return None


class _DummyModule(torch.nn.Module):
    def __init__(self, *a, **k):
        super().__init__()

    def forward(self, *a, **k):
        return torch.zeros(1)


def _finish(mod, name):
    spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    mod.__spec__ = spec
    mod.__file__ = "<stub:%s>" % name
    mod.__loader__ = None
    mod.__path__ = []
    mod.__all__ = []
    return mod


model_pkg = _finish(types.ModuleType("model"), "model")
sys.modules.setdefault("model", model_pkg)
wp = _finish(types.ModuleType("model.warplayer"), "model.warplayer")
wp.warp = _warp
sys.modules.setdefault("model.warplayer", wp)
setattr(model_pkg, "warplayer", wp)
tl = _finish(types.ModuleType("train_log"), "train_log")
tl.__path__ = sorted(set(os.path.dirname(p) for p in
                         glob.glob(os.path.join("model_src", "**", "*.py"), recursive=True)))
sys.modules.setdefault("train_log", tl)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def _load_stubs(path, name, max_stub=10):
    for _ in range(max_stub):
        try:
            return _load(path, name)
        except ModuleNotFoundError as e:
            miss = e.name
            print("  auto-stub module:", miss)
            mod = _finish(types.ModuleType(miss), miss)
            mod.__getattr__ = lambda a: _noop
            sys.modules[miss] = mod
            parent = miss.rsplit(".", 1)[0]
            if parent in sys.modules:
                setattr(sys.modules[parent], miss.rsplit(".", 1)[1], mod)
    raise RuntimeError("too many auto-stubs")


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
            mod.__dict__[name] = _DummyModule
            print("  injected dummy global:", name)
    return fn()


py = None
for p in glob.glob(os.path.join("model_src", "**", "RIFE_HDv3.py"), recursive=True):
    py = p
    break
print("py:", py)
mod = _load_stubs(py, "prov_rifehdv3")
cls = getattr(mod, "Model")
net = _call_with_injection(mod, lambda: cls())

pkls = sorted(glob.glob(os.path.join("model_src", "**", "*.pkl"), recursive=True))
sd = torch.load(pkls[0], map_location="cpu", weights_only=False)
if isinstance(sd, dict) and "state_dict" in sd:
    sd = sd["state_dict"]
clean = {}
for k, v in sd.items():
    k2 = k
    while k2.startswith("module."):
        k2 = k2[len("module."):]
    clean[k2] = v
sd_if = {k: v for k, v in clean.items() if not k.startswith("refine.")}
sd_rf = {k[len("refine."):]: v for k, v in clean.items() if k.startswith("refine.")}
miss, unexp = net.flownet.load_state_dict(sd_if, strict=False)
print("flownet load missing=%d unexpected=%d" % (len(miss), len(unexp)))
if hasattr(net, "refine") and sd_rf:
    m2, u2 = net.refine.load_state_dict(sd_rf, strict=False)
    print("refine load missing=%d unexpected=%d" % (len(m2), len(u2)))
net.eval()

fl = net.flownet
hooks = []


def mk(tag):
    def hk(m, inp, out):
        shapes = [tuple(t.shape) for t in inp if torch.is_tensor(t)]
        print("HOOK %-24s in=%s out=%s" % (tag, shapes, tuple(out.shape)))
    return hk


for i in range(5):
    blk = getattr(fl, "block%d" % i, None)
    if blk is None:
        print("no block%d" % i)
        continue
    hooks.append(blk.register_forward_hook(mk("block%d" % i)))
    c0 = blk.conv0[0][0]
    hooks.append(c0.register_forward_hook(mk("block%d.conv0" % i)))
    hooks.append(blk.lastconv[0].register_forward_hook(mk("block%d.lastconv" % i)))

i0 = torch.rand(1, 3, 384, 512)
i1 = torch.rand(1, 3, 384, 512)
with torch.no_grad():
    try:
        out = _call_with_injection(mod, lambda: net.inference(i0, i1, 0.5))
    except TypeError:
        out = _call_with_injection(mod, lambda: net.inference(i0, i1))
print("inference out:", tuple(out.shape))
for h in hooks:
    h.remove()
print("DIAG DONE")
