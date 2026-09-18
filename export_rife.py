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


_wp = types.ModuleType("model.warplayer")
_wp.warp = _warp
_mpkg = types.ModuleType("model")
_mpkg.warplayer = _wp
if not os.path.isdir(os.path.join(ROOT, "model")):
    sys.modules.setdefault("model", _mpkg)
    sys.modules.setdefault("model.warplayer", _wp)


def _load_mod(path):
    name = "prov_" + os.path.basename(path)[:-3]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def score(p):
    s = 0
    if "4.26" in p or "v4" in p:
        s += 2
    if "HDv3" in p:
        s -= 1
    return s


pys = [p for p in glob.glob(os.path.join(ROOT, "**", "*.py"), recursive=True)
       if "class IFNet" in open(p, encoding="utf-8", errors="ignore").read(200000)]
pys.sort(key=score, reverse=True)
print("IFNet py candidates:", pys)
if not pys:
    raise SystemExit("no IFNet py found")

pkls = (glob.glob(os.path.join(ROOT, "**", "*.pkl"), recursive=True) +
        glob.glob(os.path.join(ROOT, "**", "*.pth"), recursive=True) +
        glob.glob(os.path.join(ROOT, "**", "*.pt"), recursive=True))
print("weights:", pkls)
if not pkls:
    raise SystemExit("no weights found")
sd = torch.load(pkls[0], map_location="cpu", weights_only=False)
if isinstance(sd, dict) and "state_dict" in sd:
    sd = sd["state_dict"]
print("sd keys sample:", list(sd.keys())[:8])

chosen_net = None
chosen_path = None
for p in pys:
    try:
        mod = _load_mod(p)
    except Exception as e:
        print("skip", p, "->", e)
        continue
    cls = getattr(mod, "IFNet", None)
    if cls is None:
        continue
    try:
        net = cls()
    except Exception as e:
        print("skip init", p, "->", e)
        continue
    miss, unexp = net.load_state_dict(sd, strict=False)
    print("arch", p, "missing", len(miss), "unexpected", len(unexp))
    if len(miss) == 0 and len(unexp) == 0:
        chosen_net, chosen_path = net, p
        break
    if chosen_net is None:
        chosen_net, chosen_path = net, p
if chosen_net is None:
    raise SystemExit("no usable IFNet arch")
print("CHOSEN arch:", chosen_path)
net = chosen_net
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
