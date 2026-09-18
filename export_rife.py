import sys, os, glob, types, importlib.util
import torch
import torch.nn as nn
import torch.nn.functional as F


def _warp(x, flow):
    N, C, H, W = x.shape
    ys = torch.arange(H, device=x.device, dtype=x.dtype)
    xs = torch.arange(W, device=x.device, dtype=x.dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    base = torch.stack([gx, gy], 0).unsqueeze(0)          # (1,2,H,W)
    vgrid = base + flow                                    # (N,2,H,W) same rank
    vgrid = torch.stack([
        2.0 * vgrid[:, 0] / max(W - 1, 1) - 1.0,
        2.0 * vgrid[:, 1] / max(H - 1, 1) - 1.0], 1)      # (N,2,H,W)
    return F.grid_sample(x, vgrid, align_corners=True)


_wp = types.ModuleType("model.warplayer")
_wp.warp = _warp
_mpkg = types.ModuleType("model")
_mpkg.warplayer = _wp
sys.modules.setdefault("model", _mpkg)
sys.modules.setdefault("model.warplayer", _wp)

pys = [p for p in glob.glob("model_src/**/*.py", recursive=True)
       if "class IFNet" in open(p, encoding="utf-8", errors="ignore").read(200000)]
print("IFNet py:", pys)
if not pys:
    raise SystemExit("no IFNet py found")
spec = importlib.util.spec_from_file_location("prov_ifnet", pys[0])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
if hasattr(m, "warp"):
    m.warp = _warp
    print("patched m.warp")

pkls = (glob.glob("model_src/**/*.pkl", recursive=True) +
        glob.glob("model_src/**/*.pth", recursive=True) +
        glob.glob("model_src/**/*.pt", recursive=True))
print("weights:", pkls)
if not pkls:
    raise SystemExit("no weights found")

net = m.IFNet()
sd = torch.load(pkls[0], map_location="cpu", weights_only=False)
if isinstance(sd, dict) and "state_dict" in sd:
    sd = sd["state_dict"]
miss, unexp = net.load_state_dict(sd, strict=False)
print("MISSING:", list(miss)[:20])
print("UNEXPECTED:", list(unexp)[:20])
net.eval()


class W(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):
        t = x[:, 6:7].mean(dim=(2, 3), keepdim=True)
        try:
            return self.n(x[:, 0:3], x[:, 3:6], timestep=t)
        except TypeError:
            return self.n(x[:, 0:3], x[:, 3:6], t)


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
