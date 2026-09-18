import sys
import os
import glob
import types
import shutil
import importlib.util

import torch
import torch.nn as nn
import torch.nn.functional as F


def _warp(x, flow):
    N, C, H, W = x.size()
    xx = torch.arange(0, W, device=x.device).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H, device=x.device).view(-1, 1).repeat(1, W)
    grid = torch.cat((xx, yy), 0).view(1, 2, H, W).repeat(N, 1, 1, 1).float()
    vgrid = grid + flow
    vgrid[:, :1] = 2.0 * vgrid[:, :1] / max(W - 1, 1) - 1.0
    vgrid[:, 1:2] = 2.0 * vgrid[:, 1:2] / max(H - 1, 1) - 1.0
    return F.grid_sample(x, vgrid, align_corners=True)


def _install_warplayer_stub():
    warpm = types.ModuleType("model.warplayer")
    warpm.warp = _warp
    pkg = types.ModuleType("model")
    pkg.warplayer = warpm
    sys.modules.setdefault("model", pkg)
    sys.modules.setdefault("model.warplayer", warpm)
    sys.modules.setdefault("warplayer", warpm)


def _find(patterns):
    out = []
    for root in ("model_src", "."):
        for pat in patterns:
            out += sorted(glob.glob(os.path.join(root, "**", pat), recursive=True))
    seen = set()
    return [p for p in out if not (p in seen or seen.add(p))]


def _load_ifnet_from_py(path):
    spec = importlib.util.spec_from_file_location("provided_ifnet", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if hasattr(mod, "IFNet"):
        return mod.IFNet
    for name in dir(mod):
        obj = getattr(mod, name)
        if isinstance(obj, type) and issubclass(obj, nn.Module) and obj is not nn.Module:
            return obj
    return None


def load_ifnet():
    _install_warplayer_stub()
    cands = []
    for p in _find(["*.py"]):
        if os.path.basename(p) == "export_rife.py":
            continue
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                head = f.read(200000)
        except Exception:
            continue
        if "class IFNet" in head:
            cands.append(p)
    print("IFNet py candidates:", cands)
    last = None
    for p in cands:
        d = os.path.dirname(p)
        if d and d not in sys.path:
            sys.path.insert(0, d)
        try:
            cls = _load_ifnet_from_py(p)
        except Exception as e:
            last = e
            print("skip", p, "->", e)
            continue
        if cls is not None:
            print("LOADED IFNet from", p)
            return cls
    raise SystemExit("cannot load IFNet from provided py: %s" % last)


class W(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        img0 = x[:, 0:3]
        img1 = x[:, 3:6]
        tmap = x[:, 6:7]
        t = tmap.mean(dim=(2, 3), keepdim=True)
        try:
            return self.net(img0, img1, timestep=t)
        except TypeError:
            try:
                return self.net(img0, img1, t)
            except TypeError:
                return self.net(torch.cat([img0, img1, tmap], 1))


def main():
    onnxes = _find(["*.onnx"])
    if onnxes:
        shutil.copy(onnxes[0], "rife_clean.onnx")
        print("USING PROVIDED ONNX:", onnxes[0])
        return
    pkls = _find(["*.pkl", "*.pth", "*.pt"])
    print("pkl candidates:", pkls)
    if not pkls:
        raise SystemExit("no pkl/pth/onnx found in model_src/")
    pkl = pkls[0]
    IFNet = load_ifnet()
    net = IFNet()
    sd = torch.load(pkl, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    missing, unexpected = net.load_state_dict(sd, strict=False)
    print("MISSING:", list(missing)[:20])
    print("UNEXPECTED:", list(unexpected)[:20])
    net.eval()
    w = W(net)
    dummy = torch.randn(1, 7, 288, 512)
    with torch.no_grad():
        torch.onnx.export(
            w, dummy, "rife_clean.onnx", opset_version=13,
            input_names=["in0"], output_names=["out0"],
            do_constant_folding=True,
        )
    print("EXPORT OK")


main()
