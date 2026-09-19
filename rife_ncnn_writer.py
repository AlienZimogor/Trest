#!/usr/bin/env python3
"""v16: ПОЛНАЯ RIFE v4.26 своим конвертером (single-input 7ch -> 3ch).
Рекурренсия снята хуками: scale0 encode x2 -> cat1(15) -> block0..block4,
warp через F.grid_sample, финал = w0*mask + w1*(1-mask)."""
import os, glob
import torch
import torch.nn as nn
import torch.nn.functional as F


def _pref(p):
    s = 0
    b = os.path.basename(p).lower()
    if "4.26" in b or "426" in b: s += 4
    if "v4" in b: s += 1
    if "flownet" in b: s -= 3
    return s


pkls = [p for p in
        glob.glob(os.path.join("model_src", "**", "*.pkl"), recursive=True) +
        glob.glob(os.path.join("model_src", "**", "*.pth"), recursive=True) +
        glob.glob(os.path.join("model_src", "**", "*.pt"), recursive=True)]
pkls.sort(key=_pref, reverse=True)
sd_raw = torch.load(pkls[0], map_location="cpu", weights_only=False)
if isinstance(sd_raw, dict) and "state_dict" in sd_raw:
    sd_raw = sd_raw["state_dict"]
sd = {}
for k, v in sd_raw.items():
    k2 = k
    while k2.startswith("module."):
        k2 = k2[len("module."):]
    sd[k2] = v


def Wt(k):
    return sd[k]


def Bt(k):
    return sd.get(k.replace(".weight", ".bias"))


def conv(key, stride=1):
    w = Wt(key)
    c = nn.Conv2d(w.shape[1], w.shape[0], w.shape[2], stride, w.shape[2] // 2,
                  bias=(Bt(key) is not None))
    with torch.no_grad():
        c.weight.copy_(w)
        if Bt(key) is not None:
            c.bias.copy_(Bt(key))
    return c


def deconv(key):
    w = Wt(key)
    c = nn.ConvTranspose2d(w.shape[0], w.shape[1], w.shape[2], 2, 1,
                           bias=(Bt(key) is not None))
    with torch.no_grad():
        c.weight.copy_(w)
        if Bt(key) is not None:
            c.bias.copy_(Bt(key))
    return c


def warp(x, flow):
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
    return F.grid_sample(x, vgrid, align_corners=True, mode="bilinear",
                         padding_mode="zeros")


class ConvBlock(nn.Module):
    def __init__(self, prefix):
        super().__init__()
        self.c = conv(prefix + ".conv.weight")
        self.register_buffer("beta", Wt(prefix + ".beta"))
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.act(x + self.c(x) * self.beta)


class Block(nn.Module):
    def __init__(self, prefix, c0, c1, factor):
        super().__init__()
        self.factor = factor
        self.c0 = conv(prefix + ".conv0.0.0.weight", stride=2)
        self.c1 = conv(prefix + ".conv0.1.0.weight", stride=2)
        self.cb = nn.ModuleList(
            [ConvBlock(prefix + ".convblock.%d" % i) for i in range(8)])
        self.last = deconv(prefix + ".lastconv.0.weight")
        self.ps = nn.PixelShuffle(2)
        self.relu = nn.ReLU()

    def forward(self, x, flow):
        if flow is not None:
            x = torch.cat([x, flow], dim=1)
        if self.factor > 1:
            x = F.interpolate(x, scale_factor=1.0 / self.factor,
                              mode="bilinear", align_corners=False)
        y = self.relu(self.c0(x))
        y = self.relu(self.c1(y))
        for cb in self.cb:
            y = cb(y)
        y = self.ps(self.last(y))
        if self.factor > 1:
            y = F.interpolate(y, scale_factor=float(self.factor),
                              mode="bilinear", align_corners=False)
        return y[:, :4], y[:, 4:5], y[:, 5:13]


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.e0 = conv("encode.cnn0.weight", stride=2)
        self.e1 = conv("encode.cnn1.weight")
        self.e2 = conv("encode.cnn2.weight")
        self.ed = deconv("encode.cnn3.weight")
        self.b0 = Block("block0", 96, 192, 16)
        self.b1 = Block("block1", 64, 128, 8)
        self.b2 = Block("block2", 48, 96, 4)
        self.b3 = Block("block3", 32, 64, 2)
        self.b4 = Block("block4", 16, 32, 1)

    def encode(self, img):
        return self.ed(self.e2(self.e1(self.e0(img))))

    def forward(self, x):
        img0 = x[:, 0:3]
        img1 = x[:, 3:6]
        ts = x[:, 6:7]
        d0 = self.encode(img0)
        d1 = self.encode(img1)
        cat1 = torch.cat([img0, img1, d0, d1, ts], dim=1)
        flow, mask, feat = self.b0(cat1, None)
        for blk in (self.b1, self.b2, self.b3, self.b4):
            w0 = warp(img0, flow[:, :2])
            w1 = warp(img1, flow[:, 2:4])
            wd0 = warp(d0, flow[:, :2])
            wd1 = warp(d1, flow[:, 2:4])
            x24 = torch.cat([w0, w1, wd0, wd1, ts, mask, feat], dim=1)
            flow, mask, feat = blk(x24, flow)
        w0 = warp(img0, flow[:, :2])
        w1 = warp(img1, flow[:, 2:4])
        return w0 * mask + w1 * (1.0 - mask)


net = Net().eval()
x = torch.rand(1, 7, 384, 512)
with torch.no_grad():
    y = net(x)
print("eager out:", tuple(y.shape), "mean=%.4f min=%.4f max=%.4f"
      % (float(y.mean()), float(y.min()), float(y.max())))
assert tuple(y.shape) == (1, 3, 384, 512)
torch.onnx.export(net, x, "rife_hand.onnx", opset_version=13,
                  input_names=["in0"], output_names=["out0"],
                  dynamo=False, do_constant_folding=True)
print("wrote rife_hand.onnx (%d B)" % os.path.getsize("rife_hand.onnx"))
