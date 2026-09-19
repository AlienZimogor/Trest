#!/usr/bin/env python3
"""Прямой эмиттер ncnn param/bin для RIFE v4.26 (без onnx/pnnx).
Warp = кастомный слой rife.Warp; beta convblock'ов = MemoryData."""
import os, glob, struct
import torch


def _pref(p):
    s = 0; b = os.path.basename(p).lower()
    if "4.26" in b or "426" in b: s += 4
    if "v4" in b: s += 1
    if "flownet" in b: s -= 3
    return s


pkls = sorted(glob.glob(os.path.join("model_src", "**", "*.pkl"), recursive=True),
              key=_pref, reverse=True)
raw = torch.load(pkls[0], map_location="cpu", weights_only=False)
if isinstance(raw, dict) and "state_dict" in raw: raw = raw["state_dict"]
sd = {}
for k, v in raw.items():
    k2 = k
    while k2.startswith("module."): k2 = k2[7:]
    sd[k2] = v


class E:
    def __init__(self):
        self.lines = []; self.bin = bytearray(); self.blobs = {}

    def b(self, n):
        if n not in self.blobs: self.blobs[n] = len(self.blobs)
        return n

    def _w(self, t):
        a = t.detach().cpu().contiguous().numpy().astype("<f4")
        self.bin += struct.pack("<i", 0) + a.tobytes()

    def L(self, typ, name, bo, to, params=""):
        for x in bo: self.b(x)
        for x in to: self.b(x)
        self.lines.append("%s %s %d %d %s%s" % (
            typ, name, len(bo), len(to), " ".join(list(bo) + list(to)),
            (" " + params) if params else ""))

    def conv(self, name, bo, to, w, bias, stride):
        o, i, k, _ = w.shape
        p = (k - 1) // 2
        self.L("Convolution", name, [bo], [to],
               "0=%d 1=%d 11=%d 12=1 13=%d 14=%d 2=1 3=%d 4=%d 5=%d 6=%d"
               % (o, k, k, stride, p, stride, p, 1 if bias is not None else 0, o * i * k * k))
        self._w(w)
        if bias is not None: self._w(bias)

    def deconv(self, name, bo, to, w, bias):
        i, o, k, _ = w.shape
        self.L("Deconvolution", name, [bo], [to],
               "0=%d 1=%d 11=%d 12=1 13=2 14=1 2=1 3=2 4=1 5=%d 6=%d"
               % (o, k, k, 1 if bias is not None else 0, i * o * k * k))
        self._w(w)
        if bias is not None: self._w(bias)

    def memdata(self, name, to, arr):
        self.L("MemoryData", name, [], [to], "0=1 1=1 2=%d" % arr.numel())
        self._w(arr)

    def write(self, pp, pb):
        with open(pp, "w") as f:
            f.write("7767577\n%d %d\n" % (len(self.lines), len(self.blobs)))
            for ln in self.lines: f.write(ln + "\n")
        with open(pb, "wb") as f:
            f.write(bytes(self.bin))
        self.selfcheck(pp, pb)

    def selfcheck(self, pp, pb):
        txt = open(pp).read().splitlines()
        assert txt[0] == "7767577", "bad magic"
        lc, bc = map(int, txt[1].split())
        assert lc == len(self.lines) and bc == len(self.blobs), "counts mismatch"
        assert len(txt) - 2 == lc, "layer lines mismatch"
        need = 0
        for ln in txt[2:]:
            t = ln.split()
            nb, nt = int(t[2]), int(t[3])
            d = {}
            for kv in t[4 + nb + nt:]:
                k, _, v = kv.partition("="); d[k] = v
            if t[0] in ("Convolution", "Deconvolution"):
                need += 4 + int(d["6"]) * 4
                if d.get("5") == "1":
                    need += 4 + int(d["0"]) * 4
            elif t[0] == "MemoryData":
                need += 4 + int(d["0"]) * int(d["1"]) * int(d["2"]) * 4
        got = os.path.getsize(pb)
        assert need == got, "bin size mismatch need=%d got=%d" % (need, got)
        assert need == len(self.bin), "bin internal mismatch need=%d len=%d" % (need, len(self.bin))
        print("SELFCHECK OK layers=%d blobs=%d bin=%d" % (lc, bc, got))


e = E()
e.L("Input", "in0", [], ["in0"])
e.L("Slice", "split_in", ["in0"], ["img0", "img1", "ts"], "-23300=3,3,1 1=0")

for tag, src in (("A", "img0"), ("B", "img1")):
    e.conv("e0_" + tag, src, "e0o" + tag, sd["encode.cnn0.weight"], sd["encode.cnn0.bias"], 2)
    e.conv("e1_" + tag, "e0o" + tag, "e1o" + tag, sd["encode.cnn1.weight"], sd["encode.cnn1.bias"], 1)
    e.conv("e2_" + tag, "e1o" + tag, "e2o" + tag, sd["encode.cnn2.weight"], sd["encode.cnn2.bias"], 1)
    e.deconv("ed_" + tag, "e2o" + tag, "d" + tag, sd["encode.cnn3.weight"], sd["encode.cnn3.bias"])

e.L("Concat", "cat0", ["img0", "img1", "dA", "dB", "ts"], ["cat0o"], "0=0")

FACT = [16, 8, 4, 2, 1]
C0 = [96, 64, 48, 32, 16]
C1 = [192, 128, 96, 64, 32]
pf = pm = pft = None

for i, (f, c0, c1) in enumerate(zip(FACT, C0, C1)):
    pre = "b%d" % i
    if i == 0:
        src = "cat0o"
    else:
        e.L("Slice", pre + "_fs", [pf], [pre + "_fa", pre + "_fb"], "-23300=2,2 1=0")
        e.L("rife.Warp", pre + "_w0", ["img0", pre + "_fa"], [pre + "_wi0"])
        e.L("rife.Warp", pre + "_w1", ["img1", pre + "_fb"], [pre + "_wi1"])
        e.L("rife.Warp", pre + "_w2", ["dA", pre + "_fa"], [pre + "_wd0"])
        e.L("rife.Warp", pre + "_w3", ["dB", pre + "_fb"], [pre + "_wd1"])
        e.L("Concat", pre + "_cat",
            [pre + "_wi0", pre + "_wi1", pre + "_wd0", pre + "_wd1", "ts", pm, pft, pf],
            [pre + "_catin"], "0=0")
        src = pre + "_catin"
    e.L("Interp", pre + "_down", [src], [pre + "_d"], "0=2 1=%f 2=%f 5=0" % (1.0 / f, 1.0 / f))
    e.conv(pre + "_c0", pre + "_d", pre + "_c0o",
           sd["block%d.conv0.0.0.weight" % i], sd["block%d.conv0.0.0.bias" % i], 2)
    e.L("ReLU", pre + "_r0", [pre + "_c0o"], [pre + "_r0o"])
    e.conv(pre + "_c1", pre + "_r0o", pre + "_c1o",
           sd["block%d.conv0.1.0.weight" % i], sd["block%d.conv0.1.0.bias" % i], 2)
    e.L("ReLU", pre + "_r1", [pre + "_c1o"], [pre + "_x"])
    x = pre + "_x"
    for n in range(8):
        cw = sd["block%d.convblock.%d.conv.weight" % (i, n)]
        cb = sd["block%d.convblock.%d.conv.bias" % (i, n)]
        bt = sd["block%d.convblock.%d.beta" % (i, n)]
        e.conv(pre + "_cb%dc" % n, x, pre + "_cb%do" % n, cw, cb, 1)
        e.memdata(pre + "_cb%db" % n, pre + "_cb%db" % n, bt)
        e.L("BinaryOp", pre + "_cb%dm" % n, [pre + "_cb%do" % n, pre + "_cb%db" % n],
            [pre + "_cb%dm" % n], "0=2 1=0 2=0")
        e.L("BinaryOp", pre + "_cb%da" % n, [pre + "_cb%dm" % n, x],
            [pre + "_cb%da" % n], "0=0 1=0 2=0")
        e.L("ReLU", pre + "_cb%dl" % n, [pre + "_cb%da" % n], [pre + "_cb%dl" % n], "0=0.2")
        x = pre + "_cb%dl" % n
    e.deconv(pre + "_last", x, pre + "_lo",
             sd["block%d.lastconv.0.weight" % i], sd["block%d.lastconv.0.bias" % i])
    e.L("PixelShuffle", pre + "_ps", [pre + "_lo"], [pre + "_pso"], "0=2")
    e.L("Interp", pre + "_up", [pre + "_pso"], [pre + "_upo"], "0=2 1=%f 2=%f 5=0" % (float(f), float(f)))
    e.L("Slice", pre + "_sp", [pre + "_upo"],
        [pre + "_flow", pre + "_mask", pre + "_feat"], "-23300=4,1,8 1=0")
    pf, pm, pft = pre + "_flow", pre + "_mask", pre + "_feat"

e.L("Slice", "fin_fs", [pf], ["fin_fa", "fin_fb"], "-23300=2,2 1=0")
e.L("rife.Warp", "fin_w0", ["img0", "fin_fa"], ["fin_w0"])
e.L("rife.Warp", "fin_w1", ["img1", "fin_fb"], ["fin_w1"])
e.L("BinaryOp", "fin_m1", ["fin_w0", pm], ["fin_m1"], "0=2 1=0 2=0")
e.L("BinaryOp", "fin_d1", [pm], ["fin_d1"], "0=1 1=1 2=1.0")
e.L("BinaryOp", "fin_m2", ["fin_w1", "fin_d1"], ["fin_m2"], "0=2 1=0 2=0")
e.L("BinaryOp", "fin_m3", ["fin_m2"], ["fin_m3"], "0=2 1=1 2=-1.0")
e.L("BinaryOp", "fin_out", ["fin_m1", "fin_m3"], ["out"], "0=0 1=0 2=0")

e.write("rife_hand.ncnn.param", "rife_hand.ncnn.bin")
print("emitted layers=%d" % len(e.lines))
