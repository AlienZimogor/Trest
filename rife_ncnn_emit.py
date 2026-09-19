#!/usr/bin/env python3
"""Прямой эмиттер ncnn param/bin из rife_v4.26.pkl. БЕЗ torch.onnx и БЕЗ pnnx.
STAGE=1: encode + block0 (словарь слоёв, уже доказанный на устройстве), выход = flow0(4ch).
STAGE=2 (после зелёного бисекта stage1): добавляет rife.Warp и блоки b1..b4 + бленд.
CI-самопроверка формата встроена: несоответствие = ненулевой код выхода."""
import os, glob, struct, sys
import numpy as np
import torch

STAGE = int(os.environ.get("EMIT_STAGE", "1"))
FACT = [16, 8, 4, 2, 1]
C0 = [96, 64, 48, 32, 16]
C1 = [192, 128, 96, 64, 32]


def _pref(p):
    s = 0; b = os.path.basename(p).lower()
    if "4.26" in b or "426" in b: s += 4
    if "v4" in b: s += 1
    if "flownet" in b: s -= 3
    return s


pkls = sorted(glob.glob(os.path.join("model_src", "**", "*.pkl"), recursive=True)
              + glob.glob(os.path.join("model_src", "**", "*.pth"), recursive=True),
              key=_pref, reverse=True)
raw = torch.load(pkls[0], map_location="cpu", weights_only=False)
if isinstance(raw, dict) and "state_dict" in raw: raw = raw["state_dict"]
sd = {}
for k, v in raw.items():
    k2 = k
    while k2.startswith("module."): k2 = k2[len("module."):]
    sd[k2] = v


class E:
    def __init__(self):
        self.lines = []; self.blobs = []; self.idx = {}; self.bin = bytearray()

    def b(self, n):
        if n not in self.idx:
            self.idx[n] = len(self.blobs); self.blobs.append(n)
        return n

    def L(self, typ, name, bottoms, tops, params="", weights=()):
        for x in bottoms: self.b(x)
        for x in tops: self.b(x)
        self.lines.append("%s %s %d %d %s%s" % (
            typ, name, len(bottoms), len(tops),
            " ".join(list(bottoms) + list(tops)),
            (" " + params) if params else ""))
        for w in weights:
            a = np.ascontiguousarray(w.detach().cpu().numpy().reshape(-1), dtype="<f4")
            self.bin += struct.pack("<i", 0) + a.tobytes()

    def conv(self, name, bot, top, w, bias, s):
        o, i, k, _ = w.shape
        self.L("Convolution", name, [bot], [top],
               "0=%d 1=%d 11=%d 12=1 13=%d 14=%d 2=1 3=%d 4=%d 5=1 6=%d"
               % (o, k, k, s, (k - 1) // 2, s, (k - 1) // 2, o * i * k * k), (w, bias))

    def deconv(self, name, bot, top, w, bias):
        i, o, k, _ = w.shape
        self.L("Deconvolution", name, [bot], [top],
               "0=%d 1=%d 11=%d 12=1 13=2 14=1 2=1 3=2 4=1 5=1 6=%d"
               % (o, k, k, i * o * k * k), (w, bias))

    def write(self, pp, pb):
        open(pp, "w").write("7767577\n%d %d\n" % (len(self.lines), len(self.blobs))
                            + "\n".join(self.lines) + "\n")
        open(pb, "wb").write(bytes(self.bin))
        self.selfcheck(pp, pb)

    def selfcheck(self, pp, pb):
        txt = open(pp).read().splitlines()
        assert txt[0] == "7767577", "bad magic"
        lc, bc = map(int, txt[1].split())
        assert lc == len(self.lines) == len(txt) - 2, "layer count mismatch"
        assert bc == len(self.blobs), "blob count mismatch"
        need = 0
        for ln in txt[2:]:
            t = ln.split()
            if t[0] in ("Convolution", "Deconvolution"):
                d = dict(kv.split("=") for kv in t[4 + int(t[2]) + int(t[3]):])
                need += 4 + int(d["6"]) * 4
                if d.get("5") == "1":
                    need += 4 + int(d["0"]) * 4
        assert need == os.path.getsize(pb), "bin size mismatch %d != %d" % (need, os.path.getsize(pb))
        print("SELFCHECK OK layers=%d blobs=%d bin=%d" % (lc, bc, need))


e = E()
e.L("Input", "in0", [], ["in0"])
e.L("Slice", "split_in", ["in0"], ["img0", "img1", "ts"], "-23300=3,3,1 1=0")

for tag, src in (("A", "img0"), ("B", "img1")):
    e.conv("e0_" + tag, src, "e0" + tag, sd["encode.cnn0.weight"], sd["encode.cnn0.bias"], 2)
    e.conv("e1_" + tag, "e0" + tag, "e1" + tag, sd["encode.cnn1.weight"], sd["encode.cnn1.bias"], 1)
    e.conv("e2_" + tag, "e1" + tag, "e2" + tag, sd["encode.cnn2.weight"], sd["encode.cnn2.bias"], 1)
    e.deconv("ed_" + tag, "e2" + tag, "d" + tag, sd["encode.cnn3.weight"], sd["encode.cnn3.bias"])

e.L("Concat", "cat0", ["img0", "img1", "dA", "dB", "ts"], ["c0in"], "0=0")
e.L("Interp", "c0down", ["c0in"], ["c0d"], "0=2 1=0.0625 2=0.0625 5=0")
e.conv("b0c0", "c0d", "b0c0o", sd["block0.conv0.0.0.weight"], sd["block0.conv0.0.0.bias"], 2)
e.L("ReLU", "b0r0", ["b0c0o"], ["b0r0o"])
e.conv("b0c1", "b0r0o", "b0c1o", sd["block0.conv0.1.0.weight"], sd["block0.conv0.1.0.bias"], 2)
e.L("ReLU", "b0r1", ["b0c1o"], ["x0"])
x = "x0"
for n in range(8):
    cw = sd["block0.convblock.%d.conv.weight" % n]
    cbias = sd["block0.convblock.%d.conv.bias" % n]
    beta = sd["block0.convblock.%d.beta" % n]
    e.conv("b0cb%d" % n, x, "b0cb%do" % n, cw, cbias, 1)
    e.L("MemoryData", "b0cb%db" % n, [], ["b0cb%db" % n], "0=1 1=1 2=%d" % beta.numel(), (beta,))
    e.L("BinaryOp", "b0cb%dm" % n, ["b0cb%do" % n, "b0cb%db" % n], ["b0cb%dm" % n], "0=2 1=0 2=0")
    e.L("BinaryOp", "b0cb%da" % n, ["b0cb%dm" % n, x], ["b0cb%da" % n], "0=0 1=0 2=0")
    e.L("ReLU", "b0cb%dl" % n, ["b0cb%da" % n], ["b0cb%dl" % n], "0=0.2")
    x = "b0cb%dl" % n
e.deconv("b0last", x, "b0lo", sd["block0.lastconv.0.weight"], sd["block0.lastconv.0.bias"])
e.L("PixelShuffle", "b0ps", ["b0lo"], ["b0pso"], "0=2")
e.L("Interp", "b0up", ["b0pso"], ["b0f13"], "0=2 1=16 2=16 5=0")
e.L("Slice", "b0sp", ["b0f13"], ["flow0", "mask0", "feat0"], "-23300=4,1,8 1=0")

if STAGE == 1:
    e.write("rife_hand.ncnn.param", "rife_hand.ncnn.bin")
    print("STAGE1 emitted (no warp). Output blob = flow0")
else:
    print("STAGE2 not implemented yet; run STAGE=1 first and gate it by bisect")
    sys.exit(2)
