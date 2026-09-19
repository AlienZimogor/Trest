#!/usr/bin/env python3
"""Заменяет все GridSample (которые pnnx сшил неверно) на rife.Warp(x, flow).
flow-блобы = все выходы Slice/Crop, кормящие Permute (10 шт = 5 блоков x A/B).
x-блобы = [img0, img1, d0, d1]; порядок warp'ов = [img0,img1,d0,d1] x 4 блока + [img0,img1]."""
import sys

PATH = sys.argv[1] if len(sys.argv) > 1 else "rife_hand.ncnn.param"

lines = open(PATH).read().splitlines()
header = lines[:2]
body = lines[2:]

recs = []
for ln in body:
    t = ln.split()
    if len(t) < 4:
        recs.append(None)
        continue
    typ, name = t[0], t[1]
    nb, nt = int(t[2]), int(t[3])
    bottoms = t[4:4 + nb]
    tops = t[4 + nb:4 + nb + nt]
    recs.append({"type": typ, "name": name, "bottoms": bottoms,
                 "tops": tops, "raw": ln})

perm_in = set()
for r in recs:
    if r and r["type"] == "Permute":
        perm_in.add(r["bottoms"][0])

flow2 = []
for r in recs:
    if r and r["type"] in ("Slice", "Crop"):
        for t in r["tops"]:
            if t in perm_in and t not in flow2:
                flow2.append(t)
print("flow slices:", len(flow2))
assert len(flow2) == 10, "expected 10 flow slices, got %d" % len(flow2)

deconvs = [r for r in recs if r and r["type"] == "Deconvolution"]
assert len(deconvs) >= 2, "need >=2 deconvs"
d0, d1 = deconvs[0]["tops"][0], deconvs[1]["tops"][0]

slice3 = None
for r in recs:
    if r and r["type"] == "Slice" and len(r["tops"]) == 3:
        slice3 = r
        break
assert slice3, "input 3-top slice not found"
img0, img1 = slice3["tops"][0], slice3["tops"][1]
x_cycle = [img0, img1, d0, d1]

gs = [r for r in recs if r and r["type"] == "GridSample"]
print("gridsamples:", len(gs))
assert len(gs) == 18, "expected 18 GridSample, got %d" % len(gs)

for i, g in enumerate(gs):
    if i < 16:
        b, pos = i // 4, i % 4
    else:
        b, pos = 4, i - 16
    x = x_cycle[pos]
    flow = flow2[2 * b + (1 if pos in (1, 3) else 0)]
    g["raw"] = "rife.Warp %s 2 1 %s %s %s" % (g["name"], x, flow, g["tops"][0])
    print("warp%02d <- x=%s flow=%s out=%s" % (i, x, flow, g["tops"][0]))

out = []
for r, ln in zip(recs, body):
    out.append(r["raw"] if r else ln)
open(PATH, "w").write("\n".join(header + out) + "\n")
print("surgery done:", PATH)
