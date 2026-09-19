#!/usr/bin/env python3
"""Заменяет все GridSample-узлы, которые pnnx сшил неверно, на rife.Warp (x, flow).
Порядок warp'ов в forward: [img0,img1,d0,d1] x 4 масштаба + [img0,img1] финал = 18.
Пары flow (A=:2, B=2:4) = 2-канальные Slice/Crop, чьи выходы кормят Permute."""
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

imgs = None
for r in recs:
    if r and r["type"] == "Concat" and len(r["bottoms"]) == 5:
        imgs = r["bottoms"][0:4]
        break
assert imgs, "cat_0 (5-input Concat) not found"

perm_in = set()
for r in recs:
    if r and r["type"] == "Permute":
        perm_in.add(r["bottoms"][0])

flow2 = [r["tops"][0] for r in recs
         if r and r["type"] in ("Slice", "Crop")
         and len(r["tops"]) == 1 and r["tops"][0] in perm_in]
assert len(flow2) == 10, "expected 10 flow slices, got %d" % len(flow2)

gs = [r for r in recs if r and r["type"] == "GridSample"]
assert len(gs) == 18, "expected 18 GridSample, got %d" % len(gs)

x_cycle = imgs
for i, r in enumerate(gs):
    if i < 16:
        s, pos = i // 4, i % 4
    else:
        s, pos = 4, i - 16
    x = x_cycle[pos]
    flow = flow2[2 * s + (1 if pos in (1, 3) else 0)]
    r["raw"] = "rife.Warp %s 2 1 %s %s %s" % (r["name"], x, flow, r["tops"][0])
    print("warp%02d <- x=%s flow=%s out=%s" % (i, x, flow, r["tops"][0]))

out = []
for r, ln in zip(recs, body):
    out.append(r["raw"] if r else ln)
open(PATH, "w").write("\n".join(header + out) + "\n")
print("surgery done:", PATH)
