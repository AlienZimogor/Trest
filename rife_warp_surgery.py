import sys

src = sys.argv[1] if len(sys.argv) > 1 else "rife_sim.ncnn.param"
dst = sys.argv[2] if len(sys.argv) > 2 else "rife_warp.param"

FLOW = {
    "permutegridsample_0": "81",
    "permutegridsample_1": "159",
    "permutegridsample_2": "225",
    "permutegridsample_3": "291",
    "permutegridsample_4": "350",
}

lines = open(src).read().splitlines()
magic = lines[0]
nL, nB = map(int, lines[1].split())
body = lines[2:]
out = []
inserted = False
for ln in body:
    f = ln.split()
    if not f:
        out.append(ln); continue
    typ, name = f[0], f[1]
    if typ == "GridSample" and name in FLOW:
        nb, nt = int(f[2]), int(f[3])
        bottoms = f[4:4+nb]; tops = f[4+nb:4+nb+nt]
        x, o = bottoms[0], tops[0]
        wname = name.replace("permutegridsample", "warp")
        out.append("rife.Warp %s 2 1 %s %s %s" % (wname, x, FLOW[name], o))
        continue
    if typ == "Convolution" and name == "convrelu_0":
        f[4] = "400"
        out.append(" ".join(f)); continue
    if typ == "Reshape" and name == "reshape_133" and not inserted:
        out.append(ln)
        out.append("Slice slice_i0 1 1 11 400 -23300=1,3 1=0")
        inserted = True
        continue
    out.append(ln)
assert inserted, "reshape_133 not found"
nL += 1; nB += 1
with open(dst, "w") as fh:
    fh.write("\n".join([magic, "%d %d" % (nL, nB)] + out) + "\n")
print("wrote", dst, "layers", nL, "blobs", nB)
