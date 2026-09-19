#!/usr/bin/env python3
"""Минимальная ручная сборка ncnn param/bin для проверки rife.Warp на устройстве.
Граф: Input(7) -> Slice[3|4] -> Slice[2] -> rife.Warp(x=3ch, flow=2ch) -> out(3ch).
Весов нет, поэтому bin пустой."""
import os

PARAM = """7767577
5 5
Input in0 0 1 in0
Slice s0 1 2 in0 r0 r1 -23300=1,3 1=0
Slice s1 1 1 r1 f0 -23300=1,2 1=0
rife.Warp w0 2 1 r0 f0 out0
"""

with open("rife_warp.ncnn.param", "w") as f:
    f.write(PARAM)
with open("rife_warp.ncnn.bin", "wb") as f:
    pass  # нет весовых слоёв
print("wrote rife_warp.ncnn.param / .bin")
