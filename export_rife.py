import sys
import torch
import torch.nn as nn

sys.path.insert(0, "Practical-RIFE")
from model.IFNet import IFNet


class W(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        img0 = x[:, 0:3]
        img1 = x[:, 3:6]
        t = x[:, 6:7]
        try:
            return self.net(img0, img1, timestep=t)
        except TypeError:
            return self.net(img0, img1, t)


def main():
    pkl = sys.argv[1]
    net = IFNet()
    sd = torch.load(pkl, map_location="cpu")
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
