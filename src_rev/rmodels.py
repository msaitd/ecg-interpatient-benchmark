"""
Models for the revision experiments.

  cnn    : the original lightweight 1D-CNN (morphology only), four-class head
  cnnrr  : the same CNN with RR-interval fusion (5 RR features -> 16-unit branch)
  resnet : a deeper 1D residual network (~0.5 M parameters) used as a comparator
All models accept (x, rr) and can return the last convolutional feature map for Grad-CAM.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import rconfig as C


class ECGNet(nn.Module):
    def __init__(self, n_classes=C.K, n_rr=0):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
        )
        self.n_rr = n_rr
        if n_rr:
            self.rr_branch = nn.Sequential(nn.Linear(n_rr, 16), nn.ReLU())
            self.head = nn.Linear(64 + 16, n_classes)
        else:
            self.head = nn.Linear(64, n_classes)

    def forward(self, x, rr=None, return_feats=False):
        A = self.body(x)                    # (B, 64, 70) last conv feature map
        z = A.mean(-1)                      # global average pooling
        if self.n_rr:
            z = torch.cat([z, self.rr_branch(rr)], dim=1)
        out = self.head(z)
        return (out, A) if return_feats else out


class BasicBlock1d(nn.Module):
    def __init__(self, cin, cout, stride=1, k=7):
        super().__init__()
        p = k // 2
        self.c1 = nn.Conv1d(cin, cout, k, stride=stride, padding=p, bias=False)
        self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, k, padding=p, bias=False)
        self.b2 = nn.BatchNorm1d(cout)
        self.sc = None
        if stride != 1 or cin != cout:
            self.sc = nn.Sequential(nn.Conv1d(cin, cout, 1, stride=stride, bias=False),
                                    nn.BatchNorm1d(cout))

    def forward(self, x):
        o = F.relu(self.b1(self.c1(x)))
        o = self.b2(self.c2(o))
        return F.relu(o + (x if self.sc is None else self.sc(x)))


class ResNet1D(nn.Module):
    def __init__(self, n_classes=C.K, k=7):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(1, 32, k, padding=k // 2, bias=False),
                                  nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2))
        self.layers = nn.Sequential(
            BasicBlock1d(32, 32, 1, k), BasicBlock1d(32, 32, 1, k),
            BasicBlock1d(32, 64, 2, k), BasicBlock1d(64, 64, 1, k),
            BasicBlock1d(64, 128, 2, k), BasicBlock1d(128, 128, 1, k))
        self.head = nn.Linear(128, n_classes)
        self.n_rr = 0

    def forward(self, x, rr=None, return_feats=False):
        A = self.layers(self.stem(x))
        out = self.head(A.mean(-1))
        return (out, A) if return_feats else out


def build(family):
    if family == "cnn":
        return ECGNet()
    if family == "cnnrr":
        return ECGNet(n_rr=5)
    if family == "resnet":
        return ResNet1D()
    raise ValueError(family)


def n_params(model):
    return int(sum(p.numel() for p in model.parameters()))
