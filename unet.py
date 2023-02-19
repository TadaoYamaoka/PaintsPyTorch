import torch
import torch.nn as nn
import torch.nn.functional as F


class UNET(nn.Module):

    def __init__(self):
        super().__init__()
        self.c0 = nn.Conv2d(4, 32, 3, 1, 1)
        self.c1 = nn.Conv2d(32, 64, 4, 2, 1)
        self.c2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.c3 = nn.Conv2d(64, 128, 4, 2, 1)
        self.c4 = nn.Conv2d(128, 128, 3, 1, 1)
        self.c5 = nn.Conv2d(128, 256, 4, 2, 1)
        self.c6 = nn.Conv2d(256, 256, 3, 1, 1)
        self.c7 = nn.Conv2d(256, 512, 4, 2, 1)
        self.c8 = nn.Conv2d(512, 512, 3, 1, 1)

        self.dc8 = nn.ConvTranspose2d(1024, 512, 4, 2, 1)
        self.dc7 = nn.Conv2d(512, 256, 3, 1, 1)
        self.dc6 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.dc5 = nn.Conv2d(256, 128, 3, 1, 1)
        self.dc4 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.dc3 = nn.Conv2d(128, 64, 3, 1, 1)
        self.dc2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.dc1 = nn.Conv2d(64, 32, 3, 1, 1)
        self.dc0 = nn.Conv2d(64, 3, 3, 1, 1)

        self.bnc0 = nn.BatchNorm2d(32)
        self.bnc1 = nn.BatchNorm2d(64)
        self.bnc2 = nn.BatchNorm2d(64)
        self.bnc3 = nn.BatchNorm2d(128)
        self.bnc4 = nn.BatchNorm2d(128)
        self.bnc5 = nn.BatchNorm2d(256)
        self.bnc6 = nn.BatchNorm2d(256)
        self.bnc7 = nn.BatchNorm2d(512)
        self.bnc8 = nn.BatchNorm2d(512)

        self.bnd8 = nn.BatchNorm2d(512)
        self.bnd7 = nn.BatchNorm2d(256)
        self.bnd6 = nn.BatchNorm2d(256)
        self.bnd5 = nn.BatchNorm2d(128)
        self.bnd4 = nn.BatchNorm2d(128)
        self.bnd3 = nn.BatchNorm2d(64)
        self.bnd2 = nn.BatchNorm2d(64)
        self.bnd1 = nn.BatchNorm2d(32)

    def forward(self, x):
        e0 = F.relu(self.bnc0(self.c0(x)))
        e1 = F.relu(self.bnc1(self.c1(e0)))
        e2 = F.relu(self.bnc2(self.c2(e1)))
        del e1
        e3 = F.relu(self.bnc3(self.c3(e2)))
        e4 = F.relu(self.bnc4(self.c4(e3)))
        del e3
        e5 = F.relu(self.bnc5(self.c5(e4)))
        e6 = F.relu(self.bnc6(self.c6(e5)))
        del e5
        e7 = F.relu(self.bnc7(self.c7(e6)))
        e8 = F.relu(self.bnc8(self.c8(e7)))

        d8 = F.relu(self.bnd8(self.dc8(torch.cat([e7, e8], dim=1))))
        del e7, e8
        d7 = F.relu(self.bnd7(self.dc7(d8)))
        del d8
        d6 = F.relu(self.bnd6(self.dc6(torch.cat([e6, d7], dim=1))))
        del d7, e6
        d5 = F.relu(self.bnd5(self.dc5(d6)))
        del d6
        d4 = F.relu(self.bnd4(self.dc4(torch.cat([e4, d5], dim=1))))
        del d5, e4
        d3 = F.relu(self.bnd3(self.dc3(d4)))
        del d4
        d2 = F.relu(self.bnd2(self.dc2(torch.cat([e2, d3], dim=1))))
        del d3, e2
        d1 = F.relu(self.bnd1(self.dc1(d2)))
        del d2
        d0 = self.dc0(torch.cat([e0, d1], dim=1))

        return d0


class DIS(nn.Module):

    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 32, 4, 2, 1)
        self.c2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.c3 = nn.Conv2d(32, 64, 4, 2, 1)
        self.c4 = nn.Conv2d(64, 64, 3, 1, 1)
        self.c5 = nn.Conv2d(64, 128, 4, 2, 1)
        self.c6 = nn.Conv2d(128, 128, 3, 1, 1)
        self.c7 = nn.Conv2d(128, 256, 4, 2, 1)
        self.l8l = nn.Linear(8 * 8 * 256, 2)

        self.bnc1 = nn.BatchNorm2d(32)
        self.bnc2 = nn.BatchNorm2d(32)
        self.bnc3 = nn.BatchNorm2d(64)
        self.bnc4 = nn.BatchNorm2d(64)
        self.bnc5 = nn.BatchNorm2d(128)
        self.bnc6 = nn.BatchNorm2d(128)
        self.bnc7 = nn.BatchNorm2d(256)

    def forward(self, x):
        h = F.relu(self.bnc1(self.c1(x)))
        h = F.relu(self.bnc2(self.c2(h)))
        h = F.relu(self.bnc3(self.c3(h)))
        h = F.relu(self.bnc4(self.c4(h)))
        h = F.relu(self.bnc5(self.c5(h)))
        h = F.relu(self.bnc6(self.c6(h)))
        h = F.relu(self.bnc7(self.c7(h)))
        return self.l8l(h.flatten(1))
