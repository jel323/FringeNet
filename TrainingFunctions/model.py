import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
from prettytable import PrettyTable as pt


### Functions on Models
def count_params(model: nn.Module):
    table = pt(["Modules", "Parameters"])
    t_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        t_params += params
    print(table)
    print(f"Total Trainable Parameters : {t_params}")
    return


### Models
class RecurrentConv(nn.Module):
    def name(self):
        return "Recurrent Convolution"

    def __init__(self, channels, kernel_size, n=2, bn=True):
        super().__init__()
        self.n = n
        self.conv = nn.Sequential(nn.Conv2d(channels, channels, kernel_size, stride=1))
        if bn:
            self.conv.append(nn.BatchNorm2d(channels))
        self.conv.append(nn.ReLU(inplace=True))
        return

    def forward(self, x):
        x1 = self.conv(x)
        for k in range(1, self.n):
            x1 = self.conv(x[:, :, k:-k, k:-k] + x1)
        return x1


class ConvBlock(nn.Module):
    # (convolution => [BN] => ReLU) * 2
    def name(self):
        return "Convolution Block"

    def __init__(
        self,
        in_channels,
        out_channels,
        nconvs=2,
        kernel_size=3,
        init_kernel_size=3,
        bn=True,
    ):
        super().__init__()
        self.sfix = int(nconvs * (kernel_size - 1) / 2)
        self.convs = nn.Sequential()
        self.convs.append(nn.Conv2d(in_channels, out_channels, init_kernel_size))
        if bn:
            self.convs.append(nn.BatchNorm2d(out_channels))
        for _ in range(nconvs - 1):
            self.convs.append(nn.LeakyReLU(0.2, inplace=True))
            self.convs.append(nn.Conv2d(out_channels, out_channels, kernel_size))
            if bn:
                self.convs.append(nn.BatchNorm2d(out_channels))
        self.end = nn.LeakyReLU(0.2, inplace=True)
        self.featurematch = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x1 = self.convs(x)
        x2 = self.featurematch(x)[:, :, self.sfix : -self.sfix, self.sfix : -self.sfix]
        return self.end(x1 + x2)


class RecurrentBlock(nn.Module):
    # (convolution => [BN] => ReLU) * 2
    def name(self):
        return "Recurrent Convolution Block"

    def __init__(
        self,
        in_channels,
        out_channels,
        nconvs=2,
        nrepititions=2,
        kernel_size=3,
        init_kernel_size=3,
        bn=True,
    ):
        super().__init__()
        self.sfix = int((1 + (nconvs - 1) * nrepititions) * (kernel_size - 1) / 2)
        self.conv = nn.Sequential()
        self.conv.append(nn.Conv2d(in_channels, out_channels, init_kernel_size))
        if bn:
            self.conv.append(nn.BatchNorm2d(out_channels))
        self.conv.append(nn.ReLU(inplace=True))
        self.convs = nn.Sequential()
        for _ in range(nconvs - 1):
            self.convs.append(
                RecurrentConv(out_channels, kernel_size, nrepititions, bn)
            )
        self.end = nn.ReLU(inplace=True)
        self.featurematch = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x1 = self.conv(x)
        x1 = self.convs(x1)
        x2 = self.featurematch(x)[:, :, self.sfix : -self.sfix, self.sfix : -self.sfix]
        return self.end(x1 + x2)


class AttentionBlock1(nn.Module):
    def name(self):
        return "Attention Block"

    def __init__(self, in_channels_g, in_channels_l, out_channels, bn=True):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(in_channels_g, out_channels, kernel_size=1, stride=1, padding=0)
        )
        if bn:
            self.W_g.append(nn.BatchNorm2d(out_channels))
        self.W_l = nn.Sequential(
            nn.Conv2d(in_channels_l, out_channels, kernel_size=1, stride=1, padding=0)
        )
        if bn:
            self.W_l.append(nn.BatchNorm2d(out_channels))
        self.sig = nn.Sequential(
            nn.Conv2d(out_channels, 1, kernel_size=1, stride=1, padding=0)
        )
        if bn:
            self.sig.append(nn.BatchNorm2d(1))
        self.sig.append(nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)
        return

    def forward(self, g, l):
        g1 = self.W_g(g)
        l1 = self.W_l(l)
        x = self.relu(g1 + l1)
        x = self.sig(x)
        return l * x


class ConvBlockOld(nn.Module):
    # (convolution => [BN] => ReLU) * 2
    def name(self):
        return "DoubleConvolution"

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    # Downscaling with maxpool then double conv
    def name(self):
        return "Down"

    def __init__(
        self,
        in_channels,
        out_channels,
        nconvs=2,
        nrepititions=2,
        kernel_size=3,
        init_kernel_size=3,
        bn=True,
        recurrent=False,
        dropout=False,
        pconv=True,
    ):
        super().__init__()
        if recurrent:
            self.conv = RecurrentBlock(
                in_channels,
                out_channels,
                nconvs,
                nrepititions,
                kernel_size,
                init_kernel_size,
                bn,
            )
        else:
            self.conv = ConvBlock(
                in_channels,
                out_channels,
                nconvs,
                kernel_size,
                init_kernel_size,
                bn,
            )
        self.out = nn.Sequential()
        if pconv:
            self.out.append(nn.Conv2d(out_channels, out_channels, 2, 2))
        else:
            self.out.append(nn.MaxPool2d(2, stride=2))
        if dropout:
            self.out.append(nn.Dropout2d(0.1, inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return self.out(x), x


class Up(nn.Module):
    # Upscaling then double conv
    def name(self):
        return "Up"

    def __init__(
        self,
        in_channels,
        out_channels,
        nconvs=2,
        bn=True,
        recurrent=False,
        dropout=False,
    ):
        super().__init__()
        self.inc = in_channels
        self.ouc = out_channels
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.out = nn.Sequential()
        if dropout:
            self.out.append(nn.Dropout2d(0.05, inplace=True))
        if recurrent:
            self.out.append(RecurrentBlock(in_channels, out_channels, nconvs, bn=bn))
        else:
            self.out.append(
                ConvBlock(
                    in_channels,
                    out_channels,
                    nconvs,
                    bn=bn,
                )
            )
        """self.attention = AttentionBlock1(
            out_channels, out_channels, int(out_channels / 2), bn
        )"""

    def forward(self, x1, x2):
        x1 = self.up(x1)
        _, _, W, H = x1.size()
        Crop = transforms.CenterCrop(size=(W, H))
        """x2 = self.attention(x1, Crop(x2))
        _, _, W, H = x1.size()
        Crop = transforms.CenterCrop(size=(W, H))"""
        x = torch.cat((x1, Crop(x2)), dim=1)
        return self.out(x)


class OutConv(nn.Module):
    def name(self):
        return "Out Convolution"

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.out = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.out(x)


class UNetFront(nn.Module):
    def name(self):
        return "UNet Front"

    def __init__(
        self,
        n_channels,
        n_classes,
        n_layers=4,
        init_kernel_size=3,
        kernel_size=3,
        nconvs=2,
        nrepititions=2,
        bn=True,
        recurrent=False,
        dropout=False,
    ):
        super(UNetFront, self).__init__()
        self.n_layers = n_layers
        self.down = nn.ModuleList()
        self.down.append(
            Down(
                n_channels,
                64,
                nconvs,
                nrepititions,
                kernel_size,
                init_kernel_size,
                bn=bn,
                recurrent=recurrent,
                dropout=dropout,
                pconv=True,
            )
        )
        for k in range(n_layers - 1):
            self.down.append(
                Down(
                    (2**k) * 64,
                    (2 ** (k + 1) * 64),
                    nconvs,
                    kernel_size,
                    bn=bn,
                    recurrent=recurrent,
                    dropout=dropout,
                    pconv=True,
                )
            )
        if recurrent:
            self.mid = RecurrentBlock(
                (2 ** (k + 1)) * 64,
                (2 ** (k + 2)) * 64,
                nconvs,
                nrepititions,
                kernel_size,
                bn=bn,
            )
        else:
            self.mid = ConvBlock(
                (2 ** (k + 1)) * 64,
                (2 ** (k + 2)) * 64,
                nconvs,
                kernel_size,
                bn=bn,
            )
        return

    def forward(self, x):
        xs = []
        for k in range(self.n_layers):
            x, ph = self.down[k](x)
            xs.append(ph)
        x = self.mid(x)
        return x


class UNet(nn.Module):
    def name(self):
        return "UNet"

    def __init__(
        self,
        n_channels=3,
        n_classes=1,
        n_layers=4,
        mult=16,
        init_kernel_size=3,
        kernel_size=3,
        nconvs=2,
        nrepititions=2,
        bn=True,
        recurrent=False,
        recurrent_mid=False,
        dropout=False,
    ):
        super(UNet, self).__init__()
        self.n_layers = n_layers
        self.down = nn.ModuleList()
        self.down.append(
            Down(
                n_channels,
                mult,
                nconvs,
                nrepititions,
                kernel_size=kernel_size,
                init_kernel_size=init_kernel_size,
                bn=bn,
                recurrent=recurrent,
                dropout=dropout,
                pconv=True,
            )
        )
        for k in range(n_layers - 1):
            self.down.append(
                Down(
                    (2**k) * mult,
                    (2 ** (k + 1) * mult),
                    nconvs,
                    kernel_size,
                    bn=bn,
                    recurrent=recurrent,
                    dropout=dropout,
                    pconv=True,
                )
            )
        if recurrent_mid:
            self.mid = RecurrentBlock(
                (2 ** (k + 1)) * mult,
                (2 ** (k + 2)) * mult,
                1,
                nrepititions,
                kernel_size,
                bn=bn,
            )
        else:
            self.mid = ConvBlock(
                (2 ** (k + 1)) * mult,
                (2 ** (k + 2)) * mult,
                nconvs,
                kernel_size,
                bn=bn,
            )
        self.up = nn.ModuleList()
        a = [k for k in range(n_layers)]
        a.reverse()
        for k in a:
            self.up.append(
                Up(
                    (2 ** (k + 1)) * mult,
                    (2**k) * mult,
                    nconvs,
                    bn=bn,
                    recurrent=recurrent,
                    dropout=dropout,
                )
            )
        self.outc = OutConv(mult, n_classes)
        return

    def forward(self, x: torch.Tensor):
        xs = []
        for k in range(self.n_layers):
            x, ph = self.down[k](x)
            xs.append(ph)
        x = self.mid(x)
        for k in range(self.n_layers):
            x = self.up[k](x, xs[-k - 1])
        return self.outc(x)

    def hyperparams(self):
        return

    def partial(self, x: torch.Tensor, n: int):
        xs = []
        lr1 = range(self.n_layers) if n >= self.n_layers else range(n)
        for k in lr1:
            x, ph = self.down[k](x)
            xs.append(ph)
        if n >= 5:
            x = self.mid(x)
        lr2 = range(self.n_layers) if n >= 2 * self.n_layers + 1 else range(n - 5)
        for k in lr2:
            x, ph = self.up[k](x, xs[-k - 1])
        if n >= 2 * self.n_layers + 2:
            return self.outc(x)
        else:
            return x


def _fit_len(arr: np.ndarray, len: int, dtype=np.int32) -> np.ndarray:
    if arr.shape[0] != len:
        arr = np.ones((len,), dtype=dtype) * arr
    return arr


class NUNet(nn.Module):
    def name(self):
        return "NUNet"

    @staticmethod
    def default_params():
        return (2, 3, 1, np.array([4, 4]), np.array([16, 16]))

    def __init__(
        self,
        n_unets: int = 2,
        n_channels: int = 3,
        n_classes: int = 1,
        n_layers: np.ndarray = np.array([4, 3], dtype=np.int32),
        mult: np.ndarray = np.array([16], dtype=np.int32),
        init_kernel_size: np.ndarray = np.array([3], dtype=np.int32),
        kernel_size: int = 3,
        nconvs: np.ndarray = np.array([2], dtype=np.int32),
        nrepititions: np.ndarray = np.array([2], dtype=np.int32),
        bn=True,
        recurrent=False,
        recurrent_mid=False,
        dropout=False,
        img_size=(400, 600),
    ):
        super(NUNet, self).__init__()
        self.n_unets = n_unets
        self.n_channels = n_channels
        self.n_classes = n_classes
        n_layers = _fit_len(n_layers, n_unets)
        mult = _fit_len(mult, n_unets)
        init_kernel_size = _fit_len(init_kernel_size, n_unets)
        nconvs = _fit_len(nconvs, n_unets)
        nrepititions = _fit_len(nrepititions, n_unets)

        self.nets = nn.ModuleList()
        for k in range(n_unets):
            if k == 0:
                self.nets.append(
                    UNet(
                        n_channels,
                        n_classes,
                        n_layers=n_layers[k],
                        mult=mult[k],
                        init_kernel_size=init_kernel_size[k],
                        kernel_size=kernel_size,
                        nconvs=nconvs[k],
                        nrepititions=nrepititions[k],
                        bn=bn,
                        recurrent=recurrent,
                        recurrent_mid=recurrent_mid,
                        dropout=dropout,
                    )
                )
            else:
                self.nets.append(
                    UNet(
                        n_channels + n_classes,
                        n_classes,
                        n_layers=n_layers[k],
                        mult=mult[k],
                        init_kernel_size=init_kernel_size[k],
                        kernel_size=kernel_size,
                        nconvs=nconvs[k],
                        nrepititions=nrepititions[k],
                        bn=bn,
                        recurrent=recurrent,
                        recurrent_mid=recurrent_mid,
                        dropout=dropout,
                    )
                )
        self.pads = nn.ModuleList()
        if img_size == (400, 600):
            for k in range(self.n_unets):
                if n_layers[k] == 4:
                    self.pads.append(torchvision.transforms.Pad((94, 98)))
                elif n_layers[k] == 3:
                    self.pads.append(torchvision.transforms.Pad((46, 46)))
        elif img_size == (200, 250):
            for k in range(self.n_unets):
                if n_layers[k] == 4:
                    self.pads.append(torchvision.transforms.Pad((95, 94)))
                elif n_layers[k] == 3:
                    self.pads.append(torchvision.transforms.Pad((47, 46)))
        return

    def forward(self, x: torch.Tensor):
        x1 = self.nets[0](x)
        for k in range(1, self.n_unets):
            x2 = torch.empty(
                (x.size(0), self.n_channels + self.n_classes, x.size(2), x.size(3)),
                device=x.device,
            )
            x2[:, : self.n_channels, :, :] = x
            x2[:, -self.n_classes :, :, :] = self.pads[k - 1](x1)
            x1 = self.nets[k](x2)
        return x1

    def first(self, x: torch.Tensor):
        return self.nets[0](x)

    def partial(self, x: torch.Tensor, full: int, part: int):
        if full > 0:
            x1 = self.nets[0](x)
        else:
            return self.nets[0].partial(x, part)
        if full > self.n_unets:
            full = self.n_unets
        for k in range(1, full):
            x2 = torch.empty(
                (x.size(0), self.n_channels + self.n_classes, x.size(2), x.size(3)),
                device=x.device,
            )
            x2[:, : self.n_channels, :, :] = x
            x2[:, -self.n_classes :, :, :] = self.pad(x1)
            x1 = self.nets[k](x2)
        if part == 0:
            return x1
        else:
            x2 = torch.empty(
                (x.size(0), self.n_channels + self.n_classes, x.size(2), x.size(3)),
                device=x.device,
            )
            x2[:, : self.n_channels, :, :] = x
            x2[:, -self.n_classes :, :, :] = self.pad(x1)
            return self.nets[k + 1].partial(x2, part)
