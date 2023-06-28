import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
from prettytable import PrettyTable as pt


### Functions on Models
def count_params(model):
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
        n_channels,
        n_classes,
        n_layers=4,
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
                64,
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
        if recurrent_mid:
            self.mid = RecurrentBlock(
                (2 ** (k + 1)) * 64,
                (2 ** (k + 2)) * 64,
                1,
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
        self.up = nn.ModuleList()
        a = [k for k in range(n_layers)]
        a.reverse()
        for k in a:
            self.up.append(
                Up(
                    (2 ** (k + 1)) * 64,
                    (2**k) * 64,
                    nconvs,
                    bn=bn,
                    recurrent=recurrent,
                    dropout=dropout,
                )
            )
        self.outc = OutConv(64, n_classes)
        return

    def forward(self, x):
        xs = []
        for k in range(self.n_layers):
            x, ph = self.down[k](x)
            xs.append(ph)
        x = self.mid(x)
        for k in range(self.n_layers):
            x = self.up[k](x, xs[-k - 1])
        return self.outc(x)


class DConv(nn.Module):
    def name(self):
        return "Discriminator Convolution"

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        init_kernel_size=3,
        neg_slope=0.2,
        nconvs=2,
        bn=True,
    ):
        super().__init__()
        self.conv = ConvBlock(
            in_channels,
            out_channels,
            nconvs,
            kernel_size,
            init_kernel_size,
            bn,
        )
        self.out = nn.Conv2d(out_channels, out_channels, 2, 2)
        return

    def forward(self, x):
        x = self.conv(x)
        return self.out(x)


class EConv(nn.Module):
    def name(self):
        return "End Convolution"

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        bn=True,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size),
            nn.ReLU(True),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.Sigmoid(),
        )
        return

    def forward(self, x):
        return self.conv(x)


class DiscTest(nn.Module):
    def name(self):
        return "Discriminator Testing"

    def __init__(
        self,
        imgshape,
        n_channels,
        n_layers=4,
        init_kernel_size=3,
        kernel_size=3,
        nconvs=2,
        nrepititions=2,
        neg_slope=0.2,
        bn=True,
    ):
        super().__init__()
        self.pad = torchvision.transforms.Pad(0)
        self.imgshape = imgshape
        imgshapestr = str(imgshape[0]) + "x" + str(imgshape[1])
        self.n_layers = n_layers
        self.nconvs = nconvs
        self.kernel_size = kernel_size
        self.init_kernel_size = init_kernel_size
        self.disc = nn.Sequential(
            DConv(
                n_channels,
                64,
                kernel_size,
                init_kernel_size,
                neg_slope,
                nconvs,
                bn,
            ),
        )
        for k in range(n_layers - 1):
            self.disc.append(
                DConv(
                    64 * 2**k,
                    64 * 2 ** (k + 1),
                    kernel_size,
                    neg_slope=neg_slope,
                    nconvs=nconvs,
                    bn=bn,
                )
            )
        return

    def forward(self, x):
        x = self.pad(x)
        return self.disc(x)


class Discriminator(nn.Module):
    def name(self):
        return "Discriminator"

    def __init__(
        self,
        imgshape,
        n_channels,
        n_layers=4,
        init_kernel_size=3,
        kernel_size=3,
        nconvs=2,
        nrepititions=2,
        neg_slope=0.2,
        bn=True,
    ):
        super().__init__()
        self.pad = torchvision.transforms.Pad(100)
        self.imgshape = imgshape
        imgshapestr = str(imgshape[0]) + "x" + str(imgshape[1])
        self.n_layers = n_layers
        self.nconvs = nconvs
        self.kernel_size = kernel_size
        self.init_kernel_size = init_kernel_size
        self.disc = nn.Sequential(
            DConv(
                n_channels,
                64,
                kernel_size,
                init_kernel_size,
                neg_slope,
                nconvs,
                bn,
            ),
        )
        for k in range(n_layers - 1):
            self.disc.append(
                DConv(
                    64 * 2**k,
                    64 * 2 ** (k + 1),
                    kernel_size,
                    neg_slope=neg_slope,
                    nconvs=nconvs,
                    bn=bn,
                )
            )
        k_size = {}
        k_size["600x800"] = {
            4: (33, 46),
            3: (76, 103),
            2: (157, 211),
        }
        k_size["400x500"] = {4: (100, 100)}
        k_size["150x200"] = {4: (8, 11)}
        k_size["300x350"] = {4: (15, 18)}
        k_size["400x450"] = {4: (21, 24)}
        self.end = EConv(64 * 2 ** (k + 1), 1, k_size[imgshapestr][n_layers], bn)
        return

    def forward(self, x):
        x = self.pad(x)
        x = self.disc(x)
        return self.end(x)

    def calcimgshape(self):
        imgshape = np.int64(
            (
                self.imgshape
                - (self.init_kernel_size - 1)
                - (self.nconvs - 1) * (self.kernel_size - 1)
            )
            / 2
        )
        for _ in range(self.n_layers - 1):
            imgshape = np.int64(
                (imgshape - 2 * self.nconvs * (self.kernel_size - 1)) / 2
            )
        return tuple(imgshape)


class UNetOld(nn.Module):
    def name(self):
        return "UNetOld"

    def __init__(self, n_channels, n_classes, init_kernel_size=3, kernel_size=3):
        super(UNetOld, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.init_kernel_size = init_kernel_size
        self.down1 = Down(n_channels, 64, init_kernel_size)
        self.down2 = Down(64, 128, kernel_size)
        self.down3 = Down(128, 256, kernel_size)
        self.down4 = Down(256, 512, kernel_size)
        self.mid = ConvBlockOld(512, 1024, kernel_size)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        d, x1 = self.down1(x)
        d, x2 = self.down2(d)
        d, x3 = self.down3(d)
        d, x4 = self.down4(d)
        mid = self.mid(d)
        u = self.up1(mid, x4)
        u = self.up2(u, x3)
        u = self.up3(u, x2)
        u = self.up4(u, x1)
        return self.outc(u)


class UNet_vgg(nn.Module):
    def name(self):
        return "UNet with vgg"

    def __init__(self, n_channels, n_classes):
        super(UNet_vgg, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.encoder = torchvision.models.vgg16(True).features
        for i in [0, 2, 5, 7, 10, 12, 14]:
            self.encoder[i].padding = (0, 0)
            self.encoder[i].requires_grad = False
        self.down1 = nn.Sequential(
            self.encoder[0], self.relu, self.encoder[2], self.relu
        )
        self.down2 = nn.Sequential(
            self.encoder[5], self.relu, self.encoder[7], self.relu
        )
        self.down3 = nn.Sequential(
            self.encoder[10],
            self.relu,
            self.encoder[12],
            self.relu,
            self.encoder[14],
            self.relu,
        )
        self.middle = ConvBlockOld(256, 512)
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        conv1 = self.down1(x)
        conv2 = self.down2(self.pool(conv1))
        conv3 = self.down3(self.pool(conv2))
        mid = self.middle(self.pool(conv3))
        x = self.up1(mid, conv3)
        x = self.up2(x, conv2)
        x = self.up3(x, conv1)
        return self.outc(x)


class SmallNet(nn.Module):
    def name(self):
        return "SmallNet"

    def __init__(self, n_channels, n_classes):
        super(SmallNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.down1 = Down(n_channels, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.mid = ConvBlockOld(256, 512)
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        d, x1 = self.down1(x)
        d, x2 = self.down2(d)
        d, x3 = self.down3(d)
        mid = self.mid(d)
        u = self.up1(mid, x3)
        u = self.up2(u, x2)
        u = self.up3(u, x1)
        return self.outc(u)


class XSmallNet(nn.Module):
    def name(self):
        return "XSmallNet"

    def __init__(self, n_channels, n_classes):
        super(XSmallNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.down1 = Down(n_channels, 64)
        self.down2 = Down(64, 128)
        self.mid = ConvBlockOld(128, 256)
        self.up1 = Up(256, 128)
        self.up2 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        d, x1 = self.down1(x)
        d, x2 = self.down2(d)
        mid = self.mid(d)
        u = self.up1(mid, x2)
        u = self.up2(u, x1)
        return self.outc(u)
