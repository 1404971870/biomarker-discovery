import torch.nn as nn
import torch.nn.functional as F
import torch


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           U-NET
##############################


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# 尺寸变小4倍
class simple_UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(simple_UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 4, 0, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetReserve(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetReserve, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 5, 1, 2, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x

class simple_UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(simple_UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 4, 0, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x

class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet, self).__init__()


        self.down1 = UNetDown(in_channels, 64, normalize=False)# (3,2048)--(64,1024)
        self.down2 = UNetDown(64, 128)        # (64,1024)--(128,512)
        self.down3 = UNetDown(128, 256)        # (128,512)--(256,256)
        self.down4 = UNetDown(256, 512, dropout=0.5)         # (256,256)--(512,128)
        self.down5 = UNetDown(512, 512, dropout=0.5)        # (512,128)--(512,64)
        self.down6 = UNetDown(512, 512, dropout=0.5)        # (512,64)--(512,32)
        self.down7 = UNetDown(512, 512, dropout=0.5)        # (512,32)--(512,16)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)        # (512,16)--(512,8)

        self.simdown1 = simple_UNetDown(in_channels, 128)
        self.simdown2 = simple_UNetDown(128, 256)
        self.simdown3 = simple_UNetDown(256, 512)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.simup1 = simple_UNetUp(512,256)

        self.re1 = UNetReserve(1024,512)
        self.re2 = UNetReserve(512,256)
        self.re3 = UNetReserve(256,128)

        self.simre1 = UNetReserve(512,256)
        self.simre2 = UNetReserve(256,128)

        self.final = nn.Sequential(
            # 首先尺寸变为2倍，由128变为256
            nn.Upsample(scale_factor=2),
            # 然后在左边和上边padding1，尺寸变为257
            nn.ZeroPad2d((1, 0, 1, 0)),
            # 卷积后尺寸不变，变通道
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        # # 如果输入是2048，到u4的时候已经是128了
        # re1 = self.re1(u4)
        # re2 = self.re2(re1)
        # re3 = self.re3(re2)

        # d1 = self.simdown1(x)
        # d2 = self.simdown2(d1)
        # d3 = self.simdown3(d2)
        # u1 = self.simup1(d3,d2)
        # re1 = self.simre1(u1)
        # re2 = self.simre2(re1)


        return self.final(u7)

##############################
#        Discriminator
##############################


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            # 尺寸减半
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        # 将由high_wsi生成来的msi与原始的low_wsi在通道维相加作为输入
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)
