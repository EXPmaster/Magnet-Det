""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class ConcatLayer(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.data_ebd = nn.Linear(4, 128)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_dim + 128, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 4)
        )

    def forward(self, img, data):
        data = self.data_ebd(data)
        img = self.pool(img).view(data.shape[0], -1)
        concat_x = torch.cat((img, data), 1)
        return self.fc(concat_x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.outt = ConcatLayer(1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, self.n_classes)

    def forward(self, img, data):
        x1 = self.inc(img)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        out_txt_feature = self.outt(x5.detach(), data)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return (
            torch.sigmoid(logits).view(img.shape[0], img.shape[-2], img.shape[-1]),
            torch.tanh(out_txt_feature)
        )