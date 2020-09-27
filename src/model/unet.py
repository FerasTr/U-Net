from model import model_layers as layers
from torch import nn

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.conv1 = layers.DoubleConv(n_channels, 64)
        self.down1 = layers.Down(64, 128)
        self.down2 = layers.Down(128, 256)
        self.down3 = layers.Down(256, 512)
        self.down4 = layers.Down(512, 1024)
        self.up1 = layers.Up(1024, 512)
        self.up2 = layers.Up(512, 256)
        self.up3 = layers.Up(256, 128)
        self.up4 = layers.Up(128, 64)
        self.outc = layers.OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x4, x5)
        x = self.up2(x3, x)
        x = self.up3(x2, x)
        x = self.up4(x1, x)
        logits = self.outc(x)
        return logits
