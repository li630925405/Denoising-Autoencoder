import torch.nn as nn
import torch

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel, padding=padding)
        nn.init.kaiming_uniform_(self.conv.weight)
        self.act = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        x = self.bn(x)

        return x


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.dense = nn.Linear(1953, 1953)
        self.conv_block1 = ConvBlock(1, 4, 1)
        # self.conv_block2 = ConvBlock(4, 16, 3)
        # self.conv_block3 = ConvBlock(16, 64, 3)
        # self.conv_block4 = ConvBlock(64, 128, 3)
        # self.conv_block5 = ConvBlock(128, 256, 3)

    def forward(self, x):
        # print(x.shape)[16, 513, 1953] -> [8208 x 1953]
        # x = self.dense(x)
        x = self.conv_block1(x)
        # x = self.conv_block2(x)
        # x = self.pool(x)
        # print(x.shape)
        # x = self.conv_block3(x)
        # x = self.conv_block4(x)
        # x = self.pool(x)
        # print(x.shape)
        # x = self.conv_block5(x)
        x = self.pool(x)

        return x

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        # self.deconv1 = nn.ConvTranspose2d(256, 256, 3, stride=2)
        # nn.init.kaiming_uniform_(self.deconv1.weight)
        # self.deconv2 = nn.ConvTranspose2d(128, 128, 3, stride=2)
        # nn.init.kaiming_uniform_(self.deconv2.weight)
        self.deconv3 = nn.ConvTranspose2d(64, 64, 3, stride=2)
        nn.init.kaiming_uniform_(self.deconv3.weight)
        self.deconv4 = nn.ConvTranspose2d(16, 16, 3, stride=2)
        nn.init.kaiming_uniform_(self.deconv4.weight)
        self.deconv5 = nn.ConvTranspose2d(1, 1, 3, stride=2)
        nn.init.kaiming_uniform_(self.deconv5.weight)

        # self.conv_block1 = ConvBlock(256, 128, 3)
        # self.conv_block2 = ConvBlock(128, 64, 3)
        self.conv_block3 = ConvBlock(64, 16, 3)
        self.conv_block4 = ConvBlock(16, 4, 3)
        self.conv_block5 = ConvBlock(4, 1, 1)

        self.interp3 = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(64, 64, kernel_size=1),
            )
        self.interp4 = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(16, 16, kernel_size=1),
            )
        self.interp5 = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(4, 4, kernel_size=1),
            )
    
    def forward(self, x):
        # x = self.interp(x)
        # print(x.shape)
        # x = self.deconv1(x)
        # x = self.conv_block1(x)
        # print(x.shape)
        # x = self.interp(x)
        # x = self.deconv2(x)
        # x = self.conv_block2(x)
        # print(x.shape)

        # x = self.interp3(x)
        # x = self.deconv3(x)
        # x = self.conv_block3(x)

        # x = self.interp4(x)
        # x = self.deconv4(x)
        # x = self.conv_block4(x)

        x = self.interp5(x)
        # x = self.deconv5(x)
        x = self.conv_block5(x)

        return x


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        # (1954->1945) -> (1953->1941)?
        # torch.Size([16, 513, 1953])
        # torch.Size([16, 1, 255, 975])
        # torch.Size([16, 1, 126, 486])
        # torch.Size([16, 1, 251, 971])
        # torch.Size([16, 1, 501, 1941]) 

        return x