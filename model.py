import torch
import torch.nn as nn




class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, padding, kernel_size, leaky_relu_factor):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        
        
        self.proj = nn.Conv2d(self.in_channels, out_channels=out_channels, kernel_size=1, padding=padding, stride=stride) # handles the missmatch betweem in/out #channels

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=stride, padding=padding, kernel_size=kernel_size)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.lrelu = nn.LeakyReLU(negative_slope=leaky_relu_factor)


    def forward(self, x):
        identity = x
        if self.in_channels != self.out_channels:
            identity = self.proj(x)

        o = self.conv(x)
        o = self.bn(o)
        o = self.lrelu(o)

        return o+identity




class SiameseRIZZNet(nn.Module):
    def __init__(self, embedding_size: int,  leaky_relu_factor: float):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=1, out_channels=32, stride=1, padding='same', kernel_size=3)
        self.resblock1 = ResBlock(in_channels=32, out_channels=64, stride=1, padding='same', kernel_size=3, leaky_relu_factor=leaky_relu_factor)
        self.resblock2 = ResBlock(in_channels=64, out_channels=64, stride=1, padding='same', kernel_size=3, leaky_relu_factor=leaky_relu_factor)
        self.pool1     = nn.MaxPool2d(kernel_size=2, stride=2) 
        # self.resblock3 = ResBlock(in_channels=64, out_channels=64, stride=1, padding='same', kernel_size=3)
        self.resblock4 = ResBlock(in_channels=64, out_channels=128, stride=1, padding='same', kernel_size=3, leaky_relu_factor=leaky_relu_factor)
        self.pool2     = nn.MaxPool2d(kernel_size=2, stride=2) 
        self.resblock5 = ResBlock(in_channels=128, out_channels=128, stride=1, padding='same', kernel_size=3, leaky_relu_factor=leaky_relu_factor)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear_1 = nn.Linear(in_features=128, out_features=embedding_size)
        self.bn1 = nn.BatchNorm1d(num_features=embedding_size)
        self.lrelu_1 = nn.LeakyReLU(negative_slope=leaky_relu_factor)
        self.linear_2 = nn.Linear(in_features=embedding_size, out_features=embedding_size)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.pool1(x)
        # x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.pool2(x)
        x = self.resblock5(x)
        x = self.gap(x) # Bx128x1x1
        B, D, _, _ = x.shape
        x = x.squeeze(-1)
        x = x.squeeze(-1)

        x = self.linear_1(x)
        x = self.bn1(x)
        x = self.lrelu_1(x)
        x = self.linear_2(x)

        return x


