import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout2d = nn.Dropout2d(0.1)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.dropout2d(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CNNWithResidual(nn.Module):
    def __init__(self, input_channels=3, num_classes=100):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, 2, 1)
        self.bn1 = nn.BatchNorm2d(32)

        self.res1 = ResidualBlock(32, 32, 1)
        self.res2 = ResidualBlock(32, 64, 2)
        self.res3 = ResidualBlock(64, 128, 2)
        self.res4 = ResidualBlock(128, 256, 2)
        self.res5 = ResidualBlock(256, 256, 1)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
