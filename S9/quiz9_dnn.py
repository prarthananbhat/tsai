
import torch.nn as nn
import torch.nn.functional as F


# x1 = Input
# x2 = Conv(x1)
# x3 = Conv(x1 + x2)
# x4 = MaxPooling(x1 + x2 + x3)
# x5 = Conv(x4)
# x6 = Conv(x4 + x5)
# x7 = Conv(x4 + x5 + x6)
# x8 = MaxPooling(x5 + x6 + x7)
# x9 = Conv(x8)
# x10 = Conv (x8 + x9)
# x11 = Conv (x8 + x9 + x10)
# x12 = GAP(x11)
# x13 = FC(x12)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # FIRST LAYER
        self.conv_block_1 = nn.Sequential(nn.Conv2d(in_channels=3,
                                                    out_channels=64,
                                                    kernel_size=3,
                                                    stride=1,
                                                    padding=1,
                                                    bias=False
                                                    ),  # output size = 26 Receptive field = 3
                                          nn.ReLU(),
                                          nn.BatchNorm2d(64),
                                          nn.Dropout(0.10)
                                          )
        self.conv_block_2 = nn.Sequential(nn.Conv2d(in_channels=64,
                                                    out_channels=64,
                                                    kernel_size=3,
                                                    stride=1,
                                                    padding=1,
                                                    bias=False
                                                    ),  # output size = 26 Receptive field = 3
                                          nn.ReLU(),
                                          nn.BatchNorm2d(64),
                                          nn.Dropout(0.10)
                                          )
        self.conv_block_3 = nn.Sequential(nn.Conv2d(in_channels=64,
                                                    out_channels=64,
                                                    kernel_size=3,
                                                    stride=1,
                                                    padding=1,
                                                    bias=False
                                                    ),  # Output size = 22 Receptive field = 7
                                          nn.ReLU(),
                                          nn.BatchNorm2d(64)
                                          )
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # #Output size = 11 Receptive field = 8

        # SECOND LAYER
        self.conv_block_4 = nn.Sequential(nn.Conv2d(in_channels=64,
                                                    out_channels=64,
                                                    kernel_size=3,
                                                    stride=1,
                                                    padding=1,
                                                    bias=False
                                                    ),  # output size = 26 Receptive field = 3
                                          nn.ReLU(),
                                          nn.BatchNorm2d(64),
                                          nn.Dropout(0.10)
                                          )
        self.conv_block_5 = nn.Sequential(nn.Conv2d(in_channels=64,
                                                    out_channels=64,
                                                    kernel_size=3,
                                                    stride=1,
                                                    padding=1,
                                                    bias=False
                                                    ),  # output size = 26 Receptive field = 3
                                          nn.ReLU(),
                                          nn.BatchNorm2d(64),
                                          nn.Dropout(0.10)
                                          )
        self.conv_block_6 = nn.Sequential(nn.Conv2d(in_channels=64,
                                                    out_channels=64,
                                                    kernel_size=3,
                                                    stride=1,
                                                    padding=1,
                                                    bias=False
                                                    ),  # Output size = 22 Receptive field = 7
                                          nn.ReLU(),
                                          nn.BatchNorm2d(64)
                                          )
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # #Output size = 11 Receptive field = 8
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        # x1 = Input
        x1 = self.conv_block_1(x)
        # print(x1.size())

        # x2 = Conv(x1)
        x2 = self.conv_block_2(x1)
        # print(x2.size())

        # x3 = Conv(x1 + x2)
        # x3 = self.conv_block_3(x3temp)
        x3 = self.conv_block_3(x1 + x2)
        # print(x3.size())

        # x4 = MaxPooling(x1 + x2 + x3)
        # x4 = self.pool_1(x4temp)
        x4 = self.pool_1(x1 + x2 + x3)
        # print(x4.size())

        # x5 = Conv(x4)
        x5 = self.conv_block_4(x4)
        # print(x5.size())

        # x6 = Conv(x4 + x5)
        # x6 = self.conv_block_5(x6temp)
        x6 = self.conv_block_5(x4 + x5)
        # print(x6.size())

        # x7 = Conv(x4 + x5 + x6)
        x7 = self.conv_block_6(x4 + x5 + x6)
        # print(x7.size())

        # x8 = MaxPooling(x5 + x6 + x7)
        x8 = self.pool_2(x5 + x6 + x7)
        # print(x8.size())

        # x9 = Conv(x8)
        x9 = self.conv_block_4(x8)
        # print(x9.size())

        # x10 = Conv (x8 + x9)
        # x10 = self.conv_block_5(x10temp)
        x10 = self.conv_block_5(x8 + x9)
        # print(x10.size())

        # x11 = Conv (x8 + x9 + x10)
        x11 = self.conv_block_6(x8 + x9 + x10)
        # print(x11.size())

        # x12 = GAP(x11)
        x12 = self.gap(x11)
        # print(x12.size())
        x12 = x12.view(-1, 64)
        # x13 = FC(x12)
        x13 = self.fc(x12)

        return x13


