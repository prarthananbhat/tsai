import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(p=0.09)
        )  # output_size = 30

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(p=0.09)
        )  # output_size = 28

        # CONVOLUTION BLOCK 2
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(p=0.09)
        )  # output_size = 26

        # CONVOLUTION BLOCK 3
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=230, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(230),
            nn.ReLU(),
            nn.Dropout(p=0.03)
        )  # output_size = 24

        # TRANSITION BLOCK 1
        self.convblock1_1 = nn.Sequential(
            nn.Conv2d(in_channels=230, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU()
        )  # output_size = 24
        self.pool1 = nn.MaxPool2d(2, 2)  # output_size = 12

        # CONVOLUTION BLOCK 5
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(p=0.03)
        )  # output_size = 10

        # CONVOLUTION BLOCK 6
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(p=0.03)
        )  # output_size = 8

        # CONVOLUTION BLOCK 7
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(p=0.03)
        )  # output_size = 6

        # CONVOLUTION BLOCK 8
        self.convblock_depthwise = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), groups=256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(p=0.03)
        )

        # TRANSITION BLOCK 2
        self.convblock1_2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU()
        )  # output_size = 24
        self.pool2 = nn.MaxPool2d(2, 2)  # output_size = 12

        # CONVOLUTION BLOCK 8
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(p=0.03)
        )  # output_size = 4

        # DILATED CONVOLUTION BLOCK
        self.convblock_dilated = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=2, dilation=(2, 2), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # CONVOLUTION BLOCK 9
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(p=0.03)
        )  # output_size = 4

        # TRANSITION BLOCK 3
        self.convblock1_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU()
        )  # output_size = 24
        self.pool3 = nn.MaxPool2d(2, 2)  # output_size = 12

        # CONVOLUTION BLOCK 10
        self.convblock10 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(p=0.03)
        )  # output_size = 4

        # CONVOLUTION BLOCK 11
        self.convblock11 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.Dropout(p=0.03)
        )  # output_size = 4

        # OUTPUT BLOCK
        self.convblock1_4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Dropout(p=0.03)
        )  # output_size = 4

        self.avg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
        )  # output_size = 1
        self.convblock_out = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        )  # output_size = 1

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.convblock1_1(x)
        x = self.pool1(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.convblock_depthwise(x)
        x = self.convblock1_2(x)
        x = self.pool2(x)
        x = self.convblock8(x)
        x = self.convblock_dilated(x)
        x = self.convblock9(x)
        x = self.convblock1_3(x)
        x = self.pool3(x)
        x = self.convblock10(x)
        x = self.convblock11(x)
        x = self.convblock1_4(x)
        x = self.avg(x)
        x = self.convblock_out(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)