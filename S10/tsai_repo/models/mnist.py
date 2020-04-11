import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # FIRST LAYER
        self.conv_block_1 = nn.Sequential(nn.Conv2d(in_channels=1,
                                                    out_channels=4,
                                                    kernel_size=3,
                                                    stride=1,
                                                    padding=0,
                                                    bias=True
                                                    ),  # output size = 26 Receptive field = 3
                                          nn.ReLU()
                                          )
        # CONVOLUTION BLOCK
        self.conv_block_2 = nn.Sequential(nn.Conv2d(in_channels=4,
                                                    out_channels=8,
                                                    kernel_size=3,
                                                    stride=1,
                                                    padding=0,
                                                    bias=True),  # Output Size = 24, Receptive Field = 5
                                          nn.ReLU()
                                          )
        self.conv_block_3 = nn.Sequential(nn.Conv2d(in_channels=8,
                                                    out_channels=16,
                                                    kernel_size=3,
                                                    stride=1,
                                                    padding=0,
                                                    bias=True),  # Output size = 22 Receptive field = 7
                                          nn.ReLU()
                                          )

        # TRANSITION BLOCK
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # #Output size = 11 Receptive field = 8
        self.conv_block_4 = nn.Sequential(nn.Conv2d(in_channels=16,
                                                    out_channels=4,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0,
                                                    bias=True),  # Output size = 11 Receptive field = 10
                                          nn.ReLU()
                                          )

        # CONVOLUTION BLOCK
        self.conv_block_5 = nn.Sequential(nn.Conv2d(in_channels=4,
                                                    out_channels=8,
                                                    kernel_size=3,
                                                    stride=1,
                                                    padding=0,
                                                    bias=True),  # Output size = 9 Receptive field = 14
                                          nn.ReLU()
                                          )
        self.conv_block_6 = nn.Sequential(nn.Conv2d(in_channels=8,
                                                    out_channels=16,
                                                    kernel_size=3,
                                                    stride=1,
                                                    padding=0,
                                                    bias=True),  # Output size = 7 Receptive field = 18
                                          nn.ReLU()
                                          )

        # OUTPUT LAYER
        self.conv_block_7 = nn.Sequential(nn.Conv2d(in_channels=16,
                                                    out_channels=10,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0,
                                                    bias=True),  # Output size = 7 Receptive field = 20
                                          nn.ReLU()
                                          )
        self.conv_block_8 = nn.Sequential(nn.Conv2d(in_channels=10,
                                                    out_channels=10,
                                                    kernel_size=7,
                                                    stride=1,
                                                    padding=0,
                                                    bias=True)  # Output size = 1 Receptive field = 26
                                          )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.pool_1(x)
        x = self.conv_block_4(x)
        x = self.conv_block_5(x)
        x = self.conv_block_6(x)
        x = self.conv_block_7(x)
        x = self.conv_block_8(x)
        x = x.view(-1, 10)
        final_x = F.log_softmax(x, dim=-1)
        return final_x
