import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.mxp1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.mxp2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv3 = nn.ConvTranspose2d(64, 64, kernel_size=7, stride=2, padding=0, bias=False)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.mxp1(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.mxp2(out)
        out = F.relu(self.conv3(out))
        print(out.size)
        return out

def mytest():
    model = MyNet()
    summary(model, input_size=(3, 128, 128))

mytest()
